"""
bss_taint_tracker_arm64.py: find AArch64 functions that walk a [start, end) range read
from an adjacent pair of pointer-sized BSS globals.

ARM64 counterpart of bss_taint_tracker.py (same BSSSymbol input, same report dict).
Primary use: locating decrypt / checksum loops over initrd_start / initrd_end style
globals in stripped arm64 kernels and firmware.

AArch64 differences from the x86 tracker:
  - globals are addressed with `adrp xB, page` + `ldr xD, [xB, #lo12]` (or `ldp` for
    both halves of a pair at once), not `mov reg, [abs]` / `[rip+disp]`, so a small
    constant pass follows adrp / adr / add #imm / mov to the load address
  - loop bounds are `cmp xA, xB` + `b.cond`, or `subs` + `b.cond`

Algorithm (per function CFG from cfg_arm64):
  1. BSS loads: a 64-bit ldr / ldp whose address is a known BSS symbol (size >= 8)
  2. Pairs: two symbols 8 bytes apart loaded in the same function (optimised code
     reloads the end pointer in the loop head, a different block from the start)
  3. Forward taint over the CFG (union at merges): loaded registers carry their
     symbol through mov / add / sub and through sp / x29 stack slots (-O0 spills)
  4. Loops (back edges): report when a tainted register is used as a memory base and
     a tainted register is compared before a conditional branch in the loop

Usage:
    from ablation.analyzers.bss_taint_tracker_arm64 import (
        ARM64BssTaintTracker, build_function_arm64, bss_symbols_from_elf)

    syms = bss_symbols_from_elf('/path/to/vmlinux')        # symtab, or 8-byte slots
    tracker = ARM64BssTaintTracker(syms)
    report = tracker.analyze(build_function_arm64(data, fn_va, fn_size, load_base))
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .bss_taint_tracker import BSSSymbol, POINTER_SIZE
from .cfg_arm64 import CFG, build_cfg
from .insn_arm64 import Insn, make_insn
from .isa_arm64 import CALLER_SAVED, ISA

_COMPARES = frozenset({"cmp", "cmn", "subs", "adds", "ccmp", "ccmn", "tst"})


def build_function_arm64(data: bytes, fn_va: int, fn_size: int, load_base: int = 0) -> CFG:
    """CFG for fn_va..fn_va+fn_size, reading data at fn_va - load_base (flat image)."""
    import capstone
    arch = getattr(capstone, "CS_ARCH_AARCH64", None) or getattr(capstone, "CS_ARCH_ARM64")
    md = capstone.Cs(arch, capstone.CS_MODE_ARM)
    md.skipdata = True
    off = fn_va - load_base
    insns = [make_insn(i.address, i.mnemonic, i.op_str, size=i.size)
             for i in md.disasm(data[off: off + fn_size], fn_va)]
    return build_cfg(insns, entry=fn_va)


def bss_symbols_from_elf(path: str) -> List[BSSSymbol]:
    """BSS objects from .symtab; for stripped images, every 8-byte slot of .bss."""
    import lief
    binary = lief.parse(path)
    bss = binary.get_section(".bss") if binary else None
    if bss is None:
        return []
    lo, hi = int(bss.virtual_address), int(bss.virtual_address) + int(bss.size)
    out = [BSSSymbol(int(s.value), int(s.size), s.name) for s in binary.symbols
           if s.name and lo <= int(s.value) < hi and int(s.size) >= POINTER_SIZE]
    if not out:
        out = [BSSSymbol(a, POINTER_SIZE, f"bss_{a:x}") for a in range(lo, hi - 7, POINTER_SIZE)]
    return sorted(out, key=lambda s: s.addr)


class ARM64BssTaintTracker:
    """Adjacent BSS pointer pair -> loop that walks [start, end). See module doc."""

    def __init__(self, bss_symbols: List[BSSSymbol]):
        self._syms = sorted(bss_symbols, key=lambda s: s.addr)
        self._by_addr = {s.addr: s for s in self._syms if s.size >= POINTER_SIZE}

    def analyze(self, fn: CFG, name: str = "") -> Optional[dict]:
        loads = self._collect_bss_loads(fn)
        if not loads:
            return None
        pairs = self._find_pairs(loads)
        if not pairs:
            return None
        taint = self._forward_taint(fn, loads)
        loops = self._detect_loops(fn, taint, pairs)
        if not loops:
            return None
        return {
            "fn_start": fn.entry,
            "fn_name": name,
            "bss_pairs": [(s.addr, e.addr) for _, _, s, e in pairs],
            "loops": loops,
        }

    # ---- 1. BSS loads (constant pass per block) -----------------------------

    def _load_targets(self, insn: Insn, consts: Dict[str, int]) -> List[Tuple[str, int]]:
        """[(dest_reg, address)] for a 64-bit ldr / ldp off a constant base."""
        m, mem = insn.mnemonic, insn.first_mem()
        if mem is None or mem.index is not None or mem.base not in consts:
            return []
        addr = consts[mem.base] + (0 if mem.writeback == "post" else mem.offset)
        r0, r1 = insn.reg(0), insn.reg(1)          # destinations (may equal the base)
        if m in ("ldr", "ldur") and r0 is not None and r0.bits == 64:
            return [(r0.name, addr)]
        if m == "ldp" and r0 is not None and r1 is not None and r0.bits == 64:
            return [(r0.name, addr), (r1.name, addr + 8)]
        return []

    @staticmethod
    def _step_consts(insn: Insn, consts: Dict[str, int]) -> None:
        m, rd = insn.mnemonic, insn.reg(0)
        if m in ISA.call_mnems:
            for r in CALLER_SAVED:
                consts.pop(r, None)
            return
        if rd is None:
            return
        if m in ("adrp", "adr") and insn.imm(1) is not None:
            consts[rd.name] = insn.imm(1)
        elif m in ("add", "sub") and insn.reg(1) is not None and insn.imm(2) is not None \
                and insn.reg(1).name in consts:
            d = insn.imm(2) if m == "add" else -insn.imm(2)
            consts[rd.name] = consts[insn.reg(1).name] + d
        elif m in ("mov", "movz") and insn.imm(1) is not None:
            consts[rd.name] = insn.imm(1)
        elif not m.startswith(("st", "cmp", "cmn", "tst", "b", "cb", "tb", "ret")):
            consts.pop(rd.name, None)
            if m == "ldp" and insn.reg(1) is not None:
                consts.pop(insn.reg(1).name, None)

    def _consts_at(self, fn: CFG) -> Dict[int, Dict[str, int]]:
        """Register constants before each instruction (CFG dataflow, intersection join)."""
        at: Dict[int, Dict[str, int]] = {}
        entry: Dict[int, Dict[str, int]] = {fn.entry: {}}
        wq = deque([fn.entry])
        visits: Dict[int, int] = defaultdict(int)
        while wq:
            b = wq.popleft()
            if b not in fn.blocks or visits[b] > 8:
                continue
            visits[b] += 1
            consts = dict(entry[b])
            for insn in fn.blocks[b].insns:
                at[insn.address] = dict(consts)
                self._step_consts(insn, consts)
            for s in fn.blocks[b].succs:
                new = consts if s not in entry else {r: v for r, v in entry[s].items()
                                                     if consts.get(r) == v}
                if s not in entry or new != entry[s]:
                    entry[s] = dict(new)
                    wq.append(s)
        return at

    def _collect_bss_loads(self, fn: CFG) -> Dict[int, List[Tuple[int, BSSSymbol, str]]]:
        loads: Dict[int, List[Tuple[int, BSSSymbol, str]]] = defaultdict(list)
        consts_at = self._consts_at(fn)
        for bstart, bb in fn.blocks.items():
            for insn in bb.insns:
                for reg, addr in self._load_targets(insn, consts_at.get(insn.address, {})):
                    sym = self._by_addr.get(addr)
                    if sym is not None:
                        loads[bstart].append((insn.address, sym, reg))
        return loads

    # ---- 2. pairs -----------------------------------------------------------

    @staticmethod
    def _find_pairs(loads):
        """Adjacent (addr, addr+8) symbols loaded anywhere in the function."""
        first: Dict[int, Tuple[int, int, BSSSymbol]] = {}
        for bstart, lst in loads.items():
            for va, sym, _ in lst:
                if sym.addr not in first or va < first[sym.addr][1]:
                    first[sym.addr] = (bstart, va, sym)
        pairs = []
        for addr, (bstart, va, sym) in sorted(first.items()):
            nxt = first.get(addr + POINTER_SIZE)
            if nxt is not None:
                pairs.append((bstart, va, sym, nxt[2]))
        return pairs

    # ---- 3. forward taint ---------------------------------------------------

    def _forward_taint(self, fn: CFG, loads) -> Dict[int, Dict[int, Dict[str, Set]]]:
        taint: Dict[int, Dict[int, Dict[str, Set]]] = defaultdict(dict)
        in_state: Dict[int, Dict[str, Set]] = {b: defaultdict(set) for b in fn.blocks}
        wq = deque([fn.entry] if fn.entry in fn.blocks else list(fn.blocks))
        queued = set(wq)
        while wq:
            bstart = wq.popleft()
            queued.discard(bstart)
            bb = fn.blocks[bstart]
            rt: Dict = defaultdict(set, {r: set(v) for r, v in in_state[bstart].items()})
            for insn in bb.insns:
                taint[bstart][insn.address] = {r: set(v) for r, v in rt.items() if v}
                self._propagate(insn, rt)
                for va, sym, reg in loads.get(bstart, []):
                    if va == insn.address:
                        rt[reg] = rt[reg] | {sym.addr}     # BSSSymbol is unhashable
            for succ in bb.succs:
                grew = False
                for r, v in rt.items():
                    if not v <= in_state[succ][r]:
                        in_state[succ][r] |= v
                        grew = True
                if grew and succ not in queued:
                    wq.append(succ)
                    queued.add(succ)
        return taint

    @staticmethod
    def _slot(insn: Insn):
        mem = insn.first_mem()
        if mem is None or mem.base not in ("sp", "x29") or mem.index is not None or mem.writeback:
            return None
        return ("slot", mem.base, mem.offset)

    def _propagate(self, insn: Insn, rt: Dict) -> None:
        m, rd = insn.mnemonic, insn.reg(0)
        if m in ISA.call_mnems:
            for r in CALLER_SAVED:
                rt.pop(r, None)
            return
        if m.startswith("st"):
            slot = self._slot(insn)
            if slot is not None and rd is not None:
                rt[slot] = set(rt.get(rd.name, set())) if rd.bits == 64 else set()
            return
        if rd is None or m.startswith(("cmp", "cmn", "tst", "b.", "cb", "tb")) or m in ("b", "br", "ret"):
            return
        if m in ("ldr", "ldur") and self._slot(insn) is not None and rd.bits == 64:
            rt[rd.name] = set(rt.get(self._slot(insn), set()))       # spill reload
            return
        srcs = [r.name for r in insn.regs()[1:]]
        if m in ("mov", "add", "sub", "adds", "subs", "orr") and srcs:
            rt[rd.name] = set().union(*(rt.get(x, set()) for x in srcs))
        else:
            rt[rd.name] = set()
            if m == "ldp" and insn.reg(1) is not None:
                rt[insn.reg(1).name] = set()

    # ---- 4. loops -----------------------------------------------------------

    def _detect_loops(self, fn: CFG, taint, pairs) -> List[dict]:
        back_edges = [(tail, head) for tail, bb in fn.blocks.items() for head in bb.succs if head <= tail]
        pair_syms = [(s, e) for _, _, s, e in pairs]
        reports = []
        for tail, head in back_edges:
            blocks = sorted(b for b in fn.blocks if head <= b <= tail)
            mem_hit, cmp_hit, involved = False, False, set()
            for b in blocks:
                insns = fn.blocks[b].insns
                for idx, insn in enumerate(insns):
                    ts = taint[b].get(insn.address, {})
                    mem = insn.first_mem()
                    if mem is not None and ts.get(mem.base):
                        mem_hit = True
                        involved |= ts[mem.base]
                    if insn.mnemonic in _COMPARES:
                        nxt = insns[idx + 1].mnemonic if idx + 1 < len(insns) else ""
                        if nxt.startswith("b.") or nxt in ("csel", "cset", "cinc"):
                            regs = insn.regs()[1:] if insn.mnemonic in ("subs", "adds") else insn.regs()
                            for r in regs:
                                if ts.get(r.name):
                                    cmp_hit = True
                                    involved |= ts[r.name]
            if not (mem_hit and cmp_hit):
                continue
            matched = [(s.addr, e.addr) for s, e in pair_syms if s.addr in involved or e.addr in involved]
            if matched:
                reports.append({"head": head, "tail": tail, "bss_pairs": matched})
        return reports
