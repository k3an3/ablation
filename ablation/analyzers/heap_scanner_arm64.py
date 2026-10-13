"""
heap_scanner_arm64.py: heap memory-corruption scanner for AArch64.

ARM64 counterpart of heap_vuln_scanner.py and heap_uaf_scanner.py, with the same
HeapFinding output (heap_vuln_scanner.HeapFinding):

  uaf                        load / store through a freed pointer, or a freed
                             pointer passed to another call
  double_free                free() of an already freed pointer
  off_by_one_alloc           malloc(strlen(s)) then strcpy/strcat into it, or a
                             memcpy of strlen(s)+k bytes into it
  heap_overflow              attacker-controlled length copied / read into a
                             constant-size allocation without a bound <= size
  int_overflow_before_alloc  ARM64IntOverflowScanner findings (tainted multiply
                             reaching an allocator size)

How pointers are followed: every allocator call gives x0 a fresh region
"@heap<call_va>" in the ARM64 taint state. The state's frame_ptrs / mem_ptrs / eq
alias classes then say which registers and stack slots hold that pointer, through
copies, spills and reloads (the x86 scanners track registers only).

Freed-ness is a must-analysis over the function's CFG: at a merge a region stays
freed only if it is freed on every incoming path. `if (err) { free(p); return; }
use(p);` is therefore not reported, while straight-line or both-arms frees are.

Intraprocedural for uaf / double_free / off_by_one / heap_overflow (like the x86
scanners); int_overflow_before_alloc is interprocedural.

Usage:
    from ablation.analyzers.heap_scanner_arm64 import ARM64HeapScanner

    s = ARM64HeapScanner.from_path_full('/path/to/target_a64.so')
    print(s.report(s.scan()))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .arm64_tag_scanner import attacker
from .cfg_arm64 import build_cfg
from .heap_vuln_scanner import HeapFinding
from .insn_arm64 import Insn
from .isa_arm64 import ARG_REGS
from .taint_tracker_arm64 import ARM64TaintTracker, Taint, TaintEngine, TaintState, _name_at

# allocator -> (size regs multiplied together, pointer-in reg for realloc-style)
_ALLOCS: Dict[str, Tuple[Tuple[str, ...], Optional[str]]] = {
    "malloc": (("x0",), None), "calloc": (("x0", "x1"), None), "realloc": (("x1",), "x0"),
    "xmalloc": (("x0",), None), "xcalloc": (("x0", "x1"), None), "xrealloc": (("x1",), "x0"),
    "g_malloc": (("x0",), None), "g_malloc0": (("x0",), None), "g_realloc": (("x1",), "x0"),
    "_Znwm": (("x0",), None), "_Znam": (("x0",), None),
    "kmalloc": (("x0",), None), "__kmalloc": (("x0",), None), "kzalloc": (("x0",), None),
    "kcalloc": (("x0", "x1"), None), "kvmalloc": (("x0",), None), "vmalloc": (("x0",), None),
    "vzalloc": (("x0",), None), "krealloc": (("x1",), "x0"),
    "strdup": ((), None), "strndup": ((), None),
}
_FREES = frozenset({"free", "kfree", "vfree", "kvfree", "g_free", "kfree_sensitive",
                    "_ZdlPv", "_ZdaPv", "_ZdlPvm", "_ZdaPvm"})
_STRLENS = frozenset({"strlen", "strnlen", "wcslen"})
_STR_COPIES = frozenset({"strcpy", "strcat", "stpcpy", "sprintf", "vsprintf"})   # dst = x0
# copy-like: callee -> (dst reg, length reg)
_SIZED_COPIES: Dict[str, Tuple[str, str]] = {
    "memcpy": ("x0", "x2"), "memmove": ("x0", "x2"), "strncpy": ("x0", "x2"),
    "memset": ("x0", "x2"), "read": ("x1", "x2"), "recv": ("x1", "x2"),
    "recvfrom": ("x1", "x2"), "pread": ("x1", "x2"), "copy_from_user": ("x0", "x2"),
}
_HEAP = "@heap"
_SLEN, _SLEN_MOD = "%slen@", "%slen+@"
_MAX_VISITS = 4


@dataclass
class _Alloc:
    va: int
    sym: str
    size: Optional[int]          # constant byte size, if known
    from_strlen: bool            # size is exactly strlen(...) (no +1)


@dataclass
class _Flow:
    state: TaintState
    freed: Dict[str, int] = field(default_factory=dict)     # region -> free() call VA

    def copy(self) -> "_Flow":
        return _Flow(self.state.copy(), dict(self.freed))

    def join(self, other: "_Flow", widen: bool) -> "_Flow":
        freed = {r: min(v, other.freed[r]) for r, v in self.freed.items() if r in other.freed}
        return _Flow(self.state.join(other.state, widen), freed)

    def same_as(self, other: "_Flow") -> bool:
        return self.freed == other.freed and self.state.same_as(other.state)


class ARM64HeapScanner(ARM64TaintTracker):
    """UAF, double free, off-by-one allocation and heap overflow on AArch64."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._out: Dict[Tuple[str, int], HeapFinding] = {}
        self._allocs: Dict[str, _Alloc] = {}

    # ---- public API ---------------------------------------------------------

    def scan(self, include_int_overflow: bool = True) -> List[HeapFinding]:
        self._out = {}
        starts = self._get_func_starts()
        for i, fva in enumerate(starts):
            fend = starts[i + 1] if i + 1 < len(starts) else self._text_end
            self._scan_heap_func(fva, fend)
        findings = list(self._out.values())
        if include_int_overflow:
            findings += self._int_overflow_findings()
        return sorted(findings, key=lambda f: (f.func_va, f.site_va, f.kind))

    def report(self, findings: List[HeapFinding]) -> str:
        if not findings:
            return "ARM64 heap scan: no findings."
        high = sum(1 for f in findings if f.severity == "HIGH")
        lines = [f"ARM64 heap scan: {len(findings)} finding(s) ({high} HIGH)"]
        lines += [f.fmt() for f in findings]
        return "\n".join(lines)

    # ---- per-function CFG pass ----------------------------------------------

    def _scan_heap_func(self, fva: int, fend: int) -> None:
        insns = self._insns_for_func(fva, fend)
        if not insns:
            return
        try:
            cfg = build_cfg(insns, entry=fva)
        except ValueError:
            return
        self._allocs = {}
        entry: Dict[int, _Flow] = {cfg.entry: _Flow(TaintState())}
        visits: Dict[int, int] = {}
        work = [cfg.entry]
        while work:
            b = work.pop(0)
            if b not in cfg.blocks:
                continue
            visits[b] = visits.get(b, 0) + 1
            flow = entry[b].copy()
            self._exec_block(flow, cfg.blocks[b].insns, fva, fend)
            for s in cfg.blocks[b].succs:
                new = flow.copy() if s not in entry else entry[s].join(flow, widen=visits.get(s, 0) >= 2)
                if s not in entry or not new.same_as(entry[s]):
                    if visits.get(s, 0) >= _MAX_VISITS:
                        continue
                    entry[s] = new
                    if s not in work:
                        work.append(s)

    def _exec_block(self, flow: _Flow, insns: List[Insn], fva: int, fend: int) -> None:
        eng = TaintEngine()
        eng.state = flow.state
        for insn in insns:
            m = insn.mnemonic
            target = insn.imm(0) if m in ("bl", "b") else None
            if m == "b" and (target is None or fva <= target < fend):
                continue                                   # intra-function branch
            if target is not None:
                self._heap_call(eng, flow, insn, target, _name_at(self._syms, target), fva)
                continue
            self._check_access(eng, flow, insn, fva)
            slen = self._slen_sources(eng, insn)
            eng.step(insn)
            if slen and insn.reg(0) is not None:
                self._mark_slen_modified(eng, insn, slen)
        flow.state = eng.state

    # ---- checks -------------------------------------------------------------

    def _region(self, st: TaintState, reg: str) -> Optional[str]:
        loc = st.frame_ptrs.get(reg)
        return loc[0] if loc and loc[0].startswith(_HEAP) else None

    def _add(self, kind: str, fva: int, site: int, desc: str, sev: str,
             region: Optional[str] = None, freed_at: int = 0) -> None:
        a = self._allocs.get(region or "")
        self._out.setdefault((kind, site), HeapFinding(
            kind=kind, func_va=fva, site_va=site, description=desc, severity=sev,
            alloc_sym=a.sym if a else "", freed_at=freed_at))

    def _check_access(self, eng: TaintEngine, flow: _Flow, insn: Insn, fva: int) -> None:
        mem = insn.first_mem()
        if mem is None:
            return
        region = self._region(eng.state, mem.base)
        if region is None:
            return
        rw = "write" if insn.mnemonic.startswith("st") else "read"
        if region in flow.freed:
            self._add("uaf", fva, insn.address,
                      f"{rw} through {mem.base} after free (pointer from {self._alloc_desc(region)})",
                      "HIGH", region, flow.freed[region])
        # constant-offset access past a constant-size object (incl. inlined memcpy/memset)
        a = self._allocs.get(region)
        if a is None or a.size is None or mem.index is not None:
            return
        k = eng.state.frame_ptrs[mem.base][1] + (0 if mem.writeback == "post" else mem.offset)
        regs = [r for r in insn.regs() if r.name != mem.base]
        pair = insn.mnemonic.startswith(("stp", "ldp", "stnp", "ldnp"))
        width = {"b": 1, "h": 2}.get(insn.mnemonic[-1], (regs[0].bits // 8) if regs else 8)
        end = k + width * (2 if pair else 1)
        if k < 0 or end > a.size:
            self._add("heap_overflow", fva, insn.address,
                      f"{rw} of bytes [{k}, {end}) of {a.size}-byte {a.sym}@0x{a.va:x}",
                      "HIGH" if rw == "write" else "MEDIUM", region)

    def _alloc_desc(self, region: str) -> str:
        a = self._allocs.get(region)
        return f"{a.sym}@0x{a.va:x}" if a else region

    def _heap_call(self, eng: TaintEngine, flow: _Flow, insn: Insn, target: int,
                   callee: Optional[str], fva: int) -> None:
        st = eng.state
        name = callee or f"0x{target:x}"

        # free / realloc of a pointer
        freed_reg = "x0" if callee in _FREES else _ALLOCS.get(callee or "", ((), None))[1]
        if freed_reg:
            region = self._region(st, freed_reg)
            if region is not None:
                if region in flow.freed:
                    kind = "double_free" if callee in _FREES else "uaf"
                    self._add(kind, fva, insn.address,
                              f"{name}({freed_reg}) of pointer already freed "
                              f"(from {self._alloc_desc(region)})",
                              "HIGH", region, flow.freed[region])
                else:
                    flow.freed[region] = insn.address

        # freed pointer handed to any other call
        if callee not in _FREES:
            for r in ARG_REGS:
                region = self._region(st, r)
                if region in flow.freed and r != freed_reg:
                    self._add("uaf", fva, insn.address,
                              f"freed pointer passed to {name}({r}) "
                              f"(from {self._alloc_desc(region)})",
                              "MEDIUM", region, flow.freed[region])

        # writes into an allocation
        if callee in _STR_COPIES:
            a = self._allocs.get(self._region(st, "x0") or "")
            if a is not None and a.from_strlen:
                self._add("off_by_one_alloc", fva, insn.address,
                          f"{name} into {a.sym}(strlen(...)) @0x{a.va:x} without +1 "
                          "for the NUL terminator", "HIGH", self._region(st, "x0"))
        if callee in _SIZED_COPIES:
            dst, lreg = _SIZED_COPIES[callee]
            region = self._region(st, dst)
            a = self._allocs.get(region or "")
            if a is not None:
                lt = st.get(lreg)
                n = st.consts.get(lreg)
                if a.from_strlen and any(l.startswith(_SLEN_MOD) for l in lt.labels):
                    self._add("off_by_one_alloc", fva, insn.address,
                              f"{name} of strlen(...)+k bytes into {a.sym}(strlen(...)) @0x{a.va:x}",
                              "HIGH", region)
                elif (a.size is not None and attacker(lt.labels)
                      and (lt.bound is None or lt.bound >= a.size + 1)):
                    self._add("heap_overflow", fva, insn.address,
                              f"{name} of attacker length ({','.join(sorted(attacker(lt.labels)))})"
                              f" into {a.size}-byte {a.sym}@0x{a.va:x}", "HIGH", region)
                elif a.size is not None and n is not None and n > a.size:
                    self._add("heap_overflow", fva, insn.address,
                              f"{name} of {n} bytes into {a.size}-byte {a.sym}@0x{a.va:x}",
                              "HIGH", region)

        # allocation: size facts before the call clobbers the args
        alloc = None
        if callee in _ALLOCS:
            regs, _ = _ALLOCS[callee]
            size: Optional[int] = 1 if regs else None
            for r in regs:
                c = st.consts.get(r)
                size = None if (c is None or size is None) else size * c
            from_strlen = bool(regs) and any(l.startswith(_SLEN) for l in st.get(regs[0]).labels)
            alloc = _Alloc(insn.address, callee, size, from_strlen)

        # taint semantics (sources fill buffers, copies propagate) + call clobbers
        ARM64TaintTracker._on_call(self, eng, insn, target, callee, fva, name, [], 0, None)

        if alloc is not None:
            region = f"{_HEAP}{insn.address:x}"
            st.frame_ptrs["x0"] = (region, 0)
            flow.freed.pop(region, None)                  # new object on every execution
            self._allocs[region] = alloc
        elif callee in _STRLENS:
            t = st.get("x0")
            st.regs["x0"] = Taint(t.labels | {f"{_SLEN}{insn.address:#x}"}, None)

    # ---- strlen arithmetic --------------------------------------------------

    @staticmethod
    def _slen_sources(eng: TaintEngine, insn: Insn) -> Set[str]:
        if insn.mnemonic in ("mov", "ldr", "str", "ldur", "stur", "stp", "ldp"):
            return set()                                  # copies / spills keep the tag
        return {l for r in insn.regs()[1:] for l in eng.state.get(r.name).labels
                if l.startswith(_SLEN)}

    @staticmethod
    def _mark_slen_modified(eng: TaintEngine, insn: Insn, slen: Set[str]) -> None:
        rd = insn.reg(0).name
        t = eng.state.get(rd)
        labels = {l.replace(_SLEN, _SLEN_MOD) if l in slen else l for l in t.labels}
        eng.state.regs[rd] = Taint(frozenset(labels), t.bound)

    # ---- integer overflow (delegated) ---------------------------------------

    def _int_overflow_findings(self) -> List[HeapFinding]:
        from .intoverflow_scanner_arm64 import ARM64IntOverflowScanner
        ov = ARM64IntOverflowScanner(self._data, self._base_va, self._text_start,
                                     self._text_end, dict(self._syms))
        out = []
        for f in ov.scan():
            out.append(HeapFinding(
                kind="int_overflow_before_alloc", func_va=f.func_va, site_va=f.mul_va,
                description=(f"{f.mnemonic}/{f.width} on {f.wire_evidence} data -> "
                             f"{f.alloc_sym}({f.size_arg})@0x{f.alloc_va:x}"
                             + (" (checked)" if f.bounds_checked else "")),
                severity="MEDIUM" if f.bounds_checked else "HIGH", alloc_sym=f.alloc_sym))
        return out
