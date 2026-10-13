"""
arm64_reg_annotator.py: AAPCS64 forward register annotator for AArch64.

Tracks x0-x7 at call sites through one linear pass over a function window and
produces the same CallSite/AnnotationResult types as reg_annotator.py, so
FuncProfiler and IPRegAnnotator use it transparently for arm64 binaries.

Address materialization resolved:
  adr  xD, label                         -> const / string
  adrp xD, page ; add xD, xD, #lo12      -> const / string
  adrp xB, page ; ldr xD, [xB, #lo12]    -> pointer slot: string / const / &import
  ldr  xD, =literal (PC-relative)        -> const / string
  mov / movz / movn / movk               -> const (movk patches 16-bit lanes)
  mov  xD, xN  /  mov xD, xzr            -> copy / zero
  add / sub xD, sp|x29, #imm             -> stack address (mem 'sp+0x..')

Calls recorded:
  bl   target          direct call (PLT stub, export, or hex)
  blr  xN              indirect; named when xN came from an import's GOT slot
                       (-fno-plt: adrp+ldr GOT, blr) or a known function VA
  b    target          tail call: target is a PLT stub, a known function start,
                       or outside [func_va, func_end_va)
  br   xN              tail call when xN holds a known import / function

After a call x0 = ret(callee) and the rest of the AAPCS64 caller-saved set
(x1-x17, x30) is unknown; x19-x28 survive. Linear pass only: branches and
loops are not followed (each instruction visited once).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .binary_context import BinaryContext, PointerSlots, image_regions
from .insn_arm64 import Imm, Insn, Reg, make_insn
from .isa_arm64 import ARG_REGS, CALLER_SAVED
from .reg_annotator import AnnotationResult, CallSite, RegVal

try:
    import lief
    _LIEF_OK = True
except ImportError:
    _LIEF_OK = False

_MASK64 = (1 << 64) - 1
_LSL_RE = re.compile(r"lsl\s*#(\d+)")

# Mnemonic prefixes whose first register operand is read, not written
_NO_DEST = ("st", "cmp", "cmn", "tst", "ccmp", "ccmn", "fcmp", "fccmp",
            "cbz", "cbnz", "tbz", "tbnz", "b.", "prfm", "dmb", "dsb", "isb",
            "msr", "hint", "nop", "bti", "pac", "aut", "ret", "br", "svc")


def _capstone_a64():
    import capstone
    arch = getattr(capstone, 'CS_ARCH_AARCH64', None) or getattr(capstone, 'CS_ARCH_ARM64')
    md = capstone.Cs(arch, capstone.CS_MODE_ARM)
    md.skipdata = True
    return md


class ARM64RegAnnotator:
    """
    Forward register pass for AArch64 (AAPCS64: x0-x7 args, x0 return).

    Build with from_context(ctx) (preferred: reuses ctx.strings / ctx.plt /
    ctx.exports) or from_path(path).
    """

    def __init__(
        self,
        binary,
        strings: Dict[int, str],
        plt: Dict[int, str],
        exports: Optional[Dict[str, int]] = None,
        func_starts: Optional[List[int]] = None,
    ):
        self._binary = binary
        self._func_starts = frozenset(func_starts or ())
        self._strings = strings
        self._plt = plt
        self._exports_by_va: Dict[int, str] = {va: n for n, va in (exports or {}).items()}
        self._slots = PointerSlots(binary)
        self._md = _capstone_a64()
        self._segs: List[Tuple[int, bytes]] = image_regions(binary)
        # PE: self._plt holds IAT slot VAs (-> import name); slots hold on-disk VAs
        self._is_pe = _LIEF_OK and isinstance(binary, lief.PE.Binary)

    @classmethod
    def from_context(cls, ctx: BinaryContext) -> "ARM64RegAnnotator":
        if not _LIEF_OK:
            raise ImportError("lief is required for ARM64RegAnnotator")
        binary = lief.parse(Path(ctx.path).read_bytes())
        return cls(binary, ctx.strings, ctx.plt, ctx.exports, ctx.func_starts)

    @classmethod
    def from_path(cls, path: str) -> "ARM64RegAnnotator":
        return cls.from_context(BinaryContext.load_or_build(path))

    # ── query API ─────────────────────────────────────────────────────────────

    def annotate_calls(
        self,
        func_va: int,
        func_end_va: int = 0,
        window: int = 2048,
        seed_args: bool = True,
        entry_regs: Optional[Dict[str, RegVal]] = None,
    ) -> AnnotationResult:
        """
        Forward pass from func_va to func_end_va (or window bytes).

        seed_args : pre-seed x0..x7 = arg0..arg7 at entry.
        entry_regs: explicit entry state (IPRegAnnotator seeds callees with the
                    caller's call-site args); overrides seed_args.
        """
        if func_end_va <= func_va:
            func_end_va = func_va + window
        result = AnnotationResult(func_va=func_va, func_end_va=func_end_va)
        code = self._read(func_va, func_end_va - func_va)
        if not code:
            return result

        regs: Dict[str, RegVal] = {}
        if entry_regs is not None:
            regs.update(entry_regs)
        elif seed_args:
            for i, r in enumerate(ARG_REGS):
                regs[r] = RegVal.arg(i, r)

        for ci in self._md.disasm(code, func_va):
            if ci.address >= func_end_va:
                break
            if ci.address in self._plt and ci.address != func_va:
                break       # ran into a PLT stub / import thunk: past the function's end
            insn = make_insn(ci.address, ci.mnemonic, ci.op_str, size=ci.size)
            self._step(insn, regs, result)
        return result

    def fmt_calls(self, func_va: int, func_end_va: int = 0, window: int = 2048) -> str:
        return self.annotate_calls(func_va, func_end_va, window).fmt()

    # ── forward pass step ─────────────────────────────────────────────────────

    def _step(self, insn: Insn, regs: Dict[str, RegVal], result: AnnotationResult) -> None:
        mn = insn.mnemonic
        ops = insn.ops

        if mn == 'bl':
            target = insn.imm(0)
            if target is not None:
                self._call(insn.address, target & _MASK64, regs, result)
            return

        if mn == 'blr' or mn.startswith('blra'):
            r = insn.reg(0)
            target, name = self._indirect_target(regs.get(r.name) if r else None)
            if not name:
                name = f"blr {r.name}" if r else "blr"
            self._call(insn.address, target, regs, result, name=name)
            return

        if mn == 'br' or mn.startswith('bra'):
            # tail call through a register: record it when the target is known
            # (import pointer from a GOT / IAT slot, or a function / PLT VA)
            r = insn.reg(0)
            target, name = self._indirect_target(regs.get(r.name) if r else None)
            if name:
                self._call(insn.address, target, regs, result, name=name)
            return

        if mn == 'b':
            target = insn.imm(0)
            if target is not None and (
                    target in self._plt
                    or (target in self._func_starts and target != result.func_va)
                    or not (result.func_va <= target < result.func_end_va)):
                self._call(insn.address, target & _MASK64, regs, result)
            return

        if mn.startswith(_NO_DEST) or not ops or not isinstance(ops[0], Reg):
            # stores / compares / branches: only a writeback base can change
            self._writeback(insn, regs)
            return

        rd = ops[0]
        val: Optional[RegVal] = None

        if mn == 'adrp':
            # a page base, not a pointer: a string that happens to start the page is
            # resolved by the completing add / ldr, which needs the constant
            v = insn.imm(1)
            val = RegVal.const(v & _MASK64) if v is not None else RegVal.unknown()
        elif mn == 'adr':
            val = self._addr_val(insn.imm(1))

        elif mn in ('mov', 'movz', 'movn') and len(ops) >= 2:
            src = ops[1]
            if isinstance(src, Imm):
                v = src.value
                if mn == 'movn':
                    v = ~v
                val = self._const_or_string(v & (0xFFFFFFFF if rd.bits == 32 else _MASK64))
            elif isinstance(src, Reg):
                val = self._reg_val(src, regs)

        elif mn == 'movk' and isinstance(insn.imm(1), int):
            prev = regs.get(rd.name)
            if prev is not None and prev.kind in ('const', 'zero'):
                # the IR already folds "lsl #N" into the immediate
                m = _LSL_RE.search(insn.raw)
                lane = 0xFFFF << (int(m.group(1)) if m else 0)
                base = prev.int_val if prev.kind == 'const' else 0
                val = self._const_or_string(((base & ~lane) | (insn.imm(1) & lane)) & _MASK64)
            else:
                val = RegVal.unknown()

        elif mn in ('add', 'sub') and len(ops) >= 3 and isinstance(ops[1], Reg) and isinstance(ops[2], Imm):
            imm = ops[2].value if mn == 'add' else -ops[2].value
            base = ops[1].name
            bval = regs.get(base)
            if bval is not None and bval.kind == 'const':
                val = self._const_or_string((bval.int_val + imm) & _MASK64)
            elif base in ('sp', 'x29'):
                val = RegVal.mem(f"{base}+0x{imm:x}" if imm >= 0 else f"{base}-0x{-imm:x}")
            elif rd.name == 'sp':
                return

        elif mn.startswith('ld'):
            val = self._load(insn, regs)
            if mn.startswith('ldp') and len(ops) >= 2 and isinstance(ops[1], Reg):
                regs[ops[1].name] = RegVal.unknown()
            self._writeback(insn, regs)

        if rd.name in ('sp', 'xzr'):
            return
        if val is None:
            val = RegVal.unknown()
        regs[rd.name] = val
        if val.kind != 'unknown':
            result.reg_trace.append((insn.address, rd.name, val))

    # ── value helpers ─────────────────────────────────────────────────────────

    def _reg_val(self, src: Reg, regs: Dict[str, RegVal]) -> RegVal:
        if src.name == 'xzr':
            return RegVal.zero()
        if src.name == 'sp':
            return RegVal.mem('sp')
        sv = regs.get(src.name)
        if sv is None or sv.kind == 'unknown':
            return RegVal.unknown()
        if sv.kind in ('arg', 'string', 'const', 'zero'):
            if src.bits == 32 and sv.kind == 'const':
                return RegVal.const(sv.int_val & 0xFFFFFFFF)
            return sv
        return RegVal.copy(src.name, sv)

    def _const_or_string(self, v: int) -> RegVal:
        s = self._strings.get(v)
        return RegVal.string(s) if s else RegVal.const(v)

    def _addr_val(self, va: Optional[int]) -> RegVal:
        if va is None:
            return RegVal.unknown()
        return self._const_or_string(va & _MASK64)

    def _load(self, insn: Insn, regs: Dict[str, RegVal]) -> RegVal:
        mem = insn.mem(1)
        if mem is None:
            # ldr xD, #literal  (PC-relative literal pool)
            lit = insn.imm(1)
            rd = insn.reg(0)
            if lit is not None and rd is not None and insn.mnemonic == 'ldr':
                raw = self._read(lit, 4 if rd.bits == 32 else 8)
                if raw:
                    return self._const_or_string(int.from_bytes(raw, 'little'))
            return RegVal.unknown()

        base_val = regs.get(mem.base)
        off = mem.offset if mem.writeback != 'post' else 0
        if (insn.mnemonic == 'ldr' and mem.index is None and base_val is not None
                and base_val.kind == 'const' and insn.reg(0) and insn.reg(0).bits == 64):
            slot = (base_val.int_val + off) & _MASK64
            if self._is_pe:
                if slot in self._plt:                       # IAT entry
                    return RegVal.mem(f"&{self._plt[slot]}")
                raw = self._read(slot, 8)
                ptr = int.from_bytes(raw, 'little') if len(raw) == 8 else None
            else:
                ptr = self._slots.ptr(slot)
            if ptr is not None:
                return self._const_or_string(ptr)
            sym = None if self._is_pe else self._slots.symbol(slot)
            if sym:
                return RegVal.mem(f"&{sym}")
            return RegVal.mem(f"0x{slot:x}")

        desc = mem.base if mem.index is None else f"{mem.base}+{mem.index.name}"
        if off:
            desc += f"+0x{off:x}" if off >= 0 else f"-0x{-off:x}"
        if base_val is not None and base_val.kind == 'arg':
            return RegVal.mem(f"[{desc}] (from {base_val.display()})")
        return RegVal.mem(desc)

    def _writeback(self, insn: Insn, regs: Dict[str, RegVal]) -> None:
        mem = insn.first_mem()
        if mem is not None and mem.writeback and mem.base not in ('sp', 'x29'):
            regs[mem.base] = RegVal.unknown()

    def _indirect_target(self, v: Optional[RegVal]) -> Tuple[int, str]:
        if v is None:
            return 0, ""
        if v.kind == 'mem' and v.str_val.startswith('&'):
            return 0, v.str_val[1:]
        if v.kind == 'const':
            va = v.int_val
            return va, self._plt.get(va) or self._exports_by_va.get(va, "")
        return 0, ""

    def _call(
        self,
        site: int,
        target: int,
        regs: Dict[str, RegVal],
        result: AnnotationResult,
        name: str = "",
    ) -> None:
        if not name:
            name = self._plt.get(target) or self._exports_by_va.get(target) or f"0x{target:x}"
        result.calls.append(CallSite(
            site_va=site,
            target_va=target,
            target_name=name,
            args={r: regs.get(r, RegVal.unknown()) for r in ARG_REGS},
        ))
        for r in CALLER_SAVED:
            if r.startswith('x'):
                regs[r] = RegVal.unknown()
        regs['x0'] = RegVal.ret(name)

    # ── VA read ───────────────────────────────────────────────────────────────

    def _read(self, va: int, size: int) -> bytes:
        for sva, data in self._segs:
            off = va - sva
            if 0 <= off < len(data):
                return data[off: off + size]
        return b""
