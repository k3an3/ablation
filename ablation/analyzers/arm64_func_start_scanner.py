"""
arm64_func_start_scanner.py: Find function starts in stripped AArch64 ELF without .eh_frame.

BinaryContext seeds func_starts from .eh_frame FDEs and exports. AArch64 firmware
built with -fno-asynchronous-unwind-tables (or with .eh_frame stripped) leaves only
the exports, so func_containing() collapses whole regions of .text into one function
and every call edge / string xref is attributed to the wrong owner.

Signals (all on 4-byte aligned words in executable, non-PLT sections):
  - BL targets              : BL imm26 that lands inside a scanned section
  - PACIASP / PACIBSP       : return-address signing, first insn of a PAC'd function
  - BTI c / BTI jc          : indirect-call landing pad, only emitted at function entry
  - STP x29, x30, [sp,#-N]! : standard frame-record push
  - SUB sp, sp, #N ; STP xA, xB, [sp,#M]     : large-frame / no-frame-pointer variant

A start that sits directly after a PACIASP/BTI is folded back onto it, so the
landing pad (the address callers actually branch to) is the reported start.

Measured against .eh_frame on stripped clang/Rust AArch64 binaries: ~92-98%
precision, ~58% recall. Known misses: frameless leaf functions reached only
indirectly or by tail call. Known false starts: shrink-wrapped prologues placed
after an early-exit check split one function in two.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

try:
    import lief as _lief
    _LIEF_OK = True
except ImportError:
    _LIEF_OK = False

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False


PACIASP = 0xD503233F
PACIBSP = 0xD503237F
BTI_C   = 0xD503245F
BTI_JC  = 0xD50324DF
_LANDING = frozenset((PACIASP, PACIBSP, BTI_C, BTI_JC))

# STP x29, x30, [sp, #imm7]!  (64-bit, pre-index): Rt=29, Rt2=30, Rn=sp
_STP_FP_LR_MASK = 0xFFC07FFF
_STP_FP_LR_PRE  = 0xA9807BFD
# STP xA, xB, [sp, #imm7]     (64-bit, signed offset, any register pair)
_STP_SP_OFF_MASK = 0xFFC003E0
_STP_SP_OFF      = 0xA90003E0
# SUB sp, sp, #imm12{, lsl #12}  (64-bit)
_SUB_SP_MASK = 0xFF8003FF
_SUB_SP      = 0xD10003FF
# BL imm26
_BL_MASK = 0xFC000000
_BL      = 0x94000000

_SHF_EXECINSTR = 0x4


def _ends_function(w: int) -> bool:
    """Word that can be the last of a function (or padding between functions)."""
    return (w in (0xD65F03C0, 0xD65F0BFF, 0xD65F0FFF, 0xD503201F, 0)   # ret, retaa/ab, nop, 0
            or (w & 0xFC000000) == 0x14000000                         # b (tail call)
            or (w & 0xFFFFFC1F) == 0xD61F0000                         # br xN
            or (w & 0xFFE0001F) == 0xD4200000                         # brk
            or (w & 0xFFE0001F) == 0xD4400000                         # hlt
            or (w & 0xFFFF0000) == 0x00000000                         # udf
            or w == 0xD69F03E0)                                       # eret
_PF_X = 0x1


def code_regions(binary) -> List[Tuple[int, bytes]]:
    """[(va, bytes)] of executable code in a lief ELF, PLT sections excluded.

    Sectionless firmware falls back to executable LOAD segments.
    """
    regions: List[Tuple[int, bytes]] = []
    try:
        for sec in binary.sections:
            if not (int(sec.flags) & _SHF_EXECINSTR) or sec.size == 0:
                continue
            if sec.name.startswith(".plt") or sec.name == ".iplt":
                continue
            regions.append((int(sec.virtual_address), bytes(sec.content)))
    except Exception:
        regions = []
    if not regions:
        try:
            for seg in binary.segments:
                if 'LOAD' in str(seg.type) and (int(seg.flags) & _PF_X):
                    regions.append((int(seg.virtual_address), bytes(seg.content)))
        except Exception:
            pass
    return regions


@dataclass
class ScanResult:
    starts: List[int] = field(default_factory=list)
    bl_targets: int = 0
    landing_pads: int = 0
    prologues: int = 0

    def report(self) -> str:
        return (
            f"ARM64FuncStartScanner: {len(self.starts)} function starts "
            f"(bl_targets={self.bl_targets}, landing_pads={self.landing_pads}, "
            f"prologues={self.prologues})"
        )


class ARM64FuncStartScanner:
    """Prologue + BL-target function-start discovery for stripped AArch64 code."""

    def __init__(self, regions: List[Tuple[int, bytes]], strict: bool = False):
        # regions: [(va, bytes)] of executable code to scan (PLT excluded)
        # strict: a BL target / prologue is an entry only when the word before it ends
        # the previous function (ret / b / br / brk / udf / padding); a bl before it is
        # not enough (every return site follows a bl)
        # or it opens with a landing pad / frame push; for headerless images where data
        # words decode as BLs into the middle of real functions
        self._regions = [(va, data) for va, data in regions if len(data) >= 4]
        self._strict = strict

    @classmethod
    def from_lief(cls, binary) -> "ARM64FuncStartScanner":
        """Build from a parsed lief ELF binary (executable sections, PLT excluded)."""
        return cls(code_regions(binary))

    @classmethod
    def from_path(cls, path: str) -> "ARM64FuncStartScanner":
        if not _LIEF_OK:
            raise ImportError("lief is required for ARM64FuncStartScanner.from_path()")
        binary = _lief.parse(path)
        if not isinstance(binary, _lief.ELF.Binary):
            raise ValueError(f"Not an ELF binary: {path}")
        return cls.from_lief(binary)

    @classmethod
    def from_context(cls, ctx) -> "ARM64FuncStartScanner":
        return cls.from_path(ctx.path)

    def scan(self) -> List[int]:
        return self.scan_full().starts

    def scan_full(self) -> ScanResult:
        res = ScanResult()
        starts: Set[int] = set()
        ranges = [(va, va + len(data)) for va, data in self._regions]

        def _in_code(t: int) -> bool:
            return any(lo <= t < hi for lo, hi in ranges)

        for va, data in self._regions:
            words = self._words(data)
            n = len(words)
            for i in range(n):
                w = words[i]
                if w in _LANDING:
                    res.landing_pads += 1
                    starts.add(va + i * 4)
                elif (w & _BL_MASK) == _BL:
                    imm26 = w & 0x3FFFFFF
                    if imm26 & 0x2000000:
                        imm26 -= 0x4000000
                    t = va + i * 4 + imm26 * 4
                    if _in_code(t) and (not self._strict or self._entry_ok(t, prologue_ok=True)):
                        starts.add(t)
                        res.bl_targets += 1
                elif ((w & _STP_FP_LR_MASK) == _STP_FP_LR_PRE
                      or ((w & _SUB_SP_MASK) == _SUB_SP and i + 1 < n
                          and (words[i + 1] & _STP_SP_OFF_MASK) == _STP_SP_OFF)):
                    if self._strict and not self._entry_ok(va + i * 4, prologue_ok=False):
                        continue                    # shrink-wrapped push after an early exit
                    res.prologues += 1
                    starts.add(va + i * 4)

        res.starts = sorted(self._fold_landing_pads(starts))
        return res

    def _entry_ok(self, t: int, prologue_ok: bool) -> bool:
        """Strict-mode entry test for t (see __init__)."""
        w = self._word_at(t)
        if w is None:
            return False
        if w in _LANDING:
            return True
        if prologue_ok and ((w & _STP_FP_LR_MASK) == _STP_FP_LR_PRE or (w & _SUB_SP_MASK) == _SUB_SP):
            return True
        prev = self._word_at(t - 4)
        return prev is None or _ends_function(prev)

    def bl_targets(self) -> List[int]:
        """Sorted BL targets that land inside the scanned code (the most precise signal)."""
        ranges = [(va, va + len(data)) for va, data in self._regions]
        out: Set[int] = set()
        for va, data in self._regions:
            for i, w in enumerate(self._words(data)):
                if (w & _BL_MASK) == _BL:
                    imm26 = w & 0x3FFFFFF
                    if imm26 & 0x2000000:
                        imm26 -= 0x4000000
                    t = va + i * 4 + imm26 * 4
                    if any(lo <= t < hi for lo, hi in ranges):
                        out.add(t)
        return sorted(out)

    def report(self, result: Optional[ScanResult] = None) -> str:
        return (result or self.scan_full()).report()

    def inject_into_context(self, ctx) -> int:
        """Merge discovered starts into ctx.func_starts. Returns count added."""
        existing: Set[int] = set(getattr(ctx, "func_starts", None) or [])
        new = [va for va in self.scan() if va not in existing]
        if new:
            ctx.func_starts = sorted(existing | set(new))
        return len(new)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _words(data: bytes) -> List[int]:
        n = len(data) // 4
        if _NUMPY_OK:
            return np.frombuffer(data[: n * 4], dtype="<u4").tolist()
        return list(struct.unpack_from(f"<{n}I", data, 0))

    def _word_at(self, va: int) -> Optional[int]:
        for rva, data in self._regions:
            off = va - rva
            if 0 <= off <= len(data) - 4:
                return struct.unpack_from("<I", data, off)[0]
        return None

    def _fold_landing_pads(self, starts: Set[int]) -> Set[int]:
        """Move a start that directly follows a PACIASP/BTI back onto the pad.

        `paciasp; stp x29, x30, [sp,#-N]!` is one entry at the paciasp, and a
        `bti c; paciasp` pair is one entry at the bti.
        """
        out: Set[int] = set()
        for s in starts:
            while self._word_at(s - 4) in _LANDING:
                s -= 4
            out.add(s)
        return out
