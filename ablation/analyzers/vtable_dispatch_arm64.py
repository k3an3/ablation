"""
vtable_dispatch_arm64.py: C++ virtual-call dispatch sites per vtable slot (AArch64).

ARM64 counterpart of vtable_dispatch_scanner.py (same scan / scan_offset / report API
and DispatchSite / DispatchReport types). An Itanium-ABI virtual call on AArch64 is

    ldr  x8, [x0]              ; vptr
    ldr  x8, [x8, #slot]       ; <- slot offset
    blr  x8                    ; (br x8 for a tail dispatch, blraa/blraaz with PAC)

so a slot is live when a 64-bit `ldr xT, [xV, #slot]` (or `ldur` for unscaled
offsets) is followed within a few instructions by an indirect branch through xT,
with xT not overwritten in between.

Usage:
    s = ARM64VtableDispatchScanner.from_path('/path/to/lib.so')
    report = s.scan({'Shape::area': 0x0, 'Shape::draw': 0x8})
    print(ARM64VtableDispatchScanner.report(report))
"""

from __future__ import annotations

import struct
from typing import List

from .vtable_dispatch_scanner import DispatchSite, VtableDispatchScanner

_WINDOW = 8          # instructions between the slot load and the branch


def _indirect_branch_reg(x: int):
    """(register, kind) for BLR / BR / BLRAA(Z) / BRAA(Z) words, else None."""
    rn = (x >> 5) & 0x1F
    if (x & 0xFFFFFC1F) == 0xD63F0000:
        return rn, "blr"
    if (x & 0xFFFFFC1F) == 0xD61F0000:
        return rn, "br"
    if (x & 0xFFFFF81F) == 0xD63F081F or (x & 0xFFFFF800) == 0xD73F0800:
        return rn, "blra"                                  # blraaz/blrabz, blraa/blrab
    if (x & 0xFFFFF81F) == 0xD61F081F or (x & 0xFFFFF800) == 0xD71F0800:
        return rn, "bra"
    return None


def _writes(x: int, reg: int) -> bool:
    """Approximate: does word x write general register `reg`?"""
    if (x & 0x1F) != reg:
        return False
    cls = (x >> 25) & 0b1111
    if (cls & 0b0101) == 0b0100:                           # loads / stores
        return bool((x >> 22) & 1)                         # L bit: load writes Rt
    if (cls & 0b1110) == 0b1010:                           # branches / system
        return False
    return True                                            # data processing


class ARM64VtableDispatchScanner(VtableDispatchScanner):
    """Virtual-call dispatch sites per slot offset in AArch64 ELF executable sections."""

    def scan_offset(self, slot_offset: int) -> List[DispatchSite]:
        sites: List[DispatchSite] = []
        seen = set()
        for va_start, va_end, f_off in self._exec_regions:
            n = (va_end - va_start) // 4
            words = struct.unpack_from(f"<{n}I", self._data, f_off)
            for i, w in enumerate(words):
                if (w & 0xFFC00000) == 0xF9400000 and ((w >> 10) & 0xFFF) * 8 == slot_offset:
                    enc = "ldr"                            # LDR Xt, [Xn, #imm12*8]
                elif (w & 0xFFE00C00) == 0xF8400000:       # LDUR Xt, [Xn, #simm9]
                    imm9 = (w >> 12) & 0x1FF
                    if (imm9 - 0x200 if imm9 & 0x100 else imm9) != slot_offset:
                        continue
                    enc = "ldur"
                else:
                    continue
                rt, rv = w & 0x1F, (w >> 5) & 0x1F
                for j in range(i + 1, min(i + 1 + _WINDOW, n)):
                    br = _indirect_branch_reg(words[j])
                    if br is not None and br[0] == rt:
                        va = va_start + j * 4
                        if va not in seen:
                            seen.add(va)
                            sites.append(DispatchSite(call_va=va, register=f"x{rv}",
                                                      slot_offset=slot_offset,
                                                      encoding=f"{enc}+{br[1]}"))
                        break
                    if _writes(words[j], rt) or br is not None:
                        break
        return sorted(sites, key=lambda s: s.call_va)
