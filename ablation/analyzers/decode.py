"""
decode.py: one `decode(code, va)` per binary for every pass built on insn_arm64.Insn.

Why this exists
  1. The AArch64 analysis stack (taint, tag scanners, fmtstr, lengths, wrappers,
     reachability, egress, frame layout, call annotation, path solver) decoded with
     Capstone ARM64 at a dozen sites, so it could only ever see AArch64.
  2. decoder_for(ctx) picks the front end once: native AArch64, AArch64 PE (dllimport
     calls `adrp ; ldr ; blr` through the IAT become `bl <slot>`, so they resolve to the
     import like a PLT call), or the x86-64 lifter (Win64 for PE, SysV for ELF).
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List

from .insn_arm64 import Insn, from_capstone, make_insn

Decoder = Callable[..., List[Insn]]


def arm64_decode(code: bytes, va: int, entry: bool = True) -> List[Insn]:
    return list(from_capstone(code, va))


def _iat_calls(insns: List[Insn], iat: FrozenSet[int]) -> List[Insn]:
    """`adrp xB, P ; ldr xT, [xB, #o] ; ... blr xT` with P+o an IAT slot -> `bl #slot`
    (and `br xT` -> `b #slot`, a tail call through the IAT)."""
    pages: Dict[str, int] = {}
    slots: Dict[str, int] = {}
    out: List[Insn] = []
    for insn in insns:
        m = insn.mnemonic
        rd = insn.reg(0)
        if m in ("blr", "br") and rd is not None and rd.name in slots:
            out.append(make_insn(insn.address, "bl" if m == "blr" else "b",
                                 f"#{slots[rd.name]:#x}", size=insn.size))
            if m == "blr":
                slots.pop(rd.name, None)
            continue
        if m == "adrp" and rd is not None and insn.imm(1) is not None:
            pages[rd.name] = insn.imm(1)
            slots.pop(rd.name, None)
        elif m == "ldr" and rd is not None and rd.bits == 64:
            mem = insn.mem(1)
            slot = None
            if mem is not None and mem.index is None and mem.writeback is None and mem.base in pages:
                slot = pages[mem.base] + mem.offset
            if slot in iat:
                slots[rd.name] = slot
            else:
                slots.pop(rd.name, None)
            pages.pop(rd.name, None)
        elif rd is not None:
            pages.pop(rd.name, None)
            slots.pop(rd.name, None)
        if m in ("bl", "blr"):
            for r in [r for r in pages if r not in ("x19", "x20", "x21", "x22", "x23", "x24",
                                                    "x25", "x26", "x27", "x28", "x29")]:
                pages.pop(r, None)
            for r in [r for r in slots if r not in ("x19", "x20", "x21", "x22", "x23", "x24",
                                                    "x25", "x26", "x27", "x28", "x29")]:
                slots.pop(r, None)
        out.append(insn)
    return out


def decoder_for(ctx) -> Decoder:
    """decode(code, va, entry=True) -> [Insn] for this context's image."""
    from .winapi import merge
    merge()
    arch = getattr(ctx, "arch", None)
    if arch == "x86_64":
        from .lift_x86_64 import lifter_for
        return lifter_for(ctx)
    from .lift_x86_64 import is_pe
    if arch == "arm64" and not getattr(ctx, "raw", None) and is_pe(ctx):
        iat = frozenset(getattr(ctx, "plt", {}).keys())
        return lambda code, va, entry=True: _iat_calls(arm64_decode(code, va), iat)
    return arm64_decode


def supported(ctx) -> bool:
    """Whether the Insn-model passes can analyse this context."""
    return getattr(ctx, "arch", None) in ("arm64", "x86_64")
