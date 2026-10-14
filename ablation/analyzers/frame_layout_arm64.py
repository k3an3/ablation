"""
frame_layout_arm64.py: AArch64 stack-frame layout of a function, and what a format string's
conversions read / how far a stack buffer reaches.

Why this exists
  1. A format-string read is only triageable once you know what comes back. With a
     controllable format at a variadic call, conversions 1..n read the remaining GP
     argument registers (stale values), then the caller's stack at sp: its locals, and
     its saved {x29, x30}. Conversion N rendering the saved x30 leaks a return address,
     i.e. image_base + K, which defeats PIE. That took hand arithmetic from the frame
     size and the va_list's gr_offs.
  2. A width-less %s / %[ into a stack buffer is a stack overflow whose exploitability is
     its byte budget: the bytes to the next stack object and to the saved return address.

FrameLayout.of(code, fva)      prologue: sp adjustments, stp / str of x29 / x30 /
                               callee-saved registers, add x29, sp, #k; whole body:
                               address-taken stack objects (add xD, sp|x29, #k)
conversion_map(...)            conversion index (sequential and %N$) -> what it reads
extent(...)                    bytes from a stack buffer to the next object / saved x30

Offsets are relative to the function's entry sp (0 = entry sp, frame below it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .insn_arm64 import Insn, from_capstone

_STACK_ADDR = re.compile(r"^(sp|x29)(?:([+-])0x([0-9a-f]+))?$")


@dataclass
class FrameLayout:
    fva: int
    sp_body: int = 0                       # entry-relative sp after the prologue (<= 0)
    fp: Optional[int] = None               # entry-relative x29 after `add x29, sp, #k`
    saved: Dict[int, str] = field(default_factory=dict)    # entry-relative offset -> reg
    objects: List[int] = field(default_factory=list)       # address-taken stack objects

    @property
    def size(self) -> int:
        return -self.sp_body

    @classmethod
    def of(cls, code: bytes, fva: int) -> "FrameLayout":
        lay = cls(fva)
        insns = list(from_capstone(code, fva))
        sp = 0
        fp = None
        in_prologue = True
        objects = set()
        for insn in insns:
            m = insn.mnemonic
            mem = insn.first_mem()
            if in_prologue:
                if m in ("bl", "blr", "b", "ret", "br") or m.startswith("b.") or m in ("cbz", "cbnz", "tbz", "tbnz"):
                    in_prologue = False
                elif m == "sub" and insn.reg(0) is not None and insn.reg(0).name == "sp" \
                        and insn.reg(1) is not None and insn.reg(1).name == "sp" and insn.imm(2) is not None:
                    sp -= insn.imm(2) << (12 if "lsl #12" in (insn.raw or "") else 0)
                    continue
                elif m in ("stp", "str") and mem is not None and mem.base == "sp":
                    if mem.writeback == "pre":
                        sp += mem.offset
                        at = sp
                    elif mem.writeback == "post":
                        at = sp
                        sp += mem.wb_amount or mem.offset
                    else:
                        at = sp + mem.offset
                    regs = [r for r in insn.regs() if r.name != "sp"]
                    for k, r in enumerate(regs):
                        if r.bits == 64 and (r.name in ("x29", "x30") or re.fullmatch(r"x(19|2[0-8])", r.name)):
                            lay.saved[at + 8 * k] = r.name
                    continue
                elif m in ("add", "mov") and insn.reg(0) is not None and insn.reg(0).name == "x29" \
                        and insn.reg(1) is not None and insn.reg(1).name == "sp":
                    fp = sp + (insn.imm(2) or 0)
                    continue
            # address-taken stack objects anywhere in the body
            rd, rn = insn.reg(0), insn.reg(1)
            if m in ("add", "sub") and rd is not None and rn is not None and rd.name not in ("sp", "x29") \
                    and rn.name in ("sp", "x29") and insn.imm(2) is not None:
                k = insn.imm(2) if m == "add" else -insn.imm(2)
                base = sp if rn.name == "sp" else fp
                if base is not None:
                    objects.add(base + k)
            elif m == "mov" and rd is not None and rn is not None and rn.name == "sp" and rd.name != "x29":
                objects.add(sp)
        lay.sp_body, lay.fp = sp, fp
        lay.objects = sorted(objects)
        return lay

    def entry_off(self, desc: str) -> Optional[int]:
        """Entry-relative offset of a stack address described as 'sp+0x10' / 'x29-0x8'."""
        m = _STACK_ADDR.match(desc.strip("[]"))
        if not m:
            return None
        k = int(m.group(3), 16) if m.group(3) else 0
        k = -k if m.group(2) == "-" else k
        if m.group(1) == "sp":
            return self.sp_body + k
        return self.fp + k if self.fp is not None else None

    def lr_slot(self) -> Optional[int]:
        return next((o for o, r in self.saved.items() if r == "x30"), None)

    def extent(self, off: int) -> Tuple[Optional[int], str, Optional[int], str]:
        """(bytes to the next stack object or saved register, what it is, bytes to this
        function's saved x30 (None if the record is below the buffer), note)."""
        bounds = sorted({o for o in self.objects if o > off} | {o for o in self.saved if o > off})
        nxt = bounds[0] if bounds else 0
        what = (f"saved {self.saved[nxt]}" if nxt in self.saved else
                "next stack object" if bounds else "entry sp (caller's frame)")
        lr = self.lr_slot()
        if lr is not None and lr > off:
            return nxt - off, what, lr - off, ""
        note = ("frame record is below the buffer (pre-indexed stp x29, x30): an overflow "
                f"leaves this frame after {-off} bytes and runs into the caller's")
        return nxt - off, what, None, note


def _slot_class(v, code_lo: int, code_hi: int, base: int) -> Tuple[str, str]:
    """(class, text) of a stale register value."""
    while v is not None and v.kind == "copy":
        v = v.chain
    if v is None or v.kind == "unknown":
        return "stale", "stale argument register"
    if getattr(v, "is_stack_addr", False):
        return "stack", f"stack address ({v.display()}): stack disclosure"
    if v.kind == "const" and code_lo <= v.int_val < code_hi:
        return "image", f"image pointer 0x{v.int_val:x} = base+{v.int_val - base:#x}: PIE disclosure"
    if v.kind == "string":
        return "image", f"pointer to {v.display()} in the image: PIE disclosure"
    return "stale", f"stale argument register ({v.display()})"


def conversion_map(layout: FrameLayout, args: Dict, fmt_index: int, gp_slots: Optional[int],
                   ret_sites: Sequence[int], image: Tuple[int, int], base: int,
                   tainted_slots: Optional[Dict[int, Iterable[str]]] = None,
                   max_conv: int = 64) -> Tuple[List[str], Dict]:
    """What conversion i (1-based, the same for %i$) of a controllable format reads at a
    variadic call whose format is argument x<fmt_index> of the callee, made from a function
    with `layout`. args: the call site's register values. gp_slots: GP registers in the
    callee's save area (default 7 - fmt_index). Returns (lines, leak summary)."""
    n_gp = gp_slots if gp_slots is not None else 7 - fmt_index
    lo, hi = image
    out: List[Tuple[int, str, str]] = []                # (index, class, text)
    for i in range(1, n_gp + 1):
        reg = f"x{fmt_index + i}"
        cls, text = _slot_class(args.get(reg), lo, hi, base)
        out.append((i, cls, f"{reg} {text}"))
    lr = layout.lr_slot()
    tainted_slots = tainted_slots or {}
    j = 0
    while n_gp + j + 1 <= max_conv:
        i = n_gp + j + 1
        o = layout.sp_body + 8 * j
        if o >= 16:
            break
        reg = layout.saved.get(o)
        if reg == "x30":
            ks = (", ".join(f"base+{r - base:#x}" for r in sorted(set(ret_sites))[:4])
                  or "base+K, K unknown: no direct caller in the binary")
            out.append((i, "code", f"[sp+{8 * j:#x}] saved x30: return address into the caller ({ks}): "
                                   f"code pointer disclosure, defeats PIE"))
        elif reg == "x29":
            out.append((i, "stack", f"[sp+{8 * j:#x}] saved x29: stack address disclosure (defeats stack ASLR)"))
        elif reg:
            out.append((i, "saved", f"[sp+{8 * j:#x}] saved {reg} (the caller's callee-saved register)"))
        elif j in tainted_slots:
            out.append((i, "attacker", f"[sp+{8 * j:#x}] attacker bytes ({','.join(sorted(tainted_slots[j]))}): "
                                       f"a %n / %s target can be placed here"))
        elif o >= 0:
            out.append((i, "unknown", f"[sp+{8 * j:#x}] caller's caller frame (unknown)"))
        else:
            out.append((i, "local", f"[sp+{8 * j:#x}] local slot (stale data)"))
        j += 1
        if lr is not None and o > lr + 8:
            break
    # compress runs of plain locals / stale registers
    lines: List[str] = []
    k = 0
    while k < len(out):
        i, cls, text = out[k]
        e = k
        while e + 1 < len(out) and cls in ("local", "stale", "attacker", "unknown") and out[e + 1][1] == cls:
            e += 1
        if e > k:
            if cls == "attacker":
                what = text.split("] ", 1)[1].split(":")[0] + ": %n / %s targets can be placed here"
            else:
                what = {"local": "local slots (stale data)", "stale": "stale argument registers",
                        "unknown": "caller's caller frame (unknown)"}[cls]
            lines.append(f"%{i}$..%{out[e][0]}$: {what}")
        else:
            lines.append(f"%{i}$: {text}")
        k = e + 1
    leak: Dict[str, str] = {}
    for i, cls, text in out:
        if cls == "code" and "code" not in leak:
            leak["code"] = f"%{i}$p renders the saved return address ({text.split('caller (')[1].split('):')[0]})"
        elif cls == "image" and "image" not in leak:
            leak["image"] = f"%{i}$p renders an image pointer"
        elif cls == "stack" and "stack" not in leak:
            leak["stack"] = f"%{i}$p renders a stack address"
        elif cls == "attacker" and "attacker" not in leak:
            leak["attacker"] = f"%{i}$ reads attacker-controlled bytes"
    return lines, leak
