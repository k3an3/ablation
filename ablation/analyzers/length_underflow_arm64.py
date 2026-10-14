"""
length_underflow_arm64.py: protocol-parser length underflow (C12 class) for AArch64.

ARM64 counterpart of length_underflow.py. The bug:

    len = *(u16 *)(pkt + 2);          // attacker length field
    parse(pkt + 16, len - 16);        // header size subtracted, no len >= 16 check

For len < 16 the subtraction wraps: to 0xFFFFFFF0.. in a W register, or to
0xFFF0.. after a uint16 truncation (uxth / and #0xffff, x86's movzwl).

On ARM64 there is usually no truncation, because the W-register subtraction wraps
by itself, so the x86 scanner's "lea -N; movzwl; call" byte pattern does not apply.
This scanner uses the ARM64 taint pass instead:

  site   : sub/subs rd, rn, #N   (or a register holding constant N), 4 <= N <= 256,
           rn attacker-tainted (recv/read buffer contents, spills, aliases, callees)
  sink   : any call (bl / tail-call b) that receives the result in x0..x7;
           HIGH when it is a copy / read length (memcpy x2, recv x2, ...)
  guard  : a compare / cbz / tbz on rn before the subtraction, on the result after
           it, or the subtraction itself being a flag-setting subs

Usage:
    from ablation.analyzers.length_underflow_arm64 import ARM64LengthUnderflowScanner

    s = ARM64LengthUnderflowScanner.from_path_full('/path/to/target_a64.so')
    for f in s.scan_unguarded():
        print(f.fmt())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .arm64_tag_scanner import ARM64TaggingScanner, TagSite, attacker
from .insn_arm64 import Insn
from .isa_arm64 import ARG_REGS
from .taint_tracker_arm64 import TaintEngine

_HEADER_SIZE_RANGE = range(4, 257)

# callee -> argument registers that are lengths (wrapped value there = HIGH)
_LENGTH_ARGS: Dict[str, Tuple[str, ...]] = {
    "memcpy": ("x2",), "memmove": ("x2",), "memset": ("x2",), "memcmp": ("x2",),
    "strncpy": ("x2",), "strncat": ("x2",), "strncmp": ("x2",), "strndup": ("x1",),
    "read": ("x2",), "recv": ("x2",), "recvfrom": ("x2",), "pread": ("x2",),
    "write": ("x2",), "send": ("x2",), "sendto": ("x2",), "fread": ("x1", "x2"),
    "fwrite": ("x1", "x2"), "snprintf": ("x1",), "vsnprintf": ("x1",),
    "malloc": ("x0",), "calloc": ("x0", "x1"), "realloc": ("x1",),
    "copy_from_user": ("x2",), "copy_to_user": ("x2",), "__memcpy": ("x2",),
}


@dataclass
class UnderflowFinding64:
    func_va: int        # function containing the subtraction
    sub_va: int
    sub_const: int      # N (header size)
    sub_reg: str        # register the attacker length was in
    width: int          # 32 (W) / 64 (X) subtraction
    truncate_va: int    # uxth / and #0xffff after the subtraction, 0 if none
    call_va: int
    call_func_va: int   # function containing the call (callee of func_va when different)
    call_target: int
    call_label: str
    arg: str            # argument register carrying the wrapped value
    severity: str       # HIGH (length argument) / MEDIUM (other argument)
    wire_evidence: str
    guarded: bool
    note: str = ""

    def as_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        for k in ("func_va", "sub_va", "truncate_va", "call_va", "call_func_va", "call_target"):
            d[k] = hex(d[k]) if d[k] else ""
        return d

    def fmt(self) -> str:
        tag = "GUARDED" if self.guarded else "UNGUARDED"
        label = f" ({self.call_label})" if self.call_label else ""
        trunc = f"0x{self.truncate_va:x}  uxth/and #0xffff" if self.truncate_va else "none (W-register wrap)"
        return (
            f"UnderflowFinding [{tag}] {self.severity}\n"
            f"  func    : 0x{self.func_va:x}\n"
            f"  sub     : 0x{self.sub_va:x}  {self.sub_reg} - 0x{self.sub_const:x} ({self.sub_const}), {self.width}-bit\n"
            f"  truncate: {trunc}\n"
            f"  call    : 0x{self.call_va:x}  -> 0x{self.call_target:x}{label}  ({self.arg})\n"
            f"  wire    : {self.wire_evidence}\n"
            + (f"  note    : {self.note}\n" if self.note else "")
        )


class ARM64LengthUnderflowScanner(ARM64TaggingScanner):
    """attacker_len - header_size reaching a call argument without a len >= N check."""

    KIND = "sub"

    def __init__(self, *args, snprintf_rejects_huge: Optional[bool] = None, **kw):
        super().__init__(*args, **kw)
        self._hits: List[UnderflowFinding64] = []
        # True: this libc's snprintf rejects size > INT_MAX and writes nothing (musl);
        # False: it trusts the size (glibc); None: unknown
        self.snprintf_rejects_huge = snprintf_rejects_huge

    # ---- public API ---------------------------------------------------------

    def scan(self, depth: int = 3) -> List[UnderflowFinding64]:
        self._reset()
        self._hits = []
        self.run_interprocedural(depth=depth)
        out: Dict[Tuple[int, int, str], UnderflowFinding64] = {}
        for f in self._hits:
            label = self._tag(f.sub_va)
            f.guarded = self.guarded(label, self._sites[label])
            f.truncate_va = self._sites[label].extra.get("truncate_va", 0)
            if f.call_label in ("snprintf", "vsnprintf") and f.arg == "x1" and not f.truncate_va:
                # a wrapped size is > INT_MAX: safe or fatal depending on the libc
                if self.snprintf_rejects_huge is True:
                    f.severity, f.note = "LOW", "this libc's snprintf rejects size > INT_MAX and writes nothing"
                elif self.snprintf_rejects_huge is None:
                    f.severity = "MEDIUM"
                    f.note = ("depends on the libc: some snprintf builds (e.g. musl) reject size > "
                              "INT_MAX and write nothing, others (glibc) trust it")
            out.setdefault((f.sub_va, f.call_va, f.arg), f)
        return sorted(out.values(), key=lambda f: (f.func_va, f.sub_va, f.call_va))

    def scan_unguarded(self, depth: int = 3) -> List[UnderflowFinding64]:
        return [f for f in self.scan(depth) if not f.guarded]

    def report(self, findings: Optional[List[UnderflowFinding64]] = None) -> str:
        findings = self.scan() if findings is None else findings
        if not findings:
            return "No ARM64 length underflow candidates."
        ung = [f for f in findings if not f.guarded]
        lines = [f"ARM64 Length Underflow Scan: {len(findings)} candidate(s) ({len(ung)} unguarded)",
                 "=" * 70]
        lines += [f.fmt() for f in findings]
        return "\n".join(lines)

    # ---- sites / sinks ------------------------------------------------------

    def _site(self, eng: TaintEngine, insn: Insn, fva: int) -> Optional[TagSite]:
        m = insn.mnemonic
        if m not in ("sub", "subs") or len(insn.ops) < 3:
            return None
        rd, rn = insn.reg(0), insn.reg(1)
        if rd is None or rn is None or rd.name in ("xzr", "sp") or rn.name in ("sp", "x29"):
            return None
        n = insn.imm(2)
        if n is None and insn.reg(2) is not None and insn.reg(2).shift is None:
            n = eng.state.consts.get(insn.reg(2).name)
        if n is None or n not in _HEADER_SIZE_RANGE:
            return None
        src = attacker(eng.state.get(rn.name).labels)
        if not src:
            return None
        return TagSite(fva, insn.address, m, [rn.name], src, {"n": n, "width": rd.bits})

    def _after_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        # uint16 / uint8 truncation of a wrapped value (x86's movzwl)
        m = insn.mnemonic
        if m in ("uxth", "uxtb") or (m == "and" and insn.imm(2) in (0xFF, 0xFFFF)):
            src = insn.reg(1)
            if src is not None:
                for _label, site in self.sites_in(eng.state.get(src.name)):
                    site.extra.setdefault("truncate_va", insn.address)
        super()._after_step(eng, insn, fva)

    def _on_call(self, eng: TaintEngine, insn: Insn, target: int, callee: Optional[str],
                 fva: int, func_name: str, findings, depth: int, enqueue) -> None:
        length_args = _LENGTH_ARGS.get(callee or "", ()) or self._wrapper_regs(target, callee, "length")
        for r in ARG_REGS:
            for _label, site in self.sites_in(eng.state.get(r)):
                self._hits.append(UnderflowFinding64(
                    func_va=site.func_va, sub_va=site.va, sub_const=site.extra["n"],
                    sub_reg=site.regs[0], width=site.extra["width"], truncate_va=0,
                    call_va=insn.address, call_func_va=fva, call_target=target,
                    call_label=callee or "", arg=r,
                    severity="HIGH" if r in length_args else "MEDIUM",
                    wire_evidence=",".join(sorted(site.src_labels)), guarded=False,
                ))
        super()._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)
