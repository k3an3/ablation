"""
format_string_scanner_arm64.py: non-literal format strings in printf-family calls (AArch64).

ARM64 counterpart of format_string_scanner.py, same FmtStringFinding output.

Pass 1 (provenance, like the x86 scanner): ARM64RegAnnotator gives the value of the
format register at every printf / fprintf / snprintf / syslog / err / warn call,
including tail calls (`b printf`) and calls through a GOT slot:

  string literal / .rodata address     -> safe, not reported
  entry argument (x0..x7 at entry)     -> HIGH    'arg_passthrough'
  stack slot / memory load             -> MEDIUM  'stack_load' / 'memory_load'
  another call's return value          -> MEDIUM  'call_result'
  anything else                        -> MEDIUM  'register' / 'unknown'

Pass 2 (taint, ARM64 only): the ARM64 taint pass marks call sites whose format
register, or the memory it points at, holds recv/read data. Those become HIGH
'tainted' with the source named, whatever pass 1 said, so a format built from
received bytes is reported even when it looks like a plain local buffer.

puts() is not a format function and is not checked (the x86 scanner flags it
"for review").

Usage:
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner

    s = ARM64FormatStringScanner.from_path_full('/path/to/target_a64.so')
    print(s.report(s.scan()))
"""

from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Set, Tuple

from .format_string_scanner import FmtStringFinding
from .insn_arm64 import Insn
from .taint_tracker_arm64 import ARM64TaintTracker, TaintEngine

# format function -> register carrying the format string
FMT_ARG: Dict[str, str] = {
    "printf": "x0", "vprintf": "x0",
    "fprintf": "x1", "vfprintf": "x1", "sprintf": "x1", "vsprintf": "x1",
    "snprintf": "x2", "vsnprintf": "x2", "asprintf": "x1", "vasprintf": "x1",
    "dprintf": "x1", "vdprintf": "x1",
    "syslog": "x1", "vsyslog": "x1",
    "err": "x1", "errx": "x1", "warn": "x0", "warnx": "x0",
    "verr": "x1", "verrx": "x1", "vwarn": "x0", "vwarnx": "x0",
    "__printf_chk": "x1", "__fprintf_chk": "x2", "__sprintf_chk": "x3",
    "__snprintf_chk": "x4", "__vsnprintf_chk": "x4", "__syslog_chk": "x2",
    "printk": "x0", "_printk": "x0", "panic": "x0",
    "log_printf": "x1", "debug_printf": "x0", "trace_printf": "x0",
}
_MAX_FUNC_BYTES = 4096


class ARM64FormatStringScanner(ARM64TaintTracker):
    """Non-literal (and attacker-tainted) format strings at printf-family call sites."""

    @classmethod
    def from_context(cls, ctx, custom_sinks=None, custom_sources=None) -> "ARM64FormatStringScanner":
        inst = super().from_context(ctx, custom_sinks, custom_sources)
        inst._ctx = ctx
        return inst

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._ctx = None
        self._tainted: Dict[int, Tuple[str, Set[str]]] = {}   # call_va -> (reg, labels)

    # ---- public API ---------------------------------------------------------

    def scan(self, depth: int = 3) -> List[FmtStringFinding]:
        if self._ctx is None:
            raise ValueError("build with from_context / from_path_full (needs a BinaryContext)")
        self._tainted = {}
        self.run_interprocedural(depth=depth)
        findings = {f.call_va: f for f in self._provenance_pass()}
        for call_va, (reg, labels) in self._tainted.items():
            src = ",".join(sorted(labels))
            f = findings.get(call_va)
            if f is None:
                continue
            f.provenance, f.severity = "tainted", "HIGH"
            f.description = f"format string in {reg} carries received data ({src})"
        return sorted(findings.values(), key=lambda f: (f.func_va, f.call_va))

    def report(self, findings: List[FmtStringFinding]) -> str:
        if not findings:
            return "ARM64 format string scan: no findings."
        high = sum(1 for f in findings if f.severity == "HIGH")
        lines = [f"ARM64 format string scan: {len(findings)} finding(s) ({high} HIGH)"]
        lines += [f.fmt() for f in findings]
        return "\n".join(lines)

    # ---- pass 1: provenance -------------------------------------------------

    def _provenance_pass(self) -> List[FmtStringFinding]:
        from .arm64_reg_annotator import ARM64RegAnnotator
        ctx = self._ctx
        ra = ARM64RegAnnotator.from_context(ctx)
        starts = ctx.func_starts
        out: List[FmtStringFinding] = []
        for fva in starts:
            if fva in ctx.plt:
                continue
            i = bisect.bisect_right(starts, fva)
            end = starts[i] if i < len(starts) and starts[i] - fva <= _MAX_FUNC_BYTES else fva + _MAX_FUNC_BYTES
            for call in ra.annotate_calls(fva, end).calls:
                reg = FMT_ARG.get(call.target_name)
                if reg is None:
                    continue
                prov, sev, desc = self._classify(call.args.get(reg), reg)
                if prov is None:
                    continue
                out.append(FmtStringFinding(fva, call.site_va, call.target_name, reg, prov, sev, desc))
        return out

    def _classify(self, v, reg: str) -> Tuple[Optional[str], str, str]:
        if v is None or v.kind == "unknown":
            return "unknown", "MEDIUM", f"format in {reg} has unknown provenance"
        if v.kind == "copy" and v.chain is not None:
            return self._classify(v.chain, reg)
        if v.kind == "string":
            return None, "", ""                                    # literal: safe
        if v.kind == "const":
            if self._is_rodata(v.int_val):
                return None, "", ""                                # address inside .rodata
            return "register", "MEDIUM", f"format in {reg} is constant 0x{v.int_val:x} outside .rodata"
        if v.kind == "arg":
            return "arg_passthrough", "HIGH", f"format in {reg} is the caller's {v.display()}"
        if v.kind == "mem":
            if v.str_val.startswith(("sp", "x29")):
                return "stack_load", "MEDIUM", f"format in {reg} loaded from stack [{v.str_val}]"
            return "memory_load", "MEDIUM", f"format in {reg} loaded from memory {v.display()}"
        if v.kind == "ret":
            return "call_result", "MEDIUM", f"format in {reg} is the return value of {v.str_val}"
        return "register", "MEDIUM", f"format in {reg} = {v.display()}"

    def _is_rodata(self, va: int) -> bool:
        ranges = getattr(self, "_rodata", None)
        if ranges is None:
            import lief
            ranges = []
            try:
                binary = lief.parse(self._ctx.path)      # keep alive while iterating sections
                for sec in binary.sections:
                    if sec.name.startswith((".rodata", ".data.rel.ro")) and sec.size:
                        lo = int(sec.virtual_address)
                        ranges.append((lo, lo + int(sec.size)))
            except Exception:
                pass
            self._rodata = ranges
        return any(lo <= va < hi for lo, hi in ranges)

    # ---- pass 2: taint ------------------------------------------------------

    def _on_call(self, eng: TaintEngine, insn: Insn, target: int, callee: Optional[str],
                 fva: int, func_name: str, findings, depth: int, enqueue) -> None:
        reg = FMT_ARG.get(callee or "")
        if reg is not None:
            st = eng.state
            labels = st.get(reg).labels | st.pointee(reg).labels
            if labels:
                prev = self._tainted.get(insn.address, (reg, set()))[1]
                self._tainted[insn.address] = (reg, set(prev) | set(labels))
        super()._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)
