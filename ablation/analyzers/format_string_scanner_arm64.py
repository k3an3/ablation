"""
format_string_scanner_arm64.py: non-literal format strings in printf-family calls (AArch64).

ARM64 counterpart of format_string_scanner.py, same FmtStringFinding output.

Pass 1 (provenance, like the x86 scanner): ARM64RegAnnotator gives the value of the
format register at every printf / fprintf / snprintf / syslog / err / warn call,
including tail calls (`b printf`) and calls through a GOT slot:

  string literal / .rodata address     -> safe, not reported
  entry argument (x0..x7 at entry)     -> wrapper: judged by its callers (below)
  stack slot / memory load             -> MEDIUM  'stack_load' / 'memory_load'
  another call's return value          -> MEDIUM  'call_result'
  anything else                        -> MEDIUM  'register' / 'unknown'

A format that is the function's own argument makes the function a printf wrapper; the
bug, if any, is at the callers. Every call site of the wrapper (nested wrappers
followed) decides the verdict:

  some caller passes recv/read data      -> HIGH    'tainted' (caller site named)
  some caller passes a non-literal       -> MEDIUM  'arg_passthrough'
  no caller in this binary (exported)    -> LOW     'arg_passthrough'
  every caller passes a literal          -> not reported

Pass 3 (unbounded conversions): a literal format with %s / %[ and no width or
precision writing into a stack buffer:

  sprintf / vsprintf (dst = x0 on the stack)     -> 'unbounded_s'
  sscanf / scanf / fscanf (%s target on stack)   -> 'unbounded_s'

HIGH when the data is attacker-controlled (a %s source / the sscanf input holds
recv/read data, or scanf reading stdin), else MEDIUM.

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
from .taint_tracker_arm64 import ARG_REGS, ARM64TaintTracker, TaintEngine

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

# unbounded-conversion check: callee -> (format reg, first vararg index or None for a
# va_list, dst reg for the print family / input reg for sscanf, family)
_UNBOUNDED: Dict[str, Tuple[str, Optional[int], Optional[str], str]] = {
    "sprintf": ("x1", 2, "x0", "print"), "vsprintf": ("x1", None, "x0", "print"),
    "sscanf": ("x1", 2, "x0", "scan"), "scanf": ("x0", 1, None, "scan"),
    "fscanf": ("x1", 2, None, "scan"), "__isoc99_sscanf": ("x1", 2, "x0", "scan"),
    "__isoc99_scanf": ("x0", 1, None, "scan"), "__isoc99_fscanf": ("x1", 2, None, "scan"),
}
_STDIN_SCANS = frozenset({"scanf", "__isoc99_scanf"})


def unbounded_conversions(fmt: str, family: str) -> List[Tuple[int, str]]:
    """[(vararg index, spec)] of %s / %[ conversions with no bound in fmt.

    print family: %s without a precision (and without a '*' precision). scan family:
    %s / %[ without a width, not assignment-suppressed (%*s) or allocating (%ms).
    The index counts every argument the format consumes, '*' widths included.
    """
    out: List[Tuple[int, str]] = []
    i, n, arg = 0, len(fmt), 0
    while i < n:
        if fmt[i] != "%":
            i += 1
            continue
        j = i + 1
        if j < n and fmt[j] == "%":
            i = j + 1
            continue
        suppress = alloc = False
        width = prec = False
        if family == "scan" and j < n and fmt[j] == "*":
            suppress, j = True, j + 1
        while j < n and fmt[j] in "-+ #0'":
            j += 1
        if j < n and fmt[j] == "*":
            if family == "print":
                arg += 1
            width, j = True, j + 1
        while j < n and fmt[j].isdigit():
            width, j = True, j + 1
        if j < n and fmt[j] == ".":
            prec, j = True, j + 1
            if j < n and fmt[j] == "*":
                arg, j = arg + 1, j + 1
            while j < n and fmt[j].isdigit():
                j += 1
        if family == "scan" and j < n and fmt[j] == "m":
            alloc, j = True, j + 1
        while j < n and fmt[j] in "hlLqjzt":
            j += 1
        if j >= n:
            break
        conv = fmt[j]
        if conv == "[":                                  # scanset: skip to the closing ]
            k = j + 1
            if k < n and fmt[k] == "^":
                k += 1
            if k < n and fmt[k] == "]":
                k += 1
            k = fmt.find("]", k)
            j = n - 1 if k < 0 else k
        spec = fmt[i:j + 1]
        consumes = not suppress
        if consumes:
            unbounded = (conv == "s" and not prec) if family == "print" else \
                (conv in "s[" and not width and not alloc)
            if unbounded:
                out.append((arg, spec))
            arg += 1
        i = j + 1
    return out


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
        self._watch: Dict[int, Set[str]] = {}                  # wrapper VA -> arg regs
        self._watch_calls: Set[int] = set()                    # unbounded-check call sites
        self._arg_taint: Dict[Tuple[int, str], Set[str]] = {}  # (call_va, reg) -> labels

    # ---- public API ---------------------------------------------------------

    def scan(self, depth: int = 3) -> List[FmtStringFinding]:
        if self._ctx is None:
            raise ValueError("build with from_context / from_path_full (needs a BinaryContext)")
        self._tainted, self._arg_taint = {}, {}
        prov = self._provenance_pass()           # also collects wrappers + unbounded sites
        self.run_interprocedural(depth=depth)    # taint at those call sites
        findings: Dict[int, FmtStringFinding] = {}
        for f in prov:
            if f.provenance == "arg_passthrough":
                f = self._judge_wrapper(f)
                if f is None:
                    continue
            findings[f.call_va] = f
        for f in self._unbounded_findings():
            findings.setdefault(f.call_va, f)
        for call_va, (reg, labels) in self._tainted.items():
            src = ",".join(sorted(labels))
            f = findings.get(call_va)
            if f is None or f.provenance in ("unbounded_s", "tainted"):
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
        self._callers: Dict[int, List[Tuple[int, object]]] = {}   # callee VA -> [(caller, CallSite)]
        self._unb_sites: List[Tuple[int, object]] = []
        self._wrapper_arg: Dict[int, int] = {}                     # finding call_va -> arg index
        for fva in starts:
            if fva in ctx.plt:
                continue
            i = bisect.bisect_right(starts, fva)
            end = starts[i] if i < len(starts) and starts[i] - fva <= _MAX_FUNC_BYTES else fva + _MAX_FUNC_BYTES
            for call in ra.annotate_calls(fva, end).calls:
                tva = ctx.exports.get(call.target_name, call.target_va)
                self._callers.setdefault(tva, []).append((fva, call))
                if call.target_name in _UNBOUNDED:
                    self._unb_sites.append((fva, call))
                    self._watch_calls.add(call.site_va)
                reg = FMT_ARG.get(call.target_name)
                if reg is None:
                    continue
                prov, sev, desc = self._classify(call.args.get(reg), reg)
                if prov is None:
                    continue
                if prov == "arg_passthrough":
                    self._wrapper_arg[call.site_va] = self._arg_index(call.args.get(reg))
                out.append(FmtStringFinding(fva, call.site_va, call.target_name, reg, prov, sev, desc))
        # wrappers (nested ones too) whose call sites the taint pass must watch
        todo = [(f.func_va, self._wrapper_arg[f.call_va]) for f in out
                if f.call_va in self._wrapper_arg]
        while todo:
            w, idx = todo.pop()
            if idx < 0 or f"x{idx}" in self._watch.get(w, ()):
                continue
            self._watch.setdefault(w, set()).add(f"x{idx}")
            for caller, call in self._callers.get(w, ()):
                v = call.args.get(f"x{idx}")
                if self._classify(v, f"x{idx}")[0] == "arg_passthrough":
                    todo.append((caller, self._arg_index(v)))
        return out

    @staticmethod
    def _arg_index(v) -> int:
        while v is not None and v.kind == "copy":
            v = v.chain
        return v.int_val if v is not None and v.kind == "arg" else -1

    # ---- wrappers: judged by their callers -----------------------------------

    def _caller_verdict(self, w: int, idx: int, seen: Set[Tuple[int, int]]) -> Tuple[str, str]:
        """('tainted' | 'nonliteral' | 'nocallers' | 'literal', detail) over w's callers."""
        if (w, idx) in seen or idx < 0:
            return "literal", ""
        seen.add((w, idx))
        reg = f"x{idx}"
        callers = self._callers.get(w, [])
        if not callers:
            return "nocallers", f"no caller of 0x{w:x} in this binary (exported?)"
        rank = {"literal": 0, "nocallers": 1, "nonliteral": 2, "tainted": 3}
        best = ("literal", "")
        for caller, call in callers:
            labels = self._arg_taint.get((call.site_va, reg))
            if labels:
                res = ("tainted", f"{self._fn(caller)} passes received data "
                       f"({','.join(sorted(labels))}) in {reg} at 0x{call.site_va:x}")
            else:
                v = call.args.get(reg)
                prov, _sev, desc = self._classify(v, reg)
                if prov is None:
                    res = ("literal", "")
                elif prov == "arg_passthrough":
                    res = self._caller_verdict(caller, self._arg_index(v), seen)
                else:
                    res = ("nonliteral", f"{self._fn(caller)} at 0x{call.site_va:x}: {desc}")
            if rank[res[0]] > rank[best[0]]:
                best = res
        return best

    def _judge_wrapper(self, f: FmtStringFinding) -> Optional[FmtStringFinding]:
        idx = self._wrapper_arg.get(f.call_va, -1)
        verdict, detail = self._caller_verdict(f.func_va, idx, set())
        base = f"{f.callee} wrapper: format in {f.fmt_reg} is the caller's x{idx}"
        if verdict == "tainted":
            f.provenance, f.severity = "tainted", "HIGH"
            f.description = f"{base}; {detail}"
        elif verdict == "nonliteral":
            f.severity, f.description = "MEDIUM", f"{base}; {detail}"
        elif verdict == "nocallers":
            f.severity, f.description = "LOW", f"{base}; {detail}"
        else:
            return None                                       # every caller passes a literal
        return f

    def _fn(self, va: int) -> str:
        return self._syms.get(va) or f"0x{va:x}"

    # ---- pass 3: unbounded %s / %[ ------------------------------------------

    def _unbounded_findings(self) -> List[FmtStringFinding]:
        out: List[FmtStringFinding] = []
        for fva, call in self._unb_sites:
            name = call.target_name
            freg, first, io_reg, family = _UNBOUNDED[name]
            fmt = self._literal(call.args.get(freg))
            if fmt is None:
                continue                                  # non-literal: the fmt check covers it
            convs = unbounded_conversions(fmt, family)
            if not convs:
                continue
            labels: Set[str] = set()
            if family == "print":
                dst = call.args.get(io_reg)
                if not self._on_stack(dst):
                    continue
                where = f"stack buffer {dst.display()}"
                for idx, _spec in convs:
                    if first is not None and first + idx < 8:
                        labels |= self._arg_taint.get((call.site_va, f"x{first + idx}"), set())
                spec = convs[0][1]
                tgt = io_reg
            else:
                hits = [(f"x{first + idx}", spec) for idx, spec in convs
                        if first is not None and first + idx < 8
                        and self._on_stack(call.args.get(f"x{first + idx}"))]
                if not hits:
                    continue
                tgt, spec = hits[0]
                where = f"stack buffer {call.args.get(tgt).display()}"
                if io_reg:
                    labels |= self._arg_taint.get((call.site_va, io_reg), set())
                elif name in _STDIN_SCANS:
                    labels.add("stdin")
            sev = "HIGH" if labels else "MEDIUM"
            src = f" from {','.join(sorted(labels))}" if labels else ""
            out.append(FmtStringFinding(
                fva, call.site_va, name, tgt, "unbounded_s", sev,
                f"{name} '{spec}' (no {'precision' if family == 'print' else 'width'}) "
                f"into {where}{src}"))
        return out

    def _literal(self, v) -> Optional[str]:
        while v is not None and v.kind == "copy":
            v = v.chain
        if v is None:
            return None
        if v.kind == "string":
            return v.str_val
        if v.kind == "const" and self._ctx is not None:
            return self._ctx.strings.get(v.int_val) or self._cstr(v.int_val)
        return None

    def _cstr(self, va: int) -> Optional[str]:
        """NUL-terminated text at va (formats like "%s" are below the string table's
        minimum length)."""
        regions = getattr(self, "_regions", None)
        if regions is None:
            regions = self._regions = self._ctx.image_regions()
        for base, data in regions:
            off = va - base
            if 0 <= off < len(data):
                end = data.find(b"\0", off, off + 512)
                raw = data[off:end] if end >= 0 else b""
                if raw and all(32 <= c < 127 or c in (9, 10, 13) for c in raw):
                    return raw.decode()
                return None
        return None

    @staticmethod
    def _on_stack(v) -> bool:
        while v is not None and v.kind == "copy":
            v = v.chain
        return v is not None and getattr(v, "is_stack_addr", False)

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
        if ranges is None and getattr(self._ctx, "raw", None):
            # raw image: no sections; any address inside the image is a literal
            ranges = self._rodata = [(self._base_va, self._base_va + len(self._data))]
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
        wrapped = self._watch.get(target) or self._watch.get(
            self._ctx.exports.get(callee or "", -1) if self._ctx is not None else -1)
        regs = set(wrapped or ())
        if insn.address in self._watch_calls:
            regs |= set(ARG_REGS)
        for r in regs:
            st = eng.state
            labels = st.get(r).labels | st.pointee(r).labels
            if labels:
                self._arg_taint.setdefault((insn.address, r), set()).update(labels)
        reg = FMT_ARG.get(callee or "")
        if reg is not None:
            st = eng.state
            labels = st.get(reg).labels | st.pointee(reg).labels
            if labels:
                prev = self._tainted.get(insn.address, (reg, set()))[1]
                self._tainted[insn.address] = (reg, set(prev) | set(labels))
        super()._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)
