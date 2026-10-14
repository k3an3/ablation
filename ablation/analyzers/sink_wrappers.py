"""
sink_wrappers.py: infer vendor wrappers of dangerous functions (log(fmt, ...) -> vsnprintf,
run(cmd) -> system, buf_put(b, p, n) -> memcpy) so their *call sites* are checked like the
libc sink itself.

Why this exists
  1. Sink sets are keyed on libc names. A helper defined inside the target, e.g.
     log_event(int kind, const char *fmt, ...) forwarding to vsnprintf, is invisible to
     them, and so is every call site of it. Only the vsnprintf inside the helper is seen,
     where the format is legitimately the caller's problem. In firmware the dangerous
     functions are usually vendor wrappers one or two levels above libc, and that is
     where unsanitised data gets passed.
  2. The shape is crisp on AArch64: the wrapper hands one of its own incoming
     parameter registers, unchanged (copies and spills followed), to a sink argument.
     A varargs wrapper also builds the AAPCS64 register save area (stp x1..x7 /
     q0..q7 to consecutive frame slots) and initialises a va_list (gr_offs / vr_offs).

What it infers (fixpoint: wrappers of wrappers too)
  format   param -> the format argument of a printf / v*printf (or format wrapper)
  exec     param -> system / popen / exec* command arguments
  length   param -> a length argument (memcpy x2, read x2, malloc x0, snprintf x1 ...)
  copy     param -> a pointer argument whose pointee must not be attacker data
           (strcpy src, memcpy src, sprintf args ...)

Each wrapper records: which sink it reaches (the libc name at the end of the chain), the
parameter registers per kind, and for varargs functions the save-area facts
(gr_offs = -8 * GP slots saved, vr_offs = -16 * FP/SIMD slots saved; named GP params =
8 - gr_offs / -8).

Results: ctx.wrappers {va: SinkWrapper.as_dict()} (cached), consumed by
ARM64FormatStringScanner (findings at the wrapper's call sites), ARM64TaintTracker
(wrappers become sinks) and the length scanners. --sink-map entries "format:x1",
"exec:x0", "length:x2" override or add wrappers by address.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .fortify import FMT_CHK_FLAG
from .insn_arm64 import from_capstone

_MAX_FUNC = 8192

_EXEC_ARGS: Dict[str, Tuple[str, ...]] = {
    "system": ("x0",), "popen": ("x0",), "execve": ("x0", "x1"), "execv": ("x0", "x1"),
    "execvp": ("x0", "x1"), "execvpe": ("x0", "x1"), "execl": ("x0", "x1", "x2"),
    "execlp": ("x0", "x1", "x2"), "execle": ("x0", "x1", "x2"),
}
KINDS = ("format", "exec", "length", "copy")


def base_sink_args(name: str) -> Dict[str, Tuple[str, ...]]:
    """{kind: arg regs} for a libc sink name ({} if none)."""
    from .format_string_scanner_arm64 import fmt_reg
    from .length_underflow_arm64 import _LENGTH_ARGS
    from .taint_tracker_arm64 import _SINK_POINTER_ARGS
    out: Dict[str, Tuple[str, ...]] = {}
    r = fmt_reg(name)
    if r:
        out["format"] = (r,)
    if name in _EXEC_ARGS:
        out["exec"] = _EXEC_ARGS[name]
    if name in _LENGTH_ARGS:
        out["length"] = _LENGTH_ARGS[name]
    if name in _SINK_POINTER_ARGS and name not in _EXEC_ARGS:
        out["copy"] = _SINK_POINTER_ARGS[name]
    return out


@dataclass
class SinkWrapper:
    va: int
    name: str
    wraps: str                                   # libc sink at the end of the chain
    via: str                                     # direct callee (sink or inner wrapper)
    site: int                                    # call site inside the wrapper
    kinds: Dict[str, List[str]] = field(default_factory=dict)   # kind -> param regs
    varargs: bool = False
    gr_offs: Optional[int] = None
    vr_offs: Optional[int] = None
    source: str = "inferred"                     # or "sink-map"
    chk_flag: Optional[int] = None               # FORTIFY flag when the chain ends in __*printf_chk

    @property
    def fmt_reg(self) -> Optional[str]:
        regs = self.kinds.get("format")
        return regs[0] if regs else None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["va"] = f"{self.va:#x}"
        d["site"] = f"{self.site:#x}"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SinkWrapper":
        d = dict(d)
        d["va"] = int(str(d["va"]), 16)
        d["site"] = int(str(d.get("site", "0")), 16)
        return cls(**d)

    def fmt(self) -> str:
        kinds = ", ".join(f"{k} {'/'.join(v)}" for k, v in self.kinds.items())
        va = ""
        if self.varargs:
            gp = f"{-self.gr_offs // 8} GP" if self.gr_offs is not None else "? GP"
            vr = f", {-self.vr_offs // 16} FP" if self.vr_offs is not None else ""
            va = f"  varargs ({gp}{vr} slots saved)"
        via = f" via {self.via}" if self.via != self.wraps else ""
        return f"  0x{self.va:x}  {self.name:<24s} -> {self.wraps}{via}  [{kinds}]{va}  ({self.source})"


def _arg_index(v) -> int:
    while v is not None and v.kind == "copy":
        v = v.chain
    return v.int_val if v is not None and v.kind == "arg" else -1


_STP = re.compile(r"^(stp|str)$")


def varargs_layout(code: bytes, va: int) -> Tuple[bool, Optional[int], Optional[int]]:
    """(is_varargs, gr_offs, vr_offs) from the prologue: x(n)..x7 saved to consecutive
    slots (with x7 among them) is the GP register save area; q-register saves are the FP
    area; `mov wN, #-k` constants matching their sizes are gr_offs / vr_offs."""
    saved: Dict[str, int] = {}
    qsaved: Set[str] = set()
    consts: List[int] = []
    for i, insn in enumerate(from_capstone(code[:4 * 48], va)):
        m = insn.mnemonic
        mem = insn.first_mem()
        if _STP.match(m) and mem is not None and mem.base in ("sp", "x29", "x8", "x9", "x10"):
            for k, r in enumerate(insn.regs()):
                if r.name == mem.base:
                    continue
                if re.fullmatch(r"x[1-7]", r.name) and r.bits == 64:
                    saved[r.name] = mem.offset + 8 * k
                elif re.fullmatch(r"q[0-7]", r.name) or re.fullmatch(r"v[0-7]", r.name):
                    qsaved.add(r.name)
        elif m in ("mov", "movn") and insn.reg(0) is not None and insn.imm(1) is not None:
            v = insn.imm(1)
            if v < 0 or v >= 0x80000000:
                consts.append(v if v < 0 else v - (1 << 32))
        if m in ("bl", "blr", "ret", "b"):
            break
    if "x7" not in saved:
        return False, None, None
    n = 7
    while n >= 1 and f"x{n}" in saved:
        n -= 1
    gp = 7 - n                                       # x(n+1)..x7 saved
    if gp < 1:
        return False, None, None
    gr = -8 * gp
    vr = -16 * len(qsaved) if qsaved else None
    if consts:
        gr = next((c for c in consts if c == -8 * gp), gr)
        vr = next((c for c in consts if c % 16 == 0 and -0x80 <= c < 0 and c != gr), vr)
    return True, gr, vr


class SinkWrapperInferer:
    """Infer SinkWrapper records for one BinaryContext (arm64)."""

    def __init__(self, ctx, max_rounds: int = 6):
        self._ctx = ctx
        self._max_rounds = max_rounds

    def infer(self, apply: bool = True) -> Dict[int, SinkWrapper]:
        from .arm64_reg_annotator import ARM64RegAnnotator
        ctx = self._ctx
        names = ctx.call_names()
        ra = ARM64RegAnnotator.from_context(ctx)
        starts = ctx.func_starts
        regions = ctx.image_regions()

        # 1) one annotation pass: calls that pass one of the caller's own parameters
        compact: Dict[int, List[Tuple[int, str, int, Dict[str, int]]]] = {}
        for i, fva in enumerate(starts):
            if fva in ctx.plt or base_sink_args(names.get(fva, "")):
                continue                       # libc functions themselves are not wrappers
            end = starts[i + 1] if i + 1 < len(starts) and starts[i + 1] - fva <= _MAX_FUNC else fva + _MAX_FUNC
            rows = []
            for c in ra.annotate_calls(fva, end).calls:
                pm = {r: _arg_index(v) for r, v in c.args.items() if _arg_index(v) >= 0}
                if pm:
                    # -fPIC: a call to this binary's own export goes through its PLT stub
                    tva = ctx.exports.get(c.target_name, c.target_va) if c.target_name else c.target_va
                    flag = None
                    freg = FMT_CHK_FLAG.get(c.target_name or "")
                    fv = c.args.get(freg) if freg else None
                    if fv is not None and fv.kind in ("const", "zero"):
                        flag = fv.int_val if fv.kind == "const" else 0
                    rows.append((tva, c.target_name, c.site_va, pm, flag))
            if rows:
                compact[fva] = rows

        # 2) fixpoint over the compact rows
        wrappers: Dict[int, SinkWrapper] = {}
        for _ in range(self._max_rounds):
            changed = False
            for fva, rows in compact.items():
                for tva, tname, site, pm, flag in rows:
                    inner = wrappers.get(tva)
                    if inner is not None:
                        sargs = {k: tuple(v) for k, v in inner.kinds.items()}
                        wraps, via, flag = inner.wraps, inner.name, inner.chk_flag
                    else:
                        sargs = base_sink_args(tname or names.get(tva, ""))
                        wraps = via = tname or names.get(tva, "")
                    for kind, regs in sargs.items():
                        mapped = sorted({f"x{pm[r]}" for r in regs if r in pm})
                        if not mapped:
                            continue
                        w = wrappers.get(fva)
                        if w is None:
                            w = wrappers[fva] = SinkWrapper(
                                fva, names.get(fva) or f"fn_{fva:x}", wraps, via, site)
                            changed = True
                        if kind == "format" and flag is not None:
                            w.chk_flag = flag
                        cur = set(w.kinds.get(kind, ()))
                        if not set(mapped) <= cur:
                            w.kinds[kind] = sorted(cur | set(mapped))
                            changed = True
            if not changed:
                break

        # 3) varargs layout of each wrapper
        for w in wrappers.values():
            code = b""
            for rva, data in regions:
                if 0 <= w.va - rva < len(data):
                    code = data[w.va - rva:w.va - rva + 4 * 48]
                    break
            w.varargs, w.gr_offs, w.vr_offs = varargs_layout(code, w.va)

        if apply:
            merged = {va: SinkWrapper.from_dict(d) for va, d in getattr(ctx, "wrappers", {}).items()
                      if d.get("source") == "sink-map"}
            ctx.wrappers = {va: w.as_dict() for va, w in wrappers.items()}
            ctx.wrappers.update({va: w.as_dict() for va, w in merged.items()})
        return wrappers

    @staticmethod
    def report(wrappers: Dict[int, "SinkWrapper"]) -> str:
        if not wrappers:
            return "sink wrappers: none"
        lines = [f"sink wrappers: {len(wrappers)} (param -> sink argument; override with --sink-map "
                 f'\'{{"0xVA": "format:x1"}}\')']
        lines += [w.fmt() for _va, w in sorted(wrappers.items())]
        return "\n".join(lines)


def wrappers_of(ctx) -> Dict[int, SinkWrapper]:
    """ctx.wrappers as SinkWrapper objects."""
    return {va: SinkWrapper.from_dict(d) for va, d in (getattr(ctx, "wrappers", None) or {}).items()}
