"""
libc_roles.py: infer libc-equivalent roles (memcpy, sprintf, ...) of unnamed functions in a
statically linked or headerless AArch64 image, so name-keyed sink oracles still resolve.

Why this exists
  1. A raw RTOS image or a static ELF has no PLT and no imports. Every engine that matches
     sinks by imported name (taint, fmtstr, heap, overflow) found nothing to match and
     returned "clean", which reads as a cleared surface.
  2. libc in such images is the same small set of shapes whatever the vendor or libc
     (newlib, musl, uClibc, glibc, Bionic): a copy loop, a fill loop, a NUL-terminated
     walk, and printf / scanf families recognisable from their call sites.

Body shape (one function, <= 4 KB, loops = backward branches):
  memcpy   loop loads and stores through two different bases, storing the loaded value,
           not NUL-terminated; reads x0, x1, x2 at entry
  memset   loop stores, no loads, stored value from x1 (or xzr); reads x0, x1, x2
  strcpy   NUL-terminated loop (cbz / cbnz / tbz on the loaded value) that stores it
  strlen   NUL-terminated loop that only loads
  (leaf functions only: a body that calls out is a wrapper, not the primitive)

Call sites (ARM64RegAnnotator over callers that reference format strings):
  argument k is a literal with % conversions at >= 50% of a function's sites
    k = 0                     printf
    k = 1, x0 a stack buffer  sprintf
    k = 1, conversions' args are stack addresses (scan family)   sscanf
    k = 1, otherwise          fprintf (x0 a stream / fd / level)
    k = 2, x1 not a string    snprintf
    other                     __fmt_x<k> (format in xk; understood by the fmt scanner)

Usage:
    ctx = BinaryContext.from_raw('rtos.bin', 'arm64', base=0x40080000)
    roles = LibcRoleInferer(ctx).infer()          # {va: 'memcpy', ...}; also ctx.roles
    print(LibcRoleInferer.report(roles, ctx))
"""

from __future__ import annotations

import bisect
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .insn_arm64 import Insn, Mem, Reg, from_capstone

_FMT_RE = re.compile(r"%[-+ #0']*(\d+|\*)?(\.(\d+|\*))?(hh|h|ll|l|j|z|t|L|q)?[diouxXeEfgGaAcspn\[]")
_SCAN_CONV = re.compile(r"%(?!%)(\*)?\d*(hh|h|ll|l|j|z|t|L|q|m)?([diouxXeEfgGaAcspn]|\[)")
_BRANCHES = ("b", "cbz", "cbnz", "tbz", "tbnz")
_MAX_BODY = 4096


@dataclass
class _Body:
    loads: int = 0
    stores: int = 0
    copy: bool = False          # a loop stores a value it loaded, through another base
    fill: bool = False          # a loop stores, without loading
    nul: bool = False           # a loop exits on a loaded value == 0
    fill_from_x1: bool = False
    live_in: Set[str] = field(default_factory=set)
    calls: int = 0


def _branch_target(insn: Insn) -> Optional[int]:
    m = insn.mnemonic
    if m == "b" or m.startswith("b.") or m in _BRANCHES:
        return insn.imm(len(insn.ops) - 1)
    return None


def analyse_body(insns: List[Insn]) -> _Body:
    """Shape facts of one function body (see module docstring)."""
    b = _Body()
    written: Set[str] = set()
    for insn in insns:                                       # live-in argument registers
        regs = insn.regs()
        srcs = regs[1:] if regs and not insn.mnemonic.startswith(("st", "cb", "tb", "cmp", "tst")) else regs
        for r in srcs:
            if r.name in ("x0", "x1", "x2", "x3") and r.name not in written:
                b.live_in.add(r.name)
        mem = insn.first_mem()
        if mem is not None and mem.base in ("x0", "x1", "x2") and mem.base not in written:
            b.live_in.add(mem.base)
        if regs and not insn.mnemonic.startswith(("st", "cb", "tb", "cmp", "tst", "b")):
            written.add(regs[0].name)

    b.calls = sum(1 for insn in insns if insn.mnemonic in ("bl", "blr"))
    # loops: [target, branch] for each backward branch
    for i, insn in enumerate(insns):
        t = _branch_target(insn)
        if t is None or t > insn.address:
            continue
        body = [x for x in insns if t <= x.address <= insn.address]
        loaded: Dict[str, str] = {}            # value reg -> base
        load_bases: Set[str] = set()
        stored: List[Tuple[str, str]] = []     # (value reg, base)
        for x in body:
            mem = x.first_mem()
            m = x.mnemonic
            if mem is not None and m.startswith(("ldr", "ldp", "ld1", "ldur")):
                b.loads += 1
                load_bases.add(mem.base)
                for r in x.regs():
                    if r.name != mem.base and (mem.index is None or r.name != mem.index.name):
                        loaded[r.name] = mem.base
            elif mem is not None and m.startswith(("str", "stp", "st1", "stur")):
                b.stores += 1
                for r in x.regs():
                    if r.name != mem.base and (mem.index is None or r.name != mem.index.name):
                        stored.append((r.name, mem.base))
            elif m in ("cbz", "cbnz", "tbz", "tbnz") and x.reg(0) is not None and x.reg(0).name in loaded:
                b.nul = True
            elif m == "cmp" and x.reg(0) is not None and x.reg(0).name in loaded and x.imm(1) == 0:
                b.nul = True
        for val, base in stored:
            if val in loaded and loaded[val] != base:
                b.copy = True
        if stored and not load_bases:
            b.fill = True
            b.fill_from_x1 = any(v in ("x1", "xzr") for v, _ in stored) or "x1" in b.live_in
    return b


def body_role(b: _Body) -> Optional[str]:
    if b.calls:                                  # libc primitives are leaves
        return None
    if b.copy and not b.nul and {"x0", "x1", "x2"} <= b.live_in:
        return "memcpy"
    if b.fill and not b.nul and {"x0", "x2"} <= b.live_in and b.fill_from_x1:
        return "memset"
    if b.copy and b.nul and {"x0", "x1"} <= b.live_in:
        return "strcpy"
    if b.nul and b.loads and not b.stores and "x0" in b.live_in and not b.calls:
        return "strlen"
    return None


def _literal(v, cstr=None) -> Optional[str]:
    """String value of v; a constant pointer is read through cstr (formats like "%s" are
    shorter than the string table's minimum)."""
    while v is not None and v.kind == "copy":
        v = v.chain
    if v is not None and v.kind == "string":
        return v.str_val
    if v is not None and v.kind == "const" and cstr is not None:
        return cstr(v.int_val)
    return None


def _stack(v) -> bool:
    while v is not None and v.kind == "copy":
        v = v.chain
    return v is not None and getattr(v, "is_stack_addr", False)


class LibcRoleInferer:
    """Name unnamed functions by libc role from body shape and call-site arguments."""

    def __init__(self, ctx, max_callers: int = 4000, min_sites: int = 3):
        self._ctx = ctx
        self._max_callers = max_callers
        self._min_sites = min_sites
        self._regions = ctx.image_regions()

    def _code(self, va: int, end: int) -> bytes:
        for rva, data in self._regions:
            off = va - rva
            if 0 <= off < len(data):
                return data[off:off + (end - va)]
        return b""

    def _cstr(self, va: int) -> Optional[str]:
        s = self._ctx.strings.get(va)
        if s is not None:
            return s
        for rva, data in self._regions:
            off = va - rva
            if 0 <= off < len(data):
                end = data.find(b"\0", off, off + 256)
                raw = data[off:end] if end > off else b""
                if raw and all(32 <= c < 127 or c in (9, 10, 13) for c in raw):
                    return raw.decode()
                return None
        return None

    def infer(self, apply: bool = True) -> Dict[int, str]:
        ctx = self._ctx
        named = set(ctx.plt) | set(ctx.exports.values())
        roles: Dict[int, str] = {}
        indeg = Counter(t for _a, t, _l in ctx.call_edges)
        starts = ctx.func_starts

        # 1) call-site arguments for printf / scanf families (strongest evidence)
        roles.update(self._format_roles(named))

        # 2) body shape, for called leaf functions not named yet
        for i, fva in enumerate(starts):
            if fva in named or fva in roles or indeg[fva] < 1:
                continue
            end = starts[i + 1] if i + 1 < len(starts) else fva + _MAX_BODY
            if end - fva > _MAX_BODY:
                continue
            insns = list(from_capstone(self._code(fva, end), fva))
            role = body_role(analyse_body(insns))
            if role:
                roles[fva] = role

        if apply:
            ctx.roles = {**getattr(ctx, "roles", {}), **roles}
        return roles

    def _format_roles(self, skip: Set[int]) -> Dict[int, str]:
        from .arm64_reg_annotator import ARM64RegAnnotator
        ctx = self._ctx
        fmt_strings = {va for va, s in ctx.strings.items() if _FMT_RE.search(s)}
        callers = sorted({f for va in fmt_strings for site in ctx._str_xref_idx.get(va, ())
                          for f in [ctx.func_containing(site)] if f is not None})
        if len(callers) > self._max_callers:
            step = len(callers) / self._max_callers
            callers = [callers[int(i * step)] for i in range(self._max_callers)]
        ra = ARM64RegAnnotator.from_context(ctx)
        starts = ctx.func_starts
        sites: Dict[int, List] = defaultdict(list)
        done: Set[int] = set()

        def annotate(fvas) -> None:
            for fva in fvas:
                if fva in done:
                    continue
                done.add(fva)
                i = bisect.bisect_right(starts, fva)
                end = starts[i] if i < len(starts) and starts[i] - fva <= 8192 else fva + 8192
                for call in ra.annotate_calls(fva, end).calls:
                    if call.target_va and call.target_va not in skip:
                        sites[call.target_va].append(call)

        # round 1: functions referencing a format string; round 2: the other callers of
        # anything that received one (short formats such as "%s" are not in the string
        # table, so their callers have no string xref)
        annotate(callers)
        hit = {t for t, cs in sites.items()
               if any(_FMT_RE.search(_literal(c.args.get(f"x{k}"), self._cstr) or "")
                      for c in cs for k in range(4))}
        more = sorted({a for a, t, _l in ctx.call_edges if t in hit} - done)
        annotate(more[: self._max_callers])
        out: Dict[int, str] = {}
        for tva, calls in sites.items():
            if len(calls) < self._min_sites:
                continue
            votes = Counter()
            for c in calls:
                for k in range(8):
                    s = _literal(c.args.get(f"x{k}"), self._cstr)
                    if s and _FMT_RE.search(s):
                        votes[k] += 1
                        break
            if not votes:
                continue
            k, n = votes.most_common(1)[0]
            if n < max(self._min_sites, len(calls) // 2):
                continue
            fmt_calls = [c for c in calls if _literal(c.args.get(f"x{k}"), self._cstr)]
            out[tva] = self._classify(k, fmt_calls, self._cstr)
        return out

    @staticmethod
    def _classify(k: int, calls, cstr=None) -> str:
        n = len(calls)

        def frac(pred) -> float:
            return sum(1 for c in calls if pred(c)) / n

        if k == 0:
            return "printf"
        if k == 1:
            def scan_shape(c) -> bool:
                convs = [m for m in _SCAN_CONV.finditer(_literal(c.args.get("x1"), cstr) or "")
                         if not m.group(1)]
                return bool(convs) and all(_stack(c.args.get(f"x{2 + j}"))
                                           for j in range(min(len(convs), 6)))
            if frac(scan_shape) >= 0.5 and frac(lambda c: _stack(c.args.get("x0"))) < 0.5:
                return "sscanf"
            if frac(lambda c: _stack(c.args.get("x0"))) >= 0.5:
                return "sprintf"
            return "fprintf"
        if k == 2 and frac(lambda c: _literal(c.args.get("x1"), cstr) is None) >= 0.8:
            return "snprintf"
        return f"__fmt_x{k}"

    @staticmethod
    def report(roles: Dict[int, str], ctx=None) -> str:
        if not roles:
            return "libc roles: none inferred"
        by = defaultdict(list)
        for va, r in roles.items():
            by[r].append(va)
        lines = [f"libc roles: {len(roles)} function(s) inferred"]
        for r in sorted(by):
            vas = sorted(by[r])
            more = f" (+{len(vas) - 6})" if len(vas) > 6 else ""
            lines.append(f"  {r:<12} " + ", ".join(f"0x{v:x}" for v in vas[:6]) + more)
        return "\n".join(lines)
