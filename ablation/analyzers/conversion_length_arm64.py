"""
conversion_length_arm64.py: width / signedness conversion bugs in lengths (AArch64).

Why this exists
  1. Four real bugs on one AArch64 target shared one shape no engine modelled:
     a value is checked at one width or signedness and used at another. They are
     conversions, not arithmetic overflow, so ARM64IntOverflowScanner (multiplies)
     and ARM64LengthUnderflowScanner (len - N) both stayed silent.
  2. The AArch64 markers are cheap and exact: sxtw / sxth / sxtb / ldrsw / ldrsh /
     ldrsb (and `, sxtw` operand extends) into a size; and wT, wN, #0xffff / uxth
     before a check while wN is used after; uxtb / and #0xff counters in a loop
     compare.
  3. snprintf's verdict depends on the libc: some reject size > INT_MAX and write
     nothing, others trust it. The same wrapped size into sprintf-like copies is fatal.

Kinds
  sext_signed_check   a value bounded only by a signed compare (b.gt / b.ge / ...:
                      a negative value passes) is sign-extended into a size / count
  sext_unchecked      an attacker value sign-extended into a size with no check (HIGH);
                      a signed 32/16/8-bit field loaded with ldrsw / ldrsh / ldrsb straight
                      into a size, source unknown (LOW: e.g. a length stored to an int global)
  masked_check        the bound was checked on len & 0xFFFF (or uxth / uxtb), the
                      size uses the full-width len
  byte_counter_loop   a loop compares a byte-width counter (uxtb / and #0xff / ldrb)
                      against a limit wider than 255: a limit above 255 never ends
  signed_clamp        min(len, MAX) done with a signed csel (lt / le / gt / ge) reaches a
                      size: a negative len passes the clamp
  (not reported: a mask of a value already zero-extended by ldrh / ldurh / uxth, and
   capacity - (used + ret) after the *printf return was checked against the remaining
   capacity by a signed compare: the cmp / ccmp ... ge + b.le truncation idiom)
  narrowing_store     a 64-bit length (strlen / read / recv result, or attacker data)
                      stored with a 32-bit str: the high half is lost (attacker data:
                      any store; a library length: stores outside the stack frame)

Guards: an unsigned compare (b.hi / b.hs / b.lo / b.ls / cs / cc) of the value, or a
signed upper bound together with a lower bound (cmp #0 + b.lt / b.ge, tbnz / tbz #31),
makes a sign extension safe.

Severity: HIGH when the value carries recv / read (or --sources) data, else MEDIUM
(structural: still a bug, the source is just unknown). A size reaching snprintf /
vsnprintf x1 follows snprintf_rejects_huge: True -> LOW (that libc writes nothing),
False -> unchanged, None (unknown) -> MEDIUM with a note.

Usage:
    s = ARM64ConversionLengthScanner.from_context(ctx)
    print(s.report(s.scan()))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .arm64_tag_scanner import TAG, ARM64TaggingScanner, TagSite, _FLAG_USERS, attacker
from .insn_arm64 import Insn
from .isa_arm64 import ARG_REGS
from .length_underflow_arm64 import _LENGTH_ARGS
from .taint_tracker_arm64 import Taint, TaintEngine

_SEXT = frozenset({"sxtw", "sxth", "sxtb", "ldrsw", "ldrsh", "ldrsb", "ldursw", "ldursh", "ldursb"})
_SIGNED = frozenset({"gt", "ge", "lt", "le", "mi", "pl"})
_UNSIGNED = frozenset({"hi", "hs", "cs", "lo", "cc", "ls"})
_LEN_RETURNS = frozenset({"strlen", "strnlen", "read", "recv", "recvfrom", "pread", "fread",
                          "ftell", "lseek", "readlink", "wcslen", "__strlen_chk"})
_PRINTF_RET = frozenset({"snprintf", "vsnprintf", "sprintf", "vsprintf", "__snprintf_chk",
                         "__vsnprintf_chk", "__sprintf_chk", "__vsprintf_chk"})
_CSEL = frozenset({"csel", "csinc", "csinv", "csneg"})
_SNPRINTF = frozenset({"snprintf", "vsnprintf", "__snprintf_chk", "__vsnprintf_chk"})
# libcs whose snprintf rejects size > INT_MAX (returns -1, writes nothing): True;
# trusts the size: False. Only implementations whose source was checked are listed.
LIBC_SNPRINTF_REJECTS_HUGE: Dict[str, bool] = {"musl": True, "glibc": False}


@dataclass
class ConversionFinding:
    kind: str
    func_va: int
    site_va: int            # the conversion (or the loop compare)
    sink_va: int            # call receiving the converted value (0 for byte_counter_loop)
    sink: str
    arg: str
    severity: str
    detail: str
    sources: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "func_va": hex(self.func_va), "site_va": hex(self.site_va),
                "sink_va": hex(self.sink_va) if self.sink_va else "", "sink": self.sink,
                "arg": self.arg, "severity": self.severity, "detail": self.detail,
                "sources": self.sources, "note": self.note}

    def fmt(self) -> str:
        src = f"  source={self.sources}" if self.sources else ""
        sink = f" -> {self.sink}({self.arg}) @0x{self.sink_va:x}" if self.sink_va else ""
        note = f"\n      note: {self.note}" if self.note else ""
        return (f"  {self.severity:6s}  {self.kind:18s}  0x{self.func_va:x}  site 0x{self.site_va:x}"
                f"{sink}{src}\n      {self.detail}{note}")


@dataclass
class _Cmp:
    func_va: int
    va: int
    regs: Tuple[str, ...]
    labels: frozenset
    cond: str = ""            # condition code that consumed the flags ("" = cbz / tbz)
    zero: bool = False        # compared against 0 / tested the sign bit


class ARM64ConversionLengthScanner(ARM64TaggingScanner):
    """Checked at one width / signedness, used at another (see module docstring)."""

    KIND = "conv"

    def __init__(self, *args, snprintf_rejects_huge: Optional[bool] = None, **kw):
        super().__init__(*args, **kw)
        self.snprintf_rejects_huge = snprintf_rejects_huge
        self._found: Dict[Tuple, ConversionFinding] = {}

    def _reset(self) -> None:
        super()._reset()
        self._cmps: List[_Cmp] = []
        self._pending_cmp: Optional[_Cmp] = None
        self._found = {}
        self._wide: Dict[str, bool] = {}           # register -> last written as 64-bit
        self._wide_fva = None
        self._zext: Dict[str, int] = {}            # register -> known zero-extended width (bits)
        self._trunc_checked: Set[str] = set()      # printf returns tested by a signed compare

    # ---- public API ---------------------------------------------------------

    def scan(self, depth: int = 3) -> List[ConversionFinding]:
        self._reset()
        self.run_interprocedural(depth=depth)
        return sorted(self._found.values(), key=lambda f: (f.func_va, f.site_va, f.sink_va))

    def report(self, findings: List[ConversionFinding]) -> str:
        if not findings:
            return "ARM64 conversion-length scan: no findings."
        high = sum(1 for f in findings if f.severity == "HIGH")
        return "\n".join([f"ARM64 conversion-length scan: {len(findings)} finding(s) ({high} HIGH)"]
                         + [f.fmt() for f in findings])

    # ---- sites --------------------------------------------------------------

    def _site(self, eng: TaintEngine, insn: Insn, fva: int) -> Optional[TagSite]:
        m = insn.mnemonic
        rd = insn.reg(0)
        if rd is None or rd.name in ("xzr", "sp"):
            return None
        st = eng.state
        if m in _SEXT and rd.bits == 64:
            src = insn.reg(1) if not m.startswith("ld") else None
            regs = [src.name] if src is not None else []
            labels = st.get(src.name).labels if src is not None else frozenset()
            return TagSite(fva, insn.address, m, regs, attacker(labels), {"kind": "sext"})
        # `, sxtw` operand extend in an add / sub / mov producing a 64-bit value
        for r in insn.regs()[1:]:
            if r.shift == "sxtw" and rd.bits == 64:
                return TagSite(fva, insn.address, m, [r.name], attacker(st.get(r.name).labels),
                               {"kind": "sext"})
        return None

    def _before_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        m = insn.mnemonic
        st = eng.state
        if fva != self._wide_fva:
            self._wide, self._wide_fva = {}, fva
            self._zext = {}
        # capacity - x: the minuend must be a constant *before* the sub overwrites it
        rn0 = insn.reg(1)
        self._capsub_at = (insn.address if m == "sub" and rn0 is not None and insn.reg(2) is not None
                           and st.consts.get(rn0.name) is not None else None)
        if m in ("str", "stur") and insn.reg(0) is not None and insn.reg(0).bits == 32:
            self._narrow_store(eng, insn, fva)
        # flag consumer: finish the pending compare with its condition
        if self._pending_cmp is not None and (m.startswith("b.") or m in _FLAG_USERS):
            c = self._pending_cmp
            c.cond = m[2:] if m.startswith("b.") else insn.raw.rsplit(",", 1)[-1].strip() if insn.raw else ""
            self._cmps.append(c)
            self._pending_cmp = None
            if c.cond in _SIGNED:
                # snprintf truncation check (cmp / ccmp of a capacity against the return)
                self._trunc_checked |= {l for l in c.labels if l.startswith(f"{TAG}pfret@")}
            if m.startswith("b."):
                self._byte_loop(eng, insn, c, fva)
        if m in ("cmp", "cmn", "subs", "tst", "ccmp"):
            regs = [r for r in (insn.regs()[1:] if m == "subs" else insn.regs()) if r.name != "xzr"]
            labels = frozenset().union(*(st.get(r.name).labels for r in regs)) if regs else frozenset()
            ops = [str(o) for o in insn.ops]
            zero = insn.imm(len(insn.ops) - 1) == 0 or any(o in ("wzr", "xzr") for o in ops[1:])
            byte = {r.name for r in regs if r.shift in ("uxtb",)}
            self._pending_cmp = _Cmp(fva, insn.address, tuple(r.name for r in regs), labels, zero=zero)
            self._pending_cmp_insn = (insn, byte)
        elif m in ("cbz", "cbnz", "tbz", "tbnz") and insn.reg(0) is not None:
            r = insn.reg(0)
            bit = insn.imm(1) if m.startswith("tb") else None
            self._cmps.append(_Cmp(fva, insn.address, (r.name,), st.get(r.name).labels,
                                   zero=(m.startswith("cb") or bit in (31, 63))))
        if m in ("bl", "blr", "ret"):
            self._pending_cmp = None
        super()._before_step(eng, insn, fva)

    def _after_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        m = insn.mnemonic
        st = eng.state
        rd, rn = insn.reg(0), insn.reg(1)
        masked = (m in ("uxth", "uxtb") or (m == "and" and insn.imm(2) in (0xFF, 0xFFFF)))
        mask_w = 8 if (m == "uxtb" or insn.imm(2) == 0xFF) else 16
        if masked and rn is not None and self._zext.get(rn.name, 64) <= mask_w:
            masked = False                       # already zero-extended (ldrh / ldurh): a no-op
        self._track_zext(insn, m, rd, rn, st)
        if m in _CSEL and rd is not None and rd.name != "xzr":
            cond = (insn.raw or "").rsplit(",", 1)[-1].strip()
            if cond in _SIGNED:
                va = insn.address
                ops = [r.name for r in insn.regs()[1:3]]
                t = st.get(rd.name)
                lbl = f"{TAG}clamp@{va:#x}"
                st.regs[rd.name] = Taint(t.labels | {lbl}, t.bound)
                self._sites[lbl] = TagSite(fva, va, m, ops, attacker(t.labels),
                                           {"kind": "clamp", "cond": cond})
        if rd is not None and rd.name not in ("xzr", "sp") and not m.startswith(("st", "cmp", "b", "tst")):
            self._wide[rd.name] = rd.bits == 64
        if masked and rd is not None and rn is not None and rd.name != rn.name:
            va = insn.address
            width = 8 if (m == "uxtb" or insn.imm(2) == 0xFF) else 16
            t = st.get(rd.name)
            st.regs[rd.name] = Taint(t.labels | {f"{TAG}masked@{va:#x}"}, t.bound)
            o = st.get(rn.name)
            st.regs[rn.name] = Taint(o.labels | {f"{TAG}mask@{va:#x}"}, o.bound)
            self._sites[f"{TAG}mask@{va:#x}"] = TagSite(fva, va, m, [rn.name], attacker(o.labels),
                                                       {"kind": "mask", "width": width})
        if rd is not None and rd.name != "xzr" and (
                m in ("uxtb", "ldrb", "ldrsb", "ldurb") or (m == "and" and insn.imm(2) == 0xFF)):
            t = st.get(rd.name)
            st.regs[rd.name] = Taint(t.labels | {f"{TAG}byte@{insn.address:#x}"}, t.bound)
        elif rd is not None and rd.name != "xzr" and (
                m in ("uxth", "ldrh", "ldrsh", "ldurh") or (m == "and" and insn.imm(2) == 0xFFFF)):
            t = st.get(rd.name)
            st.regs[rd.name] = Taint(t.labels | {f"{TAG}half@{insn.address:#x}"}, t.bound)
        super()._after_step(eng, insn, fva)

    def _track_zext(self, insn: Insn, m: str, rd, rn, st) -> None:
        """Known zero-extended width of rd after insn (ldrh / ldurh / ldrb / uxth / and #imm;
        kept by rev16 / mov copies); a capacity-minus-x subtraction is tagged %capsub."""
        if rd is None or rd.name in ("xzr", "sp") or m.startswith(("st", "cmp", "cmn", "tst", "b", "cb", "tb")):
            return
        w = None
        if m in ("ldrh", "ldurh", "uxth"):
            w = 16
        elif m in ("ldrb", "ldurb", "uxtb"):
            w = 8
        elif m == "and" and insn.imm(2) is not None and insn.imm(2) in (0xFF, 0xFFFF):
            w = 8 if insn.imm(2) == 0xFF else 16
        elif m in ("rev16", "mov") and rn is not None and insn.imm(1) is None:
            w = self._zext.get(rn.name)
        if w is None:
            self._zext.pop(rd.name, None)
        else:
            self._zext[rd.name] = w
        if getattr(self, "_capsub_at", None) == insn.address:
            t = st.get(rd.name)
            st.regs[rd.name] = Taint(t.labels | {f"{TAG}capsub@{insn.address:#x}"}, t.bound)

    # ---- byte-width loop counters ------------------------------------------

    def _byte_loop(self, eng: TaintEngine, br: Insn, c: _Cmp, fva: int) -> None:
        target = br.imm(0)
        if target is None or target >= br.address or target < fva:
            return                                     # not a backward (loop) branch
        insn, byte_ext = getattr(self, "_pending_cmp_insn", (None, set()))
        if insn is None or insn.mnemonic != "cmp" or len(insn.ops) < 2:
            return
        st = eng.state
        a = insn.reg(0)
        b = insn.reg(1)
        lim_imm = insn.imm(1)

        def is_byte(r) -> bool:
            return r is not None and (r.name in byte_ext or r.shift == "uxtb"
                                      or any(l.startswith(f"{TAG}byte@") for l in st.get(r.name).labels))

        def wide(r) -> Tuple[bool, str]:
            if r is None:
                return (lim_imm is not None and lim_imm > 0xFF, f"#{lim_imm:#x}" if lim_imm else "")
            labels = st.get(r.name).labels
            if any(l.startswith(f"{TAG}half@") for l in labels):
                return True, f"{r.name} (16-bit)"
            k = st.consts.get(r.name)
            if k is not None and k > 0xFF:
                return True, f"{r.name}={k:#x}"
            return bool(attacker(labels)), f"{r.name} (attacker)"

        for ctr, lim in ((a, b), (b, a)):
            if ctr is None or not is_byte(ctr):
                continue
            ok, desc = wide(lim if lim is not None else None) if lim is not None or lim_imm is not None else (False, "")
            if ok:
                labels = attacker(st.get(lim.name).labels) if lim is not None else frozenset()
                self._add(ConversionFinding(
                    "byte_counter_loop", fva, insn.address, 0, "", "",
                    "HIGH" if labels else "MEDIUM",
                    f"loop at 0x{target:x}..0x{br.address:x} compares byte-width {ctr.name} "
                    f"against {desc}: a limit above 255 never terminates",
                    ",".join(sorted(labels))))
                return

    # ---- sinks --------------------------------------------------------------

    def _on_call(self, eng: TaintEngine, insn: Insn, target: int, callee: Optional[str],
                 fva: int, func_name: str, findings, depth: int, enqueue) -> None:
        self._pending_cmp = None
        size_args = _LENGTH_ARGS.get(callee or "", ()) or self._wrapper_regs(target, callee, "length")
        st = eng.state
        for r in size_args:
            t = st.get(r)
            sites = [(l, self._sites[l]) for l in t.labels if l in self._sites]
            clamped = False
            for label, site in sites:
                if site.extra.get("kind") == "clamp":
                    clamped |= self._check_clamp(label, site, t, insn, callee, r)
            for label, site in sites:
                if site.extra.get("kind") == "sext" and not clamped:
                    self._check_sext(label, site, t, insn, callee, r, fva)
                elif site.extra.get("kind") == "mask":
                    self._check_mask(label, site, t, insn, callee, r, fva)
        # structural tags (no attacker label) stay in this function: they must not make
        # arguments look tainted to the interprocedural queue or the return value
        for r in ARG_REGS:
            t = st.get(r)
            if t.labels and not attacker(t.labels):
                st.regs[r] = Taint(frozenset(), t.bound)
        super()._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)
        # 64-bit length results (size_t / ssize_t), for narrowing stores
        self._wide = {k: v for k, v in self._wide.items() if k not in ("x0", "x1")}
        if callee in _PRINTF_RET or self._wrapper_regs(target, callee, "format"):
            t = st.get("x0")
            st.regs["x0"] = Taint(t.labels | {f"{TAG}pfret@{insn.address:#x}"}, t.bound)
        if callee in _LEN_RETURNS:
            t = st.get("x0")
            st.regs["x0"] = Taint(t.labels | {f"{TAG}len64@{insn.address:#x}"}, t.bound)
            self._wide["x0"] = True

    def _narrow_store(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        r = insn.reg(0)
        if r.name in ("xzr",) or not self._wide.get(r.name):
            return
        t = eng.state.get(r.name)
        src = attacker(t.labels)
        lens = [l for l in t.labels if l.startswith(f"{TAG}len64@")]
        mem = insn.first_mem()
        on_stack = mem is not None and mem.base in ("sp", "x29")
        if not src and not (lens and not on_stack):
            return
        where = str(mem) if mem is not None else "memory"
        what = (f"attacker 64-bit value ({','.join(sorted(src))})" if src else
                f"64-bit length (result of the call at {lens[0].split('@')[1]})")
        self._add(ConversionFinding(
            "narrowing_store", fva, insn.address, 0, "", "",
            "HIGH" if src else "LOW",
            f"{what} stored as 32 bits ({insn.mnemonic} {r.name.replace('x', 'w', 1)}, {where}): "
            f"the high half is dropped; a reload (ldrsw / ldr w) sees a different length",
            ",".join(sorted(src))))

    def _check_clamp(self, label, site, t, insn, callee, arg) -> bool:
        g = self._guards_on(label, site)
        if any(c.cond in _UNSIGNED for c in g) or any(c.zero and c.va != site.va for c in g):
            return False
        src = attacker(t.labels)
        self._add(self._snprintf(ConversionFinding(
            "signed_clamp", site.func_va, site.va, insn.address, callee or "", arg,
            "HIGH" if src else "MEDIUM",
            f"{site.mnemonic} ... {site.extra['cond']} at 0x{site.va:x} clamps with a signed "
            f"compare: a negative length passes the clamp and becomes a huge {callee} size",
            ",".join(sorted(src)))))
        return True

    def _guards_on(self, label: str, site: TagSite) -> List[_Cmp]:
        out = []
        for c in self._cmps:
            if label in c.labels and c.va > site.va:
                out.append(c)                                       # the converted value
            elif c.func_va == site.func_va and c.va < site.va and set(c.regs) & set(site.regs):
                out.append(c)                                       # its source, before
        return out

    def _trunc_bounded(self, labels) -> bool:
        """capacity - (... + printf return) after that return was truncation-checked:
        the checked return is below the remaining capacity, so the difference is positive."""
        return (any(l.startswith(f"{TAG}capsub@") for l in labels)
                and any(l in self._trunc_checked for l in labels))

    def _check_sext(self, label, site, t, insn, callee, arg, fva) -> None:
        if self._trunc_bounded(t.labels):
            return
        g = self._guards_on(label, site)
        unsigned = any(c.cond in _UNSIGNED for c in g)
        upper = [c for c in g if c.cond in _SIGNED and not c.zero]
        lower = [c for c in g if c.zero and (c.cond in _SIGNED or c.cond == "")]
        src = attacker(t.labels)
        if unsigned or (upper and lower):
            return
        if upper:
            kind, detail = "sext_signed_check", (
                f"{site.mnemonic} at 0x{site.va:x} after a signed bound check "
                f"(b.{upper[0].cond} at 0x{upper[0].va:x}): a negative value passes and becomes "
                f"a huge {callee} size")
            sev = "HIGH" if src else "MEDIUM"
        elif src or site.mnemonic.startswith("ld"):
            what = "attacker value" if src else "narrow signed field"
            kind, detail = "sext_unchecked", (
                f"{what} sign-extended ({site.mnemonic} at 0x{site.va:x}) into {callee} "
                f"{arg} with no bound check")
            sev = "HIGH" if src else "LOW"
        else:
            return
        self._add(self._snprintf(ConversionFinding(kind, site.func_va, site.va, insn.address,
                                                   callee or "", arg, sev, detail,
                                                   ",".join(sorted(src)))))

    def _check_mask(self, label, site, t, insn, callee, arg, fva) -> None:
        masked = f"{TAG}masked@{site.va:#x}"
        if masked in t.labels:
            return                                         # the masked value itself is used
        checked_masked = [c for c in self._cmps if masked in c.labels]
        checked_full = [c for c in self._cmps if label in c.labels and c.va > site.va]
        if not checked_masked or checked_full:
            return
        src = attacker(t.labels)
        w = site.extra.get("width", 16)
        self._add(self._snprintf(ConversionFinding(
            "masked_check", site.func_va, site.va, insn.address, callee or "", arg,
            "HIGH" if src else "MEDIUM",
            f"bound checked on the low {w} bits ({site.mnemonic} at 0x{site.va:x}, compare at "
            f"0x{checked_masked[0].va:x}); {callee} {arg} uses the full-width value",
            ",".join(sorted(src)))))

    def _snprintf(self, f: ConversionFinding) -> ConversionFinding:
        if f.sink in _SNPRINTF and f.arg == "x1":
            if self.snprintf_rejects_huge is True:
                f.severity = "LOW"
                f.note = "this libc's snprintf rejects size > INT_MAX and writes nothing"
            elif self.snprintf_rejects_huge is None:
                f.severity = "MEDIUM" if f.severity == "HIGH" else f.severity
                f.note = ("depends on the libc: some snprintf builds (e.g. musl) reject size > "
                          "INT_MAX and write nothing, others (glibc) trust it; pass --libc")
        return f

    def _add(self, f: ConversionFinding) -> None:
        self._found.setdefault((f.kind, f.site_va, f.sink_va, f.arg), f)
