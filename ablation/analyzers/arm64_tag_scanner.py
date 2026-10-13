"""
arm64_tag_scanner.py: shared base for ARM64 "tainted operation -> sink" scanners.

ARM64IntOverflowScanner and ARM64LengthUnderflowScanner have the same shape:

  1. ARM64TaintTracker's pass makes recv/read buffer contents, spills, aliases and
     callees attacker-tainted.
  2. A subclass picks interesting operations (_site): a multiply that can wrap, a
     header-size subtraction, ... The result register gets a tag label
     ("%<kind>@<va>"), which then travels with the value like any taint label:
     through copies, spills and calls.
  3. The subclass checks sinks in _on_call (allocator size, call argument, ...)
     for tag labels.
  4. Conditionals that test tainted values are recorded as guards (a cmp / subs
     only once a b.cond / csel / ccmp actually consumes its flags); guarded() says
     whether a site's value (after it) or its operands (before it) were tested,
     or whether an umulh/smulh overflow check over them was tested.

Tag labels start with "%" so they never collide with source labels (which may
contain "@", e.g. "read@0x1234" from svc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from .insn_arm64 import Insn
from .isa_arm64 import ISA
from .taint_tracker_arm64 import ARM64TaintTracker, Taint, TaintEngine

TAG = "%"
_COMPARES = frozenset({"cmp", "cmn", "tst", "ccmp", "ccmn", "subs", "adds", "ands"})
_REG_BRANCH = frozenset({"cbz", "cbnz", "tbz", "tbnz"})
_HI_MULS = frozenset({"umulh", "smulh"})
# read NZCV: a flag-setting compare only guards anything if one of these follows
_FLAG_USERS = frozenset({"csel", "csinc", "csinv", "csneg", "cset", "csetm", "cinc", "cinv",
                         "cneg", "ccmp", "ccmn", "adc", "adcs", "sbc", "sbcs", "fcsel"})


def attacker(labels: FrozenSet[str]) -> FrozenSet[str]:
    """Source labels only (drop scanner tags)."""
    return frozenset(l for l in labels if not l.startswith(TAG))


@dataclass
class TagSite:
    func_va: int
    va: int
    mnemonic: str
    regs: List[str]                       # operand registers at the site
    src_labels: FrozenSet[str]            # attacker labels on the operands
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class _Guard:
    func_va: int
    va: int
    regs: Tuple[str, ...]
    labels: FrozenSet[str]


class ARM64TaggingScanner(ARM64TaintTracker):
    """Base: tag operation results, record guards; subclasses define sites and sinks."""

    KIND = "tag"            # tag label = f"%{KIND}@{va:#x}"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._reset()

    def _reset(self) -> None:
        self._sites: Dict[str, TagSite] = {}
        self._guards: List[_Guard] = []
        self._pending: Optional[Tuple[str, TagSite]] = None
        self._flag_guard: Optional[_Guard] = None    # compare waiting for a flag consumer

    # ---- subclass interface -------------------------------------------------

    def _site(self, eng: TaintEngine, insn: Insn, fva: int) -> Optional[TagSite]:
        """Return a TagSite if insn's result should be tagged."""
        return None

    def _tag(self, va: int) -> str:
        return f"{TAG}{self.KIND}@{va:#x}"

    def sites_in(self, t: Taint) -> List[Tuple[str, TagSite]]:
        """(label, site) for every tag of this scanner carried by t."""
        return [(l, self._sites[l]) for l in t.labels if l in self._sites]

    def guarded(self, label: str, site: TagSite) -> bool:
        for g in self._guards:
            if label in g.labels and g.va > site.va:
                return True                       # value tested after the operation
            if (g.func_va == site.func_va and g.va <= site.va
                    and set(g.regs) & set(site.regs) and g.labels & site.src_labels):
                return True                       # operand checked before it (or by a subs site)
            if g.func_va == site.func_va and any(l.startswith(f"{TAG}hi@") for l in g.labels):
                return True                       # umulh / smulh overflow check tested
        return False

    # ---- tracker hooks ------------------------------------------------------

    def _before_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        st, m = eng.state, insn.mnemonic
        self._pending = None
        if self._flag_guard is not None and (m.startswith("b.") or m in _FLAG_USERS):
            self._guards.append(self._flag_guard)        # the compare's flags are used
            self._flag_guard = None
        if m in ISA.ret_mnems:
            self._flag_guard = None
        if m in _COMPARES or m in _REG_BRANCH:
            regs = [r for r in (insn.regs()[1:] if m in ("subs", "adds", "ands") else insn.regs())
                    if r.name != "xzr"]
            if m in _REG_BRANCH:
                regs = regs[:1]
            labels = frozenset().union(*(st.get(r.name).labels for r in regs)) if regs else frozenset()
            g = _Guard(fva, insn.address, tuple(r.name for r in regs), labels) if labels else None
            if m in _REG_BRANCH:
                if g is not None:
                    self._guards.append(g)                # cbz / tbz test directly
            else:
                self._flag_guard = g                      # replaces any unconsumed compare
            if m not in ("subs", "adds", "ands"):
                return
        site = self._site(eng, insn, fva)
        if site is not None:
            self._pending = (self._tag(insn.address), site)

    def _after_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        st = eng.state
        if insn.mnemonic in _HI_MULS and insn.reg(0) is not None:
            t = st.get(insn.reg(0).name)
            if attacker(t.labels):
                st.regs[insn.reg(0).name] = Taint(t.labels | {f"{TAG}hi@{insn.address:#x}"}, t.bound)
        if self._pending is None:
            return
        label, site = self._pending
        self._pending = None
        rd = insn.reg(0)
        if rd is None or rd.name == "xzr":
            return
        t = st.get(rd.name)
        st.regs[rd.name] = Taint(t.labels | {label}, None)
        self._sites[label] = site

    def _on_call(self, eng: TaintEngine, insn: Insn, target: int, callee, fva: int,
                 func_name: str, findings, depth: int, enqueue) -> None:
        self._flag_guard = None                           # a call clobbers NZCV
        super()._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)

