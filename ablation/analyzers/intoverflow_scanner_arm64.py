"""
intoverflow_scanner_arm64.py: AArch64 integer overflow in allocation sizing.

ARM64 counterpart of intoverflow_scanner_arm32.py, built on the ARM64 taint tracker
so "wire-controlled" means data flow, not proximity:

  1. Sources fill buffers (recv/read -> *x1, fgets/fread -> *x0) and set x0; loads
     from those buffers, spills, register aliases and callees carry the taint
     (ARM64TaintTracker / TaintEngine).
  2. At each multiply-like instruction with a tainted operand (mul, madd, msub,
     mneg, lsl, add/sub with a shifted tainted operand) the operand bounds decide
     whether the product can overflow at the operation's width (32-bit W or 64-bit
     X). If it can, the result is tagged with a "mul@<va>" label.
  3. A finding is a tagged product reaching an allocator's size argument (malloc,
     realloc, aligned_alloc, operator new, kmalloc, ...), in the same function or
     through calls (interprocedural pass).
  4. bounds_checked is True when a compare / cbz / tbz tests the product after the
     multiply, or tests a tainted operand register before it, or a umulh / smulh
     overflow check over the operands is tested.

32x32->64 widening multiplies (umull/smull/umaddl/smaddl) cannot overflow and are
ignored, as are products whose operand bounds keep them below 2**width (a 16-bit
field times 8, a u32 zero-extended to 64 bits then shifted by 4).

Usage:
    from ablation.analyzers.intoverflow_scanner_arm64 import ARM64IntOverflowScanner

    scanner = ARM64IntOverflowScanner.from_path_full('/path/to/target_a64.so')
    findings = scanner.scan()
    print(scanner.report(findings))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .insn_arm64 import Imm, Insn, Reg
from .isa_arm64 import ISA
from .taint_tracker_arm64 import ARM64TaintTracker, Taint, TaintEngine

_MUL_TAG = "mul@"

# allocator -> size argument registers
_ALLOC_SIZE_ARGS: Dict[str, Tuple[str, ...]] = {
    "malloc": ("x0",), "calloc": ("x0", "x1"), "realloc": ("x1",),
    "aligned_alloc": ("x1",), "memalign": ("x1",), "posix_memalign": ("x2",),
    "valloc": ("x0",), "pvalloc": ("x0",),
    "_Znwm": ("x0",), "_Znam": ("x0",),                  # operator new / new[]
    "xmalloc": ("x0",), "xrealloc": ("x1",), "g_malloc": ("x0",), "g_malloc0": ("x0",),
    "g_realloc": ("x1",), "CRYPTO_malloc": ("x0",), "CRYPTO_zalloc": ("x0",),
    "kmalloc": ("x0",), "__kmalloc": ("x0",), "kzalloc": ("x0",), "kvmalloc": ("x0",),
    "vmalloc": ("x0",), "vzalloc": ("x0",), "kvmalloc_node": ("x0",),
}

_MUL_MNEMS = frozenset({"mul", "madd", "msub", "mneg"})
_WIDE_MULS = frozenset({"umull", "smull", "umaddl", "smaddl", "umsubl", "smsubl", "umnegl", "smnegl"})
_HI_MULS = frozenset({"umulh", "smulh"})
_COMPARES = frozenset({"cmp", "cmn", "tst", "ccmp", "ccmn", "subs", "adds", "ands"})
_REG_BRANCH = frozenset({"cbz", "cbnz", "tbz", "tbnz"})


@dataclass
class IntOverflowFinding64:
    func_va: int
    mul_va: int
    mnemonic: str
    operand_regs: List[str]
    width: int               # 32 (W) or 64 (X) multiply
    alloc_va: int
    alloc_func_va: int       # function containing the allocator call (may differ: callee)
    alloc_sym: str
    size_arg: str
    wire_evidence: str       # source labels reaching the operands
    bounds_checked: bool

    def __str__(self) -> str:
        regs = ', '.join(self.operand_regs)
        checked = " [GUARDED]" if self.bounds_checked else " [UNGUARDED]"
        where = "" if self.alloc_func_va == self.func_va else f" in 0x{self.alloc_func_va:x}"
        return (f"INT_OVERFLOW  func=0x{self.func_va:x}  mul@0x{self.mul_va:x}"
                f"  {self.mnemonic}/{self.width}({regs}) -> {self.alloc_sym}({self.size_arg})"
                f"@0x{self.alloc_va:x}{where}  wire=[{self.wire_evidence}]{checked}")


@dataclass
class _MulSite:
    func_va: int
    va: int
    mnemonic: str
    regs: List[str]
    width: int
    src_labels: FrozenSet[str]


@dataclass
class _Guard:
    func_va: int
    va: int
    regs: Tuple[str, ...]
    labels: FrozenSet[str]


def _attacker(labels: FrozenSet[str]) -> FrozenSet[str]:
    return frozenset(l for l in labels if not l.startswith(_MUL_TAG))


class ARM64IntOverflowScanner(ARM64TaintTracker):
    """Tainted multiply -> allocator size, on top of ARM64TaintTracker's taint pass."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._muls: Dict[str, _MulSite] = {}
        self._guards: List[_Guard] = []
        self._pending: Optional[Tuple[str, _MulSite]] = None
        self._ovf: List[IntOverflowFinding64] = []

    # ---- public API ---------------------------------------------------------

    def scan(self, depth: int = 3) -> List[IntOverflowFinding64]:
        self._muls, self._guards, self._ovf = {}, [], []
        self.run_interprocedural(depth=depth)
        # One finding per (allocation size arg, multiplying function): compilers split
        # n*12 into (n + n<<1) << 2, which tags two steps of the same computation.
        groups: Dict[Tuple[int, str, int], List[IntOverflowFinding64]] = {}
        for f in self._ovf:
            groups.setdefault((f.alloc_va, f.size_arg, f.func_va), []).append(self._finalize(f))
        out: List[IntOverflowFinding64] = []
        for fs in groups.values():
            steps = sorted({f.mul_va: f for f in fs}.values(), key=lambda f: f.mul_va)
            head = steps[0]
            head.mnemonic = " -> ".join(f.mnemonic for f in steps)
            head.width = max(f.width for f in steps)
            head.bounds_checked = any(f.bounds_checked for f in steps)
            out.append(head)
        return sorted(out, key=lambda f: (f.func_va, f.mul_va))

    def report(self, findings: List[IntOverflowFinding64]) -> str:
        if not findings:
            return "No ARM64 integer overflow candidates."
        unguarded = [f for f in findings if not f.bounds_checked]
        lines = [
            f"ARM64 Integer Overflow Scan: {len(findings)} candidate(s)"
            f" ({len(unguarded)} unguarded)",
            "=" * 70,
        ]
        lines += [str(f) for f in findings]
        return "\n".join(lines)

    # ---- tracker hooks ------------------------------------------------------

    def _before_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        st, m = eng.state, insn.mnemonic
        self._pending = None

        # guards: conditionals that test tainted values
        if m in _COMPARES or m in _REG_BRANCH:
            regs = [r for r in (insn.regs()[1:] if m in ("subs", "adds", "ands") else insn.regs())
                    if r.name != "xzr"]
            if m in _REG_BRANCH:
                regs = regs[:1]
            labels = frozenset().union(*(st.get(r.name).labels for r in regs)) if regs else frozenset()
            if labels:
                self._guards.append(_Guard(fva, insn.address, tuple(r.name for r in regs), labels))
            return

        site = self._mul_site(eng, insn, fva)
        if site is not None:
            self._pending = (f"{_MUL_TAG}{insn.address:#x}", site)

    def _after_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        st = eng.state
        if insn.mnemonic in _HI_MULS and insn.reg(0) is not None:
            # umulh result marks an overflow check over the operands once tested
            t = st.get(insn.reg(0).name)
            if _attacker(t.labels):
                tag = f"hi@{insn.address:#x}"
                st.regs[insn.reg(0).name] = Taint(t.labels | {tag}, t.bound)
        if self._pending is None:
            return
        label, site = self._pending
        self._pending = None
        rd = insn.reg(0)
        if rd is None or rd.name == "xzr":
            return
        t = st.get(rd.name)
        st.regs[rd.name] = Taint(t.labels | {label}, None)
        self._muls[label] = site

    def _on_call(self, eng: TaintEngine, insn: Insn, target: int, callee: Optional[str],
                 fva: int, func_name: str, findings, depth: int, enqueue) -> None:
        args = _ALLOC_SIZE_ARGS.get(callee or "")
        if args:
            for r in args:
                for label in eng.state.get(r).labels:
                    site = self._muls.get(label)
                    if site is None:
                        continue
                    self._ovf.append(IntOverflowFinding64(
                        func_va=site.func_va, mul_va=site.va, mnemonic=site.mnemonic,
                        operand_regs=site.regs, width=site.width,
                        alloc_va=insn.address, alloc_func_va=fva, alloc_sym=callee,
                        size_arg=r, wire_evidence=",".join(sorted(site.src_labels)),
                        bounds_checked=False,
                    ))
        super()._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)

    # ---- multiply analysis --------------------------------------------------

    def _mul_site(self, eng: TaintEngine, insn: Insn, fva: int) -> Optional[_MulSite]:
        """A multiply-like insn with an attacker operand that can overflow its width."""
        m = insn.mnemonic
        rd = insn.reg(0)
        if rd is None or rd.name.startswith("v") or m in _WIDE_MULS:
            return None
        width = rd.bits
        limit = 1 << width
        st = eng.state

        def tb(r: Reg) -> Taint:
            return eng._src(r)       # shift / extend already applied to the bound

        if m in _MUL_MNEMS and len(insn.ops) >= 3:
            a, b = insn.reg(1), insn.reg(2)
            if a is None or b is None:
                return None
            ta, tb_ = tb(a), tb(b)
            src = _attacker(ta.labels | tb_.labels)
            if not src:
                return None
            ba = ta.bound if ta.tainted else st.consts.get(a.name)
            bb = tb_.bound if tb_.tainted else st.consts.get(b.name)
            if ba is not None and bb is not None and ba * bb < limit:
                return None
            return _MulSite(fva, insn.address, m, [a.name, b.name], width, src)

        if m == "lsl" and len(insn.ops) >= 3 and insn.reg(1) is not None:
            a = insn.reg(1)
            ta = tb(a)
            src = _attacker(ta.labels)
            if not src:
                return None
            k = insn.imm(2)
            if k is None:                         # shift by register: unbounded
                return _MulSite(fva, insn.address, m, [a.name], width, src)
            if ta.bound is not None and (ta.bound << k) < limit:
                return None
            return _MulSite(fva, insn.address, f"lsl#{k}", [a.name], width, src)

        if m in ("add", "sub") and len(insn.ops) >= 3:
            sh = insn.reg(2)
            if sh is None or sh.shift != "lsl" or not sh.amount:
                return None
            t_sh = self._raw(eng, sh)
            src = _attacker(t_sh.labels)
            if not src:
                return None
            if t_sh.bound is not None and (t_sh.bound << sh.amount) * 2 < limit:
                return None
            return _MulSite(fva, insn.address, f"{m}+lsl#{sh.amount}",
                            [r.name for r in insn.regs()[1:3]], width, src)
        return None

    @staticmethod
    def _raw(eng: TaintEngine, r: Reg) -> Taint:
        t = eng.state.get(r.name)
        return Taint(t.labels, t.bound if r.bits == 64 or t.bound is None else min(t.bound, 0xFFFFFFFF))

    def _finalize(self, f: IntOverflowFinding64) -> IntOverflowFinding64:
        label = f"{_MUL_TAG}{f.mul_va:#x}"
        site = self._muls.get(label)
        guarded = False
        for g in self._guards:
            if label in g.labels and g.va > f.mul_va:
                guarded = True                    # product tested after the multiply
            elif (site and g.func_va == site.func_va and g.va < f.mul_va
                  and set(g.regs) & set(site.regs) and g.labels & site.src_labels):
                guarded = True                    # operand range-checked before it
            elif any(l.startswith("hi@") for l in g.labels) and site and g.func_va == site.func_va:
                guarded = True                    # umulh / smulh overflow check tested
            if guarded:
                break
        f.bounds_checked = guarded
        return f
