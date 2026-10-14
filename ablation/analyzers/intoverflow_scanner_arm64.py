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
     X). If it can, the result is tagged with a "%mul@<va>" label.
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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .arm64_tag_scanner import ARM64TaggingScanner, TagSite, attacker
from .insn_arm64 import Insn, Reg
from .taint_tracker_arm64 import Taint, TaintEngine

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
    # RTOS / embedded allocators
    "pvPortMalloc": ("x0",), "pvPortCalloc": ("x0", "x1"), "k_malloc": ("x0",),
    "k_calloc": ("x0", "x1"), "k_aligned_alloc": ("x1",), "rt_malloc": ("x0",),
    "rt_calloc": ("x0", "x1"), "rt_realloc": ("x1",), "os_malloc": ("x0",), "os_zalloc": ("x0",),
    "os_calloc": ("x0", "x1"), "os_realloc": ("x1",), "mem_malloc": ("x0",), "mem_calloc": ("x0", "x1"),
    "osal_malloc": ("x0",), "tx_byte_allocate": ("x2",), "heap_caps_malloc": ("x0",),
}

_MUL_MNEMS = frozenset({"mul", "madd", "msub", "mneg"})
_WIDE_MULS = frozenset({"umull", "smull", "umaddl", "smaddl", "umsubl", "smsubl", "umnegl", "smnegl"})


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


class ARM64IntOverflowScanner(ARM64TaggingScanner):
    """Tainted multiply -> allocator size, on top of ARM64TaintTracker's taint pass."""

    KIND = "mul"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._ovf: List[IntOverflowFinding64] = []

    # ---- public API ---------------------------------------------------------

    def scan(self, depth: int = 3) -> List[IntOverflowFinding64]:
        self._reset()
        self._ovf = []
        self.run_interprocedural(depth=depth)
        # One finding per (allocation size arg, multiplying function): compilers split
        # n*12 into (n + n<<1) << 2, which tags two steps of the same computation.
        groups: Dict[Tuple[int, str, int], List[IntOverflowFinding64]] = {}
        for f in self._ovf:
            f.bounds_checked = self.guarded(self._tag(f.mul_va), self._sites[self._tag(f.mul_va)])
            groups.setdefault((f.alloc_va, f.size_arg, f.func_va), []).append(f)
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

    def _on_call(self, eng: TaintEngine, insn: Insn, target: int, callee: Optional[str],
                 fva: int, func_name: str, findings, depth: int, enqueue) -> None:
        args = _ALLOC_SIZE_ARGS.get(callee or "")
        if args:
            for r in args:
                for _label, site in self.sites_in(eng.state.get(r)):
                    self._ovf.append(IntOverflowFinding64(
                        func_va=site.func_va, mul_va=site.va, mnemonic=site.mnemonic,
                        operand_regs=site.regs, width=site.extra["width"],
                        alloc_va=insn.address, alloc_func_va=fva, alloc_sym=callee,
                        size_arg=r, wire_evidence=",".join(sorted(site.src_labels)),
                        bounds_checked=False,
                    ))
        super()._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)

    # ---- multiply analysis --------------------------------------------------

    def _site(self, eng: TaintEngine, insn: Insn, fva: int) -> Optional[TagSite]:
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
            src = attacker(ta.labels | tb_.labels)
            if not src:
                return None
            ba = ta.bound if ta.tainted else st.consts.get(a.name)
            bb = tb_.bound if tb_.tainted else st.consts.get(b.name)
            if ba is not None and bb is not None and ba * bb < limit:
                return None
            return TagSite(fva, insn.address, m, [a.name, b.name], src, {'width': width})

        if m == "lsl" and len(insn.ops) >= 3 and insn.reg(1) is not None:
            a = insn.reg(1)
            ta = tb(a)
            src = attacker(ta.labels)
            if not src:
                return None
            k = insn.imm(2)
            if k is None:                         # shift by register: unbounded
                return TagSite(fva, insn.address, m, [a.name], src, {'width': width})
            if ta.bound is not None and (ta.bound << k) < limit:
                return None
            return TagSite(fva, insn.address, f"lsl#{k}", [a.name], src, {'width': width})

        if m in ("add", "sub") and len(insn.ops) >= 3:
            sh = insn.reg(2)
            if sh is None or sh.shift != "lsl" or not sh.amount:
                return None
            t_sh = self._raw(eng, sh)
            src = attacker(t_sh.labels)
            if not src:
                return None
            if t_sh.bound is not None and (t_sh.bound << sh.amount) * 2 < limit:
                return None
            return TagSite(fva, insn.address, f"{m}+lsl#{sh.amount}",
                           [r.name for r in insn.regs()[1:3]], src, {'width': width})
        return None

    @staticmethod
    def _raw(eng: TaintEngine, r: Reg) -> Taint:
        t = eng.state.get(r.name)
        return Taint(t.labels, t.bound if r.bits == 64 or t.bound is None else min(t.bound, 0xFFFFFFFF))
