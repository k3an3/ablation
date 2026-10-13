"""
taint_tracker_arm64.py: AArch64 data-flow taint tracker.

Labeled-taint model: Taint(labels, bound).
Key AArch64 semantics:
  - Every W-register write zero-extends: result is bounded to 0xFFFFFFFF.
  - ``and rd, rn, #imm``: bitmask with top bit set = alignment mask = no bound;
    positive mask = bounds rd.
  - Shift/extend modifiers on source operands scale the bound.
  - ldp/stp pair instructions; pre/post-index writeback.
  - Linux arm64 syscall: x8=nr, x0-x5=args, x0=result; read/recvfrom taints buffer.

Sources: read/recv/recvfrom taint *x1, fgets/gets/fread taint *x0 (the buffer filled),
         x0 carries the call's result; Linux svc read/recvfrom taint *x1 likewise.
Sinks:   system/execve/strcpy/sprintf/memcpy and other dangerous libc functions; an arg
         counts when its register or the memory it points at is tainted ("*x0").

Pointer / alias state (parity with the ARM32 labeled tracker): frame_ptrs and spilled
mem_ptrs keep "register holds address base+k"; eq alias classes over registers and stack
slots share a buffer region between copies of a pointer; ra_regs follows the return address.

AAPCS64: x0-x7 args, x0-x1 return, x19-x28 callee-saved, x30=lr, x29=fp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple

from .insn_arm64 import (
    Cond, Imm, Insn, Mem, Reg, Sym,
    from_capstone,
    from_listing as _from_listing,
    from_objdump as _from_objdump,
    make_insn,
)
from .isa_arm64 import (
    ARG_REGS, CALLER_SAVED, FLAGS, FRAME_BASES, FRAME_REG, INPUT_SYSCALLS, ISA, LINK_REG,
    MASK64, RET_REGS, SYSCALL_ARG_REGS, SYSCALL_NR_REG, SYSCALL_RET_REG, SYSCALLS,
    VEC_ARG_REGS, VEC_RET_REGS, access_bytes, and_bound, ext_bound, w_cap,
)

Labels = FrozenSet[str]
EMPTY: Labels = frozenset()

_SOURCE_NAMES: frozenset = frozenset({
    "recv", "recvfrom", "recvmsg", "read", "fread", "fgets", "gets", "getchar", "fgetc",
})
_SINK_NAMES: frozenset = frozenset({
    "system", "execve", "execl", "execvp", "execle", "execvpe", "popen",
    "strcpy", "strcat", "sprintf", "vsprintf", "snprintf", "vsnprintf",
    "memcpy", "memmove", "gets",
    "sscanf", "vsscanf", "__isoc99_sscanf", "__isoc99_vsscanf",
})
# attacker text parsed by sscanf: the bug is an unbounded %s / %[ (ablation fmtstr shows
# which); reported MEDIUM, and the parsed fields (x2..x7 outputs) become tainted
_SCAN_SINKS = frozenset({"sscanf", "vsscanf", "__isoc99_sscanf", "__isoc99_vsscanf"})
# label carried by values that came out of a call with no model (--summarise-opaque)
OPAQUE = "~opaque"
_PLT_TOL = 16

# Sources that fill a caller buffer: name -> (buffer arg, length arg or None)
_SOURCE_BUFFERS: Dict[str, Tuple[str, Optional[str]]] = {
    "read": ("x1", "x2"), "recv": ("x1", "x2"), "recvfrom": ("x1", "x2"),
    "fread": ("x0", None), "fgets": ("x0", "x1"), "gets": ("x0", None),
}
# Sink args whose pointee (not the register) carries the attacker data
_SINK_POINTER_ARGS: Dict[str, Tuple[str, ...]] = {
    "system": ("x0",), "popen": ("x0",),
    "execve": ("x0", "x1"), "execl": ("x0", "x1", "x2"), "execvp": ("x0", "x1"),
    "execle": ("x0", "x1", "x2"), "execvpe": ("x0", "x1"),
    "strcpy": ("x1",), "strcat": ("x1",), "memcpy": ("x1",), "memmove": ("x1",),
    "sprintf": tuple(f"x{i}" for i in range(1, 8)), "vsprintf": ("x1", "x2"),
    "snprintf": tuple(f"x{i}" for i in range(2, 8)), "vsnprintf": ("x2", "x3"),
    "sscanf": ("x0",), "vsscanf": ("x0",), "__isoc99_sscanf": ("x0",), "__isoc99_vsscanf": ("x0",),
}
# Copies: dst pointee takes the src args' (pointee) taint
_COPY_FUNCS: Dict[str, Tuple[str, Tuple[str, ...], Optional[str]]] = {
    "strcpy": ("x0", ("x1",), None), "strncpy": ("x0", ("x1",), "x2"),
    "strcat": ("x0", ("x1",), None), "strncat": ("x0", ("x1",), "x2"),
    "memcpy": ("x0", ("x1",), "x2"), "memmove": ("x0", ("x1",), "x2"),
    "sprintf": ("x0", tuple(f"x{i}" for i in range(1, 8)), None),
    "snprintf": ("x0", tuple(f"x{i}" for i in range(2, 8)), "x1"),
}
_EXEC_SINKS = frozenset({"system", "execve", "execl", "execvp", "popen"})


# ---------------------------------------------------------------------------
# Taint value
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Taint:
    labels: Labels = EMPTY
    bound: Optional[int] = None

    @property
    def tainted(self) -> bool:
        return bool(self.labels)

    def __str__(self) -> str:
        if not self.labels:
            return "clean"
        s = "{" + ",".join(sorted(self.labels)) + "}"
        return s + (f"<={self.bound:#x}" if self.bound is not None else "")


CLEAN = Taint()


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass
class TaintFindingARM64:
    """Binary-path source-to-sink finding (API matches other ablation trackers)."""
    func_va:      int
    func_name:    str
    sink_va:      int
    sink_name:    str
    tainted_args: List[str]
    source_name:  str
    severity:     str = "HIGH"
    note:         str = ""      # e.g. "length", "via opaque call (lower confidence)"

    def __str__(self) -> str:
        args = ", ".join(self.tainted_args)
        note = f" [{self.note}]" if self.note else ""
        return (f"[{self.severity}] 0x{self.func_va:x} ({self.func_name}): "
                f"tainted {{{args}}} -> {self.sink_name} @ 0x{self.sink_va:x} "
                f"(source: {self.source_name}){note}")


@dataclass
class Finding:
    """Generic finding from the labeled-taint engine (listing/objdump paths)."""
    address: int
    kind: str
    detail: str
    labels: Labels

    def __str__(self) -> str:
        return f"{self.address:#x} [{self.kind}] {self.detail} <- {{{','.join(sorted(self.labels))}}}"


# ---------------------------------------------------------------------------
# Taint state
# ---------------------------------------------------------------------------

Loc = object   # a register name (str) or a memory slot (base, offset)


@dataclass
class TaintState:
    """
    regs       register -> Taint
    mem        (base, off) -> Taint for exact slots; ranges: tainted byte ranges
    consts     register -> known constant
    frame_ptrs register -> (base, k): register holds the address base+k, where base is
               a frame register (sp/x29) or a buffer region ("@buf<va>", "@arg_x1")
    mem_ptrs   (base, off) -> (base', k): slot holds a pointer (spilled frame_ptrs fact)
    mem_consts (base, off) -> (nbytes, value): slot holds a known constant (spilled consts)
    eq         alias classes over registers and 64-bit slots holding the same value;
               lets a buffer filled through one copy of a pointer be seen through all
    lr_slots   frame slots holding the return address
    ra_regs    registers holding the return address (x30, its copies, reloads)
    """
    regs:       Dict[str, Taint] = field(default_factory=dict)
    mem:        Dict[Tuple[str, int], Taint] = field(default_factory=dict)
    ranges:     List[Tuple[str, int, int, Taint]] = field(default_factory=list)
    consts:     Dict[str, int] = field(default_factory=dict)
    frame_ptrs: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    lr_slots:   Set[Tuple[str, int]] = field(default_factory=set)
    mem_ptrs:   Dict[Tuple[str, int], Tuple[str, int]] = field(default_factory=dict)
    eq:         Dict[Loc, Set[Loc]] = field(default_factory=dict)
    ra_regs:    Set[str] = field(default_factory=lambda: {LINK_REG})
    mem_consts: Dict[Tuple[str, int], Tuple[int, int]] = field(default_factory=dict)

    def get(self, r: Optional[str]) -> Taint:
        if r is None or r == "xzr":
            return CLEAN
        return self.regs.get(r, CLEAN)

    def set(self, r: Optional[str], t: Taint) -> None:
        if r is None or r == "xzr":
            return
        if t.tainted or t.bound is not None:
            self.regs[r] = t
        else:
            self.regs.pop(r, None)
        self.consts.pop(r, None)
        self.frame_ptrs.pop(r, None)
        self.unalias(r)
        self.ra_regs.discard(r)
        if r in FRAME_BASES:
            self.invalidate_base(r)

    def taint_reg(self, r: str, label: str) -> None:
        cur = self.get(r)
        self.regs[r] = Taint(cur.labels | {label}, cur.bound)

    def resolve(self, base: Optional[str], off: int) -> Tuple[Optional[str], int]:
        if base in self.frame_ptrs:
            b, k = self.frame_ptrs[base]
            return b, k + off
        return base, off

    def mem_get(self, base: str, off: int) -> Taint:
        t = self.mem.get((base, off), CLEAN)
        for b, lo, hi, rt in self.ranges:
            if b == base and lo <= off < hi:
                t = Taint(t.labels | rt.labels, None)
        return t

    def mem_set(self, base: str, off: int, t: Taint, ptr: Optional[Tuple[str, int]] = None) -> None:
        if t.tainted:
            self.mem[(base, off)] = t
        else:
            self.mem.pop((base, off), None)
        if ptr is not None:
            self.mem_ptrs[(base, off)] = ptr
        else:
            self.mem_ptrs.pop((base, off), None)
        self.mem_consts.pop((base, off), None)
        self.unalias((base, off))

    def taint_range(self, base: str, lo: int, hi: int, label) -> None:
        labels = frozenset({label}) if isinstance(label, str) else frozenset(label)
        self.ranges.append((base, lo, hi, Taint(labels, None)))

    def pointee(self, r: Optional[str]) -> Taint:
        """Taint of the memory r points at (r's own taint not included)."""
        if r is None:
            return CLEAN
        loc = self.frame_ptrs.get(r) or (("sp", 0) if r == "sp" else None)
        return self.mem_get(*self.resolve(*loc)) if loc is not None else CLEAN

    # ---- alias classes -------------------------------------------------------

    def unalias(self, loc: Loc) -> None:
        g = self.eq.pop(loc, None)
        if g is not None:
            g.discard(loc)

    def alias(self, dst: Loc, src: Loc) -> None:
        if dst == src:
            return
        self.unalias(dst)
        g = self.eq.setdefault(src, {src})
        g.add(dst)
        self.eq[dst] = g

    def aliases(self, loc: Loc) -> Set[Loc]:
        return set(self.eq.get(loc, {loc}))

    def point_at(self, loc: Loc, region: str, k: int = 0) -> None:
        """Every alias of loc now holds the address region+k."""
        for m in self.aliases(loc):
            if isinstance(m, tuple):
                self.mem_ptrs[m] = (region, k)
            else:
                self.frame_ptrs[m] = (region, k)

    def _map_eq(self, f) -> None:
        """Rewrite / drop slot members of eq (f returns the new loc or None)."""
        out: Dict[Loc, Set[Loc]] = {}
        for g in {id(g): g for g in self.eq.values()}.values():
            ng = {m2 for m2 in (f(m) for m in g) if m2 is not None}
            if len(ng) > 1:
                for m in ng:
                    out[m] = ng
        self.eq = out

    # ---- frame moves ---------------------------------------------------------

    def rebase(self, reg: str, delta: int) -> None:
        def rk(k):
            b, o = k
            return (b, o - delta) if b == reg else k
        self.mem = {rk(k): t for k, t in self.mem.items()}
        self.ranges = [((b, lo - delta, hi - delta, t) if b == reg else (b, lo, hi, t)) for (b, lo, hi, t) in self.ranges]
        self.lr_slots = {rk(k) for k in self.lr_slots}
        self.frame_ptrs = {r: rk(v) for r, v in self.frame_ptrs.items()}
        self.mem_ptrs = {rk(k): rk(v) for k, v in self.mem_ptrs.items()}
        self.mem_consts = {rk(k): v for k, v in self.mem_consts.items()}
        self._map_eq(lambda m: rk(m) if isinstance(m, tuple) else m)
        if reg in self.consts:
            self.consts[reg] = (self.consts[reg] + delta) & MASK64

    def invalidate_base(self, reg: str) -> None:
        self.mem = {k: v for k, v in self.mem.items() if k[0] != reg}
        self.ranges = [r for r in self.ranges if r[0] != reg]
        self.lr_slots = {k for k in self.lr_slots if k[0] != reg}
        self.frame_ptrs = {r: v for r, v in self.frame_ptrs.items() if v[0] != reg}
        self.mem_ptrs = {k: v for k, v in self.mem_ptrs.items() if k[0] != reg and v[0] != reg}
        self.mem_consts = {k: v for k, v in self.mem_consts.items() if k[0] != reg}
        self._map_eq(lambda m: None if isinstance(m, tuple) and m[0] == reg else m)

    def drop_below_sp(self) -> None:
        """A call may overwrite everything below sp."""
        below = lambda k: k[0] == "sp" and k[1] < 0
        self.mem = {k: v for k, v in self.mem.items() if not below(k)}
        self.mem_ptrs = {k: v for k, v in self.mem_ptrs.items() if not below(k)}
        self.mem_consts = {k: v for k, v in self.mem_consts.items() if not below(k)}
        self._map_eq(lambda m: None if isinstance(m, tuple) and below(m) else m)

    def snapshot(self) -> Dict[str, str]:
        return {r: str(t) for r, t in sorted(self.regs.items())}

    def _eq_groups(self) -> FrozenSet[FrozenSet]:
        return frozenset(frozenset(g) for g in self.eq.values() if len(g) > 1)

    # lattice ops (CFG fixpoint) ------------------------------------------
    def copy(self) -> "TaintState":
        out = TaintState(
            dict(self.regs), dict(self.mem), list(self.ranges),
            dict(self.consts), dict(self.frame_ptrs), set(self.lr_slots),
            dict(self.mem_ptrs), {}, set(self.ra_regs), dict(self.mem_consts),
        )
        for g in {id(g): g for g in self.eq.values()}.values():
            ng = set(g)
            for m in ng:
                out.eq[m] = ng
        return out

    def join(self, other: "TaintState", widen: bool = False) -> "TaintState":
        """Least upper bound for a control-flow merge."""
        out = TaintState()
        for r in set(self.regs) | set(other.regs):
            a, b = self.get(r), other.get(r)
            ab = 0 if not a.tainted else a.bound
            bb = 0 if not b.tainted else b.bound
            if ab is None or bb is None:
                bound = None
            elif widen and ab != bb:
                bound = None
            else:
                bound = max(ab, bb)
            t = Taint(a.labels | b.labels, bound)
            if t.tainted or t.bound is not None:
                out.regs[r] = t
        for k in set(self.mem) | set(other.mem):
            a, b = self.mem.get(k, CLEAN), other.mem.get(k, CLEAN)
            out.mem[k] = Taint(a.labels | b.labels, None)
        seen: Set[tuple] = set()
        for rng in self.ranges + other.ranges:
            key = (rng[0], rng[1], rng[2], rng[3].labels)
            if key not in seen:
                seen.add(key)
                out.ranges.append(rng)
        out.consts = {r: v for r, v in self.consts.items() if other.consts.get(r) == v}
        out.frame_ptrs = {r: v for r, v in self.frame_ptrs.items() if other.frame_ptrs.get(r) == v}
        out.mem_ptrs = {k: v for k, v in self.mem_ptrs.items() if other.mem_ptrs.get(k) == v}
        out.mem_consts = {k: v for k, v in self.mem_consts.items() if other.mem_consts.get(k) == v}
        out.lr_slots = self.lr_slots | other.lr_slots
        out.ra_regs = self.ra_regs | other.ra_regs
        # alias classes: meet of the two partitions (aliased on both paths)
        parts: Dict[Tuple[int, int], Set[Loc]] = {}
        for m in set(self.eq) & set(other.eq):
            parts.setdefault((id(self.eq[m]), id(other.eq[m])), set()).add(m)
        for g in parts.values():
            if len(g) > 1:
                for m in g:
                    out.eq[m] = g
        return out

    def same_as(self, other: "TaintState") -> bool:
        return (
            self.regs == other.regs and self.mem == other.mem
            and sorted(map(repr, self.ranges)) == sorted(map(repr, other.ranges))
            and self.consts == other.consts and self.frame_ptrs == other.frame_ptrs
            and self.lr_slots == other.lr_slots and self.mem_ptrs == other.mem_ptrs
            and self.mem_consts == other.mem_consts
            and self.ra_regs == other.ra_regs and self._eq_groups() == other._eq_groups()
        )


# ---------------------------------------------------------------------------
# Labeled-taint engine (listing / objdump / capstone path)
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    """Opt-in finding classes (defaults match the ARM32 labeled tracker)."""
    report_tainted_branch: bool = False   # "tainted-branch-condition" at b.cond/cbz/tbz
    report_bounded: bool = False          # "bounded-tainted-*-address" for bounded indices


class TaintEngine:
    """
    Forward labeled-taint tracker over a stream of AArch64 ``Insn`` objects.

    Seed with ``taint_register(r, label)`` / ``taint_pointee(r, label)`` then call
    ``run(insns)``. Findings accumulate in ``self.findings``; unknown mnemonics in
    ``self.unknown``.
    """

    def __init__(self, policy: Optional[Policy] = None):
        self.isa = ISA
        self.policy = policy or Policy()
        self.state = TaintState()
        self.findings: List[Finding] = []
        self.unknown: Set[str] = set()

    def taint_register(self, reg: str, label: str) -> None:
        from .isa_arm64 import canon_reg as _canon
        r = _canon(reg)
        self.state.taint_reg(r[0] if r else reg, label)

    def taint_memory(self, base: str, offset: int, label: str) -> None:
        self.state.mem[(base, offset)] = Taint(frozenset({label}), None)

    def taint_pointee(self, reg: str, label, length: int = 1 << 20) -> None:
        """The buffer reg points at is attacker-controlled (reg itself stays clean)."""
        self.taint_buffer(reg, length, label, region=f"@arg_{reg}")

    def taint_buffer(self, reg: str, length: int, label, region: str) -> None:
        """Taint length bytes at reg's pointee; an unknown pointee becomes `region`,
        shared by every register / slot aliased to reg."""
        st = self.state
        loc = st.frame_ptrs.get(reg) or (("sp", 0) if reg == "sp" else None)
        if loc is None:
            st.point_at(reg, region, 0)
            loc = (region, 0)
        b, k = st.resolve(*loc)            # x29-relative -> sp-relative, as loads resolve
        st.taint_range(b, k, k + length, label)

    def run(self, insns: Iterable[Insn]) -> List[Finding]:
        for insn in insns:
            self.step(insn)
        return self.findings

    # ---- helpers -----------------------------------------------------------

    def _src(self, r: Optional[Reg]) -> Taint:
        if r is None:
            return CLEAN
        t = self.state.get(r.name)
        return Taint(t.labels, ext_bound(r.shift, r.amount, t.bound, r.bits))

    def _write(self, r: Optional[Reg], t: Taint) -> None:
        if r is None:
            return
        if r.name.startswith("x") or r.name == "sp":
            t = Taint(t.labels, w_cap(t.bound, r.bits))
        self.state.set(r.name, t)

    def _find(self, insn: Insn, kind: str, detail: str, labels: Labels) -> None:
        self.findings.append(Finding(insn.address, kind, detail, labels))

    def _check_addr(self, insn: Insn, bt: Taint, kind: str, detail: str) -> None:
        if not bt.tainted:
            return
        if bt.bound is not None:
            if self.policy.report_bounded:
                self._find(insn, "bounded-" + kind,
                           f"{detail} (attacker contribution <= {bt.bound:#x})", bt.labels)
            return
        self._find(insn, kind, detail, bt.labels)

    # ---- dispatcher --------------------------------------------------------

    def step(self, insn: Insn) -> None:
        m, isa = insn.mnemonic, self.isa
        if m in isa.ret_mnems:
            self._ret(insn)
        elif m in isa.call_mnems:
            self._call(insn)
        elif m in isa.indirect_jump_mnems:
            self._indirect_jump(insn)
        elif m in isa.uncond_branch_mnems:
            pass
        elif m in isa.cond_branch_mnems:
            self._cond_branch(insn, self.state.get(FLAGS))
        elif m in isa.reg_branch_mnems:
            self._cond_branch(insn, self._src(insn.reg(0)))
        elif m == "svc":
            self._svc(insn)
        elif m in isa.nop_mnems:
            pass
        elif m in isa.sys_mnems:
            if m == "mrs":
                self._write(insn.reg(0), CLEAN)
        elif m in isa.load_pair_mnems or m in isa.store_pair_mnems:
            self._pair(insn, load=m in isa.load_pair_mnems)
        elif m in isa.load_mnems:
            self._load(insn)
        elif m in isa.store_mnems:
            self._store(insn)
        elif m in isa.atomic_rmw_mnems:
            self._atomic(insn)
        elif m == "mov":
            self._mov(insn)
        elif self._is_copy(insn):
            self._copy(insn, insn.reg(0), self._copy_src(insn))
        elif m in isa.const_mnems:
            self._const(insn)
        elif m in isa.partial_write_mnems:
            self._partial(insn)
        elif m in isa.ext_mnems:
            self._ext(insn)
        elif m in isa.cond_select_mnems:
            self._csel(insn)
        elif m in isa.flag_setting_mnems and m in ("cmp", "cmn", "tst", "ccmp", "ccmn", "fcmp", "fcmpe", "fccmp"):
            self._flags_only(insn)
        elif m in isa.fp_copy_mnems:
            self._write(insn.reg(0), Taint(self._src(insn.reg(1)).labels, None))
        elif m in isa.alu_mnems:
            self._alu(insn)
        else:
            self._unknown(insn)

    # ---- copy detection ----------------------------------------------------

    def _is_copy(self, insn: Insn) -> bool:
        m = insn.mnemonic
        if m not in self.isa.copy_mnems or len(insn.ops) != 3:
            return False
        rd, r1, r2, imm = insn.reg(0), insn.reg(1), insn.reg(2), insn.imm(2)
        if rd is None or r1 is None or (r1.shift is not None) or (r2 is not None and r2.shift is not None):
            return False
        if m in ("add", "sub") and imm == 0:
            return True
        if m in ("lsl", "lsr", "asr", "ror") and imm == 0:
            return True
        if m == "orr":
            return (r1.name == "xzr") != (r2 is not None and r2.name == "xzr")
        if m in ("eor", "sub", "add") and r2 is not None and r2.name == "xzr":
            return True
        return False

    def _copy_src(self, insn: Insn) -> Reg:
        r1, r2 = insn.reg(1), insn.reg(2)
        if insn.mnemonic == "orr" and r1.name == "xzr":
            return r2
        return r1

    # ---- mov ---------------------------------------------------------------

    def _mov(self, insn: Insn) -> None:
        st = self.state
        rd, src, imm = insn.reg(0), insn.reg(1), insn.imm(1)
        if rd is None:
            return
        if imm is not None:
            self._write(rd, CLEAN)
            st.consts[rd.name] = imm & MASK64
            return
        if src is None:
            return
        if rd.name.startswith("v") and src.name.startswith("x"):
            self._partial(insn)
            return
        if src.bits == 128 or rd.bits == 128:
            self._write(rd, Taint(self._src(src).labels, None))
            return
        self._copy(insn, rd, src)

    def _copy(self, insn: Insn, rd: Reg, rs: Reg) -> None:
        st = self.state
        if rs.name == "xzr":
            self._write(rd, CLEAN)
            st.consts[rd.name] = 0
            return
        # mov sp, x29 with x29 = sp+k: sp restores by rebasing
        fp = st.frame_ptrs.get(rs.name)
        if rd.name == "sp" and fp is not None and fp[0] == "sp":
            st.rebase("sp", fp[1])
            return
        was_ra = rs.name in st.ra_regs and rs.shift is None
        self._write(rd, self._src(rs))
        c = st.consts.get(rs.name)
        if c is not None and rd.bits == 64:
            st.consts[rd.name] = c
        if rd.bits == 64 and rs.bits == 64 and rs.shift is None and rs.name != "sp":
            st.alias(rd.name, rs.name)
            if was_ra:
                st.ra_regs.add(rd.name)
        # mov x29, sp records a live frame-pointer alias
        if rs.name in FRAME_BASES and rd.name != rs.name and rd.bits == 64:
            st.frame_ptrs[rd.name] = (rs.name, 0)
        elif fp is not None and rd.bits == 64:
            st.frame_ptrs[rd.name] = fp

    # ---- constants ---------------------------------------------------------

    def _const(self, insn: Insn) -> None:
        rd, v = insn.reg(0), insn.imm(1)
        if rd is None:
            return
        self._write(rd, CLEAN)
        if v is not None:
            m = insn.mnemonic
            self.state.consts[rd.name] = ((~v) & MASK64) if m == "movn" else (v & MASK64)

    def _partial(self, insn: Insn) -> None:
        rd = insn.reg(0)
        if rd is None:
            return
        old = self.state.get(rd.name)
        labels = old.labels
        for r in insn.regs()[1:]:
            labels |= self._src(r).labels
        c = self.state.consts.get(rd.name)
        self._write(rd, Taint(labels, None))
        if insn.mnemonic == "movk" and c is not None and insn.imm(1) is not None and not labels:
            v = insn.imm(1)
            shift = (v.bit_length() - 1) // 16 * 16 if v else 0
            self.state.consts[rd.name] = (c & ~(0xFFFF << shift) & MASK64) | (v & MASK64)

    # ---- extensions --------------------------------------------------------

    def _ext(self, insn: Insn) -> None:
        m, rd, rn = insn.mnemonic, insn.reg(0), insn.reg(1)
        if rd is None or rn is None:
            return
        src = self._src(rn)
        bound: Optional[int] = None
        if m in ("uxtb", "uxth", "uxtw"):
            bound = {"uxtb": 0xFF, "uxth": 0xFFFF, "uxtw": 0xFFFF_FFFF}[m]
            if src.bound is not None:
                bound = min(bound, src.bound)
        elif m in ("sxtb", "sxth", "sxtw"):
            bound = ext_bound(m, 0, src.bound, 64)
        elif m in ("ubfx", "ubfiz") and insn.imm(2) is not None and insn.imm(3) is not None:
            lsb, width = insn.imm(2), insn.imm(3)
            bound = (1 << width) - 1
            if m == "ubfiz":
                bound <<= lsb
        elif m == "ubfm" and insn.imm(2) is not None and insn.imm(3) is not None:
            immr, imms = insn.imm(2), insn.imm(3)
            if imms >= immr:
                bound = (1 << (imms - immr + 1)) - 1
            else:
                bound = ((1 << (imms + 1)) - 1) << (rd.bits - immr)
        self._write(rd, Taint(src.labels, bound))

    # ---- csel / cset -------------------------------------------------------

    def _csel(self, insn: Insn) -> None:
        rd = insn.reg(0)
        if rd is None:
            return
        labels = self.state.get(FLAGS).labels
        bound: Optional[int] = None
        srcs = insn.regs()[1:]
        for r in srcs:
            labels |= self._src(r).labels
        if insn.mnemonic in ("cset", "csetm"):
            bound = 1 if insn.mnemonic == "cset" else None
        elif insn.mnemonic == "csel" and len(srcs) == 2:
            bs = [self._src(r).bound for r in srcs]
            if all(b is not None for b in bs):
                bound = max(bs)
        self._write(rd, Taint(labels, bound))

    def _flags_only(self, insn: Insn) -> None:
        labels: Labels = EMPTY
        for r in insn.regs():
            labels |= self._src(r).labels
        if insn.mnemonic in ("ccmp", "ccmn", "fccmp"):
            labels |= self.state.get(FLAGS).labels
        self.state.set(FLAGS, Taint(labels, None))

    # ---- ALU ---------------------------------------------------------------

    def _alu(self, insn: Insn) -> None:
        st, m = self.state, insn.mnemonic
        rd = insn.reg(0)
        if rd is None:
            return
        srcs = insn.regs()[1:]
        imm = next((o.value for o in insn.ops[1:] if isinstance(o, Imm)), None)
        taints = [self._src(r) for r in srcs]
        labels: Labels = EMPTY
        for t in taints:
            labels |= t.labels
        tainted = [t for t in taints if t.tainted]
        all_bounded = bool(tainted) and all(t.bound is not None for t in tainted)
        bound: Optional[int] = None

        # sp / frame arithmetic
        if m in ("add", "sub") and imm is not None and len(srcs) == 1 and srcs[0].shift is None:
            rn = srcs[0]
            delta = imm if m == "add" else -imm
            if rd.name == rn.name and rd.name in FRAME_BASES:
                st.rebase(rd.name, delta)
                if taints[0].tainted:
                    st.regs[rd.name] = Taint(labels, None)
                return
            if rn.name in FRAME_BASES and rd.bits == 64:
                self._write(rd, Taint(labels, None))
                st.frame_ptrs[rd.name] = (rn.name, delta)
                return
            if rn.name in st.frame_ptrs and rd.bits == 64:
                b, k = st.frame_ptrs[rn.name]
                self._write(rd, Taint(labels, None))
                st.frame_ptrs[rd.name] = (b, k + delta)
                return

        if m == "and" and imm is not None:
            bound = and_bound(imm, rd.bits)
            if bound is not None and taints and taints[0].bound is not None:
                bound = min(bound, taints[0].bound)
        elif m in ("and", "ands"):
            bs = [t.bound for t in taints if t.bound is not None]
            bound = min(bs) if bs else None
        elif m in ("bic", "bics") and taints:
            bound = taints[0].bound
        elif m in ("add", "adds", "sub", "subs", "orr", "eor") and all_bounded:
            bound = sum(t.bound for t in tainted)
        elif m in ("lsl", "lsr", "asr") and imm is not None and len(taints) == 1 and taints[0].bound is not None:
            bound = ext_bound(m, imm, taints[0].bound, 64)
        elif m in ("add", "sub") and imm is not None and len(taints) == 1:
            bound = taints[0].bound

        if m in ("adds", "subs", "ands", "bics", "adcs", "sbcs", "negs", "ngcs"):
            st.set(FLAGS, Taint(labels, None))
        self._write(rd, Taint(labels, bound))
        if m == "add" and imm is not None and len(srcs) == 1 and srcs[0].name in st.consts and not labels:
            st.consts[rd.name] = (st.consts[srcs[0].name] + imm) & MASK64

    # ---- memory helpers ----------------------------------------------------

    def _addr(self, mem: Mem) -> Tuple[Taint, Optional[str], int, bool]:
        st = self.state
        bt = st.get(mem.base)
        exact = mem.index is None
        if mem.index is not None:
            it = self._src(mem.index)      # bound already scaled by lsl / extend
            # a clean base plus a bounded index stays bounded; a tainted base does not
            bound = it.bound if it.tainted and not bt.tainted else None
            bt = Taint(bt.labels | it.labels, bound)
        rb, roff = st.resolve(mem.base, mem.offset)
        return bt, rb, roff, exact

    def _writeback(self, mem: Mem) -> None:
        st = self.state
        if mem.writeback == "pre":
            delta = mem.offset
        elif mem.writeback == "post":
            delta = mem.wb_amount
        else:
            return
        if mem.base in FRAME_BASES:
            st.rebase(mem.base, delta)
        elif mem.base in st.frame_ptrs:
            b, k = st.frame_ptrs[mem.base]
            st.frame_ptrs[mem.base] = (b, k + delta)
        elif mem.base in st.consts:
            st.consts[mem.base] = (st.consts[mem.base] + delta) & MASK64

    def _load_value(self, rb: Optional[str], off: int, exact: bool, bt: Taint, nbytes: int, signed: bool) -> Taint:
        st = self.state
        val = st.mem_get(rb, off) if (rb is not None and exact) else CLEAN
        if bt.tainted:
            val = Taint(val.labels | bt.labels, val.bound)
        bound = None if signed or nbytes >= 8 else (1 << (8 * nbytes)) - 1
        if val.bound is not None and bound is not None:
            bound = min(bound, val.bound)
        elif val.bound is not None:
            bound = val.bound
        return Taint(val.labels, bound if val.tainted else None)

    def _load(self, insn: Insn) -> None:
        rd = insn.reg(0)
        mem = insn.first_mem()
        if rd is None:
            return
        if mem is None:
            self._write(rd, CLEAN)
            return
        if mem.writeback == "pre":
            self._writeback(mem)
            mem = Mem(mem.base, 0)
        bt, rb, roff, exact = self._addr(mem)
        self._check_addr(insn, bt, "tainted-load-address", f"{insn.mnemonic} {rd} via {mem}")
        nbytes, signed = access_bytes(insn.mnemonic, rd.bits)
        self._write(rd, self._load_value(rb, roff, exact, bt, nbytes, signed))
        self._reload_facts(rd, rb, roff, exact, nbytes)
        if mem.writeback == "post":
            self._writeback(mem)

    def _reload_facts(self, rd: Optional[Reg], rb: Optional[str], off: int, exact: bool, nbytes: int) -> None:
        """A 64-bit reload restores the slot's pointer fact, alias class and RA role."""
        st = self.state
        if rd is None or rb is None or not exact or rd.name == "xzr":
            return
        slot = (rb, off)
        c = st.mem_consts.get(slot)
        if c is not None and c[0] == nbytes:
            st.consts[rd.name] = c[1]
        if nbytes != 8:
            return
        ptr = st.mem_ptrs.get(slot)
        if ptr is not None:
            st.frame_ptrs[rd.name] = ptr
        st.alias(rd.name, slot)
        if slot in st.lr_slots:
            st.ra_regs.add(rd.name)

    def _store(self, insn: Insn) -> None:
        rt = insn.reg(0)
        mem = insn.first_mem()
        if mem is None:
            return
        if insn.mnemonic in ("stxr", "stlxr", "stxrb", "stxrh", "stlxrb", "stlxrh"):
            self._write(rt, CLEAN)
            rt = insn.reg(1)
        if mem.writeback == "pre":
            self._writeback(mem)
            mem = Mem(mem.base, 0)
        bt, rb, roff, exact = self._addr(mem)
        self._check_addr(insn, bt, "tainted-store-address", f"{insn.mnemonic} {rt} via {mem}")
        if rb is not None and exact:
            self._store_slot(insn, rb, roff, rt)
        if mem.writeback == "post":
            self._writeback(mem)

    def _store_slot(self, insn: Insn, rb: str, off: int, rt: Optional[Reg]) -> None:
        st = self.state
        val = self._src(rt) if rt is not None else CLEAN
        full = rt is not None and rt.bits == 64 and rt.name != "xzr"
        ptr = None
        if full:
            ptr = st.frame_ptrs.get(rt.name) or (("sp", 0) if rt.name == "sp" else None)
        st.mem_set(rb, off, val, ptr)
        if rt is not None and rt.name in st.consts and not val.tainted:
            st.mem_consts[(rb, off)] = (rt.bits // 8, st.consts[rt.name] & ((1 << rt.bits) - 1))
        elif rt is not None and rt.name == "xzr":
            st.mem_consts[(rb, off)] = (rt.bits // 8, 0)
        if full and rt.name != "sp":
            st.alias((rb, off), rt.name)
        if full and rt.name in st.ra_regs and rb in FRAME_BASES:
            st.lr_slots.add((rb, off))
        elif (rb, off) in st.lr_slots:
            st.lr_slots.discard((rb, off))
            if val.tainted:
                self._find(insn, "tainted-overwrite-of-saved-lr", f"{rt} -> [{rb}, #{off}]", val.labels)

    def _pair(self, insn: Insn, load: bool) -> None:
        r1, r2 = insn.reg(0), insn.reg(1)
        mem = insn.first_mem()
        if r1 is None or r2 is None or mem is None:
            return
        if insn.mnemonic in ("stxp", "stlxp"):
            self._write(r1, CLEAN)
            r1, r2 = insn.reg(1), insn.reg(2)
        if mem.writeback == "pre":
            self._writeback(mem)
            mem = Mem(mem.base, 0)
        bt, rb, roff, exact = self._addr(mem)
        kind = "tainted-load-address" if load else "tainted-store-address"
        self._check_addr(insn, bt, kind, f"{insn.mnemonic} {r1},{r2} via {mem}")
        nbytes = 4 if insn.mnemonic == "ldpsw" else (r1.bits // 8 if r1 else 8)
        if load:
            self._write(r1, self._load_value(rb, roff, exact, bt, nbytes, insn.mnemonic == "ldpsw"))
            self._write(r2, self._load_value(rb, roff + nbytes, exact, bt, nbytes, insn.mnemonic == "ldpsw"))
            self._reload_facts(r1, rb, roff, exact, nbytes)
            self._reload_facts(r2, rb, roff + nbytes, exact, nbytes)
        elif rb is not None and exact:
            self._store_slot(insn, rb, roff, r1)
            self._store_slot(insn, rb, roff + nbytes, r2)
        if mem.writeback == "post":
            self._writeback(mem)

    def _atomic(self, insn: Insn) -> None:
        rs, rt, mem = insn.reg(0), insn.reg(1), insn.first_mem()
        if rs is None or rt is None or mem is None:
            return
        bt, rb, roff, exact = self._addr(mem)
        self._check_addr(insn, bt, "tainted-store-address", f"{insn.mnemonic} via {mem}")
        old = self._load_value(rb, roff, exact, bt, rt.bits // 8, False)
        m = insn.mnemonic
        if m.startswith("cas"):
            self._write(rs, old)
            new = Taint(old.labels | self._src(rt).labels, None)
        else:
            new = self._src(rs) if m.startswith("swp") else Taint(old.labels | self._src(rs).labels, None)
            self._write(rt, old)
        if rb is not None and exact:
            self.state.mem_set(rb, roff, new)

    # ---- control flow ------------------------------------------------------

    def _cond_branch(self, insn: Insn, t: Taint) -> None:
        if self.policy.report_tainted_branch and t.tainted:
            self._find(insn, "tainted-branch-condition", str(insn).split(": ", 1)[-1], t.labels)

    def _call(self, insn: Insn) -> None:
        st = self.state
        if insn.mnemonic == "bl":
            target = insn.imm(0)
        else:
            rn = insn.reg(0)
            bt = self._src(rn)
            if bt.tainted:
                self._find(insn, "tainted-indirect-call", f"{insn.mnemonic} via {rn}", bt.labels)
            target = st.consts.get(rn.name) if rn else None
        self._call_effects(insn, target)

    def _call_effects(self, insn: Insn, target: Optional[int]) -> None:
        st = self.state
        arg_labels: Labels = EMPTY
        for r in ARG_REGS + VEC_ARG_REGS:
            arg_labels |= st.get(r).labels
        for r in CALLER_SAVED:
            st.set(r, CLEAN)
        st.set(FLAGS, CLEAN)
        if arg_labels:
            for r in RET_REGS + VEC_RET_REGS:
                st.set(r, Taint(arg_labels, None))
        st.set(LINK_REG, CLEAN)      # also drops x30's return-address role
        st.consts[LINK_REG] = (insn.address + insn.size) & MASK64
        st.drop_below_sp()

    def _indirect_jump(self, insn: Insn) -> None:
        rn = insn.reg(0)
        bt = self._src(rn)
        if bt.tainted:
            ret = rn is not None and rn.name in self.state.ra_regs
            self._find(insn, "tainted-return-address" if ret else "tainted-indirect-jump",
                       f"{insn.mnemonic} via {rn}", bt.labels)

    def _ret(self, insn: Insn) -> None:
        rn = insn.reg(0)
        name = rn.name if rn is not None else LINK_REG
        bt = self.state.get(name)
        if bt.tainted:
            self._find(insn, "tainted-return-address", f"{insn.mnemonic} via {name}", bt.labels)

    def _svc(self, insn: Insn) -> None:
        st = self.state
        nr = st.consts.get(SYSCALL_NR_REG)
        name = SYSCALLS.get(nr, f"sys_{nr}") if nr is not None else "sys_?"
        if st.get(SYSCALL_NR_REG).tainted:
            self._find(insn, "tainted-syscall-number", name, st.get(SYSCALL_NR_REG).labels)
        for r in SYSCALL_ARG_REGS:
            if st.get(r).tainted:
                self._find(insn, "tainted-syscall-arg", f"{name}({r})", st.get(r).labels)
        if name in INPUT_SYSCALLS:
            label = f"{name}@{insn.address:#x}"
            n = st.consts.get("x2")
            self.taint_buffer("x1", n if n is not None else 1 << 20, label,
                              region=f"@buf{insn.address:x}")
            st.set(SYSCALL_RET_REG, Taint(frozenset({label}), None))
        else:
            st.set(SYSCALL_RET_REG, CLEAN)

    def _unknown(self, insn: Insn) -> None:
        self.unknown.add(insn.mnemonic)
        rd = insn.reg(0)
        if rd is None:
            return
        labels: Labels = EMPTY
        for r in insn.regs()[1:]:
            labels |= self._src(r).labels
        self._write(rd, Taint(labels, None))


# ---------------------------------------------------------------------------
# Binary-path helpers
# ---------------------------------------------------------------------------

def _name_at(symbols: Dict[int, str], va: int, tol: int = _PLT_TOL) -> Optional[str]:
    if va in symbols:
        return symbols[va]
    for off in range(0, tol + 1, 4):
        if va + off in symbols:
            return symbols[va + off]
        if va - off in symbols:
            return symbols[va - off]
    return None


def _load_elf(path: str):
    try:
        import lief
        elf = lief.parse(path)
        if elf is None:
            raise ValueError(f"lief: cannot parse {path}")
        data = Path(path).read_bytes()
        text = elf.get_section(".text")
        if text:
            text_start = text.virtual_address
            text_end   = text_start + text.size
            base_va    = text.virtual_address - text.offset
        else:
            base_va = text_start = 0
            text_end = len(data)
        syms: Dict[int, str] = {}
        for sym in elf.symbols:
            if sym.name and sym.value:
                syms[sym.value] = sym.name
        return data, base_va, text_start, text_end, syms
    except ImportError:
        pass
    from elftools.elf.elffile import ELFFile
    data = Path(path).read_bytes()
    ef = ELFFile(open(path, "rb"))
    text_sh = ef.get_section_by_name(".text")
    if text_sh:
        text_start = text_sh["sh_addr"]
        text_end   = text_start + text_sh["sh_size"]
        base_va    = text_sh["sh_addr"] - text_sh["sh_offset"]
    else:
        base_va = text_start = text_end = 0
        text_end = len(data)
    syms: Dict[int, str] = {}
    for sec in ef.iter_sections():
        if sec.name in (".symtab", ".dynsym"):
            for sym in sec.iter_symbols():
                if sym.name and sym["st_value"]:
                    syms[sym["st_value"]] = sym.name
    return data, base_va, text_start, text_end, syms


# ---------------------------------------------------------------------------
# ARM64TaintTracker: binary-path API
# ---------------------------------------------------------------------------

class ARM64TaintTracker:
    """
    Intraprocedural + interprocedural taint tracker for AArch64 ELF binaries.

    Binary path: capstone disassembly -> TaintEngine.
    Source/sink: recv/read -> x0; system/strcpy -> x0-x7.

    Also exposes:
      run_from_listing(text)  -- listing text path
      run_from_objdump(text)  -- objdump -d text path
    Both return List[Finding] (generic labeled-taint findings).
    """

    def __init__(self, data: bytes, base_va: int, text_start: int, text_end: int,
                 symbols: Dict[int, str],
                 custom_sinks: Optional[Set[str]] = None,
                 custom_sources: Optional[Set[str]] = None):
        self._data        = data
        self._base_va     = base_va
        self._text_start  = text_start
        self._text_end    = text_end
        self._syms        = symbols
        self._sink_names  = _SINK_NAMES | (custom_sinks or set())
        self._source_names = _SOURCE_NAMES | (custom_sources or set())
        self._user_sources: Dict[str, str] = {}               # name -> "ret" | "val" | "xN"
        self._user_sink_args: Dict[str, Tuple[str, ...]] = {}  # name -> arg regs (() = all)
        self.summarise_opaque = False
        # name -> VA of functions defined here (a PLT call to one of them is followed)
        self._local_funcs: Dict[str, int] = {
            n: va for va, n in symbols.items() if text_start <= va < text_end}

    @classmethod
    def from_path(cls, path: str, custom_sinks: Optional[Set[str]] = None,
                  custom_sources: Optional[Set[str]] = None) -> "ARM64TaintTracker":
        data, base_va, text_start, text_end, syms = _load_elf(path)
        return cls(data, base_va, text_start, text_end, syms, custom_sinks, custom_sources)

    @classmethod
    def from_path_full(cls, path: str, custom_sinks: Optional[Set[str]] = None,
                       custom_sources: Optional[Set[str]] = None) -> "ARM64TaintTracker":
        """
        Like from_path but augments symbols from BinaryContext:
          - PLT stub VAs (so BL targets resolve to sink / source names); handles
            GNU ld and lld layouts and 16/24-byte BTI/PAC stubs
          - Function starts (.eh_frame, or ARM64FuncStartScanner when stripped)
            so all internal functions are scanned, not just exports.

        Use this instead of from_path for stripped ARM64 binaries.
        """
        from .binary_context import BinaryContext
        return cls.from_context(BinaryContext.load_or_build(path), custom_sinks, custom_sources)

    @classmethod
    def from_path_ko(cls, path: str, custom_sinks: Optional[Set[str]] = None,
                     custom_sources: Optional[Set[str]] = None) -> "ARM64TaintTracker":
        """
        Kernel module variant (.ko).

        Kernel modules are relocatable objects — BL instructions targeting external
        symbols have displacement=0 before the kernel loader patches them.  The
        .rela.text section contains R_AARCH64_CALL26/JUMP26 entries that map
        instruction offsets to their real external symbol names.  This method injects
        those entries into _syms so the taint engine resolves them transparently.
        """
        import struct
        import lief as _lief

        data, base_va, text_start, text_end, syms = _load_elf(path)

        try:
            binary = _lief.parse(path)
            syms_list = list(binary.symtab_symbols)
            rela_text = next(
                (s for s in binary.sections if s.name == ".rela.text"), None
            )
            if rela_text:
                rdata = bytes(rela_text.content)
                n = len(rdata) // 24
                for i in range(n):
                    off, info, addend = struct.unpack_from("<QQq", rdata, i * 24)
                    sym_idx = info >> 32
                    rtype = info & 0xFFFFFFFF
                    if rtype in (282, 283):  # R_AARCH64_JUMP26 / CALL26
                        if sym_idx < len(syms_list):
                            sname = syms_list[sym_idx].name
                            if sname:
                                # BL at text_start+off resolves to external symbol sname.
                                # The unrelocated BL has displacement=0, so capstone reports
                                # target = text_start+off.  Map that address to the symbol.
                                syms[text_start + off] = sname
        except Exception:
            pass

        return cls(data, base_va, text_start, text_end, syms, custom_sinks, custom_sources)

    @classmethod
    def from_context(cls, ctx, custom_sinks: Optional[Set[str]] = None,
                     custom_sources: Optional[Set[str]] = None) -> "ARM64TaintTracker":
        """Build from a BinaryContext: file layout from the ELF (or the raw image), names from ctx."""
        if getattr(ctx, "raw", None):
            data = Path(ctx.path).read_bytes()
            base_va = text_start = ctx.base_va
            text_end, syms = base_va + len(data), {}
        else:
            data, base_va, text_start, text_end, syms = _load_elf(ctx.path)
        syms.update(ctx.plt)                                   # stub VA -> import name
        syms.update({va: name for name, va in ctx.exports.items()})
        for va, role in getattr(ctx, "roles", {}).items():     # inferred libc roles
            syms.setdefault(va, role)
        for fva in ctx.func_starts:
            if text_start <= fva < text_end and fva not in syms:
                syms[fva] = f"fn_{fva:x}"
        return cls(data, base_va, text_start, text_end, syms, custom_sinks, custom_sources)

    # ---- user sources / sinks ---------------------------------------------

    def _spec_name(self, tok: str, prefix: str) -> str:
        """'name' or '0xVA' -> the callee name the tracker will see at that VA."""
        if tok.lower().startswith("0x"):
            va = int(tok, 16)
            name = self._syms.get(va)
            if name is None or name.startswith("fn_"):
                name = f"{prefix}_{va:x}"
                self._syms[va] = name
            return name
        return tok

    def add_sources(self, specs) -> List[str]:
        """Attacker-input functions beyond recv/read: framework accessors (HTTP / MQTT /
        RTSP request getters, protobuf field readers ...). Each spec is NAME or 0xVA, with:
          (none) / :ret   result and the memory it points at are tainted (char *get_param())
          :val            scalar result only (int get_content_length())
          :xN             fills the buffer at argument xN (int read_body(req, buf, n) -> :x1)
        Returns the callee names used."""
        out = []
        for spec in ([specs] if isinstance(specs, str) else specs):
            for item in str(spec).split(","):
                item = item.strip()
                if not item:
                    continue
                tok, _, mode = item.partition(":")
                name = self._spec_name(tok, "src")
                self._user_sources[name] = mode or "ret"
                self._source_names = self._source_names | {name}
                out.append(name)
        return out

    def add_sinks(self, specs) -> List[str]:
        """Extra sinks: NAME or 0xVA, optionally :xN[+xM] = the arguments whose value or
        pointee must not be tainted (default: any argument's value, x0..x3 pointee)."""
        out = []
        for spec in ([specs] if isinstance(specs, str) else specs):
            for item in str(spec).split(","):
                item = item.strip()
                if not item:
                    continue
                tok, _, regs = item.partition(":")
                name = self._spec_name(tok, "sink")
                self._user_sink_args[name] = tuple(r for r in regs.split("+") if r)
                self._sink_names = self._sink_names | {name}
                out.append(name)
        return out

    # ---- listing / objdump entry points ------------------------------------

    def run_from_listing(self, text: str, label: str = "input") -> List[Finding]:
        eng = TaintEngine()
        for r in ARG_REGS:
            eng.taint_register(r, label)
        return eng.run(_from_listing(text))

    def run_from_objdump(self, text: str, label: str = "input") -> List[Finding]:
        eng = TaintEngine()
        for r in ARG_REGS:
            eng.taint_register(r, label)
        return eng.run(_from_objdump(text))

    # ---- binary scan -------------------------------------------------------

    def _get_func_starts(self) -> List[int]:
        starts: Set[int] = set()
        for va in self._syms:
            if self._text_start <= va < self._text_end:
                starts.add(va)
        if not starts:
            starts.add(self._text_start)
        return sorted(starts)

    def _insns_for_func(self, func_va: int, func_end: int) -> List[Insn]:
        offset = func_va - self._base_va
        length = func_end - func_va
        if offset < 0 or offset + length > len(self._data):
            return []
        snippet = self._data[offset:offset + length]
        return list(from_capstone(snippet, func_va))

    def _scan_func_binary(self, func_va: int, func_end: int,
                          init_labels: Optional[Set[str]] = None) -> List[TaintFindingARM64]:
        return self._scan(func_va, func_end, set(init_labels or ()), {}, 0, None)

    def _scan(
        self,
        fva: int,
        fend: int,
        reg_labels: Set[str],
        ptr_seed: Dict[str, FrozenSet[str]],
        depth: int,
        enqueue,
    ) -> List[TaintFindingARM64]:
        """
        Linear taint pass over one function.

        reg_labels: labels seeded on every arg register (caller passed tainted values)
        ptr_seed:   arg register -> labels of the buffer it points at
        enqueue:    callback(callee_va, reg_labels, ptr_seed) for interprocedural runs
        """
        func_name = self._syms.get(fva, f"fn_0x{fva:x}")
        findings: List[TaintFindingARM64] = []
        eng = TaintEngine()
        for r in ARG_REGS:
            for lbl in reg_labels:
                eng.taint_register(r, lbl)
        for r, labels in ptr_seed.items():
            eng.taint_pointee(r, labels)

        for insn in self._insns_for_func(fva, fend):
            m = insn.mnemonic
            target = insn.imm(0) if m in ("bl", "b") else None
            if m == "b" and (target is None or fva <= target < fend):
                eng.step(insn)          # intra-function branch
                continue
            if target is not None:
                callee = _name_at(self._syms, target)
                self._on_call(eng, insn, target, callee, fva, func_name, findings, depth, enqueue)
            else:
                opq: Set[str] = set()
                if self.summarise_opaque and m.startswith("blr"):
                    for r in ARG_REGS:
                        opq |= eng.state.get(r).labels | eng.state.pointee(r).labels
                self._before_step(eng, insn, fva)
                eng.step(insn)          # ret is checked and the linear pass continues
                self._after_step(eng, insn, fva)
                if opq:
                    self._opaque_result(eng, opq, insn.address)
        return findings

    def _before_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        """Hook for subclasses: called before each non-call instruction is stepped."""

    def _after_step(self, eng: TaintEngine, insn: Insn, fva: int) -> None:
        """Hook for subclasses: called after each non-call instruction is stepped."""

    def _on_call(self, eng: TaintEngine, insn: Insn, target: int, callee: Optional[str],
                 fva: int, func_name: str, findings: List[TaintFindingARM64],
                 depth: int, enqueue) -> None:
        """bl / tail-call b: sink check, copy + source modelling, callee queueing."""
        st = eng.state
        if callee in self._sink_names:
            user = self._user_sink_args.get(callee)
            regs = user if user else ARG_REGS
            ptr_regs = user if user else (_SINK_POINTER_ARGS.get(callee)
                                          or (("x0", "x1", "x2", "x3") if user is not None else ()))
            tainted = [r for r in regs if st.get(r).tainted and callee not in _SCAN_SINKS]
            tainted += [f"*{r}" for r in ptr_regs if st.pointee(r).tainted and r not in tainted]
            if tainted:
                labels: Set[str] = set()
                sure = False                    # some argument is tainted without an opaque hop
                for a in tainted:
                    al = (st.pointee(a[1:]) if a.startswith("*") else st.get(a)).labels
                    labels |= al
                    sure |= OPAQUE not in al
                if sure:
                    labels.discard(OPAQUE)
                sev = ("CRITICAL" if callee in _EXEC_SINKS else
                       "MEDIUM" if callee in _SCAN_SINKS else "HIGH")
                note = "parsed with sscanf: check %s width (ablation fmtstr)" if callee in _SCAN_SINKS else ""
                self._finding(findings, fva, func_name, insn.address, callee, tainted, labels, sev, note)

        # read / recv whose length came from attacker data (a parsed header field)
        if callee in self._source_names and callee in _SOURCE_BUFFERS:
            nreg = _SOURCE_BUFFERS[callee][1]
            if nreg and st.get(nreg).tainted:
                self._finding(findings, fva, func_name, insn.address, callee, [nreg],
                              set(st.get(nreg).labels), "HIGH",
                              "length from attacker data into a fixed buffer")

        # sscanf of attacker text: every output it writes is attacker data
        if callee in _SCAN_SINKS and st.pointee("x0").tainted:
            for r in ARG_REGS[2:]:
                if r in st.frame_ptrs:
                    eng.taint_buffer(r, 1 << 8, st.pointee("x0").labels, region=f"@scan{insn.address:x}")

        # user sources that fill an argument buffer
        mode = self._user_sources.get(callee or "")
        if mode and mode.startswith("x") and mode in ARG_REGS:
            eng.taint_buffer(mode, 1 << 20, callee, region=f"@buf{insn.address:x}")

        # unmodelled call fed attacker data: summarise conservatively (lower confidence)
        opaque_labels: Set[str] = set()
        if (self.summarise_opaque and callee not in self._sink_names
                and callee not in self._source_names and callee not in _COPY_FUNCS):
            for r in ARG_REGS:
                opaque_labels |= st.get(r).labels | st.pointee(r).labels

        if callee in _COPY_FUNCS:
            dst, srcs, nreg = _COPY_FUNCS[callee]
            labels = frozenset().union(*(st.get(r).labels | st.pointee(r).labels for r in srcs))
            if labels:
                n = st.consts.get(nreg) if nreg else None
                eng.taint_buffer(dst, n if n is not None else 1 << 12, labels,
                                 region=f"@buf{insn.address:x}")
        elif callee in self._source_names and callee in _SOURCE_BUFFERS:
            buf, nreg = _SOURCE_BUFFERS[callee]
            n = st.consts.get(nreg) if nreg else None
            eng.taint_buffer(buf, n if n is not None else 1 << 20, callee,
                             region=f"@buf{insn.address:x}")
        elif callee not in self._sink_names and enqueue is not None and depth > 0:
            reg_in = [r for r in ARG_REGS if st.get(r).tainted]
            ptr_in = {r: st.pointee(r).labels for r in ARG_REGS if st.pointee(r).tainted}
            if reg_in or ptr_in:
                # -fPIC: a call to this binary's own export goes through its PLT stub
                callee_va = self._local_funcs.get(callee, target) if callee else target
                enqueue(callee_va, set().union(*(st.get(r).labels for r in reg_in)), ptr_in)

        eng._call_effects(insn, target)
        if callee in self._source_names:
            # scalar result in x0 only: read/recv/fgetc/... do not return in x1
            st.set("x1", CLEAN)
            st.set("x0", Taint(frozenset({callee}), None))
            if self._user_sources.get(callee) == "ret":       # pointer to request data
                eng.taint_buffer("x0", 1 << 20, callee, region=f"@ret{insn.address:x}")
        elif opaque_labels:
            self._opaque_result(eng, opaque_labels, insn.address)

    def _opaque_result(self, eng: TaintEngine, labels: Set[str], site: int) -> None:
        st = eng.state
        lab = frozenset(labels) | {OPAQUE}
        st.set("x0", Taint(lab, None))
        eng.taint_buffer("x0", 1 << 12, lab, region=f"@opq{site:x}")

    def _finding(self, findings, fva, func_name, site, callee, tainted, labels, sev, note="") -> None:
        if OPAQUE in labels:                       # passed through a call with no model
            sev = {"CRITICAL": "HIGH", "HIGH": "MEDIUM"}.get(sev, "LOW")
            note = "; ".join(x for x in (note, "via opaque call (lower confidence)") if x)
        real = sorted(l for l in labels if l != OPAQUE)
        findings.append(TaintFindingARM64(
            func_va=fva, func_name=func_name, sink_va=site, sink_name=callee,
            tainted_args=list(tainted), source_name=next(iter(real), "unknown"),
            severity=sev, note=note))

    # ---- public API -------------------------------------------------------

    def run(self) -> List[TaintFindingARM64]:
        starts = self._get_func_starts()
        findings: List[TaintFindingARM64] = []
        for i, fva in enumerate(starts):
            fend = starts[i + 1] if i + 1 < len(starts) else self._text_end
            findings.extend(self._scan_func_binary(fva, fend))
        return findings

    def run_interprocedural(self, depth: int = 4) -> List[TaintFindingARM64]:
        starts   = self._get_func_starts()
        func_end: Dict[int, int] = {}
        for i, fva in enumerate(starts):
            func_end[fva] = starts[i + 1] if i + 1 < len(starts) else self._text_end

        findings: List[TaintFindingARM64] = []
        queue: List[Tuple[int, int, Set[str], Dict[str, FrozenSet[str]]]] = [
            (fva, depth, set(), {}) for fva in starts]
        visited: Dict[Tuple, int] = {}

        while queue:
            fva, d, reg_labels, ptr_seed = queue.pop(0)
            key = (fva, tuple(sorted(reg_labels)),
                   tuple(sorted((r, tuple(sorted(l))) for r, l in ptr_seed.items())))
            if visited.get(key, -1) >= d:
                continue
            visited[key] = d

            def enqueue(target: int, labels: Set[str], ptrs: Dict[str, FrozenSet[str]], _d=d) -> None:
                c_va = target
                for off in range(0, _PLT_TOL + 1, 4):
                    if c_va + off in func_end:
                        c_va += off; break
                    if c_va - off in func_end and c_va - off > 0:
                        c_va -= off; break
                if c_va in func_end:
                    queue.append((c_va, _d - 1, labels, ptrs))

            findings.extend(self._scan(fva, func_end.get(fva, self._text_end),
                                       reg_labels, ptr_seed, d, enqueue))

        seen: Set[Tuple] = set()
        unique: List[TaintFindingARM64] = []
        for f in findings:
            k = (f.func_va, f.sink_va, f.sink_name)
            if k not in seen:
                seen.add(k)
                unique.append(f)
        return unique

    def report(self, findings: List[TaintFindingARM64]) -> str:
        if not findings:
            return "ARM64 taint: no findings."
        lines = [f"ARM64 taint: {len(findings)} finding(s)\n"]
        for f in sorted(findings, key=lambda x: (x.func_va, x.sink_va)):
            lines.append(str(f))
        return "\n".join(lines)
