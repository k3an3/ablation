"""
path_solver_arm64.py: Bounded symbolic execution + Z3 path feasibility for AArch64.

AArch64 backend for PathSolver. Same pipeline and result types:

  cfg_arm64  ->  taint tracker / FuncProfiler (finds sink VA)  ->  ARM64PathSolver (finds input)

For each simple CFG path entry -> ... -> block containing target_va:
  - Registers x0..x30, sp are Z3 BitVec(64) symbols at entry (x0_0, x1_0, ...)
  - Instructions update register / flag / stack expressions
  - Each conditional branch on the path adds its direction as a constraint
  - Execution stops at target_va itself (any instruction VA, e.g. the `bl system`)
Z3 solves the conjunction; a SAT model gives concrete entry register values.

Differences from the x86-64 PathSolver:
  - NZCV flags are modelled exactly (subs/adds/ands/ccmp/ccmn, 32- and 64-bit),
    so b.cond, csel/cset/csinc/... evaluate the architectural condition instead
    of the most recent compare's operands.
  - cbz/cbnz/tbz/tbnz constrain the tested register directly.
  - Stores to a concrete address expression (sp/x29 frame slots, or any base
    whose expression is identical) are forwarded to later loads of the same
    address and width, so -O0 argument spills keep their constraints.
  - target_va may be any instruction, not only a block start.

Limits (same spirit as PathSolver): loads without a forwarded store are fresh
symbols (sound over-approximation); calls give x0..x18/x30 fresh values; a store
through one register does not invalidate a slot reached through another
(aliasing ignored); vector/FP registers are not tracked.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import z3

from .cfg_arm64 import CFG, build_cfg
from .insn_arm64 import Cond, Imm, Insn, Mem, Reg, from_listing, make_insn
from .isa_arm64 import ISA, access_bytes
from .path_solver import PathResult, PathSolver, StructSeed

_GPRS = [f"x{i}" for i in range(31)] + ["sp"]
_LSL_RE = re.compile(r"lsl\s*#(\d+)")
_CALL_CLOBBER = [f"x{i}" for i in range(19)] + ["x30"]

# Mnemonics whose first register operand is a source (or that write nothing)
_NO_DEST_PREFIX = ("st", "prfm", "dmb", "dsb", "isb", "msr", "hint", "nop", "bti",
                   "pac", "aut", "xpac", "dc", "ic", "tlbi", "brk", "hlt")
_NO_DEST = (ISA.ret_mnems | ISA.indirect_jump_mnems | ISA.uncond_branch_mnems
            | ISA.cond_branch_mnems | ISA.reg_branch_mnems
            | frozenset({"cmp", "cmn", "tst", "ccmp", "ccmn", "fcmp", "fcmpe", "fccmp"}))


def _mask(bits: int) -> int:
    return (1 << bits) - 1


class _State:
    """Symbolic machine state for one path."""

    def __init__(self):
        self.regs: Dict[str, z3.ExprRef] = {r: z3.BitVec(f"{r}_0", 64) for r in _GPRS}
        self.n = z3.Bool("N_0")
        self.z = z3.Bool("Z_0")
        self.c = z3.Bool("C_0")
        self.v = z3.Bool("V_0")
        self.mem: Dict[Tuple[str, int], z3.ExprRef] = {}

    def cond(self, code: str) -> z3.BoolRef:
        n, z, c, v = self.n, self.z, self.c, self.v
        return {
            "eq": z, "ne": z3.Not(z),
            "cs": c, "hs": c, "cc": z3.Not(c), "lo": z3.Not(c),
            "mi": n, "pl": z3.Not(n), "vs": v, "vc": z3.Not(v),
            "hi": z3.And(c, z3.Not(z)), "ls": z3.Or(z3.Not(c), z),
            "ge": n == v, "lt": n != v,
            "gt": z3.And(z3.Not(z), n == v), "le": z3.Or(z, n != v),
            "al": z3.BoolVal(True), "nv": z3.BoolVal(True),
        }[code]


class ARM64PathSolver(PathSolver):
    """
    Bounded symbolic executor + Z3 path constraint solver for AArch64.

    Usage:
        ps = ARM64PathSolver.from_context(ctx, func_va)
        r = ps.solve_first_sat(target_va=0x1054c)       # any insn VA, e.g. a sink call
        print(r)       # [SAT] path=0x10518 -> 0x10530 ... model=(x0=0x..., x1=0x...)

        # hand-written / objdump input (tests, snippets)
        ps = ARM64PathSolver.from_listing(text, base=0x1000)
    """

    def __init__(
        self,
        cfg: CFG,
        plt: Optional[Dict[int, str]] = None,
        max_path_blocks: int = 40,
        max_paths: int = 50,
        z3_timeout_ms: int = 5000,
        struct_seed: Optional[StructSeed] = None,
    ):
        # PathSolver.__init__ wants x86 bytes + capstone handle; set the shared
        # fields used by solve()/solve_first_sat()/_enum_paths() directly.
        self.cfg = cfg
        self.func_va = cfg.entry
        self.func_end_va = max(b.end for b in cfg.blocks.values())
        self.plt = plt or {}
        self.data = b""
        self.md = None
        self.max_path_blocks = max_path_blocks
        self.max_paths = max_paths
        self.z3_timeout_ms = z3_timeout_ms
        self.struct_seed: StructSeed = struct_seed or {}
        self._solver: Optional[z3.Solver] = None
        self._mem_cache: Dict[Tuple[str, int], z3.ExprRef] = {}
        self._seeded_fields_used = set()
        self._insn_block: Dict[int, int] = {
            i.address: b.start for b in cfg.blocks.values() for i in b.insns
        }

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_insns(cls, insns: Iterable[Insn], entry: Optional[int] = None, **kw) -> "ARM64PathSolver":
        return cls(build_cfg(list(insns), entry), **kw)

    @classmethod
    def from_listing(cls, text: str, base: int = 0x400000, **kw) -> "ARM64PathSolver":
        return cls.from_insns(from_listing(text, base), **kw)

    @classmethod
    def from_context(cls, ctx, func_va: int, func_end_va: int = 0, max_bytes: int = 8192, **kw) -> "ARM64PathSolver":
        """Disassemble func_va..func_end_va (default: next func start) from a BinaryContext."""
        import bisect

        import capstone
        import lief

        if not func_end_va:
            idx = bisect.bisect_right(ctx.func_starts, func_va)
            nxt = ctx.func_starts[idx] if idx < len(ctx.func_starts) else func_va + max_bytes
            func_end_va = min(nxt, func_va + max_bytes)
        binary = lief.parse(Path(ctx.path).read_bytes())
        code = b""
        for seg in binary.segments:
            sva = int(seg.virtual_address)
            off = func_va - sva
            if 'LOAD' in str(seg.type) and 0 <= off < seg.physical_size:
                code = bytes(seg.content)[off: off + (func_end_va - func_va)]
                break
        if not code:
            raise ValueError(f"0x{func_va:x} is not in a LOAD segment of {ctx.path}")
        arch = getattr(capstone, 'CS_ARCH_AARCH64', None) or getattr(capstone, 'CS_ARCH_ARM64')
        md = capstone.Cs(arch, capstone.CS_MODE_ARM)
        md.skipdata = True
        insns = [make_insn(i.address, i.mnemonic, i.op_str, size=i.size)
                 for i in md.disasm(code, func_va)]
        kw.setdefault("plt", ctx.plt)
        return cls.from_insns(insns, entry=func_va, **kw)

    # ── path enumeration: target may be mid-block ─────────────────────────────

    def _enum_paths(self, start: int, target: int):
        block = self._insn_block.get(target, target)
        yield from super()._enum_paths(start, block)

    # ── operand evaluation ────────────────────────────────────────────────────

    def _read(self, st: _State, r: Reg, bits: Optional[int] = None) -> z3.ExprRef:
        bits = bits or r.bits
        if r.name == "xzr":
            return z3.BitVecVal(0, bits)
        v = st.regs.get(r.name)
        if v is None:   # vector / untracked
            v = z3.BitVec(f"{r.name}_untracked", 64)
        return z3.Extract(bits - 1, 0, v) if bits < 64 else v

    def _write(self, st: _State, r: Reg, val: z3.ExprRef) -> None:
        if r.name == "xzr" or r.name not in st.regs:
            return
        if r.bits == 32:
            val = z3.Extract(31, 0, val) if val.size() > 32 else val
            val = z3.ZeroExt(64 - val.size(), val)
        elif val.size() < 64:
            val = z3.ZeroExt(64 - val.size(), val)
        st.regs[r.name] = val

    def _fit(self, v: z3.ExprRef, bits: int) -> z3.ExprRef:
        if v.size() > bits:
            return z3.Extract(bits - 1, 0, v)
        if v.size() < bits:
            return z3.ZeroExt(bits - v.size(), v)
        return v

    def _src(self, st: _State, op, bits: int) -> z3.ExprRef:
        """Second-operand value at `bits` width, with shift / extend applied."""
        if isinstance(op, Imm):
            return z3.BitVecVal(op.value & _mask(bits), bits)
        if not isinstance(op, Reg):
            return z3.BitVec(f"opnd_{id(op)}", bits)
        if op.shift in ("uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"):
            w = {"b": 8, "h": 16, "w": 32, "x": 64}[op.shift[-1]]
            raw = z3.Extract(w - 1, 0, self._read(st, op, 64))
            ext = z3.SignExt if op.shift.startswith("s") else z3.ZeroExt
            v = ext(bits - w, raw) if bits > w else z3.Extract(bits - 1, 0, raw)
            return v << op.amount if op.amount else v
        v = self._read(st, op, bits)
        if op.shift and op.amount:
            amt = z3.BitVecVal(op.amount, bits)
            v = {"lsl": lambda: v << amt, "lsr": lambda: z3.LShR(v, amt),
                 "asr": lambda: v >> amt, "ror": lambda: z3.RotateRight(v, amt)}.get(op.shift, lambda: v)()
        return v

    # ── flags ─────────────────────────────────────────────────────────────────

    def _flags_sub(self, st: _State, a: z3.ExprRef, b: z3.ExprRef) -> z3.ExprRef:
        bits = a.size()
        res = a - b
        st.n = z3.Extract(bits - 1, bits - 1, res) == 1
        st.z = res == 0
        st.c = z3.UGE(a, b)
        st.v = z3.Extract(bits - 1, bits - 1, (a ^ b) & (a ^ res)) == 1
        return res

    def _flags_add(self, st: _State, a: z3.ExprRef, b: z3.ExprRef) -> z3.ExprRef:
        bits = a.size()
        res = a + b
        st.n = z3.Extract(bits - 1, bits - 1, res) == 1
        st.z = res == 0
        st.c = z3.ULT(res, a)
        st.v = z3.Extract(bits - 1, bits - 1, ~(a ^ b) & (a ^ res)) == 1
        return res

    def _flags_logic(self, st: _State, res: z3.ExprRef) -> z3.ExprRef:
        bits = res.size()
        st.n = z3.Extract(bits - 1, bits - 1, res) == 1
        st.z = res == 0
        st.c = z3.BoolVal(False)
        st.v = z3.BoolVal(False)
        return res

    # ── memory ────────────────────────────────────────────────────────────────

    def _addr(self, st: _State, m: Mem) -> z3.ExprRef:
        base = st.regs.get(m.base, z3.BitVec(f"{m.base}_untracked", 64))
        if m.index is not None:
            return base + self._src(st, m.index, 64)
        if m.writeback == "post":
            return base
        return base + z3.BitVecVal(m.offset & _mask(64), 64)

    def _writeback(self, st: _State, m: Mem) -> None:
        if m.writeback and m.base in st.regs:
            amt = m.wb_amount if m.writeback == "post" else m.offset
            st.regs[m.base] = st.regs[m.base] + z3.BitVecVal(amt & _mask(64), 64)

    def _slot_key(self, addr: z3.ExprRef, nbytes: int) -> Tuple[str, int]:
        return (z3.simplify(addr).sexpr(), nbytes)

    def _load(self, st: _State, m: Mem, nbytes: int, va: int, tag: str = "") -> z3.ExprRef:
        """Value of nbytes at m: forwarded store > struct seed > fresh symbol."""
        addr = self._addr(st, m)
        key = self._slot_key(addr, nbytes)
        if key in st.mem:
            return st.mem[key]
        if self.struct_seed and m.index is None:
            skey = (m.base, m.offset if m.writeback != "post" else 0)
            mc = self.struct_seed.get(skey)
            if mc is not None:
                if skey not in self._mem_cache:
                    sym = z3.BitVec(f"{mc.name}_s", mc.width)
                    if mc.exact is not None:
                        self._solver.add(sym == z3.BitVecVal(mc.exact, mc.width))
                    else:
                        if mc.lo is not None:
                            self._solver.add(z3.UGE(sym, z3.BitVecVal(mc.lo, mc.width)))
                        if mc.hi is not None:
                            self._solver.add(z3.ULE(sym, z3.BitVecVal(mc.hi, mc.width)))
                    self._mem_cache[skey] = sym
                    self._seeded_fields_used.add(mc.name)
                return self._fit(self._mem_cache[skey], nbytes * 8)
        return z3.BitVec(f"mem@{va:#x}{tag}", nbytes * 8)

    def _store(self, st: _State, m: Mem, val: z3.ExprRef, nbytes: int, off: int = 0) -> None:
        addr = self._addr(st, m) + z3.BitVecVal(off, 64)
        st.mem[self._slot_key(addr, nbytes)] = self._fit(val, nbytes * 8)

    # ── one instruction ───────────────────────────────────────────────────────

    def _exec(self, st: _State, insn: Insn, call_idx: List[int]) -> None:
        mn, ops = insn.mnemonic, insn.ops
        va = insn.address

        if mn in ISA.call_mnems:
            call_idx[0] += 1
            for r in _CALL_CLOBBER:
                st.regs[r] = z3.BitVec(f"{r}_call{call_idx[0]}", 64)
            return
        if mn == "svc":
            st.regs["x0"] = z3.BitVec(f"x0_svc@{va:#x}", 64)
            return

        # ── flag setters without a destination ────────────────────────────
        if mn in ("cmp", "cmn", "tst") and len(ops) >= 2 and isinstance(ops[0], Reg):
            bits = ops[0].bits
            a = self._read(st, ops[0])
            b = self._src(st, ops[1], bits)
            if mn == "cmp":
                self._flags_sub(st, a, b)
            elif mn == "cmn":
                self._flags_add(st, a, b)
            else:
                self._flags_logic(st, a & b)
            return
        if mn in ("ccmp", "ccmn") and len(ops) >= 4 and isinstance(ops[0], Reg) and isinstance(ops[3], Cond):
            bits = ops[0].bits
            take = st.cond(ops[3].code)        # evaluated on the incoming flags
            nzcv = ops[2].value if isinstance(ops[2], Imm) else 0
            a, b = self._read(st, ops[0]), self._src(st, ops[1], bits)
            (self._flags_sub if mn == "ccmp" else self._flags_add)(st, a, b)
            cmp_f = (st.n, st.z, st.c, st.v)
            imm_f = tuple(z3.BoolVal(bool(nzcv >> s & 1)) for s in (3, 2, 1, 0))
            st.n, st.z, st.c, st.v = (z3.If(take, cf, imf) for cf, imf in zip(cmp_f, imm_f))
            return

        # ── stores ────────────────────────────────────────────────────────
        if mn in ISA.store_mnems or mn in ISA.store_pair_mnems:
            m = insn.first_mem()
            if m is None:
                return
            regs = [o for o in ops if isinstance(o, Reg)]
            if mn in ("stxr", "stlxr", "stxrb", "stxrh", "stlxrb", "stlxrh", "stxp", "stlxp") and regs:
                self._write(st, regs[0], z3.BitVecVal(0, 32))   # status: success
                regs = regs[1:]
            if mn in ISA.store_pair_mnems:
                for k, r in enumerate(regs[:2]):
                    self._store(st, m, self._read(st, r), r.bits // 8, off=k * (r.bits // 8))
            elif regs:
                nbytes = {"b": 1, "h": 2}.get(mn[-1], regs[0].bits // 8)   # strb / sturh / ...
                self._store(st, m, self._read(st, regs[0]), nbytes)
            self._writeback(st, m)
            return

        if mn in _NO_DEST or mn.startswith(_NO_DEST_PREFIX) or not ops or not isinstance(ops[0], Reg):
            return

        rd = ops[0]
        if rd.name.startswith("v"):
            return
        bits = rd.bits
        val: Optional[z3.ExprRef] = None

        # ── loads ─────────────────────────────────────────────────────────
        if mn in ISA.load_mnems or mn in ISA.load_pair_mnems:
            m = insn.first_mem()
            if m is None:
                self._write(st, rd, z3.BitVec(f"lit@{va:#x}", 64))
                return
            if mn in ISA.load_pair_mnems:
                regs = [o for o in ops if isinstance(o, Reg)][:2]
                nb = 4 if mn == "ldpsw" else regs[0].bits // 8
                # second element sits nb bytes after the first (post-index: first is at base)
                first_off = 0 if m.writeback == "post" else m.offset
                second = Mem(m.base, first_off + nb)
                vals = [self._load(st, m, nb, va, tag=".0"), self._load(st, second, nb, va, tag=".1")]
                for r, v in zip(regs, vals):
                    v = z3.SignExt(64 - v.size(), v) if mn == "ldpsw" else v
                    self._write(st, r, v)
                self._writeback(st, m)
                return
            nb, signed = access_bytes(mn, bits)
            v = self._load(st, m, nb, va)
            if v.size() < bits:
                v = (z3.SignExt if signed else z3.ZeroExt)(bits - v.size(), v)
            self._writeback(st, m)
            self._write(st, rd, v)
            return

        # ── moves / constants ─────────────────────────────────────────────
        if mn in ("mov", "movz") and len(ops) >= 2:
            val = self._src(st, ops[1], bits)
        elif mn == "movn" and isinstance(insn.imm(1), int):
            val = z3.BitVecVal(~insn.imm(1) & _mask(bits), bits)
        elif mn == "movk" and isinstance(insn.imm(1), int):
            m_ = _LSL_RE.search(insn.raw)
            lane = 0xFFFF << (int(m_.group(1)) if m_ else 0)
            old = self._read(st, rd, bits)
            val = (old & z3.BitVecVal(~lane & _mask(bits), bits)) | z3.BitVecVal(insn.imm(1) & lane, bits)
        elif mn in ("adr", "adrp") and isinstance(insn.imm(1), int):
            val = z3.BitVecVal(insn.imm(1) & _mask(64), 64)
        elif mn == "mvn" and len(ops) >= 2:
            val = ~self._src(st, ops[1], bits)
        elif mn in ("neg", "negs") and len(ops) >= 2:
            b = self._src(st, ops[1], bits)
            val = self._flags_sub(st, z3.BitVecVal(0, bits), b) if mn == "negs" else -b

        # ── ALU ───────────────────────────────────────────────────────────
        elif mn in ("add", "adds", "sub", "subs", "and", "ands", "orr", "eor", "bic", "bics", "orn", "eon") \
                and len(ops) >= 3 and isinstance(ops[1], Reg):
            a = self._read(st, ops[1], bits)
            b = self._src(st, ops[2], bits)
            if mn == "add":
                val = a + b
            elif mn == "adds":
                val = self._flags_add(st, a, b)
            elif mn == "sub":
                val = a - b
            elif mn == "subs":
                val = self._flags_sub(st, a, b)
            elif mn == "and":
                val = a & b
            elif mn == "ands":
                val = self._flags_logic(st, a & b)
            elif mn == "orr":
                val = a | b
            elif mn == "eor":
                val = a ^ b
            elif mn == "bic":
                val = a & ~b
            elif mn == "bics":
                val = self._flags_logic(st, a & ~b)
            elif mn == "orn":
                val = a | ~b
            else:
                val = a ^ ~b
        elif mn in ("lsl", "lsr", "asr", "ror") and len(ops) >= 3 and isinstance(ops[1], Reg):
            a = self._read(st, ops[1], bits)
            amt = self._src(st, ops[2], bits) & z3.BitVecVal(bits - 1, bits)
            val = {"lsl": a << amt, "lsr": z3.LShR(a, amt), "asr": a >> amt,
                   "ror": z3.RotateRight(a, amt)}[mn]
        elif mn in ("mul", "mneg", "madd", "msub") and len(ops) >= 3:
            p = self._read(st, ops[1], bits) * self._src(st, ops[2], bits)
            if mn == "mneg":
                val = -p
            elif mn in ("madd", "msub") and len(ops) >= 4:
                acc = self._read(st, ops[3], bits)
                val = acc + p if mn == "madd" else acc - p
            else:
                val = p
        elif mn in ("udiv", "sdiv") and len(ops) >= 3:
            a, b = self._read(st, ops[1], bits), self._src(st, ops[2], bits)
            q = z3.UDiv(a, b) if mn == "udiv" else a / b
            val = z3.If(b == 0, z3.BitVecVal(0, bits), q)     # A64: x/0 == 0
        elif mn in ("umulh", "smulh") and len(ops) >= 3:
            ext = z3.ZeroExt if mn == "umulh" else z3.SignExt
            p = ext(64, self._read(st, ops[1], 64)) * ext(64, self._read(st, ops[2], 64))
            val = z3.Extract(127, 64, p)

        # ── extends / bitfields ───────────────────────────────────────────
        elif mn in ("uxtb", "uxth", "uxtw", "sxtb", "sxth", "sxtw") and len(ops) >= 2 and isinstance(ops[1], Reg):
            w = {"b": 8, "h": 16, "w": 32}[mn[-1]]
            raw = z3.Extract(w - 1, 0, self._read(st, ops[1], 64))
            val = (z3.SignExt if mn[0] == "s" else z3.ZeroExt)(bits - w, raw)
        elif mn in ("ubfx", "sbfx") and len(ops) >= 4 and isinstance(ops[1], Reg):
            lsb, width = insn.imm(2), insn.imm(3)
            if isinstance(lsb, int) and isinstance(width, int) and width > 0:
                raw = z3.Extract(lsb + width - 1, lsb, self._read(st, ops[1], bits))
                val = (z3.SignExt if mn == "sbfx" else z3.ZeroExt)(bits - width, raw) if width < bits else raw
        elif mn == "ubfiz" and len(ops) >= 4 and isinstance(ops[1], Reg):
            lsb, width = insn.imm(2), insn.imm(3)
            if isinstance(lsb, int) and isinstance(width, int) and width > 0:
                raw = z3.Extract(width - 1, 0, self._read(st, ops[1], bits))
                val = z3.ZeroExt(bits - width, raw) << lsb if width < bits else raw << lsb

        # ── conditional select ────────────────────────────────────────────
        elif mn in ("csel", "csinc", "csinv", "csneg") and len(ops) >= 4 and isinstance(ops[3], Cond):
            a, b = self._read(st, ops[1], bits), self._read(st, ops[2], bits)
            alt = {"csel": b, "csinc": b + 1, "csinv": ~b, "csneg": -b}[mn]
            val = z3.If(st.cond(ops[3].code), a, alt)
        elif mn in ("cset", "csetm") and len(ops) >= 2 and isinstance(ops[1], Cond):
            one = z3.BitVecVal(1 if mn == "cset" else _mask(bits), bits)
            val = z3.If(st.cond(ops[1].code), one, z3.BitVecVal(0, bits))
        elif mn in ("cinc", "cinv", "cneg") and len(ops) >= 3 and isinstance(ops[2], Cond):
            a = self._read(st, ops[1], bits)
            alt = {"cinc": a + 1, "cinv": ~a, "cneg": -a}[mn]
            val = z3.If(st.cond(ops[2].code), alt, a)

        if val is None:
            val = z3.BitVec(f"{rd.name}@{va:#x}", 64)
        self._write(st, rd, val)

    # ── branch constraint ─────────────────────────────────────────────────────

    def _branch_cond(self, st: _State, insn: Insn) -> Optional[Tuple[z3.BoolRef, Optional[int]]]:
        """(condition-for-taken, taken_target) for a conditional branch, else None."""
        mn, ops = insn.mnemonic, insn.ops
        if mn in ISA.cond_branch_mnems:
            return st.cond(mn[2:]), insn.imm(0)
        if mn in ("cbz", "cbnz") and ops and isinstance(ops[0], Reg):
            r = self._read(st, ops[0])
            c = r == 0
            return (c if mn == "cbz" else z3.Not(c)), insn.imm(1)
        if mn in ("tbz", "tbnz") and len(ops) >= 3 and isinstance(ops[0], Reg):
            bit = insn.imm(1)
            r = self._read(st, ops[0], 64)
            c = z3.Extract(bit, bit, r) == 0
            return (c if mn == "tbz" else z3.Not(c)), insn.imm(2)
        return None

    # ── symbolic execution of one path ────────────────────────────────────────

    def _solve_path(self, path: List[int], target_va: int) -> PathResult:
        st = _State()
        solver = z3.Solver()
        solver.set("timeout", self.z3_timeout_ms)
        self._solver = solver
        self._mem_cache = {}
        self._seeded_fields_used = set()
        clauses = 0
        call_idx = [0]

        for step, bva in enumerate(path):
            block = self.cfg.blocks.get(bva)
            if block is None:
                continue
            nxt = path[step + 1] if step + 1 < len(path) else None
            for insn in block.insns:
                if nxt is None and insn.address == target_va:
                    break
                if insn is block.insns[-1] and nxt is not None:
                    bc = self._branch_cond(st, insn)
                    if bc is not None:
                        cond, tgt = bc
                        fall = insn.address + insn.size
                        if tgt != fall:
                            if nxt == tgt:
                                solver.add(cond)
                                clauses += 1
                            elif nxt == fall:
                                solver.add(z3.Not(cond))
                                clauses += 1
                        continue
                self._exec(st, insn, call_idx)

        status = solver.check()
        sat = status == z3.sat
        note = "timeout" if status == z3.unknown else ("" if sat else "unsat")
        model: Dict[str, int] = {}
        if sat:
            m = solver.model()
            for r in _GPRS:
                v = m.eval(z3.BitVec(f"{r}_0", 64), model_completion=True)
                if z3.is_bv_value(v):
                    model[r] = v.as_long()
        return PathResult(sat=sat, path=path, model=model, clauses=clauses, note=note,
                          seeded_fields=sorted(self._seeded_fields_used),
                          fields=self._field_values(solver) if sat else {})
