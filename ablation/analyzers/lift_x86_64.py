"""
lift_x86_64.py: x86-64 -> the backend-neutral AArch64 Insn model (insn_arm64.Insn).

Why this exists
  1. Every vuln-hunting pass built for AArch64 (taint, conversion lengths, width-less
     %s extents, format slot maps, sink-wrapper inference, reachability, egress, heap /
     overflow tags, path-sensitive call annotation) runs on insn_arm64.Insn. x86-64 PE
     (and ELF) had only the older SysV-only taint tracker and none of those passes.
  2. Re-implementing each pass for x86 would fork every fix. Lifting x86-64 into the
     same model keeps one implementation: a fix to a pass lands on both architectures.
  3. The lift is semantic, not cosmetic: register roles follow the calling convention
     (the first argument lands in x0, the return value comes back in x0), the return
     address is a real stack slot (so a stack overflow reaching it is visible to the
     same frame / extent logic as a saved x30), and IAT-indirect calls become direct
     calls to the import slot, so `call [rip+__imp_recv]` is `bl recv`.

Register map
  Win64:  rcx x0  rdx x1  r8 x2  r9 x3 | rax x9  r10 x10  r11 x11 | rbx x19  rsi x20
          rdi x21  r12..r15 x22..x25  rbp x29  rsp sp
  SysV:   rdi x0  rsi x1  rdx x2  rcx x3  r8 x4  r9 x5 | rax x9  r10 x10  r11 x11 |
          rbx x19  r12..r15 x22..x25  rbp x29  rsp sp
  xmm0..15 -> v0..v15. x16 / x17 are lift temporaries (address / loaded value).

Model
  * Function entry: `str x30, [sp, #-8]!` (size 0) stands for the return address the
    call pushed, so entry sp is the caller's sp at the call and the saved x30 slot is
    the real return-address slot. `ret` reloads it (`ldr x30, [sp], #8`) after copying
    rax into x0.
  * After every call: `mov x9, x0` (rax = the callee's result).
  * Flags: cmp / test become cmp / tst (`test r, r` is `cmp r, #0`, so je is a zero test
    and jle / js are signed lower bounds). add / sub / and set flags only when the next
    instruction reads them; other ALU ops add `cmp rd, #0` in that case.
  * jcc -> b.cond, cmovcc -> csel, setcc -> cset; movsxd / movsx / movzx -> sxtw / sxt* /
    uxt* (or ldrs* / ldr{b,h} from memory); lea -> add / sub / adrp+add.
  * Memory operands: [base + index*scale + disp] becomes `add x16, base, index, lsl #s`
    and [x16, #disp]; rip-relative is adrp x16 + [x16, #lo] (what the AArch64 passes
    already resolve for globals and strings); gs: / fs: is the TLS base (mrs tpidr_el0).
  * `mov eax, N ; call __chkstk ; sub rsp, rax` becomes `sub sp, sp, #N`.
  * Micro-ops of one x86 instruction share its address and size; reported VAs are real
    instruction addresses (byte-verifiable).
  * Unmodelled instructions become `x86.<mnemonic> rd, ...` so the engine clobbers the
    destination with the union of the source registers' taint (never silently clean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

import capstone
from capstone import x86_const as X

from .insn_arm64 import Insn, make_insn


# ── ABIs ──────────────────────────────────────────────────────────────────────

_COMMON = {
    "rax": "x9", "r10": "x10", "r11": "x11", "rbx": "x19",
    "r12": "x22", "r13": "x23", "r14": "x24", "r15": "x25", "rbp": "x29", "rsp": "sp",
}
REGMAP: Dict[str, Dict[str, str]] = {
    "win64": dict(_COMMON, rcx="x0", rdx="x1", r8="x2", r9="x3", rsi="x20", rdi="x21"),
    "sysv":  dict(_COMMON, rdi="x0", rsi="x1", rdx="x2", rcx="x3", r8="x4", r9="x5"),
}

_FAMILY: Dict[str, str] = {}
for _b, _al in [
    ("rax", "eax ax al ah"), ("rbx", "ebx bx bl bh"), ("rcx", "ecx cx cl ch"),
    ("rdx", "edx dx dl dh"), ("rsi", "esi si sil"), ("rdi", "edi di dil"),
    ("rsp", "esp sp spl"), ("rbp", "ebp bp bpl"),
] + [(f"r{n}", f"r{n}d r{n}w r{n}b") for n in range(8, 16)]:
    _FAMILY[_b] = _b
    for _a in _al.split():
        _FAMILY[_a] = _b

_JCC = {
    "e": "eq", "z": "eq", "ne": "ne", "nz": "ne", "g": "gt", "nle": "gt", "ge": "ge", "nl": "ge",
    "l": "lt", "nge": "lt", "le": "le", "ng": "le", "a": "hi", "nbe": "hi", "ae": "hs", "nb": "hs",
    "nc": "hs", "b": "lo", "c": "lo", "nae": "lo", "be": "ls", "na": "ls", "s": "mi", "ns": "pl",
    "o": "vs", "no": "vc", "p": "vs", "pe": "vs", "np": "vc", "po": "vc",
}
_FLAG_READERS = ("j", "cmov", "set", "adc", "sbb")
_NOPS = frozenset({"nop", "int3", "endbr64", "endbr32", "pause", "lfence", "mfence", "sfence",
                   "prefetcht0", "prefetcht1", "prefetcht2", "prefetchnta", "prefetchw", "fnop",
                   "cld", "std", "clc", "stc", "cmc", "wait", "fwait", "xabort", "xend"})
_VEC_MOVES = frozenset({"movups", "movaps", "movdqu", "movdqa", "movupd", "movapd", "lddqu",
                        "vmovups", "vmovaps", "vmovdqu", "vmovdqa", "movntdq", "movntps", "vmovntdq"})


@dataclass(frozen=True)
class X64ABI:
    name: str
    arg_regs: Tuple[str, ...]          # lifted names, in argument order
    stack_arg_base: int                # entry-relative offset of the first stack argument
    regmap: Dict[str, str]


ABIS = {
    "win64": X64ABI("win64", ("x0", "x1", "x2", "x3"), 0x20, REGMAP["win64"]),
    "sysv":  X64ABI("sysv", ("x0", "x1", "x2", "x3", "x4", "x5"), 0, REGMAP["sysv"]),
}


class X64Lifter:
    """Lift x86-64 machine code into insn_arm64.Insn (see the module docstring)."""

    def __init__(self, abi: str = "win64", iat: Iterable[int] = ()):
        self.abi = ABIS[abi]
        self.iat: FrozenSet[int] = frozenset(iat)
        self._md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self._md.detail = True
        self._md.skipdata = False

    # ---- public -------------------------------------------------------------

    def lift(self, code: bytes, base: int, entry: bool = True) -> List[Insn]:
        """Insns for `code` at `base`. entry: base is a function entry (adds the
        return-address push)."""
        raw = list(self._md.disasm(code, base))
        out: List[Insn] = []
        if entry and raw:
            out.append(make_insn(base, "str", "x30, [sp, #-8]!", size=0))
        self._rax_imm: Optional[int] = None
        self._rax_age = 99
        for k, ci in enumerate(raw):
            nxt = raw[k + 1] if k + 1 < len(raw) else None
            self._next_reads_flags = nxt is not None and nxt.mnemonic.startswith(_FLAG_READERS)
            self._ops: List[Tuple[str, str]] = []
            try:
                self._one(ci)
            except Exception:                    # never lose the instruction
                self._ops = [self._unknown_ops(ci)]
            for m, o in self._ops:
                out.append(make_insn(ci.address, m, o, size=ci.size))
            self._track_rax(ci)
        return out

    # ---- helpers -------------------------------------------------------------

    def _e(self, m: str, ops: str = "") -> None:
        self._ops.append((m, ops))

    def _reg(self, cs_name: str, width: Optional[int] = None) -> str:
        """Lifted register text for an x86 register, sized by `width` (bytes)."""
        n = cs_name.lower()
        if n.startswith(("xmm", "ymm", "zmm")):
            return f"q{int(n[3:]) % 32}"
        fam = _FAMILY.get(n)
        if fam is None:
            raise ValueError(f"register {n}")
        x = self.abi.regmap[fam]
        if x == "sp":
            return "sp"
        w = width if width is not None else (8 if n == fam else 4)
        return x if w == 8 else "w" + x[1:]

    def _rn(self, ci, reg_id: int, width: Optional[int] = None) -> str:
        return self._reg(ci.reg_name(reg_id), width)

    @staticmethod
    def _w(text: str, width: int) -> str:
        """Re-size a lifted GPR name."""
        if text in ("sp", "xzr", "wzr") or text[0] in "qdsv":
            return text
        return ("x" if width == 8 else "w") + text[1:]

    @staticmethod
    def _imm(v: int) -> str:
        return f"#{v:#x}" if v >= 0 else f"#-{-v:#x}"

    def _mem(self, ci, op) -> str:
        """Emit address micro-ops for an x86 memory operand; return the arm64 [..] text."""
        m = op.mem
        disp = m.disp
        seg = ci.reg_name(m.segment) if m.segment else ""
        if m.base == X.X86_REG_RIP:
            a = (ci.address + ci.size + disp) & 0xFFFFFFFFFFFFFFFF
            self._e("adrp", f"x16, #{a & ~0xFFF:#x}")
            return f"[x16, #{a & 0xFFF:#x}]" if a & 0xFFF else "[x16]"
        base = self._rn(ci, m.base, 8) if m.base else None
        if seg in ("gs", "fs"):
            self._e("mrs", "x16, tpidr_el0")
            if base:
                self._e("add", f"x16, x16, {base}")
            if m.index:
                self._e("add", f"x16, x16, {self._rn(ci, m.index, 8)}, lsl #{m.scale.bit_length() - 1}")
            base = "x16"
        elif m.index:
            idx = self._rn(ci, m.index, 8)
            sh = m.scale.bit_length() - 1
            if base:
                self._e("add", f"x16, {base}, {idx}" + (f", lsl #{sh}" if sh else ""))
            elif sh:
                self._e("lsl", f"x16, {idx}, #{sh}")
            else:
                self._e("mov", f"x16, {idx}")
            base = "x16"
        elif base is None:
            self._e("mov", f"x16, {self._imm(disp)}")
            return "[x16]"
        if disp:
            return f"[{base}, {self._imm(disp)}]"
        return f"[{base}]"

    def _load(self, ci, op, dst: str, width: int, signed: bool = False) -> None:
        """dst (lifted name, already sized to the destination) <- memory operand."""
        addr = self._mem(ci, op)
        if dst[0] in "q":
            n = dst[1:]
            self._e("ldr", f"{({16: 'q', 8: 'd', 4: 's'}).get(op.size, 'q')}{n}, {addr}")
            return
        sz = op.size
        if sz >= 8:
            self._e("ldr", f"{self._w(dst, 8)}, {addr}")
        elif sz == 4:
            if signed:
                self._e("ldrsw", f"{self._w(dst, 8)}, {addr}")
            else:
                self._e("ldr", f"{self._w(dst, 4)}, {addr}")
        else:
            m = {1: "b", 2: "h"}.get(sz, "b")
            self._e(f"ldrs{m}" if signed else f"ldr{m}",
                    f"{self._w(dst, 8 if signed and width == 8 else 4)}, {addr}")

    def _store(self, ci, op, src: str) -> None:
        addr = self._mem(ci, op)
        if src[0] == "q":
            n = src[1:]
            self._e("str", f"{({16: 'q', 8: 'd', 4: 's'}).get(op.size, 'q')}{n}, {addr}")
            return
        sz = op.size
        if sz >= 8:
            self._e("str", f"{self._w(src, 8)}, {addr}")
        elif sz == 4:
            self._e("str", f"{self._w(src, 4)}, {addr}")
        else:
            self._e("strh" if sz == 2 else "strb", f"{self._w(src, 4)}, {addr}")

    def _src(self, ci, op, width: int, tmp: str = "x17") -> str:
        """A register / immediate text for a source operand (memory loaded into tmp)."""
        if op.type == X.X86_OP_REG:
            return self._rn(ci, op.reg, width)
        if op.type == X.X86_OP_IMM:
            return self._imm(op.imm)
        t = self._w(tmp, width)
        self._load(ci, op, t, width)
        return t

    def _src_reg(self, ci, op, width: int, tmp: str = "x17") -> str:
        """Like _src, but an immediate is materialised into tmp (for mul / csel ...)."""
        if op.type == X.X86_OP_IMM:
            t = self._w(tmp, width)
            self._e("mov", f"{t}, {self._imm(op.imm)}")
            return t
        return self._src(ci, op, width, tmp)

    def _unknown_ops(self, ci) -> Tuple[str, str]:
        regs = []
        try:
            for op in ci.operands:
                if op.type == X.X86_OP_REG:
                    regs.append(self._rn(ci, op.reg))
        except Exception:
            regs = []
        return f"x86.{ci.mnemonic.replace(' ', '_')}", ", ".join(regs)

    def _track_rax(self, ci) -> None:
        if ci.mnemonic == "mov" and len(ci.operands) == 2 and ci.operands[0].type == X.X86_OP_REG \
                and _FAMILY.get(ci.reg_name(ci.operands[0].reg)) == "rax" \
                and ci.operands[1].type == X.X86_OP_IMM:
            self._rax_imm, self._rax_age = ci.operands[1].imm, 0
            return
        self._rax_age += 1
        if self._rax_age > 2:
            self._rax_imm = None

    # ---- the lift ------------------------------------------------------------

    def _one(self, ci) -> None:
        m = ci.mnemonic
        ops = ci.operands
        if m.startswith(("rep ", "repne ", "repe ")) or m.startswith(("movs", "stos", "lods", "scas", "cmps")) \
                and all(o.type == X.X86_OP_MEM for o in ops) and ops:
            self._string_op(ci)
            return
        if m.startswith("notrack "):
            m = m.split(" ", 1)[1]
        if m.startswith(("lock ", "xacquire ", "xrelease ")):
            m = m.split(" ", 1)[1]
        if m in _NOPS:
            self._e("nop")
            return
        if m in ("ud2", "int", "int1", "into", "hlt"):
            self._e("brk", "#0")
            return
        if m == "syscall":
            self._e("svc", "#0")
            return
        if m in ("mov", "movabs"):
            return self._mov(ci, ops)
        if m in ("movzx", "movsx", "movsxd"):
            return self._ext(ci, ops, m != "movzx")
        if m == "lea":
            return self._lea(ci, ops)
        if m == "push":
            src = self._src_reg(ci, ops[0], 8)
            self._e("str", f"{src}, [sp, #-8]!")
            return
        if m == "pop":
            if ops[0].type == X.X86_OP_REG:
                self._e("ldr", f"{self._rn(ci, ops[0].reg, 8)}, [sp], #8")
            else:
                self._e("ldr", "x17, [sp], #8")
                self._store(ci, ops[0], "x17")
            return
        if m == "leave":
            self._e("mov", "sp, x29")
            self._e("ldr", "x29, [sp], #8")
            return
        if m in ("ret", "retf", "retfq"):
            self._e("mov", "x0, x9")
            self._e("ldr", "x30, [sp], #8")
            if ops and ops[0].type == X.X86_OP_IMM and ops[0].imm:
                self._e("add", f"sp, sp, #{ops[0].imm:#x}")
            self._e("ret")
            return
        if m == "call":
            return self._call(ci, ops)
        if m == "jmp":
            return self._jmp(ci, ops)
        if m in ("jrcxz", "jecxz"):
            self._e("cbz", f"{self._reg('rcx', 8)}, #{ops[0].imm:#x}")
            return
        if m.startswith("j") and m[1:] in _JCC:
            self._e(f"b.{_JCC[m[1:]]}", f"#{ops[0].imm:#x}")
            return
        if m.startswith("cmov") and m[4:] in _JCC:
            w = ops[0].size
            d = self._rn(ci, ops[0].reg, w)
            s = self._src_reg(ci, ops[1], w)
            self._e("csel", f"{d}, {s}, {d}, {_JCC[m[4:]]}")
            return
        if m.startswith("set") and m[3:] in _JCC:
            if ops[0].type == X.X86_OP_REG:
                self._e("cset", f"{self._rn(ci, ops[0].reg, 4)}, {_JCC[m[3:]]}")
            else:
                self._e("cset", f"w17, {_JCC[m[3:]]}")
                self._store(ci, ops[0], "w17")
            return
        if m in ("cmp", "test"):
            return self._cmp(ci, ops, m == "test")
        if m in ("add", "sub", "and", "or", "xor", "adc", "sbb"):
            return self._alu2(ci, ops, m)
        if m in ("inc", "dec"):
            return self._alu1(ci, ops, "add" if m == "inc" else "sub", imm=1)
        if m in ("neg", "not"):
            return self._alu1(ci, ops, "neg" if m == "neg" else "mvn")
        if m in ("shl", "sal", "shr", "sar", "rol", "ror"):
            return self._shift(ci, ops, {"shl": "lsl", "sal": "lsl", "shr": "lsr", "sar": "asr",
                                          "rol": "ror", "ror": "ror"}[m])
        if m == "imul":
            return self._imul(ci, ops)
        if m == "mul":
            return self._mul1(ci, ops, signed=False)
        if m in ("div", "idiv"):
            return self._div(ci, ops, m == "idiv")
        if m in ("cdqe", "cwde", "cbw"):
            w = 8 if m == "cdqe" else 4
            self._e({"cdqe": "sxtw", "cwde": "sxth", "cbw": "sxtb"}[m],
                    f"{self._reg('rax', w)}, {self._reg('rax', 4)}")
            return
        if m in ("cdq", "cqo", "cwd"):
            w = 8 if m == "cqo" else 4
            self._e("asr", f"{self._reg('rdx', w)}, {self._reg('rax', w)}, #{63 if w == 8 else 31}")
            return
        if m == "xchg":
            return self._xchg(ci, ops)
        if m in ("bswap",):
            d = self._rn(ci, ops[0].reg, ops[0].size)
            self._e("rev", f"{d}, {d}")
            return
        if m in ("bsf", "bsr", "tzcnt", "lzcnt", "popcnt"):
            w = ops[0].size
            d = self._rn(ci, ops[0].reg, w)
            self._e("clz", f"{d}, {self._src_reg(ci, ops[1], w)}")
            return
        if m == "bt":
            w = ops[0].size
            a = self._src_reg(ci, ops[0], w, "x16")
            self._e("tst", f"{a}, {self._src_reg(ci, ops[1], w)}")
            return
        if m in ("xadd", "cmpxchg"):
            return self._alu2(ci, ops, "add" if m == "xadd" else "or")
        if m in _VEC_MOVES or m in ("movq", "movd", "movsd", "movss", "movlps", "movhps", "movlpd", "movhpd"):
            return self._vmov(ci, ops, m)
        if m in ("pxor", "xorps", "xorpd", "vpxor", "vxorps") and len(ops) >= 2 \
                and all(o.type == X.X86_OP_REG for o in ops) and len({o.reg for o in ops}) == 1:
            self._e("movi", f"v{int(ci.reg_name(ops[0].reg)[3:]) % 32}.2d, #0")
            return
        self._ops.append(self._unknown_ops(ci))

    # ---- instruction classes --------------------------------------------------

    def _mov(self, ci, ops) -> None:
        d, s = ops
        w = d.size
        if d.type == X.X86_OP_REG:
            dn = self._rn(ci, d.reg, w)
            if s.type == X.X86_OP_MEM:
                self._load(ci, s, dn, w)
            elif s.type == X.X86_OP_IMM:
                self._e("mov", f"{dn}, {self._imm(s.imm & 0xFFFFFFFF if w == 4 else s.imm)}")
            else:
                sn = self._rn(ci, s.reg, w)
                if w < 4:                               # 8/16-bit move: merge, approximated
                    self._e("bfi", f"{self._w(dn, 4)}, {self._w(sn, 4)}, #0, #{8 * w}")
                else:
                    self._e("mov", f"{dn}, {sn}")
            return
        # store
        if s.type == X.X86_OP_IMM:
            if s.imm == 0:
                self._store(ci, d, "xzr" if w == 8 else "wzr")
                return
            t = "x17" if w == 8 else "w17"
            self._e("mov", f"{t}, {self._imm(s.imm)}")
            self._store(ci, d, t)
            return
        self._store(ci, d, self._rn(ci, s.reg, w if w >= 4 else 4))

    def _ext(self, ci, ops, signed: bool) -> None:
        d, s = ops
        w = d.size
        dn = self._rn(ci, d.reg, w)
        if s.type == X.X86_OP_MEM:
            self._load(ci, s, dn, w, signed=signed)
            return
        sn = self._rn(ci, s.reg, 4)
        k = {1: "b", 2: "h", 4: "w"}[s.size]
        if not signed and k == "w":
            self._e("mov", f"{self._w(dn, 4)}, {sn}")
        elif signed:
            self._e(f"sxt{k}", f"{dn}, {sn}")
        else:
            self._e(f"uxt{k}", f"{self._w(dn, 4)}, {sn}")

    def _lea(self, ci, ops) -> None:
        d, s = ops
        w = d.size
        dn = self._rn(ci, d.reg, w)
        m = s.mem
        if m.base == X.X86_REG_RIP:
            a = (ci.address + ci.size + m.disp) & 0xFFFFFFFFFFFFFFFF
            dx = self._w(dn, 8)
            self._e("adrp", f"{dx}, #{a & ~0xFFF:#x}")
            if a & 0xFFF:
                self._e("add", f"{dx}, {dx}, #{a & 0xFFF:#x}")
            return
        base = self._rn(ci, m.base, w) if m.base else None
        if base == "sp" or (base and base.endswith("sp")):
            base = "sp"
        disp = m.disp
        if m.index:
            idx = self._rn(ci, m.index, w)
            sh = m.scale.bit_length() - 1
            if base:
                self._e("add", f"{dn}, {base}, {idx}" + (f", lsl #{sh}" if sh else ""))
            elif sh:
                self._e("lsl", f"{dn}, {idx}, #{sh}")
            else:
                self._e("mov", f"{dn}, {idx}")
            if disp:
                self._e("add" if disp > 0 else "sub", f"{dn}, {dn}, #{abs(disp):#x}")
            return
        if base is None:
            self._e("mov", f"{dn}, {self._imm(disp)}")
            return
        if disp:
            self._e("add" if disp > 0 else "sub", f"{dn}, {base}, #{abs(disp):#x}")
        else:
            self._e("mov", f"{dn}, {base}")

    def _call(self, ci, ops) -> None:
        op = ops[0]
        if op.type == X.X86_OP_IMM:
            self._e("bl", f"#{op.imm:#x}")
        elif op.type == X.X86_OP_MEM and op.mem.base == X.X86_REG_RIP and not op.mem.index:
            slot = (ci.address + ci.size + op.mem.disp) & 0xFFFFFFFFFFFFFFFF
            if slot in self.iat:
                self._e("bl", f"#{slot:#x}")
            else:
                self._load(ci, op, "x16", 8)
                self._e("blr", "x16")
        elif op.type == X.X86_OP_REG:
            self._e("blr", self._rn(ci, op.reg, 8))
        else:
            self._load(ci, op, "x16", 8)
            self._e("blr", "x16")
        self._e("mov", "x9, x0")

    def _jmp(self, ci, ops) -> None:
        op = ops[0]
        if op.type == X.X86_OP_IMM:
            self._e("b", f"#{op.imm:#x}")
        elif op.type == X.X86_OP_MEM and op.mem.base == X.X86_REG_RIP and not op.mem.index:
            slot = (ci.address + ci.size + op.mem.disp) & 0xFFFFFFFFFFFFFFFF
            if slot in self.iat:
                self._e("b", f"#{slot:#x}")
            else:
                self._load(ci, op, "x16", 8)
                self._e("br", "x16")
        elif op.type == X.X86_OP_REG:
            self._e("br", self._rn(ci, op.reg, 8))
        else:
            self._load(ci, op, "x16", 8)
            self._e("br", "x16")

    def _cmp(self, ci, ops, test: bool) -> None:
        a, b = ops
        w = a.size
        if test and a.type == X.X86_OP_REG and b.type == X.X86_OP_REG and a.reg == b.reg:
            self._e("cmp", f"{self._rn(ci, a.reg, w)}, #0")
            return
        an = self._src_reg(ci, a, w, "x16")
        bn = self._src(ci, b, w, "x17")
        self._e("tst" if test else "cmp", f"{an}, {bn}")

    def _flags_after(self, d: str, m: str) -> None:
        if self._next_reads_flags and m not in ("adds", "subs", "ands"):
            self._e("cmp", f"{d}, #0")

    def _alu2(self, ci, ops, m: str) -> None:
        d, s = ops
        w = d.size if d.size >= 4 else 4
        if m in ("xor", "sub", "sbb") and d.type == X.X86_OP_REG and s.type == X.X86_OP_REG and d.reg == s.reg:
            dn = self._rn(ci, d.reg, w)
            if m == "sbb":
                self._e("csetm", f"{dn}, lo")            # sbb r, r: 0 or -1 from the carry
            else:
                self._e("mov", f"{dn}, #0")
            return
        if d.type == X.X86_OP_REG and self._rn(ci, d.reg, 8) == "sp":
            if s.type == X.X86_OP_IMM:
                self._e(m if m in ("add", "sub", "and") else "add", f"sp, sp, {self._imm(s.imm)}")
            elif m == "sub" and self._rax_imm is not None and _FAMILY.get(ci.reg_name(s.reg)) == "rax":
                self._e("sub", f"sp, sp, #{self._rax_imm:#x}")          # __chkstk frame
            else:
                self._e(m if m in ("add", "sub") else "add", f"sp, sp, {self._src_reg(ci, s, 8)}")
            return
        am = {"add": "add", "sub": "sub", "and": "and", "or": "orr", "xor": "eor",
              "adc": "adc", "sbb": "sbc"}[m]
        if self._next_reads_flags and am in ("add", "sub", "and"):
            am += "s"
        if s.type == X.X86_OP_IMM and s.imm < 0 and am in ("add", "sub", "adds", "subs"):
            am = {"add": "sub", "sub": "add", "adds": "subs", "subs": "adds"}[am]   # add r, -4 = sub r, 4
            s_imm = -s.imm
        else:
            s_imm = None
        if d.type == X.X86_OP_REG:
            dn = self._rn(ci, d.reg, w)
            sn = (f"#{s_imm:#x}" if s_imm is not None else
                  self._src(ci, s, w) if am not in ("adc", "sbc") else self._src_reg(ci, s, w))
            self._e(am, f"{dn}, {dn}, {sn}")
            self._flags_after(dn, am)
            return
        # memory destination (x86 has no mem, mem form): load, op, store
        sn = f"#{s_imm:#x}" if s_imm is not None else self._src(ci, s, w)
        tv = self._w("x17", w)
        self._load(ci, d, tv, w)
        self._e(am, f"{tv}, {tv}, {sn}")
        self._flags_after(tv, am)
        self._store(ci, d, tv)

    def _alu1(self, ci, ops, am: str, imm: Optional[int] = None) -> None:
        d = ops[0]
        w = d.size if d.size >= 4 else 4
        if d.type == X.X86_OP_REG:
            dn = self._rn(ci, d.reg, w)
        else:
            dn = self._w("x17", w)
            self._load(ci, d, dn, w)
        if imm is not None:
            mm = am + "s" if self._next_reads_flags else am
            self._e(mm, f"{dn}, {dn}, #{imm}")
        else:
            mm = "negs" if am == "neg" and self._next_reads_flags else am
            self._e(mm, f"{dn}, {dn}")
            if mm == "mvn":
                self._flags_after(dn, mm)
        if d.type != X.X86_OP_REG:
            self._store(ci, d, dn)

    def _shift(self, ci, ops, am: str) -> None:
        d = ops[0]
        w = d.size if d.size >= 4 else 4
        if d.type == X.X86_OP_REG:
            dn = self._rn(ci, d.reg, w)
        else:
            dn = self._w("x17", w)
            self._load(ci, d, dn, w)
        if len(ops) == 1:
            amt = "#1"
        elif ops[1].type == X.X86_OP_IMM:
            amt = f"#{ops[1].imm & (63 if w == 8 else 31)}"
        else:
            amt = self._rn(ci, ops[1].reg, w)
        self._e(am, f"{dn}, {dn}, {amt}")
        self._flags_after(dn, am)
        if d.type != X.X86_OP_REG:
            self._store(ci, d, dn)

    def _imul(self, ci, ops) -> None:
        if len(ops) == 1:
            return self._mul1(ci, ops, signed=True)
        d = ops[0]
        w = d.size if d.size >= 4 else 4
        dn = self._rn(ci, d.reg, w)
        if len(ops) == 2:
            s = self._src_reg(ci, ops[1], w)
            self._e("mul", f"{dn}, {dn}, {s}")
        else:
            a = self._src_reg(ci, ops[1], w, "x16")
            self._e("mov", f"{self._w('x17', w)}, {self._imm(ops[2].imm)}")
            self._e("mul", f"{dn}, {a}, {self._w('x17', w)}")
        self._flags_after(dn, "mul")

    def _mul1(self, ci, ops, signed: bool) -> None:
        w = ops[0].size if ops[0].size >= 4 else 4
        s = self._src_reg(ci, ops[0], w)
        rax, rdx = self._reg("rax", w), self._reg("rdx", w)
        if w == 8:
            self._e("smulh" if signed else "umulh", f"{rdx}, {rax}, {s}")
        else:
            self._e("mul", f"{rdx}, {rax}, {s}")
        self._e("mul", f"{rax}, {rax}, {s}")

    def _div(self, ci, ops, signed: bool) -> None:
        w = ops[0].size if ops[0].size >= 4 else 4
        s = self._src_reg(ci, ops[0], w)
        rax, rdx = self._reg("rax", w), self._reg("rdx", w)
        dv = "sdiv" if signed else "udiv"
        self._e(dv, f"{rdx}, {rax}, {s}")              # remainder: tainted like the quotient
        self._e(dv, f"{rax}, {rax}, {s}")

    def _xchg(self, ci, ops) -> None:
        a, b = ops
        w = a.size if a.size >= 4 else 4
        if a.type == X.X86_OP_REG and b.type == X.X86_OP_REG:
            if a.reg == b.reg:
                self._e("nop")
                return
            an, bn = self._rn(ci, a.reg, w), self._rn(ci, b.reg, w)
            self._e("mov", f"{self._w('x17', w)}, {an}")
            self._e("mov", f"{an}, {bn}")
            self._e("mov", f"{bn}, {self._w('x17', w)}")
            return
        mem, reg = (a, b) if a.type == X.X86_OP_MEM else (b, a)
        rn = self._rn(ci, reg.reg, w)
        self._load(ci, mem, self._w("x17", w), w)
        self._store(ci, mem, rn)
        self._e("mov", f"{rn}, {self._w('x17', w)}")

    def _vmov(self, ci, ops, m: str) -> None:
        d, s = ops[0], ops[-1]
        def vreg(op):
            return f"v{int(ci.reg_name(op.reg)[-2:].lstrip('m')) % 32}"
        dvec = d.type == X.X86_OP_REG and ci.reg_name(d.reg).startswith(("xmm", "ymm"))
        svec = s.type == X.X86_OP_REG and ci.reg_name(s.reg).startswith(("xmm", "ymm"))
        if d.type == X.X86_OP_REG and s.type == X.X86_OP_MEM:
            if dvec:
                self._load(ci, s, "q" + vreg(d)[1:], 16)
            else:
                self._load(ci, s, self._rn(ci, d.reg, d.size), d.size)
            return
        if d.type == X.X86_OP_MEM and s.type == X.X86_OP_REG:
            self._store(ci, d, ("q" + vreg(s)[1:]) if svec else self._rn(ci, s.reg, s.size))
            return
        if dvec and svec:
            self._e("mov", f"{vreg(d)}.16b, {vreg(s)}.16b")
            return
        if dvec and s.type == X.X86_OP_REG:
            self._e("fmov", f"d{vreg(d)[1:]}, {self._rn(ci, s.reg, 8)}")
            return
        if svec and d.type == X.X86_OP_REG:
            self._e("fmov", f"{self._rn(ci, d.reg, 8)}, d{vreg(s)[1:]}")
            return
        self._ops.append(self._unknown_ops(ci))

    def _string_op(self, ci) -> None:
        """rep movs / stos and friends: a copy from [rsi] to [rdi], count rcx. Modelled as
        a load of the source's first qword and a store to the destination, which carries
        the source buffer's taint onto the destination buffer."""
        m = ci.mnemonic.split()[-1]
        di, si = self._reg("rdi", 8), self._reg("rsi", 8)
        if m.startswith("movs"):
            self._e("ldr", f"x17, [{si}]")
            self._e("str", f"x17, [{di}]")
        elif m.startswith("stos"):
            self._e("str", f"{self._reg('rax', 8)}, [{di}]")
        else:
            self._e(f"x86.{m}", "")


def lifter_for(ctx) -> Optional[Callable[[bytes, int], List[Insn]]]:
    """decode(code, va) for a BinaryContext's architecture: the x86-64 lifter (Win64 for
    PE, SysV otherwise) or None for native AArch64."""
    if getattr(ctx, "arch", None) != "x86_64":
        return None
    abi = "win64" if is_pe(ctx) else "sysv"
    lf = X64Lifter(abi, iat=getattr(ctx, "plt", {}).keys())
    return lambda code, va, entry=True: lf.lift(code, va, entry)


def is_pe(ctx) -> bool:
    try:
        with open(ctx.path, "rb") as fh:
            return fh.read(2) == b"MZ"
    except OSError:
        return False


def abi_of(ctx) -> X64ABI:
    """The lifted calling convention of a context (AAPCS64 as an X64ABI-shaped record for
    native AArch64)."""
    if getattr(ctx, "arch", None) == "x86_64":
        return ABIS["win64" if is_pe(ctx) else "sysv"]
    return AAPCS64


AAPCS64 = X64ABI("aapcs64", tuple(f"x{i}" for i in range(8)), 0, {})


# ── report rendering ──────────────────────────────────────────────────────────

_X86_32 = {"rax": "eax", "rbx": "ebx", "rcx": "ecx", "rdx": "edx", "rsi": "esi", "rdi": "edi",
           "rbp": "ebp", "rsp": "esp"}
_GPR_TOKEN = None


def x86_names(text: str, abi: X64ABI) -> str:
    """Render a report written in lifted (AArch64) register names with the x86-64 names
    they stand for: x0 -> rcx (Win64), w9 -> eax, saved x30 -> saved return address."""
    import re
    global _GPR_TOKEN
    if _GPR_TOKEN is None:
        _GPR_TOKEN = re.compile(r"(?<![\w#.$])([xw])(\d{1,2})\b")
    inv = {a: x for x, a in abi.regmap.items() if a != "sp"}
    text = text.replace("saved x30", "saved return address").replace("ARM64 ", "x86-64 ")
    text = text.replace("saved x29", "saved rbp").replace("W-register wrap", "32-bit wrap")

    def one(m):
        r = inv.get(f"x{m.group(2)}")
        if m.group(2) == "30":
            return "retaddr"
        if r is None:
            return m.group(0)
        if m.group(1) == "x":
            return r
        return _X86_32.get(r, r + "d")
    return _GPR_TOKEN.sub(one, text)
