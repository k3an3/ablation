"""
reachability_arm64.py: is a finding's code reachable, how is its handler registered, and
is it gated on configuration that this build never sets? (AArch64)

Why this exists
  1. Findings sat in a handler that is never registered and never called: no direct
     caller, its address in no relocation, data word, table or materialised constant.
     Dead code, reported like live code.
  2. The opposite error costs more: a route believed dead because its registration was
     not found suppressed re-testing of a real bug for months. So registrations
     (a handler address passed to a setter together with a string key, or a
     {key, handler} table entry) are resolved and shown.
  3. Error paths gated on a configuration slot the program never sets are dead in this
     build only. They must be labelled, never silently dropped.

What it computes
  address-taken   function starts appearing as an 8-byte word anywhere in the image
                  (relocations applied, so .data.rel.ro / init_array / GOT count) or
                  materialised in code by ADR / ADRP+ADD
  live            closure over call edges from exports, the ELF entry point and every
                  address-taken function
  registrations   handler address + string literal arguments of one call
                  (setter(key, handler)), or a data word -> handler next to a word -> string
  gate            the nearest conditional branch every path to the site must take,
                  testing a global or a field of a struct a global points at, and whether
                  any store in the binary writes it

Verdicts (classify): None (live, ungated), ("unreachable: ...", suppressed=True),
("gated: ... no store in this binary", suppressed=True), ("gated: ... written at ...",
suppressed=False). Suppressed findings are counted, with reasons, and listed with
--show-suppressed: never dropped silently.
"""

from __future__ import annotations

import bisect
import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    import numpy as np
    _NP = True
except ImportError:                                  # pragma: no cover
    _NP = False

from .insn_arm64 import from_capstone

_MAX_FUNC = 8192
_PSEUDO = 1 << 62            # address stand-in for an imported global (GOT slot | _PSEUDO)


def _adrp_page(w: int, pc: int) -> int:
    imm = ((w >> 29) & 3) | (((w >> 5) & 0x7FFFF) << 2)
    if imm & (1 << 20):
        imm -= 1 << 21
    return ((pc & ~0xFFF) + (imm << 12)) & 0xFFFFFFFFFFFFFFFF


def _adr_target(w: int, pc: int) -> int:
    imm = ((w >> 29) & 3) | (((w >> 5) & 0x7FFFF) << 2)
    if imm & (1 << 20):
        imm -= 1 << 21
    return (pc + imm) & 0xFFFFFFFFFFFFFFFF


@dataclass
class Registration:
    handler: int
    key: str
    how: str          # "setter <name> at 0x..." / "table at 0x..."


class Reachability:
    def __init__(self, ctx):
        self._ctx = ctx
        self._regions = ctx.image_regions()
        self._starts = list(ctx.func_starts)
        self._start_set = set(self._starts)
        self._taken: Optional[Set[int]] = None
        self._materialised: Dict[int, List[int]] = {}     # func -> code sites materialising it
        self._live: Optional[Set[int]] = None
        self._regs: Optional[List[Registration]] = None
        self._adrp_cache = None
        self._got: Dict[int, int] = {}                     # GOT slot -> address it holds
        self._got_sym: Dict[int, str] = {}                 # GOT slot -> imported symbol
        self._pseudo_names: Dict[int, str] = {}
        if not getattr(ctx, "raw", None):
            try:
                import lief
                from .binary_context import elf_relocations
                rel = elf_relocations(lief.parse(ctx.path))
                self._got, self._got_sym = dict(rel.values), rel.symbols
                # an imported global (defined in another module): a pseudo address
                for slot, name in rel.symbols.items():
                    self._got.setdefault(slot, _PSEUDO | slot)
                    self._pseudo_names[_PSEUDO | slot] = name
            except Exception:
                pass

    # ---- address-taken / live ----------------------------------------------

    def _code_regions(self) -> List[Tuple[int, bytes]]:
        ctx = self._ctx
        if getattr(ctx, "raw", None) and ctx.raw.get("code_ranges"):
            data = dict(self._regions)[ctx.base_va]
            return [(lo, data[lo - ctx.base_va:hi - ctx.base_va]) for lo, hi in ctx.raw["code_ranges"]]
        lo = self._starts[0] if self._starts else 0
        return [(va, d) for va, d in self._regions if va <= lo < va + len(d)] or self._regions

    def address_taken(self) -> Set[int]:
        if self._taken is not None:
            return self._taken
        taken: Set[int] = set()
        starts = self._starts
        if _NP and starts:
            sarr = np.array(starts, dtype=np.uint64)
            for va, data in self._regions:
                for align in range(0, 8, 8):
                    n = (len(data) - align) // 8
                    if n <= 0:
                        continue
                    w = np.frombuffer(data[align:align + n * 8], dtype="<u8")
                    hit = np.isin(w, sarr)
                    taken.update(int(x) for x in w[hit])
        for cva, code in self._code_regions():
            for site, target in self._materialisations(cva, code):
                if target in self._start_set:
                    taken.add(target)
                    self._materialised.setdefault(target, []).append(site)
        self._taken = taken
        return taken

    def _materialisations(self, cva: int, code: bytes):
        """(site, value) for ADR and ADRP + ADD (same register, within 8 words)."""
        n = len(code) // 4
        if n == 0:
            return
        if _NP:
            w = np.frombuffer(code[: n * 4], dtype="<u4")
            adr = np.nonzero((w & 0x9F000000) == 0x10000000)[0].tolist()
            adrp = np.nonzero((w & 0x9F000000) == 0x90000000)[0].tolist()
            words = w
        else:                                                    # pragma: no cover
            words = struct.unpack_from(f"<{n}I", code)
            adr = [i for i, x in enumerate(words) if (x & 0x9F000000) == 0x10000000]
            adrp = [i for i, x in enumerate(words) if (x & 0x9F000000) == 0x90000000]
        for i in adr:
            yield cva + 4 * i, _adr_target(int(words[i]), cva + 4 * i)
        for i in adrp:
            a = int(words[i])
            rd = a & 31
            page = _adrp_page(a, cva + 4 * i)
            for j in range(i + 1, min(i + 9, n)):
                x = int(words[j])
                if (x & 0xFF800000) == 0x91000000 and ((x >> 5) & 31) == rd:
                    imm = (x >> 10) & 0xFFF
                    if x & (1 << 22):
                        imm <<= 12
                    yield cva + 4 * j, page + imm
                    if (x & 31) == rd:
                        break
                elif (x & 31) == rd and (x & 0x9F000000) == 0x90000000:
                    break

    def roots(self) -> Set[int]:
        """Entry point, dynamically exported functions (ELF; .symtab names are not
        entries), every named function of a raw image (vector table / symbols file)."""
        ctx = self._ctx
        r: Set[int] = set()
        if getattr(ctx, "raw", None):
            r = set(ctx.exports.values()) & self._start_set
        else:
            try:
                import lief
                b = lief.parse(ctx.path)
                for sym in b.exported_functions:
                    if int(sym.address) in self._start_set:
                        r.add(int(sym.address))
            except Exception:
                r = set(ctx.exports.values()) & self._start_set
        entry = self._entry()
        if entry is not None:
            r.add(entry)
        if getattr(ctx, "raw", None) and self._starts:
            r.add(ctx.base_va if ctx.base_va in self._start_set else self._starts[0])
        return r

    def _entry(self) -> Optional[int]:
        if getattr(self._ctx, "raw", None):
            return None
        try:
            import lief
            b = lief.parse(self._ctx.path)
            e = int(b.header.entrypoint) if b is not None else None
            return e if e in self._start_set else None
        except Exception:
            return None

    def live(self) -> Set[int]:
        if self._live is not None:
            return self._live
        succ: Dict[int, Set[int]] = defaultdict(set)
        for a, b, _l in self._ctx.call_edges:
            succ[a].add(b)
        seen = set(self.roots()) | self.address_taken()
        q = deque(seen)
        while q:
            f = q.popleft()
            for t in succ.get(f, ()):
                if t not in seen:
                    seen.add(t)
                    q.append(t)
        self._live = seen
        return seen

    def function_status(self, fva: int) -> Optional[str]:
        if fva in self.live():
            return None
        callers = sorted({a for a, b, _l in self._ctx.call_edges if b == fva})
        if not callers:
            return "unreachable: no callers, address never taken"
        names = ", ".join(self._ctx.name(c) for c in callers[:3])
        return f"unreachable: only called from unreachable code ({names})"

    # ---- registrations -------------------------------------------------------

    def registrations(self) -> List[Registration]:
        if self._regs is not None:
            return self._regs
        from .arm64_reg_annotator import ARM64RegAnnotator
        ctx = self._ctx
        self.address_taken()
        out: List[Registration] = []
        ra = ARM64RegAnnotator.from_context(ctx)
        by_func: Dict[int, Set[int]] = defaultdict(set)
        for handler, sites in self._materialised.items():
            for s in sites:
                f = ctx.func_containing(s)
                if f is not None:
                    by_func[f].add(handler)
        for f, handlers in by_func.items():
            i = bisect.bisect_right(self._starts, f)
            end = self._starts[i] if i < len(self._starts) and self._starts[i] - f <= _MAX_FUNC else f + _MAX_FUNC
            for c in ra.annotate_calls(f, end).calls:
                vals = list(c.args.values())
                hs = [v.int_val for v in vals if v is not None and v.kind == "const" and v.int_val in handlers]
                keys = [v.str_val for v in vals if v is not None and v.kind == "string"]
                for h in hs:
                    for k in keys[:1]:
                        out.append(Registration(h, k, f"setter {c.target_name or hex(c.target_va)} "
                                                      f"at 0x{c.site_va:x}"))
        # {key, handler} tables in data
        strings = ctx.strings
        for va, data in self._regions:
            n = len(data) // 8
            if not n or not _NP:
                continue
            w = np.frombuffer(data[: n * 8], dtype="<u8")
            idx = np.nonzero(np.isin(w, np.array(sorted(self._start_set), dtype=np.uint64)))[0].tolist()
            for i in idx:
                h = int(w[i])
                for j in (i - 1, i + 1):
                    if 0 <= j < n and int(w[j]) in strings:
                        out.append(Registration(h, strings[int(w[j])], f"table at 0x{va + 8 * i:x}"))
                        break
        self._regs = out
        return out

    def registered_as(self, fva: int) -> List[Registration]:
        return [r for r in self.registrations() if r.handler == fva]

    # ---- config gates ----------------------------------------------------------

    def _func_insns(self, fva: int):
        i = bisect.bisect_right(self._starts, fva)
        end = self._starts[i] if i < len(self._starts) and self._starts[i] - fva <= _MAX_FUNC else fva + _MAX_FUNC
        for va, data in self._regions:
            if 0 <= fva - va < len(data):
                return list(from_capstone(data[fva - va:end - va], fva))
        return []

    def gate(self, fva: int, site: int) -> Optional[Tuple[str, bool]]:
        """(label, never_set) for the nearest branch that gates `site` on a global /
        config field, or None."""
        from .cfg_arm64 import build_cfg
        insns = self._func_insns(fva)
        if not insns:
            return None
        try:
            cfg = build_cfg(insns, entry=fva)
        except (ValueError, IndexError):
            return None
        sblock = next((b for b in cfg.blocks.values() if b.start <= site < b.end), None)
        if sblock is None:
            return None

        def reaches(skip: Tuple[int, int]) -> bool:
            seen, q = {cfg.entry}, deque([cfg.entry])
            while q:
                b = q.popleft()
                if b == sblock.start:
                    return True
                for s in cfg.blocks[b].succs:
                    if (b, s) != skip and s in cfg.blocks and s not in seen:
                        seen.add(s)
                        q.append(s)
            return False

        by_addr = {x.address: k for k, x in enumerate(insns)}
        gates = []
        for b in cfg.blocks.values():
            if len(b.succs) != 2 or b.start == sblock.start:
                continue
            for s in b.succs:
                if not reaches((b.start, s)):
                    gates.append((b, s))        # every path to the site takes b -> s
        for b, need in sorted(gates, key=lambda g: -g[0].start):
            br = b.insns[-1]
            cond = self._condition(insns, by_addr, br, need)
            if cond is None:
                continue
            reg, wants_nonzero = cond
            loc = self._def_location(insns, by_addr[br.address], reg)
            if loc is None:
                continue
            gaddr, field = loc
            gname = self._pseudo_names.get(gaddr) or self._ctx.name(gaddr)
            gname = gname if gname and not gname.startswith("0x") else f"0x{gaddr:x}"
            where = f"[{gname}]" + (f"->+{field:#x}" if field is not None else "")
            what = "non-zero / non-NULL" if wants_nonzero else "zero / NULL"
            writers = self._writers(gaddr, field)
            if writers:
                ws = ", ".join(f"0x{w:x}" for w in writers[:3])
                return (f"gated: reached only when {where} is {what} (branch at 0x{br.address:x}); "
                        f"written at {ws}", False)
            dead = wants_nonzero and not gaddr & _PSEUDO      # an import may be set elsewhere
            if wants_nonzero and gaddr & _PSEUDO:
                tail = ("an imported global never written here: its own module may set it "
                        "(not suppressed)")
            elif wants_nonzero:
                tail = ("never written in this binary: dead unless set from outside "
                        "(configuration, another process)")
            else:
                tail = "never written in this binary (stays zero: live)"
            return (f"gated: reached only when {where} is {what} (branch at 0x{br.address:x}); {tail}", dead)
        return None

    @staticmethod
    def _condition(insns, by_addr, br, need: int) -> Optional[Tuple[str, bool]]:
        """(tested register, site needs it non-zero) for cbz / cbnz / b.eq|ne after cmp #0."""
        m = br.mnemonic
        taken = br.imm(len(br.ops) - 1)
        goes_taken = need == taken
        if m in ("cbz", "cbnz"):
            nonzero_taken = m == "cbnz"
            return br.reg(0).name, (nonzero_taken if goes_taken else not nonzero_taken)
        if m in ("b.eq", "b.ne"):
            k = by_addr[br.address]
            for j in range(k - 1, max(k - 6, -1), -1):
                x = insns[j]
                if x.mnemonic in ("cmp", "tst") and x.imm(1) in (0, None) and x.reg(0) is not None:
                    if x.mnemonic == "cmp" and x.imm(1) != 0:
                        return None
                    nonzero_taken = m == "b.ne"
                    return x.reg(0).name, (nonzero_taken if goes_taken else not nonzero_taken)
                if x.mnemonic.startswith(("b", "cb", "tb")):
                    break
        return None

    def _def_location(self, insns, k: int, reg: str) -> Optional[Tuple[int, Optional[int]]]:
        """(global address, field offset or None) the tested register was loaded from.
        Follows the load chain back to an ADRP page: ldr reg, [G]; ldr reg, [*G + off];
        either through a GOT slot (adrp; ldr xP, [GOT]; ldr reg, [xP ...])."""
        def defn(r: str, upto: int) -> Optional[int]:
            for j in range(upto - 1, max(upto - 48, -1), -1):
                rd = insns[j].reg(0)
                if rd is not None and rd.name == r and not insns[j].mnemonic.startswith(("st", "cmp", "tst", "cb", "tb", "b")):
                    return j
            return None

        def address(r: str, upto: int, depth: int = 0) -> Optional[Tuple[str, int, Optional[int]]]:
            """('addr', A) when r holds address A; ('val', A, None) when r holds *A."""
            j = defn(r, upto)
            if j is None or depth > 4:
                return None
            x = insns[j]
            m = x.mnemonic
            if m == "adrp":
                return ("addr", x.imm(1), None)
            if m == "add" and x.reg(1) is not None and x.imm(2) is not None:
                inner = address(x.reg(1).name, j, depth + 1)
                if inner and inner[0] == "addr":
                    return ("addr", inner[1] + x.imm(2), None)
                return None
            if m.startswith("ldr") and x.first_mem() is not None and x.first_mem().index is None:
                mem = x.first_mem()
                inner = address(mem.base, j, depth + 1)
                if inner is None:
                    return None
                if inner[0] == "addr":
                    a = inner[1] + mem.offset
                    if a in self._got:                       # GOT slot: r = &global
                        return ("addr", self._got[a], None)
                    return ("val", a, None)                  # r = *global
                if inner[0] == "val":                        # r = (*global)->field
                    return ("field", inner[1], mem.offset)
            return None

        loc = address(reg, k + 1) if False else None
        j = defn(reg, k)
        if j is None:
            return None
        x = insns[j]
        mem = x.first_mem()
        if not x.mnemonic.startswith("ldr") or mem is None or mem.index is None and False:
            return None
        inner = address(mem.base, j)
        if inner is None:
            return None
        if inner[0] == "addr":
            a = inner[1] + mem.offset
            if a in self._got:
                return None                                  # tested a pointer to a global
            return a, None                                   # the global itself
        if inner[0] == "val":
            return inner[1], mem.offset                      # field of the object *global
        return None

    def _adrp_index(self):
        if self._adrp_cache is None:
            self._adrp_cache = []
            for cva, code in self._code_regions():
                n = len(code) // 4
                if not n or not _NP:
                    continue
                w = np.frombuffer(code[: n * 4], dtype="<u4")
                for i in np.nonzero((w & 0x9F000000) == 0x90000000)[0].tolist():
                    self._adrp_cache.append((cva, w, n, i, _adrp_page(int(w[i]), cva + 4 * i)))
        return self._adrp_cache

    def _writers(self, gaddr: int, field: Optional[int]) -> List[int]:
        """Code sites storing to the global at gaddr (field None) or to field `field` of
        the object the global points at; direct or through the global's GOT slot."""
        slots = [sl for sl, v in self._got.items() if v == gaddr]
        pages = {sl & ~0xFFF for sl in slots}
        if not gaddr & _PSEUDO:
            pages.add(gaddr & ~0xFFF)
        out: List[int] = []
        for cva, w, n, i, pg in self._adrp_index():
            if pg not in pages:
                continue
            addr: Dict[int, int] = {int(w[i]) & 31: pg}     # register -> absolute address
            ptr: Set[int] = set()                           # registers holding *global
            for j in range(i + 1, min(i + 24, n)):
                x = int(w[j])
                rn, rt = (x >> 5) & 31, x & 31
                if (x & 0x3B000000) == 0x39000000:          # ldr / str, unsigned imm
                    size = x >> 30
                    is_load = bool(x & 0x00400000)
                    imm = ((x >> 10) & 0xFFF) << size
                    at = addr[rn] + imm if rn in addr else None
                    if not is_load:
                        if field is None and at == gaddr:
                            out.append(cva + 4 * j)
                        elif field is not None and rn in ptr and imm == field:
                            out.append(cva + 4 * j)
                    else:
                        addr.pop(rt, None)
                        ptr.discard(rt)
                        if at is not None and at in self._got and self._got[at] == gaddr and size == 3:
                            addr[rt] = gaddr                # &global from its GOT slot
                        elif at == gaddr and size == 3 and field is not None:
                            ptr.add(rt)
                elif (x & 0xFF800000) == 0x91000000 and rn in addr:      # add xD, xN, #imm
                    imm = (x >> 10) & 0xFFF
                    addr[rt] = addr[rn] + (imm << 12 if x & (1 << 22) else imm)
                elif (x & 0x9F000000) == 0x90000000:
                    addr.pop(rt, None)
                if (x & 0xFC000000) in (0x94000000, 0x14000000) or x == 0xD65F03C0:
                    break
        return sorted(set(out))

    # ---- one finding ----------------------------------------------------------

    def classify(self, fva: int, site: int) -> Tuple[Optional[str], bool, Optional[str]]:
        """(reach label, suppressed, registration note) for a finding at site in fva."""
        reg = self.registered_as(fva)
        note = None
        if reg:
            note = "; ".join(f"registered as '{r.key}' via {r.how}" for r in reg[:3])
        st = self.function_status(fva)
        if st is not None and not reg:
            return st, True, note
        g = self.gate(fva, site)
        if g is not None:
            return g[0], g[1], note
        return None, False, note


def _reg_index(name: str) -> Optional[int]:
    return int(name[1:]) if name.startswith("x") and name[1:].isdigit() else None
