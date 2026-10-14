"""
egress_arm64.py: does a sink's *output* reach an attacker-visible channel? (AArch64)

Why this exists
  1. A format-string leak only matters if the rendered text can be read back. In the
     motivating case it was written to a log file that a different handler in the same
     binary served to the client verbatim. Each half had been reviewed alone and
     dismissed: the write looked like local logging, the route like static file serving.
     A leak sink and a file-serving route on the same path are one bug.
  2. String-constant identity gets most of the way: the writer and the reader reference
     the same .rodata path string.

What it computes
  writers   path -> functions that open it for writing (fopen "w"/"a"/"+", open with
            O_WRONLY / O_RDWR / O_CREAT / O_APPEND, creat), and functions that write
            through a FILE* / fd global that such an open stored into
  readers   path -> functions that open it for reading
  emitters  functions that send to a peer (send / sendto / sendmsg / sendfile /
            SSL_write / writev / write ... and --egress names), with call depth
  couple()  for a finding in function F (and the wrapper chain it calls): the paths
            written within two call levels, and for each a reader whose own code or
            callees (two levels) emit:  F -> write "/var/log/x" in log_line (0x..) ->
            read in serve_file (0x..) -> sendfile (0x..)
  Limits: paths built at runtime (snprintf'd names) and output shared through global
  buffers rather than files are not followed.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .insn_arm64 import from_capstone

_MAX_FUNC = 8192
OPEN_FUNCS: Dict[str, Tuple[str, Optional[str], str]] = {
    "fopen": ("x0", "x1", "mode"), "fopen64": ("x0", "x1", "mode"), "freopen": ("x0", "x1", "mode"),
    "open": ("x0", "x1", "flags"), "open64": ("x0", "x1", "flags"), "__open_2": ("x0", "x1", "flags"),
    "__open64_2": ("x0", "x1", "flags"), "openat": ("x1", "x2", "flags"), "creat": ("x0", None, "write"),
}
STREAM_OUT: Dict[str, str] = {
    "fprintf": "x0", "vfprintf": "x0", "__fprintf_chk": "x0", "__vfprintf_chk": "x0",
    "fputs": "x1", "fputc": "x1", "putc": "x1", "fwrite": "x3", "fflush": "x0",
    "write": "x0", "dprintf": "x0", "vdprintf": "x0", "pwrite": "x0",
}
EGRESS = frozenset({"send", "sendto", "sendmsg", "sendfile", "sendfile64", "SSL_write", "writev",
                    "BIO_write", "mbedtls_ssl_write", "wolfSSL_write", "write", "splice"})
_O_WRITE = 0x1 | 0x2 | 0x40 | 0x400            # O_WRONLY | O_RDWR | O_CREAT | O_APPEND


@dataclass
class OpenSite:
    func: int
    site: int
    path: str
    write: bool
    callee: str


class EgressMap:
    def __init__(self, ctx, extra_egress: Optional[Set[str]] = None):
        from .reachability_arm64 import Reachability
        self._ctx = ctx
        self._egress = EGRESS | set(extra_egress or ())
        self._R = Reachability(ctx)
        self._starts = ctx.func_starts
        self._regions = ctx.image_regions()
        self._built = False
        self.opens: List[OpenSite] = []
        self.writers: Dict[str, Set[int]] = defaultdict(set)
        self.readers: Dict[str, Set[int]] = defaultdict(set)
        self._emits: Dict[int, List[Tuple[int, str]]] = defaultdict(list)    # func -> [(site, name)]
        self._callees: Dict[int, Set[int]] = defaultdict(set)
        for a, b, _l in ctx.call_edges:
            self._callees[a].add(b)

    # ---- build ---------------------------------------------------------------

    def _end(self, fva: int) -> int:
        i = bisect.bisect_right(self._starts, fva)
        return self._starts[i] if i < len(self._starts) and self._starts[i] - fva <= _MAX_FUNC else fva + _MAX_FUNC

    def _insns(self, fva: int):
        end = self._end(fva)
        for va, data in self._regions:
            if 0 <= fva - va < len(data):
                return list(from_capstone(data[fva - va:end - va], fva))
        return []

    def build(self) -> "EgressMap":
        if self._built:
            return self
        from .arm64_reg_annotator import ARM64RegAnnotator
        ctx = self._ctx
        names = set(OPEN_FUNCS) | set(STREAM_OUT) | self._egress
        interesting = [a for a, _b, lbl in ctx.call_edges if lbl in names]
        name_of = ctx.call_names()                       # edges without a label: by callee VA
        for a, b, _l in ctx.call_edges:
            if name_of.get(b) in names:
                interesting.append(a)
        interesting = sorted(set(interesting))
        ra = ARM64RegAnnotator.from_context(ctx)
        file_globals: Dict[int, str] = {}                 # global holding a FILE* / fd -> path
        stream_uses: List[Tuple[int, int, int]] = []      # (func, site, global) of writes via a global
        for fva in interesting:
            calls = ra.annotate_calls(fva, self._end(fva)).calls
            insns = None
            for c in calls:
                name = c.target_name or name_of.get(c.target_va, "")
                if name in OPEN_FUNCS:
                    preg, mreg, kind = OPEN_FUNCS[name]
                    path = self._text(c.args.get(preg))
                    if not path or not path.startswith("/"):
                        continue
                    write = self._is_write(kind, c.args.get(mreg) if mreg else None)
                    self.opens.append(OpenSite(fva, c.site_va, path, write, name))
                    (self.writers if write else self.readers)[path].add(fva)
                    if write:
                        insns = insns or self._insns(fva)
                        g = self._stored_global(insns, c.site_va)
                        if g is not None:
                            file_globals[g] = path
                elif name in STREAM_OUT:
                    insns = insns or self._insns(fva)
                    k = next((i for i, x in enumerate(insns) if x.address == c.site_va), None)
                    if k is not None:
                        ch = self._R.addr_chain(insns, k, STREAM_OUT[name])
                        if ch is not None and ch[0] == "val":
                            stream_uses.append((fva, c.site_va, ch[1]))
                if name in self._egress:
                    self._emits[fva].append((c.site_va, name))
        for fva, site, g in stream_uses:
            if g in file_globals:
                self.writers[file_globals[g]].add(fva)
        self._built = True
        return self

    def _text(self, v) -> Optional[str]:
        """String value of v; constants are read from the image (short literals such as
        the mode "a" are below the string table's minimum length)."""
        while v is not None and v.kind == "copy":
            v = v.chain
        if v is None:
            return None
        if v.kind == "string":
            return v.str_val
        if v.kind != "const":
            return None
        for base, data in self._regions:
            off = v.int_val - base
            if 0 <= off < len(data):
                end = data.find(b"\0", off, off + 256)
                raw = data[off:end] if end > off else b""
                return raw.decode() if raw and all(32 <= ch < 127 for ch in raw) else None
        return None

    def _is_write(self, kind: str, v) -> bool:
        if kind == "write":
            return True
        if v is None:
            return False
        if kind == "mode":
            mode = self._text(v)
            return bool(mode) and any(ch in mode for ch in "wa+")
        if v.kind == "const":
            return bool(v.int_val & _O_WRITE)
        return False

    def _stored_global(self, insns, call_site: int) -> Optional[int]:
        """Global a fopen / open result (x0, or a copy of it) is stored into right after."""
        k = next((i for i, x in enumerate(insns) if x.address == call_site), None)
        if k is None:
            return None
        holders = {"x0"}
        for j in range(k + 1, min(k + 12, len(insns))):
            x = insns[j]
            m = x.mnemonic
            if m == "mov" and x.reg(1) is not None and x.reg(1).name in holders and x.reg(0) is not None:
                holders.add(x.reg(0).name)
            elif m in ("str", "stur") and x.reg(0) is not None and x.reg(0).name in holders:
                mem = x.first_mem()
                ch = self._R.addr_chain(insns, j, mem.base) if mem is not None else None
                if ch is not None and ch[0] == "addr":
                    return ch[1] + mem.offset
            elif m in ("bl", "blr", "ret", "b"):
                break
        return None

    # ---- queries ---------------------------------------------------------------

    def _near(self, fvas, depth: int = 2) -> Dict[int, int]:
        out = {f: 0 for f in fvas}
        frontier = list(fvas)
        for d in range(1, depth + 1):
            nxt = []
            for f in frontier:
                for c in self._callees.get(f, ()):
                    if c not in out:
                        out[c] = d
                        nxt.append(c)
            frontier = nxt
        return out

    def _emitter(self, fva: int) -> Optional[Tuple[int, int, str]]:
        for f in self._near([fva]):
            for site, name in self._emits.get(f, ()):
                return f, site, name
        return None

    def couple(self, fvas) -> List[str]:
        """Egress chains for output produced in any of fvas (a finding's function plus the
        wrapper chain it calls)."""
        self.build()
        name = self._ctx.name
        near = self._near(list(fvas))
        chains: List[str] = []
        for path, ws in self.writers.items():
            hit = [w for w in ws if w in near]
            if not hit:
                continue
            for rd in sorted(self.readers.get(path, ())):
                em = self._emitter(rd)
                if em is None:
                    continue
                ef, esite, ename = em
                via = f" in {name(ef)}" if ef != rd else ""
                chains.append(f"write in {name(hit[0])} (0x{hit[0]:x}) -> \"{path}\" -> read in "
                              f"{name(rd)} (0x{rd:x}) -> {ename} (0x{esite:x}){via}")
        return chains

    def report(self) -> str:
        self.build()
        name = self._ctx.name
        lines = [f"file paths: {len(set(self.writers) | set(self.readers))} opened by path literal"]
        for path in sorted(set(self.writers) | set(self.readers)):
            ws = ", ".join(name(w) for w in sorted(self.writers.get(path, ())))
            rs = []
            for r in sorted(self.readers.get(path, ())):
                em = self._emitter(r)
                rs.append(f"{name(r)}" + (f" -> {em[2]} (0x{em[1]:x})" if em else ""))
            flag = "  EGRESS" if ws and any(self._emitter(r) for r in self.readers.get(path, ())) else ""
            lines.append(f"  {path!r:<36} written by [{ws}]  read by [{', '.join(rs)}]{flag}")
        return "\n".join(lines)
