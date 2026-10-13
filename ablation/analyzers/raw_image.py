"""
raw_image.py: helpers for raw (headerless) firmware images, e.g. RTOS flash dumps.

Used by BinaryContext.from_raw(). Everything here works on one flat region
[base, base + len(data)):

  cortexm_vectors(data, base)   Cortex-M vector table -> {name: handler VA}
  infer_cortexm_base(data)      load base from the vector table (when not given)
  parse_symbols(path)           nm / "addr name" / "name addr" / GNU ld map lines
  thumb_bl_targets(data, base)  (site, target) for every Thumb-2 BL in the image
  arm_bl_targets(data, base)    (site, target) for every A32 BL (cond AL)
  literal_pool_xrefs(...)       (site, pointer) for Thumb / A32 PC-relative LDRs
  movw_movt_pairs(...)          (site, value) for MOVW + MOVT of the same register
  string_at(data, base, va)     NUL-terminated printable run starting exactly at va
  nul_terminated_strings(...)   printable runs followed by a NUL byte

Thumb function addresses are even here (bit 0 stripped); the Thumb state is
recorded separately in BinaryContext.thumb_funcs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

_CM_EXCEPTIONS = {
    1: "Reset_Handler", 2: "NMI_Handler", 3: "HardFault_Handler", 4: "MemManage_Handler",
    5: "BusFault_Handler", 6: "UsageFault_Handler", 11: "SVC_Handler",
    12: "DebugMon_Handler", 14: "PendSV_Handler", 15: "SysTick_Handler",
}
_CM_RESERVED = {7, 8, 9, 10, 13}
_CM_MAX_VECTORS = 16 + 240


# ── Cortex-M vector table ────────────────────────────────────────────────────

def _vector_words(data: bytes) -> List[int]:
    n = min(len(data) // 4, _CM_MAX_VECTORS)
    return list(struct.unpack_from(f"<{n}I", data, 0)) if n else []


def cortexm_vectors(data: bytes, base: int) -> Dict[str, int]:
    """{handler_name: VA} from a vector table at the start of the image.

    Entry 0 is the initial SP. Handlers are Thumb (odd) addresses inside the image.
    The table ends at the first non-zero entry that is not such an address. A
    handler shared by several IRQs (the usual Default_Handler) is named once,
    as Default_Handler.
    """
    words = _vector_words(data)
    if len(words) < 2 or not (words[1] & 1):
        return {}
    end = base + len(data)
    seen: Dict[int, List[str]] = {}
    for i, w in enumerate(words[1:], start=1):
        if i in _CM_RESERVED or w == 0:
            continue
        if not (w & 1) or not (base <= (w & ~1) < end):
            if i >= 16:
                break                       # past the end of the table
            continue
        name = _CM_EXCEPTIONS.get(i, f"IRQ{i - 16}_Handler")
        seen.setdefault(w & ~1, []).append(name)
    out: Dict[str, int] = {}
    for va, names in seen.items():
        out[names[0] if len(names) == 1 or names[0] == "Reset_Handler" else "Default_Handler"] = va
    return out


def infer_cortexm_base(data: bytes) -> Optional[int]:
    """Load base of a Cortex-M image whose vector table is at offset 0.

    Candidates are the reset vector rounded down to 256 B .. 16 MB. The answer is
    the highest candidate that puts every handler inside the image and past the
    table (the VTOR must point at the table, and handlers follow it). None if the
    image does not start with a plausible vector table.
    """
    words = _vector_words(data)
    if len(words) < 2 or not (words[1] & 1):
        return None
    handlers = sorted({w & ~1 for i, w in enumerate(words[1:16], start=1)
                       if w and (w & 1) and i not in _CM_RESERVED})
    if not handlers:
        return None
    reset = words[1] & ~1
    best = None
    for k in range(8, 25):
        cand = reset & ~((1 << k) - 1)
        if all(cand + 16 * 4 <= h < cand + len(data) for h in handlers):
            best = cand if best is None else max(best, cand)
    return best


# ── symbols file ─────────────────────────────────────────────────────────────

SYMBOL_FORMATS = """\
one entry per line; '#' starts a comment; commas / tabs count as spaces:
  VA name                       0x08001234 uart_rx         (also GNU ld map lines)
  name VA | name = VA           uart_rx 0x08001234
  VA size name                  0x08001234 0x40 uart_rx
  nm                            08001234 T uart_rx         (code types t/T/w/W only)
  readelf -sW                   12: 08001234 64 FUNC GLOBAL DEFAULT 1 uart_rx
  VA file func line             0x08001234 src/uart.c uart_rx 112
                                (VA is any address inside the function, e.g. a
                                 harvested log-call site: the name is attached to
                                 the function containing it)
VA is hex, with or without 0x."""

_HEXTOK = re.compile(r"^(?:0x)?[0-9a-fA-F]{1,16}$")
_NAMETOK = re.compile(r"^[A-Za-z_.$][\w.$@:~<>]*$")
_READELF = re.compile(r"^\s*\d+:\s+([0-9a-fA-F]+)\s+(\S+)\s+(\w+)\s+\w+\s+\w+\s+\S+\s+(\S+)")


@dataclass
class SymbolEntry:
    name: str
    va: int
    inside: bool = False          # VA lies somewhere inside the function, not at its start


@dataclass
class SymbolFile:
    entries: List[SymbolEntry] = field(default_factory=list)
    lines: int = 0                # non-empty, non-comment lines
    skipped_data: int = 0         # nm / readelf data symbols
    unrecognised: List[str] = field(default_factory=list)
    dialects: Dict[str, int] = field(default_factory=dict)

    def _add(self, dialect: str, name: str, va: int, inside: bool = False) -> None:
        self.entries.append(SymbolEntry(name, va, inside))
        self.dialects[dialect] = self.dialects.get(dialect, 0) + 1


def _looks_va(tok: str) -> bool:
    """'0x...' or 8+ hex digits: an address, not a name like 'dead' or 'add'."""
    return _HEXTOK.match(tok) is not None and (tok.lower().startswith("0x") or len(tok) >= 8)


def _hex(tok: str) -> Optional[int]:
    return int(tok, 16) if _HEXTOK.match(tok) else None


def parse_symbol_file(path: str) -> SymbolFile:
    """Parse a symbols file in any of SYMBOL_FORMATS. Thumb bit 0 is kept."""
    out = SymbolFile()
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        out.lines += 1
        m = _READELF.match(line)
        if m:
            if m.group(3) in ("FUNC", "IFUNC"):
                out._add("readelf", m.group(4), int(m.group(1), 16))
            else:
                out.skipped_data += 1
            continue
        toks = line.replace(",", " ").replace("\t", " ").replace("=", " ").split()
        va0 = _hex(toks[0])
        if len(toks) == 3 and va0 is not None and len(toks[1]) == 1 and toks[1].isalpha() \
                and _NAMETOK.match(toks[2]):
            if toks[1] in "tTwW":
                out._add("nm", toks[2], va0)
            else:
                out.skipped_data += 1
        elif len(toks) == 2 and va0 is not None and _NAMETOK.match(toks[1]) \
                and not _looks_va(toks[1]):
            out._add("va name", toks[1], va0)
        elif len(toks) == 2 and _NAMETOK.match(toks[0]) and _looks_va(toks[1]):
            out._add("name va", toks[0], _hex(toks[1]))
        elif len(toks) == 3 and va0 is not None and toks[1].isdigit() | (_hex(toks[1]) is not None) \
                and _NAMETOK.match(toks[2]) and not _looks_va(toks[2]):
            out._add("va size name", toks[2], va0)
        elif len(toks) >= 4 and va0 is not None and toks[-1].isdigit() and _NAMETOK.match(toks[-2]):
            out._add("va file func line", toks[-2], va0, inside=True)
        elif len(out.unrecognised) < 5:
            out.unrecognised.append(raw.strip())
        else:
            out.unrecognised.append("")
    return out


def parse_symbols(path: str) -> Dict[str, int]:
    """{name: address} from a symbols file (see SYMBOL_FORMATS); first entry per name wins."""
    out: Dict[str, int] = {}
    for e in parse_symbol_file(path).entries:
        out.setdefault(e.name, e.va)
    return out


# ── code-region inference ────────────────────────────────────────────────────

_A64_RET = 0xD65F03C0


def infer_code_ranges(data: bytes, base: int, mode: str, window: int = 4096,
                      max_gap: int = 4) -> List[Tuple[int, int]]:
    """[(lo, hi)] VA ranges that hold code, from return / prologue density per window.

    arm64: a window is code when it holds >= 2 of RET / stp x29,x30,[sp,#-N]! /
    PACIASP / BTI c (an exact 32-bit match never occurs by chance in data).
    thumb / arm32: bx lr / pop {..pc} together with push {..lr}, >= 4 in total.
    Code windows up to max_gap windows apart are merged (literal pools, jump tables).
    """
    if not _NUMPY_OK or len(data) < window:
        return [(base, base + len(data))]
    nwin = (len(data) + window - 1) // window
    hits = np.zeros(nwin, dtype=np.int64)
    if mode == "arm64":
        n = len(data) // 4
        w = np.frombuffer(data[: n * 4], dtype="<u4")
        m = ((w == _A64_RET) | ((w & 0xFFC07FFF) == 0xA9807BFD)
             | (w == 0xD503233F) | (w == 0xD503245F))
        np.add.at(hits, (np.nonzero(m)[0] * 4) // window, 1)
        code = hits >= 2
    else:
        n = len(data) // 2
        h = np.frombuffer(data[: n * 2], dtype="<u2")
        if mode == "thumb":
            ret = (h == 0x4770) | ((h & 0xFF00) == 0xBD00)
            push = (h & 0xFF00) == 0xB500
        else:
            w = np.frombuffer(data[: (len(data) // 4) * 4], dtype="<u4")
            ret = np.zeros(n, dtype=bool)
            push = np.zeros(n, dtype=bool)
            ret[::2][: len(w)] = (w == 0xE12FFF1E) | ((w & 0xFFFF8000) == 0xE8BD8000)
            push[::2][: len(w)] = (w & 0xFFFF4000) == 0xE92D4000
        r_hits = np.zeros(nwin, dtype=np.int64)
        p_hits = np.zeros(nwin, dtype=np.int64)
        np.add.at(r_hits, (np.nonzero(ret)[0] * 2) // window, 1)
        np.add.at(p_hits, (np.nonzero(push)[0] * 2) // window, 1)
        code = (r_hits >= 1) & (p_hits >= 1) & (r_hits + p_hits >= 4)
    # last return-ish marker per window: a range ends right after it, not at the window edge
    # (exact-match markers only: a b / pop {pc} pattern occurs in random data)
    if mode == "arm64":
        rets = np.nonzero(w == _A64_RET)[0] * 4
    elif mode == "thumb":
        rets = np.nonzero(h == 0x4770)[0] * 2
    else:
        rets = np.nonzero(ret)[0] * 2
    ranges: List[Tuple[int, int]] = []
    idx = np.nonzero(code)[0].tolist()
    for i in idx:
        lo = base + i * window
        hi = min(base + (i + 1) * window, base + len(data))
        if ranges and lo - ranges[-1][1] <= max_gap * window:
            ranges[-1] = (ranges[-1][0], hi)
        else:
            ranges.append((lo, hi))
    trimmed = []
    for lo, hi in ranges:
        k = np.searchsorted(rets, hi - base) - 1
        if k >= 0 and base + int(rets[k]) >= lo:
            hi = min(hi, base + int(rets[k]) + (4 if mode != "thumb" else 2))
        trimmed.append((lo, hi))
    return trimmed or [(base, base + len(data))]


def parse_ranges(spec: str) -> List[Tuple[int, int]]:
    """'0x1000-0x8000,0x9000-0xa000' -> [(lo, hi)]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not sep:
            raise ValueError(f"code range {part!r}: expected LO-HI")
        out.append((int(lo, 16), int(hi, 16)))
    return out


# ── branch scans ─────────────────────────────────────────────────────────────

def thumb_bl_targets(data: bytes, base: int, branch: bool = False) -> List[Tuple[int, int]]:
    """(site_va, target_va) for every Thumb-2 BL whose target is inside the image.
    branch=True scans B.W (T4, hw2 = 10 J1 1 J2 imm11) instead: tail calls.

    BL T1: hw1 = 11110 S imm10, hw2 = 11 J1 1 J2 imm11;
    target = site + 4 + SignExtend(S:I1:I2:imm10:imm11:0), I = NOT(J XOR S).
    Data in literal pools can decode as BL; callers filter targets.
    """
    n = len(data) // 2
    if n < 2:
        return []
    end = base + len(data)
    want = 0x9000 if branch else 0xD000
    out: List[Tuple[int, int]] = []
    if _NUMPY_OK:
        h = np.frombuffer(data[: n * 2], dtype="<u2").astype(np.int64)
        h1, h2 = h[:-1], h[1:]
        idx = np.where(((h1 & 0xF800) == 0xF000) & ((h2 & 0xD000) == want))[0]
        h1, h2 = h1[idx], h2[idx]
        s = (h1 >> 10) & 1
        i1 = (~(((h2 >> 13) & 1) ^ s)) & 1
        i2 = (~(((h2 >> 11) & 1) ^ s)) & 1
        off = (s << 24) | (i1 << 23) | (i2 << 22) | ((h1 & 0x3FF) << 12) | ((h2 & 0x7FF) << 1)
        off = off - ((off & (1 << 24)) << 1)
        sites = base + idx * 2
        tgts = sites + 4 + off
        ok = (tgts >= base) & (tgts < end)
        return list(zip(sites[ok].tolist(), tgts[ok].tolist()))
    hw = struct.unpack_from(f"<{n}H", data, 0)
    for i in range(n - 1):
        a, b = hw[i], hw[i + 1]
        if (a & 0xF800) != 0xF000 or (b & 0xD000) != want:
            continue
        s = (a >> 10) & 1
        i1 = (~(((b >> 13) & 1) ^ s)) & 1
        i2 = (~(((b >> 11) & 1) ^ s)) & 1
        off = (s << 24) | (i1 << 23) | (i2 << 22) | ((a & 0x3FF) << 12) | ((b & 0x7FF) << 1)
        if s:
            off -= 1 << 25
        t = base + i * 2 + 4 + off
        if base <= t < end:
            out.append((base + i * 2, t))
    return out


def arm_bl_targets(data: bytes, base: int) -> List[Tuple[int, int]]:
    """(site_va, target_va) for every A32 `BL` (cond AL, 0xEB) inside the image."""
    n = len(data) // 4
    end = base + len(data)
    words = struct.unpack_from(f"<{n}I", data, 0)
    out = []
    for i, w in enumerate(words):
        if (w >> 24) != 0xEB:
            continue
        imm = w & 0xFFFFFF
        if imm & 0x800000:
            imm -= 1 << 24
        t = base + i * 4 + 8 + imm * 4
        if base <= t < end:
            out.append((base + i * 4, t))
    return out


# ── literal pools ────────────────────────────────────────────────────────────

def literal_pool_xrefs(data: bytes, base: int, thumb: bool) -> List[Tuple[int, int]]:
    """(site_va, pointer) for PC-relative literal loads of a 32-bit word.

    Thumb: LDR Rt, [PC, #imm8*4] (16-bit, 0x48xx..0x4Fxx) and LDR.W Rt, [PC, #+/-imm12]
           (0xF8DF / 0xF85F); PC = Align(site + 4, 4).
    A32:   LDR Rt, [PC, #+/-imm12] (0xE59F / 0xE51F); PC = site + 8.
    """
    end = base + len(data)

    def word_at(va: int) -> Optional[int]:
        off = va - base
        return struct.unpack_from("<I", data, off)[0] if 0 <= off <= len(data) - 4 else None

    out: List[Tuple[int, int]] = []
    if thumb:
        n = len(data) // 2
        hw = struct.unpack_from(f"<{n}H", data, 0)
        for i, a in enumerate(hw):
            site = base + i * 2
            pool = None
            if (a & 0xF800) == 0x4800:                              # LDR Rt, [PC, #imm8*4]
                pool = ((site + 4) & ~3) + (a & 0xFF) * 4
            elif a in (0xF8DF, 0xF85F) and i + 1 < n:               # LDR.W Rt, [PC, #+/-imm12]
                imm = hw[i + 1] & 0xFFF
                pool = ((site + 4) & ~3) + (imm if a == 0xF8DF else -imm)
            if pool is not None and base <= pool < end:
                w = word_at(pool)
                if w is not None:
                    out.append((site, w))
    else:
        n = len(data) // 4
        for i, w in enumerate(struct.unpack_from(f"<{n}I", data, 0)):
            if (w & 0xFFFF0000) in (0xE59F0000, 0xE51F0000):
                imm = w & 0xFFF
                site = base + i * 4
                pool = site + 8 + (imm if (w >> 23) & 1 else -imm)
                p = word_at(pool)
                if p is not None:
                    out.append((site, p))
    return out


def nul_terminated_strings(data: bytes, base: int, min_len: int = 4) -> Dict[int, str]:
    """{va: text} for printable ASCII runs of >= min_len that end in a NUL byte.

    In a raw image code and data share one region; requiring the terminator
    removes most of the instruction bytes that happen to be printable.
    """
    out: Dict[int, str] = {}
    for m in re.finditer(rb"[\x20-\x7e\t\r\n]{%d,}\x00" % min_len, data):
        out[base + m.start()] = m.group()[:-1].decode("ascii", errors="replace")
    return out


def movw_movt_pairs(data: bytes, base: int, thumb: bool, window: int = 8) -> List[Tuple[int, int]]:
    """(movt_site, value) for MOVW Rd, #lo16 followed by MOVT Rd, #hi16 within `window`
    instructions: how Cortex-M / v7 code without literal pools builds addresses.

    Thumb-2 T3: hw1 = 11110 i 10 x 100 imm4 (MOVW x=0: 0xF240, MOVT x=1: 0xF2C0),
                hw2 = 0 imm3 Rd imm8;  imm16 = imm4:i:imm3:imm8
    A32:        cond 0011 0x00 imm4 Rd imm12 (MOVW 0x03000000, MOVT 0x03400000)
    """
    out: List[Tuple[int, int]] = []
    if thumb:
        n = len(data) // 2
        hw = struct.unpack_from(f"<{n}H", data, 0)

        def dec(i: int):
            a, b = hw[i], hw[i + 1]
            kind = {0xF240: "w", 0xF2C0: "t"}.get(a & 0xFBF0)
            if kind is None or b & 0x8000:
                return None
            imm = ((a & 0xF) << 12) | (((a >> 10) & 1) << 11) | (((b >> 12) & 7) << 8) | (b & 0xFF)
            return kind, (b >> 8) & 0xF, imm

        for i in range(n - 1):
            d = dec(i)
            if d is None or d[0] != "w":
                continue
            for j in range(i + 2, min(i + 2 + 2 * window, n - 1)):
                e = dec(j)
                if e is not None and e[1] == d[1]:
                    if e[0] == "t":
                        out.append((base + j * 2, (e[2] << 16) | d[2]))
                    break
        return out
    n = len(data) // 4
    words = struct.unpack_from(f"<{n}I", data, 0)

    def dec_a(w: int):
        kind = {0x03000000: "w", 0x03400000: "t"}.get(w & 0x0FF00000)
        if kind is None:
            return None
        return kind, (w >> 12) & 0xF, ((w >> 4) & 0xF000) | (w & 0xFFF)

    for i, w in enumerate(words):
        d = dec_a(w)
        if d is None or d[0] != "w":
            continue
        for j in range(i + 1, min(i + 1 + window, n)):
            e = dec_a(words[j])
            if e is not None and e[1] == d[1]:
                if e[0] == "t":
                    out.append((base + j * 4, (e[2] << 16) | d[2]))
                break
    return out


def string_at(data: bytes, base: int, va: int, min_len: int = 4) -> Optional[str]:
    """The NUL-terminated printable run that starts exactly at va, if any."""
    off = va - base
    if not 0 <= off < len(data):
        return None
    m = re.match(rb"[\x20-\x7e\t\r\n]{%d,}\x00" % min_len, data[off: off + 4096])
    return m.group()[:-1].decode("ascii", errors="replace") if m else None

