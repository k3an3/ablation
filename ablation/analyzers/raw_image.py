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

_HEX = r"(?:0x)?([0-9a-fA-F]{4,16})"
_NM_RE = re.compile(rf"^\s*{_HEX}\s+([A-Za-z])\s+([A-Za-z_.$][\w.$@]*)\s*$")    # nm
_ADDR_NAME_RE = re.compile(rf"^\s*{_HEX}\s+([A-Za-z_.$][\w.$@]*)\s*$")            # addr name / ld map
_NAME_ADDR_RE = re.compile(rf"^\s*([A-Za-z_.$][\w.$@]*)\s*[=:,\s]\s*{_HEX}\s*$")  # name addr / name = addr


def parse_symbols(path: str) -> Dict[str, int]:
    """{name: address} from an nm listing, "addr name" / "name addr" lines, or a GNU
    ld map file (its "0xADDR  symbol" lines). Thumb bit 0 is kept; callers strip it.
    nm lines keep code symbols only (t/T/w/W): a data symbol is not a function start."""
    out: Dict[str, int] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.split("#", 1)[0]
        for rx, name_first in ((_NM_RE, False), (_ADDR_NAME_RE, False), (_NAME_ADDR_RE, True)):
            m = rx.match(line)
            if m and rx is _NM_RE:
                if m.group(2) in "tTwW":
                    out.setdefault(m.group(3), int(m.group(1), 16))
                break
            if m:
                name, addr = (m.group(1), m.group(2)) if name_first else (m.group(2), m.group(1))
                out.setdefault(name, int(addr, 16))
                break
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

