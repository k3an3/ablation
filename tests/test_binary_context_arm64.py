"""Tests for BinaryContext on AArch64 ELF: PLT, call graph, string xrefs, func starts.

Fixtures are cross-built with clang + ld.lld (no libc needed: -shared -nostdlib).
Fixture tests skip when no AArch64-capable clang/lld is on PATH; the scanner and
decode unit tests run everywhere.
"""

import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ablation.analyzers.arm64_func_start_scanner import ARM64FuncStartScanner
from ablation.analyzers.binary_context import BinaryContext, _a64_adrp_page

sys.path.insert(0, str(Path(__file__).parent))
from a64_fixtures import VARIANTS as _VARIANTS, build_variants, symbols as _symbols


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    return build_variants(tmp_path_factory)


def _strip(path, tmp_path):
    out = tmp_path / (path.stem + ".stripped.so")
    r = subprocess.run(["llvm-objcopy", "--strip-all", str(path), str(out)],
                       capture_output=True)
    if r.returncode != 0:
        pytest.skip("llvm-objcopy required to strip fixture")
    return out


# ── decode / scanner unit tests (no toolchain needed) ────────────────────────

def test_adrp_page_decode():
    # adrp x16, 0x30000 at 0x10510: page delta 0x20 -> immhi=0x8, immlo=0
    assert _a64_adrp_page(0x90000110, 0x10510) == 0x30000
    # immlo carries the low 2 bits: delta 0x21 pages
    assert _a64_adrp_page(0xB0000110, 0x10510) == 0x31000
    # negative page offset: adrp x0, 0x0 at 0x104c8 -> immhi:immlo = -0x10
    w = 0x90000000 | ((((-0x10) >> 2) & 0x7FFFF) << 5) | (((-0x10) & 3) << 29)
    assert _a64_adrp_page(w, 0x104c8) == 0x0


def _words(*ws):
    return b"".join(struct.pack("<I", w) for w in ws)


def test_scanner_prologues_and_landing_pads():
    NOP, RET = 0xD503201F, 0xD65F03C0
    PACIASP, BTI_C = 0xD503233F, 0xD503245F
    STP_PRE = 0xA9BF7BFD           # stp x29, x30, [sp, #-16]!
    SUB_SP = 0xD10143FF            # sub sp, sp, #0x50
    STP_OFF = 0xA9047BFD           # stp x29, x30, [sp, #0x40]
    code = _words(
        STP_PRE, NOP, RET,         # 0x1000: frame-record push
        PACIASP, STP_PRE, RET,     # 0x100c: folded onto paciasp
        SUB_SP, STP_OFF, RET,      # 0x1018: large frame
        BTI_C, NOP, RET,           # 0x1024: bti c landing pad
        0x97FFFFF4,                # 0x1030: bl 0x1000  (imm26 = -12)
    )
    r = ARM64FuncStartScanner([(0x1000, code)]).scan_full()
    assert r.starts == [0x1000, 0x100C, 0x1018, 0x1024]
    assert r.bl_targets == 1


def test_scanner_ignores_bl_outside_code():
    code = _words(0x94100000, 0xD65F03C0)   # bl +0x4000000 (outside region)
    assert ARM64FuncStartScanner([(0x1000, code)]).scan() == []


# ── BinaryContext on cross-built fixtures ────────────────────────────────────

@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_arch_detected(fixtures, variant):
    assert BinaryContext.build(str(fixtures[variant])).arch == "arm64"


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_plt_stubs_named(fixtures, variant):
    ctx = BinaryContext.build(str(fixtures[variant]))
    assert set(ctx.plt.values()) == {"helper", "strcpy", "puts", "system"}
    # every stub VA lies in .plt and BL targets resolve to it
    import lief
    plt = lief.parse(str(fixtures[variant])).get_section(".plt")
    for va in ctx.plt:
        assert plt.virtual_address <= va < plt.virtual_address + plt.size


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_call_graph(fixtures, variant):
    ctx = BinaryContext.build(str(fixtures[variant]))
    syms = _symbols(fixtures[variant]) if "noeh" not in variant else None
    edges = {(ctx.name(a), lbl or ctx.name(b)) for a, b, lbl in ctx.call_edges}
    assert ("helper", "strcpy") in edges
    assert ("helper", "puts") in edges          # tail call: b puts@plt
    assert ("handler", "helper") in edges
    assert ("handler", "puts") in edges
    # static tail() reaches system via `b system@plt`
    callers = ctx.callers_of("system")
    assert len(callers) == 1
    if syms:
        assert callers[0][0] == syms["tail"]


@pytest.mark.parametrize("variant", list(_VARIANTS))
def test_string_xrefs(fixtures, variant):
    ctx = BinaryContext.build(str(fixtures[variant]))
    by_text = {s: va for va, s in ctx.strings.items()}
    handler = ctx.export_va("handler")
    helper = ctx.export_va("helper")
    # adrp + add
    assert ctx.funcs_referencing_string(by_text["handler-string-xref"]) == [handler]
    # adr (string within +-1MB of code)
    assert ctx.funcs_referencing_string(by_text["helper-called"]) == [helper]
    # adrp + ldr through a hidden pointer variable (R_AARCH64_RELATIVE slot)
    assert ctx.funcs_referencing_string(by_text["global-pointer-string"]) == [handler]
    # string passed from a static function
    assert len(ctx.funcs_referencing_string(by_text["/bin/sh -c reboot"])) == 1
    assert ("handler-string-xref" in [s for _, s in ctx.strings_in_func(handler)])


def test_func_starts_without_eh_frame(fixtures, tmp_path):
    """noeh fixture has no .eh_frame: scanner must recover every function."""
    import lief
    path = fixtures["noeh"]
    assert lief.parse(str(path)).get_section(".eh_frame") is None
    syms = _symbols(path)
    ctx = BinaryContext.build(str(_strip(path, tmp_path)))
    for name in ("helper", "handler", "tail", "framed"):
        assert syms[name] in ctx.func_starts, name


def test_func_starts_without_eh_frame_pac(fixtures, tmp_path):
    """PAC/BTI prologues: start is the paciasp / bti c, not the following stp."""
    syms = _symbols(fixtures["noeh_bti_pac"])
    ctx = BinaryContext.build(str(_strip(fixtures["noeh_bti_pac"], tmp_path)))
    for name in ("helper", "handler", "tail", "framed"):
        assert syms[name] in ctx.func_starts, name
    assert syms["helper"] + 4 not in ctx.func_starts


def test_eh_frame_starts_match_symbols(fixtures):
    syms = _symbols(fixtures["plain"])
    ctx = BinaryContext.build(str(fixtures["plain"]))
    for name in ("helper", "handler", "tail", "framed"):
        assert syms[name] in ctx.func_starts, name


def test_stale_arm64_cache_rejected(fixtures, tmp_path):
    """v2 arm64 caches were built by the x86 fallback paths and must rebuild."""
    import json
    ctx = BinaryContext.build(str(fixtures["plain"]))
    cache = tmp_path / "c.json"
    ctx._save_json(cache)
    assert BinaryContext._load_json(cache, "").plt == ctx.plt
    payload = json.loads(cache.read_text())
    payload["cache_version"] = 2
    cache.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        BinaryContext._load_json(cache, "")
