"""ARM64BssTaintTracker and ARM64VtableDispatchScanner (+ x86 BSS tracker crash fixes).

clang/clang++ + ld.lld fixtures at -O0 and -O2; skip without the toolchain.
"""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.bss_taint_tracker import BSSSymbol
from ablation.analyzers.bss_taint_tracker_arm64 import (
    ARM64BssTaintTracker, bss_symbols_from_elf, build_function_arm64,
)
from ablation.analyzers.vtable_dispatch_arm64 import ARM64VtableDispatchScanner, _indirect_branch_reg

from a64_fixtures import build_variants

# ── BSS range walks ──────────────────────────────────────────────────────────

_BSS_SRC = r'''
#define H __attribute__((visibility("hidden")))
H unsigned char *initrd_start;
H unsigned char *initrd_end;
H unsigned char *other_ptr;
H unsigned long counter;
void decrypt(unsigned char key) { for (unsigned char *p = initrd_start; p < initrd_end; p++) *p ^= key; }
unsigned long size(void) { return initrd_end - initrd_start; }
void spin(void) { for (unsigned long i = 0; i < counter; i++) other_ptr[i] = 0; }
'''


@pytest.fixture(scope="module")
def bss_bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_BSS_SRC,
                          variants={"O0": ["-O0"], "O2": ["-O2"]})


def _analyze_all(path):
    import lief
    b = lief.parse(str(path))
    data = Path(path).read_bytes()
    text = b.get_section(".text")
    base = text.virtual_address - text.offset
    syms = bss_symbols_from_elf(str(path))
    tr = ARM64BssTaintTracker(syms)
    by_name = {s.name: s.addr for s in syms}
    out = {}
    for s in b.symbols:
        if s.name in ("decrypt", "size", "spin") and s.name not in out:
            out[s.name] = tr.analyze(build_function_arm64(data, s.value, s.size, base), s.name)
    return out, by_name


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_bss_range_walk(bss_bins, opt):
    res, addr = _analyze_all(bss_bins[opt])
    assert res["decrypt"] is not None
    assert res["decrypt"]["bss_pairs"] == [(addr["initrd_start"], addr["initrd_end"])]
    assert res["decrypt"]["loops"] and res["decrypt"]["fn_name"] == "decrypt"
    assert res["size"] is None            # pair read, no loop
    assert res["spin"] is None            # single unrelated global


def test_bss_symbols_fallback_for_stripped(bss_bins, tmp_path):
    stripped = tmp_path / "s.so"
    if subprocess.run(["llvm-objcopy", "--strip-all", str(bss_bins["O2"]), str(stripped)]).returncode:
        pytest.skip("llvm-objcopy required")
    syms = bss_symbols_from_elf(str(stripped))
    assert syms and all(s.size == 8 for s in syms) and syms[0].name.startswith("bss_")


def test_x86_bss_symbol_hashable():
    # the x86 tracker puts BSSSymbol into sets; it used to raise TypeError
    assert len({BSSSymbol(1, 8), BSSSymbol(1, 8)}) == 1


def test_x86_build_function_runs():
    from ablation.analyzers.bss_taint_tracker import build_function
    code = bytes.fromhex("4889f8c3")          # mov rax, rdi ; ret   (was AttributeError .addr)
    fn = build_function(b"\x00" * 0x10 + code, 0x10, len(code))
    assert list(fn.blocks) == [0x10]


# ── vtable dispatch ──────────────────────────────────────────────────────────

_VT_SRC = r'''
struct Shape {
    virtual int area() const;
    virtual void draw() const;
    virtual int sides() const;
    virtual void unused_slot() const;
};
int Shape::area() const { return 0; }
void Shape::draw() const {}
int Shape::sides() const { return 0; }
void Shape::unused_slot() const {}
extern "C" int use_area(const Shape *s) { return s->area() + 1; }
extern "C" void use_draw(const Shape *s) { s->draw(); }
extern "C" int use_sides(const Shape *s) { int a = s->sides(); return a * s->sides(); }
'''


@pytest.fixture(scope="module")
def vt_bins(tmp_path_factory):
    if not (shutil.which("clang++") and shutil.which("ld.lld")):
        pytest.skip("clang++ + ld.lld required")
    d = tmp_path_factory.mktemp("vt")
    (d / "vt.cc").write_text(_VT_SRC)
    out = {}
    for opt in ("O0", "O2"):
        so = d / f"vt_{opt}.so"
        r = subprocess.run(["clang++", "--target=aarch64-linux-gnu", f"-{opt}", "-fPIC", "-shared",
                            "-nostdlib", "-fuse-ld=lld", "-fno-rtti", "-fno-exceptions",
                            str(d / "vt.cc"), "-o", str(so)], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(r.stderr[:200])
        out[opt] = so
    return out


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_vtable_live_and_dead_slots(vt_bins, opt):
    s = ARM64VtableDispatchScanner.from_path(str(vt_bins[opt]))
    r = s.scan({"area": 0x0, "draw": 0x8, "sides": 0x10, "unused_slot": 0x18})
    assert set(r.live()) == {"area", "draw", "sides"}
    assert r.is_dead("unused_slot")
    assert len(r.sites_for("sides")) == 2
    if opt == "O2":
        assert r.sites_for("draw")[0].encoding == "ldr+br"     # tail dispatch
    assert "LIVE" in ARM64VtableDispatchScanner.report(r)


def test_vtable_from_context(vt_bins):
    from ablation.analyzers.binary_context import BinaryContext
    ctx = BinaryContext.load_or_build(str(vt_bins["O2"]))
    assert ARM64VtableDispatchScanner.from_context(ctx).scan_offset(0x10)


def test_pac_indirect_branch_encodings():
    assert _indirect_branch_reg(0xD63F0100) == (8, "blr")      # blr x8
    assert _indirect_branch_reg(0xD61F0020) == (1, "br")       # br x1
    assert _indirect_branch_reg(0xD73F0911) == (8, "blra")     # blraa x8, x17
    assert _indirect_branch_reg(0xD63F091F) == (8, "blra")     # blraaz x8
    assert _indirect_branch_reg(0xD65F03C0) is None            # ret
