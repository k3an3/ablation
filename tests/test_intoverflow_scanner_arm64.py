"""ARM64IntOverflowScanner: tainted multiply -> allocation size.

Fixtures are cross-built with clang + ld.lld at -O0 and -O2 and skip when the
toolchain is missing. Each function's verdict is checked against intent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.intoverflow_scanner_arm64 import ARM64IntOverflowScanner

from a64_fixtures import build_variants

_SRC = r'''
typedef unsigned long size_t;
typedef unsigned int u32; typedef unsigned short u16; typedef unsigned long u64;
extern long recv(int, void *, size_t, int);
extern void *malloc(size_t);
extern void *calloc(size_t, size_t);
void *v32(int fd) { unsigned char b[16]; recv(fd, b, 16, 0); u32 n = *(u32 *)(b + 4); u32 sz = n * 12; return malloc(sz); }
void *v64(int fd) { unsigned char b[16]; recv(fd, b, 16, 0); u64 n = *(u64 *)b; return malloc(n * 24); }
void *safe16(int fd) { unsigned char b[16]; recv(fd, b, 16, 0); u16 n = *(u16 *)b; return malloc(n * 8); }
void *safewide(int fd) { unsigned char b[16]; recv(fd, b, 16, 0); u32 n = *(u32 *)b; return malloc((size_t)n * 16); }
void *guarded(int fd) { unsigned char b[16]; recv(fd, b, 16, 0); u32 n = *(u32 *)b; if (n > 1000) return 0; return malloc(n * 40); }
void *checked(int fd) { unsigned char b[16]; size_t sz; recv(fd, b, 16, 0); u64 n = *(u64 *)b; if (__builtin_mul_overflow(n, 24, &sz)) return 0; return malloc(sz); }
void *viacalloc(int fd) { unsigned char b[16]; recv(fd, b, 16, 0); u64 n = *(u64 *)b; return calloc(n, 24); }
void *clean(int k) { return malloc((u32)k * 12); }
__attribute__((noinline)) void *alloc_items(u32 n) { return malloc(n * 40u); }
void *inter(int fd) { unsigned char b[16]; recv(fd, b, 16, 0); return alloc_items(*(u32 *)b); }
'''


@pytest.fixture(scope="module")
def bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_SRC,
                          variants={"O0": ["-O0"], "O2": ["-O2"]})


def _verdicts(path):
    s = ARM64IntOverflowScanner.from_path_full(str(path))
    findings = s.scan()
    out = {}
    for f in findings:
        out[s._syms.get(f.func_va, hex(f.func_va))] = "guarded" if f.bounds_checked else "UNGUARDED"
    return out, findings


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_verdicts(bins, opt):
    v, _ = _verdicts(bins[opt])
    assert v["v32"] == "UNGUARDED"
    assert v["v64"] == "UNGUARDED"
    assert v["alloc_items"] == "UNGUARDED"        # size computed in the callee of inter()
    assert v["checked"] == "guarded"              # __builtin_mul_overflow (umulh check)
    for safe in ("safe16", "safewide", "viacalloc", "clean"):
        assert safe not in v, safe
    # -O0 keeps the 32-bit multiply after the range check; -O2 proves n*40 fits and
    # emits a widening umull, which cannot overflow
    if opt == "O0":
        assert v["guarded"] == "guarded"
    else:
        assert "guarded" not in v


def test_width_and_merged_steps(bins):
    _, findings = _verdicts(bins["O2"])
    v32 = next(f for f in findings if f.width == 32 and "lsl#2" in f.mnemonic)
    assert v32.mnemonic.count("->") == 1          # n*12 = (n + n<<1) << 2 reported once
    assert v32.alloc_sym == "malloc" and v32.size_arg == "x0"
    assert v32.wire_evidence == "recv"


def test_interprocedural_alloc_in_callee(bins):
    s = ARM64IntOverflowScanner.from_path_full(str(bins["O0"]))
    f = next(f for f in s.scan() if s._syms.get(f.func_va) == "alloc_items")
    assert f.alloc_func_va == f.func_va           # multiply and malloc both in the callee


def test_report_format(bins):
    s = ARM64IntOverflowScanner.from_path_full(str(bins["O2"]))
    text = s.report(s.scan())
    assert text.startswith("ARM64 Integer Overflow Scan:") and "[UNGUARDED]" in text
