"""ARM64HeapScanner: uaf / double_free / off_by_one_alloc / heap_overflow / int overflow.

clang + ld.lld fixtures at -O0 and -O2 (skip without the toolchain). Each function's
expected verdict is fixed; the negative controls must stay silent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.heap_scanner_arm64 import ARM64HeapScanner

from a64_fixtures import build_variants

_SRC = r'''
typedef unsigned long size_t;
extern void *malloc(size_t); extern void free(void *);
extern size_t strlen(const char *); extern char *strcpy(char *, const char *);
extern void *memcpy(void *, const void *, size_t); extern long recv(int, void *, size_t, int);
extern void use(void *); extern int check(void);
struct obj { int a; void (*cb)(void); };
int uaf_read(void) { struct obj *o = malloc(sizeof *o); o->a = 1; free(o); return o->a; }
void uaf_pass(void) { char *p = malloc(32); free(p); use(p); }
void dfree(void) { char *p = malloc(32); use(p); free(p); free(p); }
void dfree_both_arms(int c) { char *p = malloc(32); if (c) { use(p); free(p); } else { free(p); } free(p); }
int errpath(void) { char *p = malloc(32); if (!check()) { free(p); return -1; } use(p); free(p); return 0; }
void realloc_ok(void) { char *p = malloc(32); free(p); p = malloc(64); use(p); free(p); }
void loop_ok(int n) { for (int i = 0; i < n; i++) { char *p = malloc(16); use(p); free(p); } }
char *dup_bad(const char *s) { char *d = malloc(strlen(s)); strcpy(d, s); return d; }
char *dup_ok(const char *s) { char *d = malloc(strlen(s) + 1); strcpy(d, s); return d; }
char *dup_memcpy_bad(const char *s) { size_t n = strlen(s); char *d = malloc(n); memcpy(d, s, n + 1); return d; }
void hovf(int fd) { unsigned char h[8]; recv(fd, h, 8, 0); unsigned n = *(unsigned *)h; char *b = malloc(64); recv(fd, b, n, 0); use(b); }
void hok(int fd) { unsigned char h[8]; recv(fd, h, 8, 0); unsigned n = h[0] & 0x3f; char *b = malloc(64); recv(fd, b, n, 0); use(b); }
void const_ovf(void) { char src[200]; use(src); char *b = malloc(64); memcpy(b, src, 128); use(b); }
void *iovf(int fd) { unsigned char h[8]; recv(fd, h, 8, 0); unsigned n = *(unsigned *)h; return malloc(n * 12u); }
'''

_EXPECT = {
    ("uaf_read", "uaf", "HIGH"),
    ("uaf_pass", "uaf", "MEDIUM"),
    ("dfree", "double_free", "HIGH"),
    ("dfree_both_arms", "double_free", "HIGH"),
    ("dup_bad", "off_by_one_alloc", "HIGH"),
    ("dup_memcpy_bad", "off_by_one_alloc", "HIGH"),
    ("hovf", "heap_overflow", "HIGH"),
    ("const_ovf", "heap_overflow", "HIGH"),
    ("iovf", "int_overflow_before_alloc", "HIGH"),
}
_SILENT = {"errpath", "realloc_ok", "loop_ok", "dup_ok", "hok"}


@pytest.fixture(scope="module")
def bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_SRC,
                          variants={"O0": ["-O0"], "O2": ["-O2"]})


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_heap_verdicts(bins, opt):
    s = ARM64HeapScanner.from_path_full(str(bins[opt]))
    got = {(s._syms.get(f.func_va), f.kind, f.severity) for f in s.scan()}
    assert _EXPECT <= got, _EXPECT - got
    assert not {g for g in got if g[0] in _SILENT}, got


def test_uaf_records_free_site_and_allocator(bins):
    s = ARM64HeapScanner.from_path_full(str(bins["O2"]))
    f = next(f for f in s.scan() if s._syms.get(f.func_va) == "uaf_read")
    assert f.alloc_sym == "malloc" and f.func_va < f.freed_at < f.site_va
    assert "after free" in f.fmt()


def test_report_and_int_overflow_toggle(bins):
    s = ARM64HeapScanner.from_path_full(str(bins["O2"]))
    assert not any(f.kind == "int_overflow_before_alloc" for f in s.scan(include_int_overflow=False))
    assert s.report(s.scan()).startswith("ARM64 heap scan:")
