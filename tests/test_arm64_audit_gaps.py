"""Gaps found auditing a multi-binary AArch64 firmware: relocations, sinks without imports,
taint sources / sinks overrides, conversion-driven lengths, finding quality.

clang + ld.lld fixtures; skip without the toolchain.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.binary_context import BinaryContext, elf_relocations

from a64_fixtures import build_variants, symbols

# ── relocations ──────────────────────────────────────────────────────────────

_REL_SRC = r'''
extern int puts(const char *);
int op_a(int x) { return x + 1; }
int op_b(int x) { return x * 2; }
__attribute__((visibility("hidden"))) int op_c(int x) { return x - 3; }
int (*const table[])(int) = { op_a, op_b, op_c };
const char *msgs[] = { "alpha", "beta" };
int (*ext)(const char *) = puts;
int run(int i, int x) { puts(msgs[i & 1]); return table[i](x); }
'''


@pytest.fixture(scope="module")
def rel_bin(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_REL_SRC, variants={"O2": []})["O2"]


def test_abs64_table_resolves(rel_bin):
    import lief
    truth = symbols(rel_bin)
    rel = elf_relocations(lief.parse(str(rel_bin)))
    assert rel.stats == {"RELATIVE": 3, "ABS64": 2, "GLOB_DAT": 2,
                         "ABS64 (import)": 1, "JUMP_SLOT (import)": 1}
    ctx = BinaryContext.load_or_build(str(rel_bin), force_rebuild=True)
    t = truth["table"]
    assert [ctx.read_ptr(t + 8 * i) for i in range(3)] == [truth["op_a"], truth["op_b"], truth["op_c"]]
    regions = dict(ctx.image_regions())
    seg = next(va for va, d in regions.items() if va <= t < va + len(d))
    assert int.from_bytes(regions[seg][t - seg:t - seg + 8], "little") == truth["op_a"]
    assert "relocs     : applied 3 RELATIVE, 2 ABS64, 2 GLOB_DAT" in ctx.summary()
    assert rel.symbols[truth["ext"]] == "puts"


# ── sinks without imports: a stripped headerless image with its own libc ────

_ROLES_SRC = r'''
typedef unsigned long size_t; typedef __builtin_va_list va_list;
#define N __attribute__((noinline))
volatile unsigned char *const UART = (unsigned char *)0x09000000;
N void *memcpy(void *d, const void *s, size_t n) { char *a = d; const char *b = s; while (n--) *a++ = *b++; return d; }
N void *memset(void *d, int c, size_t n) { char *a = d; while (n--) *a++ = (char)c; return d; }
N char *strcpy(char *d, const char *s) { char *r = d; while ((*d++ = *s++) != 0) {} return r; }
N size_t strlen(const char *s) { const char *p = s; while (*p) p++; return (size_t)(p - s); }
N static char *emit_s(char *o, const char *s) { while (*s) *o++ = *s++; return o; }
N int vsprintf(char *o, const char *f, va_list ap) { char *b = o;
  for (; *f; f++) { if (*f != '%') { *o++ = *f; continue; } f++;
    if (*f == 's') o = emit_s(o, __builtin_va_arg(ap, const char *));
    else if (*f == 'd') { int v = __builtin_va_arg(ap, int); *o++ = '0' + (v % 10); } }
  *o = 0; return (int)(o - b); }
N int sprintf(char *o, const char *f, ...) { va_list ap; __builtin_va_start(ap, f); int r = vsprintf(o, f, ap); __builtin_va_end(ap); return r; }
N int snprintf(char *o, size_t n, const char *f, ...) { va_list ap; char t[256]; __builtin_va_start(ap, f); int r = vsprintf(t, f, ap); __builtin_va_end(ap); memcpy(o, t, n); return r; }
N int printf(const char *f, ...) { char t[256]; va_list ap; __builtin_va_start(ap, f); int r = vsprintf(t, f, ap); __builtin_va_end(ap); for (char *p = t; *p; p++) *UART = *p; return r; }
N int sscanf(const char *s, const char *f, ...) { va_list ap; __builtin_va_start(ap, f);
  for (; *f; f++) if (*f == '%') { f++; if (*f == 's') { char *d = __builtin_va_arg(ap, char *); while (*s && *s != ' ') *d++ = *s++; *d = 0; } else if (*f == 'd') { int *d = __builtin_va_arg(ap, int *); *d = *s++ - '0'; } }
  __builtin_va_end(ap); return 1; }
N long net_recv(int fd, char *buf, size_t n) { for (size_t i = 0; i < n; i++) buf[i] = (char)*UART; return (long)n; }
extern void use(char *);
N void use(char *p) { *UART = (unsigned char)p[0]; }
N void h_login(const char *user) { char o[32]; sprintf(o, "user=%s", user); use(o); }
N void h_id(int id) { char o[32]; sprintf(o, "id=%d", id); use(o); }
N void h_path(const char *p) { char o[64]; sprintf(o, "/www/%s", p); use(o); }
N void h_parse(const char *req) { char m[16]; char u[64]; int v; sscanf(req, "%s %s %d", m, u, &v); use(m); use(u); }
N void h_parse2(const char *req) { char k[16]; sscanf(req, "%s", k); use(k); }
N void h_parse3(const char *req) { int a; char k[8]; sscanf(req, "%d %s", &a, k); use(k); }
N void h_log(int x) { printf("x=%d\n", x); printf("boot %d\n", x + 1); printf("ok %s\n", "y"); }
N void h_fmt(char *o, int n) { snprintf(o, 16, "n=%d", n); snprintf(o, 16, "m=%d", n + 1); snprintf(o, 16, "%s", "z"); }
N void h_copy(char *d, const char *s, size_t n) { char t[32]; memcpy(t, s, n); memset(d, 0, 8); memcpy(d, t, 4); strcpy(d, s); memcpy(d + 1, s, 2); use(t); }
N size_t h_len(const char *s) { return strlen(s) + strlen(s + 1) + strlen("abc"); }
void _start(void) { char buf[128]; net_recv(0, buf, 127); buf[127] = 0;
  h_login(buf); h_id(1); h_path(buf); h_parse(buf); h_parse2(buf); h_parse3(buf); h_log(2); h_fmt(buf, 3); h_copy(buf, buf + 1, 9); h_len(buf); for (;;) {} }
'''

_RAW_LD = """
ENTRY(_start)
SECTIONS { . = 0x40080000; .text : { *(.text._start) *(.text*) } .rodata : { *(.rodata*) }
           .data : { *(.data*) } .bss : { *(.bss*) } }
"""
RAW_BASE = 0x40080000


def build_raw(tmp: Path, src: str, extra=()):
    if not all(shutil.which(t) for t in ("clang", "ld.lld", "llvm-objcopy", "llvm-nm")):
        pytest.skip("needs clang, ld.lld, llvm-objcopy, llvm-nm")
    (tmp / "fw.c").write_text(src)
    (tmp / "fw.ld").write_text(_RAW_LD)
    r = subprocess.run(["clang", "--target=aarch64-none-elf", "-O2", "-ffreestanding", "-fno-builtin",
                        "-nostdlib", "-ffunction-sections", "-fuse-ld=lld", "-w", *extra,
                        f"-Wl,-T,{tmp / 'fw.ld'}", "-o", str(tmp / "fw.elf"), str(tmp / "fw.c")],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"cross build failed: {r.stderr[-300:]}")
    subprocess.run(["llvm-objcopy", "-O", "binary", str(tmp / "fw.elf"), str(tmp / "fw.bin")], check=True)
    nm = subprocess.run(["llvm-nm", str(tmp / "fw.elf")], check=True, capture_output=True, text=True).stdout
    truth = {ln.split()[2]: int(ln.split()[0], 16) for ln in nm.splitlines() if ln.split()[1] in "tT"}
    return tmp / "fw.bin", truth


@pytest.fixture(scope="module")
def roles_img(tmp_path_factory):
    return build_raw(tmp_path_factory.mktemp("roles"), _ROLES_SRC)


def test_libc_roles_inferred(roles_img):
    from ablation.analyzers.libc_roles import LibcRoleInferer
    binf, truth = roles_img
    ctx = BinaryContext.from_raw(str(binf), "arm64", base=RAW_BASE, force_rebuild=True)
    assert ctx.sink_coverage({"memcpy", "sprintf"})[0] == set()
    roles = LibcRoleInferer(ctx).infer()
    want = {"memcpy": "memcpy", "memset": "memset", "strcpy": "strcpy", "strlen": "strlen",
            "sprintf": "sprintf", "snprintf": "snprintf", "printf": "printf", "sscanf": "sscanf",
            "emit_s": "strcpy"}                     # a copy-until-NUL helper is strcpy-shaped
    assert roles == {truth[f]: r for f, r in want.items()}
    assert ctx.name(truth["sscanf"]) == "sscanf"


def test_fmtstr_finds_unbounded_without_symbols(roles_img):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    binf, truth = roles_img
    ctx = BinaryContext.from_raw(str(binf), "arm64", base=RAW_BASE, force_rebuild=True)
    ctx.infer_roles(save=False)
    got = {(f.func_va, f.callee) for f in ARM64FormatStringScanner.from_context(ctx).scan()}
    assert got == {(truth["h_login"], "sprintf"), (truth["h_path"], "sprintf"),
                   (truth["h_parse"], "sscanf"), (truth["h_parse2"], "sscanf"),
                   (truth["h_parse3"], "sscanf")}


def _cli(*a):
    return subprocess.run([sys.executable, "-m", "ablation.cli", *map(str, a)], capture_output=True,
                          text=True, cwd=Path(__file__).parent.parent)


def test_cli_warns_on_zero_resolved_sinks(roles_img, tmp_path):
    binf, _truth = roles_img
    fresh = tmp_path / "fresh.bin"                         # own cache entry, no roles yet
    fresh.write_bytes(binf.read_bytes() + b"\0" * 4 + __import__("os").urandom(8))
    r = _cli("heap", fresh, "--raw", "arm64", "--base", hex(RAW_BASE), "--no-infer-sinks")
    assert "0 of" in r.stdout and "results are not meaningful" in r.stdout
    r = _cli("fmtstr", fresh, "--raw", "arm64", "--base", hex(RAW_BASE))
    assert "inferring libc roles" in r.stderr and "not meaningful" not in r.stdout
    assert r.stdout.count("unbounded") == 0 and r.stdout.count("sscanf '%s'") == 3


def test_sink_map_names_roles_cli(roles_img, tmp_path):
    binf, truth = roles_img
    sm = tmp_path / "sinks.json"
    sm.write_text(json.dumps({hex(truth["sprintf"]): "sprintf"}))
    fresh = tmp_path / "fresh2.bin"
    fresh.write_bytes(binf.read_bytes() + b"\0" * 8 + __import__("os").urandom(8))
    r = _cli("fmtstr", fresh, "--raw", "arm64", "--base", hex(RAW_BASE), "--sink-map", sm,
             "--no-infer-sinks")
    assert "sink map: 1 function" in r.stderr
    assert r.stdout.count("sprintf '%s'") == 2 and "sscanf" not in r.stdout
