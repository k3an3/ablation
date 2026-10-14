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
    assert r.stdout.count("unbounded") == 0
    assert sum(1 for l in r.stdout.splitlines() if "fmt_string" in l and "sscanf" in l and "(no width)" in l) == 3


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


# ── taint: framework sources, wider sinks, opaque summaries ──────────────────

_HTTP_SRC = r'''
typedef unsigned long size_t;
#define H __attribute__((visibility("hidden"), noinline))
extern long recv(int, void *, size_t, int);
extern char *strcpy(char *, const char *);
extern void *memcpy(void *, const void *, size_t);
extern int sprintf(char *, const char *, ...);
extern int sscanf(const char *, const char *, ...);
extern long read(int, void *, size_t);
extern void sink(void *);
struct req { int fd; char raw[512]; int clen; };
/* the framework: recv is far below the handlers */
H static long io_fill(struct req *r) { return recv(r->fd, r->raw, sizeof r->raw, 0); }
H static long io_read(struct req *r) { long n = io_fill(r); r->raw[511] = 0; return n; }
H static int parse(struct req *r) { return io_read(r) > 0 ? 0 : -1; }
H const char *http_get_param(struct req *r, const char *k) { (void)k; return r->raw + 16; }
H int http_content_length(struct req *r) { return r->clen; }
H int http_read_body(struct req *r, void *buf, size_t n) { memcpy(buf, r->raw, n); return (int)n; }
H unsigned be16(const unsigned char *p) { return (unsigned)(p[0] << 8 | p[1]); }
/* handlers: the real bugs */
void h_user(struct req *r) { char name[32]; strcpy(name, http_get_param(r, "user")); sink(name); }
void h_fmt(struct req *r) { char o[64]; sprintf(o, "id=%s", http_get_param(r, "id")); sink(o); }
void h_scan(struct req *r) { char m[8]; char u[8]; sscanf(http_get_param(r, "q"), "%s %s", m, u); char t[8]; strcpy(t, u); sink(m); sink(t); }
void h_body(struct req *r, int fd) { char body[256]; read(fd, body, http_content_length(r)); sink(body); }
void h_copy(struct req *r) { char b[64]; memcpy(b, r->raw, http_content_length(r)); sink(b); }
void h_hdr(struct req *r) { unsigned char pkt[64]; char out[16]; http_read_body(r, pkt, sizeof pkt);
    unsigned n = be16(pkt); sink(pkt); memcpy(out, r->raw, n); sink(out); }   /* length parsed from a header */
void serve(struct req *r) { if (parse(r) == 0) { h_user(r); h_fmt(r); } }
'''


@pytest.fixture(scope="module")
def http_so(tmp_path_factory):
    so = build_variants(tmp_path_factory, src_text=_HTTP_SRC, variants={"O1": ["-O1"]})["O1"]
    return so, symbols(so)


def _http_tracker(so, truth, **kw):
    from ablation.analyzers.taint_tracker_arm64 import ARM64TaintTracker
    ctx = BinaryContext.load_or_build(str(so))
    t = ARM64TaintTracker.from_context(ctx)
    t.add_sources([hex(truth["http_get_param"]), f"{hex(truth['http_content_length'])}:val",
                   f"http_read_body:x1"])
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_framework_sources_by_address(http_so):
    so, truth = http_so
    from ablation.analyzers.taint_tracker_arm64 import ARM64TaintTracker
    plain = ARM64TaintTracker.from_context(BinaryContext.load_or_build(str(so)))
    assert not plain.run_interprocedural(depth=5)          # recv is too deep: nothing
    got = {(f.func_name, f.sink_name, f.severity) for f in _http_tracker(so, truth).run_interprocedural()}
    assert ("h_user", "strcpy", "HIGH") in got
    assert ("h_fmt", "sprintf", "HIGH") in got
    assert ("h_scan", "sscanf", "MEDIUM") in got              # parsed attacker text
    assert ("h_scan", "strcpy", "HIGH") in got                # ... and its output field copied
    assert ("h_body", "read", "HIGH") in got                  # header length into a fixed buffer
    assert ("h_copy", "memcpy", "HIGH") in got
    assert not any(n == "h_hdr" for n, _s, _v in got)         # be16() result has no model


def test_summarise_opaque_lower_confidence(http_so):
    so, truth = http_so
    fs = _http_tracker(so, truth, summarise_opaque=True).run_interprocedural()
    hdr = [f for f in fs if f.func_name == "h_hdr"]
    assert len(hdr) == 1 and hdr[0].sink_name == "memcpy" and hdr[0].tainted_args == ["x2"]
    assert hdr[0].severity == "MEDIUM" and "opaque" in hdr[0].note
    assert hdr[0].source_name == "http_read_body"
    # findings with a direct path keep their severity
    assert {(f.func_name, f.severity) for f in fs if f.sink_name == "strcpy"} >= {("h_user", "HIGH")}


def test_cli_sources_sinks(http_so):
    so, truth = http_so
    r = _cli("taint", so, "--sources", f"{hex(truth['http_get_param'])}", "--sinks", "sink:x0")
    assert "strcpy" in r.stdout and "-> sink @" in r.stdout


# ── conversion-driven lengths, snprintf variants ─────────────────────────────

_CONV_SRC = r'''
typedef unsigned long size_t;
extern void *memcpy(void *, const void *, size_t);
extern int snprintf(char *, size_t, const char *, ...);
extern long recv(int, void *, size_t, int);
extern int get(void);
extern unsigned short get16(void);
extern void sink(void *);
int g_len;
void set_len(long n) { g_len = (int)n; }                                     /* 64 -> 32 store */
void use_len(char *d, const char *s) { memcpy(d, s, g_len); }               /* ldrsw -> size */
void signed_chk(char *d, const char *s) { int n = get(); if (n > 64) return; memcpy(d, s, n); }
void signed_ok(char *d, const char *s) { int n = get(); if (n > 64 || n < 0) return; memcpy(d, s, n); }
void unsigned_ok(char *d, const char *s) { int n = get(); if ((unsigned)n > 64) return; memcpy(d, s, n); }
void masked(char *d, const char *s) { unsigned n = (unsigned)get(); if ((n & 0xffff) > 64) return; memcpy(d, s, n); }
void masked_ok(char *d, const char *s) { unsigned n = (unsigned)get() & 0xffff; if (n > 64) return; memcpy(d, s, n); }
void byte_loop(char *d) { unsigned short lim = get16(); for (unsigned char i = 0; i < lim; i++) d[i] = 0; }
void byte_ok(char *d) { unsigned short lim = get16(); for (unsigned short i = 0; i < lim; i++) d[i] = 0; }
void wire(int fd, char *d) { char b[64]; recv(fd, b, sizeof b, 0); int n = *(int *)(b + 4); if (n > 32) return; memcpy(d, b, n); }
void snp(char *d, const char *s) { int n = get(); if (n > 64) return; snprintf(d, n, "%s", s); }
void und(int fd, char *d) { unsigned char b[64]; recv(fd, b, sizeof b, 0); unsigned n = b[2]; snprintf(d, n - 16, "%s", "x"); }
'''


@pytest.fixture(scope="module")
def conv_bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_CONV_SRC, variants={"O1": ["-O1"], "O2": ["-O2"]})


@pytest.mark.parametrize("opt", ["O1", "O2"])
def test_conversion_lengths(conv_bins, opt):
    from ablation.analyzers.conversion_length_arm64 import ARM64ConversionLengthScanner
    ctx = BinaryContext.load_or_build(str(conv_bins[opt]))
    got = {(ctx.name(f.func_va), f.kind, f.severity, f.sink)
           for f in ARM64ConversionLengthScanner.from_context(ctx).scan()}
    assert got == {
        ("use_len", "sext_unchecked", "LOW", "memcpy"),          # 64->32 global, ldrsw
        ("signed_chk", "sext_signed_check", "MEDIUM", "memcpy"),
        ("masked", "masked_check", "MEDIUM", "memcpy"),
        ("byte_loop", "byte_counter_loop", "MEDIUM", ""),
        ("wire", "sext_signed_check", "HIGH", "memcpy"),         # recv field, b.gt only
        ("snp", "sext_signed_check", "MEDIUM", "snprintf"),      # libc unknown: note
    }                                  # signed_ok / unsigned_ok / masked_ok / byte_ok: silent


@pytest.mark.parametrize("libc,sev", [("musl", "LOW"), ("glibc", "MEDIUM")])
def test_snprintf_libc_awareness(conv_bins, libc, sev):
    from ablation.analyzers.conversion_length_arm64 import (
        ARM64ConversionLengthScanner, LIBC_SNPRINTF_REJECTS_HUGE)
    from ablation.analyzers.length_underflow_arm64 import ARM64LengthUnderflowScanner
    ctx = BinaryContext.load_or_build(str(conv_bins["O2"]))
    s = ARM64ConversionLengthScanner.from_context(ctx)
    s.snprintf_rejects_huge = LIBC_SNPRINTF_REJECTS_HUGE[libc]
    snp = [f for f in s.scan() if f.sink == "snprintf"]
    assert [f.severity for f in snp] == [sev]
    u = ARM64LengthUnderflowScanner.from_context(ctx)
    u.snprintf_rejects_huge = LIBC_SNPRINTF_REJECTS_HUGE[libc]
    und = [f for f in u.scan_unguarded() if f.call_label == "snprintf" and f.arg == "x1"]
    assert und and und[0].severity == {"musl": "LOW", "glibc": "HIGH"}[libc]
    u.snprintf_rejects_huge = None
    und = [f for f in u.scan_unguarded() if f.call_label == "snprintf" and f.arg == "x1"]
    assert und[0].severity == "MEDIUM" and "musl" in und[0].note


def test_cli_lengths(conv_bins):
    r = _cli("lengths", conv_bins["O2"], "--libc", "musl")
    assert "sext_signed_check" in r.stdout and "masked_check" in r.stdout
    assert "byte_counter_loop" in r.stdout


# ── finding quality: runtime-built formats, C++ heap shapes ──────────────────

_CXX_SRC = r'''
typedef unsigned long size_t;
extern int printf(const char *, ...);
extern int sprintf(char *, const char *, ...);
extern void *malloc(size_t);
extern void free(void *);
extern void use(void *);
extern void *op_new(size_t) __asm__("_Znwm");
extern void op_delete(void *) __asm__("_ZdlPv");
extern void obj_ctor(void *) __asm__("_ZN3ObjC1Ev");
extern void str_append(void *, const char *) __asm__("_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6appendEPKc");
extern const char *str_c_str(void *) __asm__("_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv");
void fmt_runtime(const char *user) { char f[64]; sprintf(f, "%s:%%d", user); printf(f, 1); }
void fmt_string(void *s) { printf(str_c_str(s)); }
void fmt_lit(void) { printf("plain %d\n", 3); }
int cxx_ctor(void) { char *b = op_new(16); obj_ctor(b); int r = b[24]; op_delete(b); return r; }
int cxx_grow(void) { char *b = op_new(8); str_append(b, "x"); int r = b[12]; op_delete(b); return r; }
int c_oob(void) { char *b = malloc(8); b[12] = 1; use(b); free(b); return 0; }
'''


@pytest.fixture(scope="module")
def cxx_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_CXX_SRC, variants={"O2": ["-O2"]})["O2"]


def test_runtime_format_buffers(cxx_so):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx = BinaryContext.load_or_build(str(cxx_so))
    got = {ctx.name(f.func_va): (f.provenance, f.severity)
           for f in ARM64FormatStringScanner.from_context(ctx).scan()}
    assert got == {"fmt_runtime": ("runtime_buffer", "MEDIUM"),
                   "fmt_string": ("runtime_buffer", "MEDIUM")}


def test_cxx_objects_not_constant_size(cxx_so):
    from ablation.analyzers.heap_scanner_arm64 import ARM64HeapScanner
    ctx = BinaryContext.load_or_build(str(cxx_so))
    got = {(ctx.name(f.func_va), f.kind) for f in
           ARM64HeapScanner.from_context(ctx).scan(include_int_overflow=False)}
    assert got == {("c_oob", "heap_overflow")}


# ── round 2: vendor sink wrappers ────────────────────────────────────────────

_WRAP_SRC = r'''
typedef unsigned long size_t; typedef __builtin_va_list va_list;
extern int vsnprintf(char *, size_t, const char *, va_list);
extern long recv(int, void *, size_t, int);
extern int system(const char *);
extern void *memcpy(void *, const void *, size_t);
extern void out(const char *);
extern void get_name(char *);
#define N __attribute__((noinline))
N void log_event(int kind, const char *fmt, ...) { char buf[0x600]; va_list ap; __builtin_va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); __builtin_va_end(ap); out(buf); (void)kind; }
N static void dbg(const char *msg) { log_event(2, msg); }
N static void run(const char *c) { system(c); }
N static void put(char *d, const char *s, size_t n) { memcpy(d, s, n); }
void h_ok(void) { log_event(1, "started %d", 3); dbg("plain"); }
void h_bad(int fd) { char req[256]; recv(fd, req, 255, 0); req[255] = 0; log_event(1, req); }
void h_dbg(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; dbg(r); }
N void h_mid(void) { char t[32]; get_name(t); log_event(1, t); }
void h_ext(const char *p) { log_event(2, p); }
void h_run(int fd) { char c[64]; recv(fd, c, 63, 0); c[63] = 0; run(c); }
void h_put(int fd, char *d) { unsigned char b[64]; recv(fd, b, 64, 0); put(d, (char *)b, b[0]); }
void route(void) { h_mid(); }
'''


@pytest.fixture(scope="module")
def wrap_bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_WRAP_SRC, variants={"O0": ["-O0"], "O2": ["-O2"]})


def _wctx(path):
    ctx = BinaryContext.load_or_build(str(path), force_rebuild=True)
    ctx.infer_wrappers(save=False)
    return ctx, symbols(path)


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_wrappers_inferred(wrap_bins, opt):
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx, t = _wctx(wrap_bins[opt])
    w = wrappers_of(ctx)
    le = w[t["log_event"]]
    assert le.kinds["format"] == ["x1"] and le.wraps == "vsnprintf"
    assert le.varargs and le.gr_offs == -0x30           # x2..x7 saved: 6 GP slots
    assert w[t["dbg"]].kinds["format"] == ["x0"] and w[t["dbg"]].via == "log_event"
    assert w[t["run"]].kinds["exec"] == ["x0"]
    assert w[t["put"]].kinds["length"] == ["x2"]


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_fmtstr_reports_wrapper_call_sites(wrap_bins, opt):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx, t = _wctx(wrap_bins[opt])
    got = {(f.func_va, f.provenance, f.severity) for f in ARM64FormatStringScanner.from_context(ctx).scan()}
    assert got == {
        (t["h_bad"], "tainted", "HIGH"),                 # recv data as log_event's format
        (t["h_dbg"], "tainted", "HIGH"),                 # ... through dbg -> log_event
        (t["h_mid"], "runtime_buffer", "MEDIUM"),
        (t["h_ext"], "arg_passthrough", "LOW"),          # exported wrapper, no caller
    }                                                    # nothing at vsnprintf inside log_event


def test_taint_wrapper_sinks(wrap_bins):
    from ablation.analyzers.taint_tracker_arm64 import ARM64TaintTracker
    ctx, t = _wctx(wrap_bins["O2"])
    got = {(f.func_va, f.sink_name, f.severity) for f in ARM64TaintTracker.from_context(ctx).run_interprocedural()}
    assert (t["h_run"], "run", "CRITICAL") in got
    assert (t["h_bad"], "log_event", "HIGH") in got


def test_sink_map_wrapper_override(wrap_bins, tmp_path):
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx, t = _wctx(wrap_bins["O2"])
    ctx.apply_sink_map({hex(t["h_mid"]): "format:x0"})
    w = wrappers_of(ctx)[t["h_mid"]]
    assert w.kinds["format"] == ["x0"] and w.source == "sink-map"


def test_cli_wrappers(wrap_bins):
    r = _cli("wrappers", wrap_bins["O2"])
    assert "log_event" in r.stdout and "varargs (6 GP" in r.stdout


# ── round 2: lengths through vendor wrappers, clamps, narrowing stores ───────

_LEN2_SRC = r'''
typedef unsigned long size_t;
extern long read(int, void *, size_t);
extern long recv(int, void *, size_t, int);
extern size_t strlen(const char *);
extern long get64(void);
extern int get(void);
#define N __attribute__((noinline))
N static long net_read(int fd, void *b, size_t n) { return read(fd, b, n); }   /* vendor wrapper */
int g_len32;
void store_len(const char *s) { g_len32 = strlen(s); }                         /* 64 -> 32 store */
void store_wire(int fd) { unsigned char b[8]; recv(fd, b, 8, 0); long v = *(long *)b; g_len32 = (int)v; }
void signed64(int fd, char *b) { long n = get64(); if (n > 64) return; net_read(fd, b, (int)n); }
void clamp(int fd, char *b) { int n = get(); int m = n < 64 ? n : 64; net_read(fd, b, m); }
void clamp_ok(int fd, char *b) { unsigned n = (unsigned)get(); unsigned m = n < 64 ? n : 64; net_read(fd, b, m); }
void local_ok(const char *s) { int n = (int)strlen(s); (void)n; }
'''


@pytest.fixture(scope="module")
def len2_bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_LEN2_SRC, variants={"O1": ["-O1"], "O2": ["-O2"]})


@pytest.mark.parametrize("opt", ["O1", "O2"])
def test_lengths_wrapper_clamp_narrowing(len2_bins, opt):
    from ablation.analyzers.conversion_length_arm64 import ARM64ConversionLengthScanner
    ctx, t = _wctx(len2_bins[opt])
    got = {(f.func_va, f.kind, f.severity, f.sink) for f in
           ARM64ConversionLengthScanner.from_context(ctx).scan()}
    assert got == {
        (t["store_len"], "narrowing_store", "LOW", ""),       # strlen result -> int global
        (t["store_wire"], "narrowing_store", "HIGH", ""),
        (t["signed64"], "sext_signed_check", "MEDIUM", "net_read"),   # vendor read wrapper
        (t["clamp"], "signed_clamp", "MEDIUM", "net_read"),
    }                                                          # clamp_ok / local_ok: silent


def test_lengths_json(len2_bins, tmp_path):
    out = tmp_path / "l.json"
    r = _cli("lengths", len2_bins["O2"], "--json", out)
    assert r.returncode == 0, r.stderr[-400:]
    data = json.loads(out.read_text())
    assert {d["kind"] for d in data["conversion"]} == {"narrowing_store", "sext_signed_check", "signed_clamp"}
    assert data["underflow"] == []


def test_overflow_sink_map_rtos_allocator(roles_img, tmp_path):
    binf, truth = roles_img
    fresh = tmp_path / "ov.bin"
    fresh.write_bytes(binf.read_bytes() + b"\0" * 12 + __import__("os").urandom(8))
    sm = tmp_path / "alloc.json"
    sm.write_text(json.dumps({hex(truth["memset"]): "pvPortMalloc"}))     # any VA, named
    r = _cli("overflow", fresh, "--raw", "arm64", "--base", hex(RAW_BASE), "--sink-map", sm,
             "--no-infer-sinks")
    assert "sink oracles resolved" not in r.stdout
    r = _cli("overflow", fresh, "--raw", "arm64", "--base", hex(RAW_BASE), "--no-infer-sinks")
    assert "matches any of:" in r.stdout and "malloc" in r.stdout


# ── round 2: FORTIFY variants ────────────────────────────────────────────────

_FORT_SRC = r"""
typedef unsigned long size_t; typedef __builtin_va_list va_list;
extern int __vsnprintf_chk(char *, size_t, int, size_t, const char *, va_list);
extern int vsnprintf(char *, size_t, const char *, va_list);
extern int __printf_chk(int, const char *, ...);
extern void *__memcpy_chk(void *, const void *, size_t, size_t);
extern long recv(int, void *, size_t, int);
extern void out(const char *);
#define N __attribute__((noinline))
N void flog(const char *fmt, ...) { char b[64]; va_list ap; __builtin_va_start(ap, fmt); __vsnprintf_chk(b, 64, 1, 64, fmt, ap); __builtin_va_end(ap); out(b); }
N void plog(const char *fmt, ...) { char b[64]; va_list ap; __builtin_va_start(ap, fmt); vsnprintf(b, 64, fmt, ap); __builtin_va_end(ap); out(b); }
void via_chk(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; flog(r); }
void via_plain(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; plog(r); }
void pchk1(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; __printf_chk(1, r); }
void pchk0(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; __printf_chk(0, r); }
void cpy_known(int fd) { unsigned char b[64]; char t[16]; recv(fd, b, 64, 0); __memcpy_chk(t, b, b[0], 16); out(t); }
void cpy_unknown(int fd, char *d) { unsigned char b[64]; recv(fd, b, 64, 0); __memcpy_chk(d, b, b[0], (size_t)-1); }
"""


@pytest.fixture(scope="module")
def fort_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_FORT_SRC, variants={"O2": ["-O2"]})["O2"]


def test_fortify_percent_n(fort_so):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx, t = _wctx(fort_so)
    notes = {f.func_va: f.details.get("fortify", "") for f in ARM64FormatStringScanner.from_context(ctx).scan()}
    assert "blocked by FORTIFY" in notes[t["via_chk"]] and "flag=1" in notes[t["via_chk"]]
    assert notes[t["via_plain"]].startswith("%n: not blocked (plain vsnprintf")
    assert "blocked by FORTIFY" in notes[t["pchk1"]]
    assert notes[t["pchk0"]].startswith("%n: not blocked (__printf_chk with flag 0")


def test_fortify_memcpy_chk_is_a_sink(fort_so):
    from ablation.analyzers.taint_tracker_arm64 import ARM64TaintTracker
    ctx, t = _wctx(fort_so)
    got = {f.func_va: f for f in ARM64TaintTracker.from_context(ctx).run_interprocedural()
           if f.sink_name == "memcpy"}
    assert got[t["cpy_known"]].severity == "MEDIUM" and "aborts when the copy exceeds 0x10" in got[t["cpy_known"]].note
    # __memcpy_chk(.., (size_t)-1) is folded to plain memcpy by the compiler: no FORTIFY note
    assert got[t["cpy_unknown"]].severity == "HIGH" and "FORTIFY" not in got[t["cpy_unknown"]].note


# ── round 2: conversion -> slot map, destination extent ──────────────────────

def test_format_slot_map(wrap_bins):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx, t = _wctx(wrap_bins["O2"])
    fs = {f.func_va: f for f in ARM64FormatStringScanner.from_context(ctx).scan()}
    bad = fs[t["h_bad"]]                       # char req[256] at sp, frame record above it
    slots = bad.details["slots"]
    assert "%1$..%6$: stale argument registers" in slots         # x2..x7 (6 GP slots)
    assert any(s.startswith("%7$..%38$: attacker bytes (recv)") for s in slots)
    assert any(s.startswith("%39$: [sp+0x100] saved x29") for s in slots)
    assert any(s.startswith("%40$: [sp+0x108] saved x30") for s in slots)
    assert bad.severity == "HIGH" and "defeats PIE" in bad.description
    # through the frameless dbg -> log_event tail call: h_dbg's frame is what is read
    assert any(s.startswith("%16$: [sp+0x48] saved x30") for s in fs[t["h_dbg"]].details["slots"])
    # a caller exists: K is the return address into it
    mid = fs[t["h_mid"]]
    import subprocess
    dis = subprocess.run(["llvm-objdump", "-d", "--no-show-raw-insn", str(wrap_bins["O2"])],
                         capture_output=True, text=True).stdout
    route = dis.split("<route>:")[1]
    site = next(int(l.split(":")[0], 16) for l in route.splitlines() if "<h_mid" in l and ("bl\t" in l or "b\t" in l))
    assert f"base+{site + 4:#x}" in mid.details["leak"]


_DEST_SRC = r"""
typedef unsigned long size_t;
extern long recv(int, void *, size_t, int);
extern int sscanf(const char *, const char *, ...);
extern void use(void *);
void five(int fd) { char req[64]; char m[5]; char rest[32]; recv(fd, req, 63, 0); req[63] = 0;
                    sscanf(req, "%s %31s", m, rest); use(m); use(rest); }
"""


@pytest.fixture(scope="module")
def dest_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_DEST_SRC, variants={"O2": ["-O2"]})["O2"]


def test_scanf_destination_extent(dest_so):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    from ablation.analyzers.frame_layout_arm64 import FrameLayout
    ctx, t = _wctx(dest_so)
    (f,) = [f for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.provenance == "unbounded_s"]
    dest = f.details["dest"]
    import re as _re
    n = int(_re.search(r": (\d+) bytes to the", dest).group(1))
    assert 5 <= n <= 8                          # char m[5] (allocation rounding), not the frame
    assert "bytes to saved x30" in dest


# ── round 2: path-sensitive resolution, unresolved verdict ───────────────────

_PATH_SRC = r"""
typedef unsigned long size_t;
extern void *calloc(size_t, size_t);
extern int printf(const char *, ...);
void two_paths(int c) { const char *f; if (c) f = calloc(1, 16); else f = "lit %d\n"; printf(f, 1); }
void same_lit(int c) { const char *f = "a %d\n"; if (c) f = "a %d\n"; printf(f, 1); }
"""


@pytest.fixture(scope="module")
def path_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_PATH_SRC, variants={"O0": ["-O0"]})["O0"]


def test_path_sensitive_slot_resolution(path_so):
    from ablation.analyzers.arm64_reg_annotator import ARM64RegAnnotator
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx, t = _wctx(path_so)
    ra = ARM64RegAnnotator.from_context(ctx)
    end = min(v for v in ctx.func_starts if v > t["two_paths"])
    lin = [c for c in ra.annotate_calls(t["two_paths"], end).calls if c.target_name == "printf"][0]
    cfg = [c for c in ra.annotate_calls_cfg(t["two_paths"], end).calls if c.target_name == "printf"][0]
    assert lin.args["x0"].kind == "string"                 # linear: last store in address order
    assert cfg.args["x0"].kind == "unknown" and "different definitions" in cfg.args["x0"].str_val
    fs = {f.func_va: f for f in ARM64FormatStringScanner.from_context(ctx).scan()}
    f = fs[t["two_paths"]]
    assert f.provenance == "unresolved" and f.details["bucket"] == "unresolved"
    assert "ret(calloc)" in f.description and "lit %d" in f.description
    assert t["same_lit"] not in fs                          # same literal on both paths


# ── round 2: reachability, registrations, config gates ───────────────────────

_REACH_SRC = r'''
typedef unsigned long size_t;
extern int printf(const char *, ...);
extern long recv(int, void *, size_t, int);
extern void http_register(const char *path, void (*h)(int));
struct cfg { char pad[0x50]; const char *errlog; };
extern struct cfg *g_cfg;
int g_debug;
#define H __attribute__((noinline, used, visibility("hidden")))
H void h_live(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; printf(r); }
H void h_dead(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; printf(r); }
H void h_dead2(int fd) { h_dead(fd); }
H void h_table(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; printf(r); }
struct route { const char *k; void (*h)(int); };
const struct route routes[] = { { "/tbl", h_table } };
H void err_log(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; if (g_cfg->errlog) printf(r); }
H void dbg_log(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; if (g_debug) printf(r); }
void set_debug(int v) { g_debug = v; }
void init(void) { http_register("/live", h_live); }
__attribute__((visibility("hidden"))) struct cfg *g_local;
H void err_local(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; if (g_local->errlog) printf(r); }
void serve(int fd) { err_log(fd); dbg_log(fd); err_local(fd); }
'''


@pytest.fixture(scope="module")
def reach_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_REACH_SRC,
                          variants={"O2": ["-O2", "-Wno-format-security"]})["O2"]


def test_reachability_and_gates(reach_so):
    from ablation.analyzers.reachability_arm64 import Reachability
    ctx, t = _wctx(reach_so)
    R = Reachability(ctx)
    regs = {(r.handler, r.key) for r in R.registrations()}
    assert (t["h_live"], "/live") in regs and (t["h_table"], "/tbl") in regs
    assert R.function_status(t["h_dead2"]) == "unreachable: no callers, address never taken"
    assert R.function_status(t["h_dead"]).startswith("unreachable: only called from unreachable code")
    assert R.function_status(t["h_live"]) is None and R.function_status(t["h_table"]) is None

    def site(fn):
        from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
        return next(f.call_va for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.func_va == t[fn])

    label, sup, _ = R.classify(t["dbg_log"], site("dbg_log"))
    assert not sup and "is non-zero" in label and "written at" in label         # set_debug writes it
    label, sup, _ = R.classify(t["err_log"], site("err_log"))
    assert not sup and "[g_cfg]->+0x50" in label and "imported global" in label   # may be set elsewhere
    label, sup, _ = R.classify(t["err_local"], site("err_local"))
    assert sup and "->+0x50" in label and "dead unless set from outside" in label
    label, sup, note = R.classify(t["h_dead"], site("h_dead"))
    assert sup and label.startswith("unreachable")
    assert "registered as '/live'" in R.classify(t["h_live"], site("h_live"))[2]


def test_cli_show_suppressed(reach_so):
    r = _cli("fmtstr", reach_so)
    assert "2 finding(s) suppressed (1 unreachable, 1 gated on a never-written global)" in r.stdout
    r = _cli("fmtstr", reach_so, "--show-suppressed")
    assert r.stdout.count("SUPPRESSED") == 2 and "unreachable: only called" in r.stdout
    r = _cli("reach", reach_so)
    assert "'/live'" in r.stdout and "'/tbl'" in r.stdout


# ── round 2: egress coupling ─────────────────────────────────────────────────

_EGRESS_SRC = r"""
typedef unsigned long size_t; typedef __builtin_va_list va_list; typedef struct FILE FILE;
extern FILE *fopen(const char *, const char *);
extern int fputs(const char *, FILE *);
extern int vsnprintf(char *, size_t, const char *, va_list);
extern size_t fread(void *, size_t, size_t, FILE *);
extern long send(int, const void *, size_t, int);
extern long recv(int, void *, size_t, int);
extern int open(const char *, int, ...);
extern long read(int, void *, size_t);
#define N __attribute__((noinline))
#define LOG "/var/log/httpd.log"
static FILE *g_log;
N void log_init(void) { g_log = fopen(LOG, "a"); }
N void log_event(int kind, const char *fmt, ...) { char buf[0x600]; va_list ap; __builtin_va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap); __builtin_va_end(ap); fputs(buf, g_log); (void)kind; }
N void serve_log(int sock) { int fd = open(LOG, 0); char b[512]; long n = read(fd, b, sizeof b); send(sock, b, n, 0); }
N void serve_motd(int sock) { FILE *f = fopen("/etc/motd", "r"); char b[64]; fread(b, 1, 64, f); send(sock, b, 64, 0); }
N void cfg_write(const char *v) { FILE *f = fopen("/etc/app.conf", "w"); fputs(v, f); }
void h_req(int sock) { char r[256]; recv(sock, r, 255, 0); r[255] = 0; log_event(1, r); }
"""


@pytest.fixture(scope="module")
def egress_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_EGRESS_SRC, variants={"O2": ["-O2"]})["O2"]


def test_egress_map(egress_so):
    from ablation.analyzers.egress_arm64 import EgressMap
    ctx, t = _wctx(egress_so)
    em = EgressMap(ctx).build()
    assert em.writers["/var/log/httpd.log"] == {t["log_init"], t["log_event"]}   # via the FILE* global
    assert em.readers["/var/log/httpd.log"] == {t["serve_log"]}
    assert em.writers["/etc/app.conf"] == {t["cfg_write"]} and "/etc/app.conf" not in em.readers
    chains = em.couple({t["log_event"]})
    assert len(chains) == 1 and '"/var/log/httpd.log" -> read in serve_log' in chains[0] and "send" in chains[0]
    assert em.couple({t["cfg_write"]}) == []                 # written, never served


def test_fmtstr_finding_carries_egress(egress_so):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx, t = _wctx(egress_so)
    (f,) = [f for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.func_va == t["h_req"]]
    assert f.severity == "HIGH" and "served back to clients" in f.description
    assert "write in log_event" in f.details["egress"] and "send" in f.details["egress"]
    r = _cli("egress", egress_so)
    assert "'/var/log/httpd.log'" in r.stdout and "EGRESS" in r.stdout


# ── round 3: size-taking vendor wrappers (clamped / TLS) ─────────────────────

_SIZEW_SRC = r"""
typedef unsigned long size_t;
extern long recv(int, void *, size_t, int);
extern int SSL_read(void *, void *, int);
extern int get(void);
extern void *g_ssl;
extern void use(void *);
#define N __attribute__((noinline))
struct conn { int fd; long avail; };
N long body_read(struct conn *c, char *buf, long want) { long n = want < c->avail ? want : c->avail; return recv(c->fd, buf, n, 0); }
N int tls_read(char *buf, int len) { return SSL_read(g_ssl, buf, len); }
N long field_len(struct conn *c, char *buf) { return recv(c->fd, buf, c->avail, 0); }
void h_body(struct conn *c) { char b[64]; int len = get(); if (len > 64) return; body_read(c, b, len); use(b); }
void h_tls(void) { char b[64]; int n = get(); int m = n < 64 ? n : 64; tls_read(b, m); use(b); }
"""


@pytest.fixture(scope="module")
def sizew_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_SIZEW_SRC, variants={"O2": ["-O2"]})["O2"]


def test_size_wrappers_derived(sizew_so):
    from ablation.analyzers.sink_wrappers import wrappers_of
    from ablation.analyzers.conversion_length_arm64 import ARM64ConversionLengthScanner
    ctx, t = _wctx(sizew_so)
    w = wrappers_of(ctx)
    assert w[t["body_read"]].kinds["length"] == ["x2"] and "length" in w[t["body_read"]].derived
    assert w[t["tls_read"]].kinds["length"] == ["x1"]            # SSL_read x2 <- param x1
    assert "length" not in w.get(t["field_len"], type("", (), {"kinds": {}})()).kinds   # a field, not a param
    got = {(f.func_va, f.kind, f.sink) for f in ARM64ConversionLengthScanner.from_context(ctx).scan()}
    assert (t["h_body"], "sext_signed_check", "body_read") in got
    assert (t["h_tls"], "signed_clamp", "tls_read") in got


# ── round 3: egress through vendor open helpers, coarse fallback ─────────────

_EGRESS2_SRC = r"""
typedef unsigned long size_t; typedef __builtin_va_list va_list; typedef struct FILE FILE;
extern FILE *fopen(const char *, const char *);
extern int fputs(const char *, FILE *);
extern int vsnprintf(char *, size_t, const char *, va_list);
extern int snprintf(char *, size_t, const char *, ...);
extern long send(int, const void *, size_t, int);
extern long recv(int, void *, size_t, int);
extern int open(const char *, int, ...);
extern long read(int, void *, size_t);
#define N __attribute__((noinline))
N FILE *xfopen(const char *path, const char *mode) { return fopen(path, mode); }
N int xopen_ro(const char *p) { return open(p, 0); }
N long http_send(int s, const void *b, size_t n) { return send(s, b, n, 0); }
static FILE *g_acc;
N void acc_init(void) { g_acc = xfopen("/var/log/access.log", "a"); }
N void gpFileAccessLog(int k, const char *fmt, ...) { char buf[0x200]; va_list ap; __builtin_va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap); __builtin_va_end(ap); fputs(buf, g_acc); (void)k; }
N void serve_access(int s) { int fd = xopen_ro("/var/log/access.log"); char b[256]; long n = read(fd, b, 256); http_send(s, b, n); }
void route(int s) { char r[128]; recv(s, r, 127, 0); r[127] = 0; gpFileAccessLog(1, r); }
/* runtime-built paths: only the shared literal links writer and reader */
N void rot_log(const char *fmt, ...) { char p[64]; char buf[256]; va_list ap; snprintf(p, 64, "%s.%d", "/var/log/rot.log", 1);
    FILE *f = fopen(p, "a"); __builtin_va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); __builtin_va_end(ap); fputs(buf, f); }
N void serve_rot(int s) { char q[64]; char b[256]; snprintf(q, 64, "%s.%d", "/var/log/rot.log", 1); int fd = open(q, 0);
    long n = read(fd, b, 256); send(s, b, n, 0); }
void route2(int s) { char r[128]; recv(s, r, 127, 0); r[127] = 0; rot_log(r); }
"""


@pytest.fixture(scope="module")
def egress2_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_EGRESS2_SRC, variants={"O2": ["-O2"]})["O2"]


def test_egress_vendor_open_helpers(egress2_so):
    from ablation.analyzers.egress_arm64 import EgressMap
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx, t = _wctx(egress2_so)
    w = wrappers_of(ctx)
    assert w[t["xfopen"]].kinds["open"] == ["x0"] and w[t["xfopen"]].open_mode == "param:x1"
    assert w[t["xopen_ro"]].open_mode == "flags:0x0"
    em = EgressMap(ctx).build()
    assert t["acc_init"] in em.writers["/var/log/access.log"]
    assert t["gpFileAccessLog"] in em.writers["/var/log/access.log"]      # via the FILE* global
    assert em.readers["/var/log/access.log"] == {t["serve_access"]}
    fs = {f.func_va: f for f in ARM64FormatStringScanner.from_context(ctx).scan()}
    eg = fs[t["route"]].details["egress"]
    assert '"/var/log/access.log" -> read in serve_access' in eg and "send" in eg
    eg2 = fs[t["route2"]].details["egress"]
    assert eg2.startswith("(coarse: shared path literal)") and "serve_rot" in eg2


# ── round 3: lengths false positives (zero-extended mask, truncation guard) ───

_LFP_SRC = r"""
typedef unsigned long size_t;
extern void *memcpy(void *, const void *, size_t);
extern int snprintf(char *, size_t, const char *, ...);
extern int get(void);
#define N __attribute__((noinline))
N int rec(const unsigned char *p, char *dst, int cap) {
    unsigned short raw; memcpy(&raw, p + 1, 2); unsigned n = __builtin_bswap16(raw);
    if ((int)(n & 0xffff) >= cap) return -1;
    memcpy(dst, p + 3, n & 0xffff); return 0; }
N int two_part(char *out, const char *a, const char *b) {
    int pre = get(); int rem = 350 - pre;
    int r = snprintf(out + pre, rem, "%s", a);
    if (r < 0 || rem <= r) return -1;
    int used = pre + r; int rem2 = 440 - used;
    return snprintf(out + used, rem2, "%s", b); }
N void masked_bug(char *d, const char *s) { unsigned n = (unsigned)get(); if ((n & 0xffff) > 64) return; memcpy(d, s, n); }
"""


@pytest.fixture(scope="module")
def lfp_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_LFP_SRC, variants={"O2": ["-O2"]})["O2"]


def test_lengths_false_positives_fixed(lfp_so, monkeypatch):
    from ablation.analyzers.conversion_length_arm64 import ARM64ConversionLengthScanner
    ctx, t = _wctx(lfp_so)
    got = {(f.func_va, f.kind) for f in ARM64ConversionLengthScanner.from_context(ctx).scan()}
    assert (t["masked_bug"], "masked_check") in got           # the real bug stays
    assert not any(fn == t["rec"] for fn, _k in got)            # ldurh + rev16: mask is a no-op
    assert (t["two_part"], "sext_signed_check") not in got      # truncation check bounds rem2


_CCMP_ASM = r"""
    .text
    .globl two_part_gcc
    .type two_part_gcc, %function
two_part_gcc:                       // gcc's snprintf truncation idiom (cmp #0; ccmp ...; b.le)
    stp   x29, x30, [sp, #-0x40]!
    mov   x29, sp
    stp   x19, x20, [sp, #0x10]
    stp   x21, x22, [sp, #0x20]
    str   x24, [sp, #0x30]
    mov   x20, x0
    mov   x22, x2
    bl    get
    mov   w19, w0                   // prefix_len
    mov   w21, #0x15e               // 350
    sub   w21, w21, w0              // remaining = 350 - prefix_len
    add   x0, x20, w19, sxtw
    adr   x2, fmt
    mov   x3, x22
    sxtw  x1, w21
    bl    vsnprintf
    cmp   w0, #0
    ccmp  w21, w0, #4, ge           // if ret >= 0: remaining vs ret
    b.le  1f                        // bail on error or truncation
    add   w19, w19, w0              // used = prefix_len + ret
    mov   w21, #0x1b8               // 440
    sub   w21, w21, w19
    sxtw  x24, w21                  // flagged by 2.88
    add   x0, x20, w19, sxtw
    adr   x2, fmt
    mov   x3, x22
    mov   x1, x24
    bl    vsnprintf
1:  ldr   x24, [sp, #0x30]
    ldp   x21, x22, [sp, #0x20]
    ldp   x19, x20, [sp, #0x10]
    ldp   x29, x30, [sp], #0x40
    ret
    .size two_part_gcc, .-two_part_gcc

    .globl rec_gcc
    .type rec_gcc, %function
rec_gcc:                            // 16-bit big-endian length: ldurh + rev16 + and #0xffff
    stp   x29, x30, [sp, #-0x20]!
    mov   x29, sp
    stp   x19, x20, [sp, #0x10]
    mov   x3, x0
    mov   w19, w2                   // capacity
    ldurh w2, [x0, #1]
    rev16 w2, w2
    and   w20, w2, #0xffff          // the checked value
    cmp   w20, w19
    b.lt  2f
    mov   w0, #-1
    b     3f
2:  mov   x0, x1
    add   x1, x3, #3
    and   x2, x2, #0xffff           // the memcpy length: the same value
    bl    memcpy
    mov   w0, #0
3:  ldp   x19, x20, [sp, #0x10]
    ldp   x29, x30, [sp], #0x20
    ret
    .size rec_gcc, .-rec_gcc
fmt: .asciz "%s"
"""


@pytest.fixture(scope="module")
def ccmp_so(tmp_path_factory):
    d = tmp_path_factory.mktemp("ccmp")
    (d / "t.s").write_text(_CCMP_ASM)
    so = d / "t.so"
    r = subprocess.run(["clang", "--target=aarch64-linux-gnu", "-shared", "-nostdlib", "-fuse-ld=lld",
                        str(d / "t.s"), "-o", str(so)], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(r.stderr[-200:])
    return so


def test_ccmp_truncation_guard(ccmp_so, monkeypatch):
    from ablation.analyzers import conversion_length_arm64 as cl
    ctx = BinaryContext.load_or_build(str(ccmp_so), force_rebuild=True)
    ctx.infer_wrappers(save=False)
    kinds = lambda: {(f.kind, f.sink) for f in cl.ARM64ConversionLengthScanner.from_context(ctx).scan()}
    monkeypatch.setattr(cl.ARM64ConversionLengthScanner, "_trunc_bounded", lambda self, labels: False)
    assert ("sext_signed_check", "vsnprintf") in kinds()         # 2.88 behaviour: flagged
    monkeypatch.undo()
    assert ("sext_signed_check", "vsnprintf") not in kinds()     # bounded by the truncation check


def test_zero_extended_mask_not_a_check(ccmp_so, monkeypatch):
    from ablation.analyzers import conversion_length_arm64 as cl
    ctx = BinaryContext.load_or_build(str(ccmp_so), force_rebuild=True)
    kinds = lambda: {(f.kind, f.sink) for f in cl.ARM64ConversionLengthScanner.from_context(ctx).scan()}
    monkeypatch.setattr(cl.ARM64ConversionLengthScanner, "_track_zext", lambda *a: None)
    assert ("masked_check", "memcpy") in kinds()                 # 2.88 behaviour: flagged
    monkeypatch.undo()
    assert ("masked_check", "memcpy") not in kinds()             # ldurh already zero-extended


_VALIDATOR_ASM = r"""
    .text
    .globl validate
    .type validate, %function
validate:                           // gcc layout: frame record at the bottom of a 0x90 frame
    stp   x29, x30, [sp, #-0x90]!
    mov   x29, sp
    adr   x1, vfmt
    add   x2, sp, #0x80             // first %s: 16 bytes to the frame top
    add   x3, sp, #0x70
    add   x4, sp, #0x60
    add   x5, sp, #0x50
    add   x6, sp, #0x40
    add   x7, sp, #0x30
    bl    sscanf
    ldp   x29, x30, [sp], #0x90
    ret
    .size validate, .-validate
    .globl route_v
    .type route_v, %function
route_v:
    stp   x29, x30, [sp, #-0x20]!   // caller's record at its sp: 8 more bytes to its x30
    mov   x29, sp
    bl    validate
    ldp   x29, x30, [sp], #0x20
    ret
    .size route_v, .-route_v
vfmt: .asciz "%s %s %s %s %s %s %s %s"
"""


@pytest.fixture(scope="module")
def validator_so(tmp_path_factory):
    d = tmp_path_factory.mktemp("val")
    (d / "t.s").write_text(_VALIDATOR_ASM)
    so = d / "t.so"
    r = subprocess.run(["clang", "--target=aarch64-linux-gnu", "-shared", "-nostdlib", "-fuse-ld=lld",
                        str(d / "t.s"), "-o", str(so)], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(r.stderr[-200:])
    return so


def test_widthless_scanf_budget_into_caller(validator_so, tmp_path):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx = BinaryContext.load_or_build(str(validator_so), force_rebuild=True)
    ctx.infer_wrappers(save=False)
    (f,) = [f for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.provenance == "unbounded_s"]
    assert "6 width-less conversions" in f.description          # x2..x7 (the rest go on the stack)
    assert f.details["budget"] == "24 bytes from %s (x2) to a saved x30"
    assert "caller's saved x30 (route_v)" in f.details["dest"]
    out = tmp_path / "l.json"
    r = _cli("lengths", validator_so, "--json", out)
    assert r.returncode == 0, r.stderr[-300:]
    assert json.loads(out.read_text())["unbounded"][0]["details"]["budget"].startswith("24 bytes")


_EXE_SRC = r"""
typedef unsigned long size_t;
extern int printf(const char *, ...);
extern long recv(int, void *, size_t, int);
extern void http_register(const char *path, void (*h)(int));
#define N __attribute__((noinline))
N void h_live(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; printf(r); }
N void h_dead(int fd) { char r[64]; recv(fd, r, 63, 0); r[63] = 0; printf(r); }
N void h_dead_caller(int fd) { h_dead(fd); }
void _start(void) { http_register("/live", h_live); for (;;) {} }
"""


def test_executable_exports_not_roots(tmp_path):
    (tmp_path / "e.c").write_text(_EXE_SRC)
    exe = tmp_path / "e"
    r = subprocess.run(["clang", "--target=aarch64-linux-gnu", "-O2", "-fPIE", "-pie", "-nostdlib",
                        "-fuse-ld=lld", "-Wno-format-security", "-Wl,--export-dynamic",
                        "-Wl,-dynamic-linker,/lib/ld-linux-aarch64.so.1",
                        "-Wl,--unresolved-symbols=ignore-all", "-o", str(exe), str(tmp_path / "e.c")],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.skip(r.stderr[-200:])
    from ablation.analyzers.reachability_arm64 import Reachability
    ctx = BinaryContext.load_or_build(str(exe), force_rebuild=True)
    t = symbols(exe)
    R = Reachability(ctx)
    assert R.function_status(t["h_live"]) is None                       # address taken
    assert R.function_status(t["h_dead_caller"]) == "unreachable: no callers, address never taken"
    assert R.function_status(t["h_dead"]).startswith("unreachable: only called from unreachable code")
    assert not R.exports_are_roots                                      # -rdynamic, no dl* imports


_WFMT_SRC = r"""
extern int printf(const char *, ...);
extern void set_fmt(char *);
char g_fmt[32] = "default %d\n";
void use_wfmt(void) { printf(g_fmt, 1); }
void use_lit(void) { printf("lit %d\n", 1); }
"""


@pytest.fixture(scope="module")
def wfmt_so(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_WFMT_SRC,
                          variants={"O2": ["-O2", "-Wno-format-security"]})["O2"]


def test_writable_format_and_explain(wfmt_so, wrap_bins):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx, t = _wctx(wfmt_so)
    s = ARM64FormatStringScanner.from_context(ctx)
    fs = {f.func_va: f for f in s.scan()}
    f = fs[t["use_wfmt"]]
    assert f.provenance == "writable_format" and "initially 'default %d" in f.description
    assert t["use_lit"] not in fs
    lit_site = next(c.site_va for _f, c in s._site_calls.values() if _f == t["use_lit"])
    assert "literal (not reported)" in s.explain(lit_site)
    ctx2, t2 = _wctx(wrap_bins["O2"])
    s2 = ARM64FormatStringScanner.from_context(ctx2)
    s2.scan()
    bad_site = next(f.call_va for f in s2.scan() if f.func_va == t2["h_bad"])
    ex = s2.explain(bad_site)
    assert "wrapper of vsnprintf" in ex and "taint at the call: recv" in ex
    inner = s2.explain(next(site for site, (fv, c) in s2._site_calls.items()
                            if fv == t2["log_event"] and c.target_name == "vsnprintf"))
    assert "forwarding inside the format wrapper log_event" in inner
    r = _cli("fmtstr", wrap_bins["O2"], "--explain", hex(bad_site))
    assert "taint at the call: recv" in r.stdout and "reach:" in r.stdout


# ── round 4: the abl_repro corpus (lengths through a vendor body_read; nine %s) ─

_REPRO_SRC = r'''
/* ablation lengths / width-less %s corpus (synthetic; rebuilt from the round-4 description).
 * The size sink is a vendor wrapper: body_read(conn, dst, len) saturates len up to INT_MAX
 * and forwards it to read(). Nothing calls read() directly with an attacker length. */
typedef unsigned long size_t;
typedef long ssize_t;
#define N __attribute__((noinline))
#define INT_MAX 0x7fffffff
struct conn { int fd; long avail; };
N ssize_t read(int fd, void *buf, size_t n) { register long x0 __asm__("x0") = fd; register void *x1 __asm__("x1") = buf;
    register size_t x2 __asm__("x2") = n; register long x8 __asm__("x8") = 63;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x1), "r"(x2), "r"(x8) : "memory"); return x0; }
N int sscanf(const char *s, const char *f, ...) { __asm__ volatile("" :: "r"(s), "r"(f) : "memory"); return 0; }
N long hdr_int(struct conn *c) { long v; read(c->fd, &v, sizeof v); return v; }
N ssize_t body_read(struct conn *c, char *dst, long len) { if (len > INT_MAX) len = INT_MAX; return read(c->fd, dst, (size_t)len); }
N void sink(void *p) { __asm__ volatile("" :: "r"(p) : "memory"); }
int g_truncated_len;
volatile long g_consumed;
/* 1: zero-only cbz, then a SIGNED cap check, then sxtw into the wrapper */
N void case1(struct conn *c) { char buf[6001]; int len = (int)hdr_int(c); if (len == 0) return; if (len >= 6001) return; body_read(c, buf, len); sink(buf); }
/* 2: 64 -> 32 truncating store to a global, used as a size in another function */
N void case2_store(struct conn *c) { long total = hdr_int(c); g_truncated_len = (int)total; }
N void case2_use(struct conn *c) { char buf[256]; body_read(c, buf, g_truncated_len); sink(buf); }
/* 3: signed csel min that keeps a negative, guarded only by cbz */
N void case3(struct conn *c, int cap) { char buf[256]; long rem = g_truncated_len - g_consumed; int n = rem < cap ? (int)rem : cap;
    if (n == 0) return; body_read(c, buf, n); sink(buf); }
/* 4 (negative control): reject <= 0, then a true clamp to the cap */
N void case4(struct conn *c, int cap) { char buf[256]; int n = (int)hdr_int(c); if (n <= 0) return; if (n > cap) n = cap;
    body_read(c, buf, n); sink(buf); }
/* 5: nine width-less %s: destinations 1-6 in x2-x7, 7-9 on the stack (the smallest) */
N void case5(struct conn *c) { char req[256]; char a[64], b2[48], c3[32], d[24], e[16], f[12], g[8], h[5], i[4];
    read(c->fd, req, 255); req[255] = 0;
    sscanf(req, "%s %s %s %s %s %s %s %s %s", a, b2, c3, d, e, f, g, h, i);
    sink(a); sink(b2); sink(c3); sink(d); sink(e); sink(f); sink(g); sink(h); sink(i); }
void _start(void) { struct conn c = { 0, 0 }; case1(&c); case2_store(&c); case2_use(&c); case3(&c, 64); case4(&c, 64); case5(&c); for (;;) {} }
'''


@pytest.fixture(scope="module")
def repro_bin(tmp_path_factory):
    d = tmp_path_factory.mktemp("repro")
    (d / "abl_repro.c").write_text(_REPRO_SRC)
    out = d / "abl_repro"
    r = subprocess.run(["clang", "--target=aarch64-linux-gnu", "-nostdlib", "-static", "-O2",
                        "-fno-stack-protector", "-fomit-frame-pointer", "-fuse-ld=lld",
                        str(d / "abl_repro.c"), "-o", str(out)], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(r.stderr[-200:])
    return out


def test_repro_corpus_lengths(repro_bin):
    from ablation.analyzers.conversion_length_arm64 import ARM64ConversionLengthScanner
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx = BinaryContext.load_or_build(str(repro_bin), force_rebuild=True)
    ctx.infer_wrappers(save=False)
    e = ctx.exports
    w = wrappers_of(ctx)
    assert w[e["body_read"]].kinds["length"] == ["x2"]
    assert e["case1"] not in w                         # a callee's result is not a parameter
    got = {(f.func_va, f.kind, f.sink) for f in ARM64ConversionLengthScanner.from_context(ctx).scan()}
    assert got == {
        (e["case1"], "sext_signed_check", "body_read"),     # cbz is not a lower bound
        (e["case2_use"], "narrowing_store", "body_read"),   # linked to case2_store's 32-bit store
        (e["case3"], "signed_clamp", "body_read"),          # min(rem, cap) keeps a negative
    }                                                       # case4 (<= 0 rejected) and body_read's
                                                            # own INT_MAX saturation: silent


def test_repro_corpus_nine_dests(repro_bin):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx = BinaryContext.load_or_build(str(repro_bin), force_rebuild=True)
    ctx.infer_wrappers(save=False)
    (f,) = [f for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.provenance == "unbounded_s"]
    assert "9 width-less conversions" in f.description
    dest = f.details["dest"]
    for reg in ("x2", "x7", "stack+0x0", "stack+0x8", "stack+0x10"):
        assert f"-> {reg} [sp+" in dest
    assert f.details["smallest"] == "%s (stack+0x10): 4 bytes before the next stack object"


_SPCOPY_ASM = r"""
    .text
    .globl val_spcopy
    .type val_spcopy, %function
val_spcopy:                         // destinations addressed through a copy of sp
    stp   x29, x30, [sp, #-0x40]!
    mov   x29, sp
    str   x19, [sp, #0x10]
    add   x19, sp, #0x18
    adr   x1, sfmt
    add   x2, x19, #0x0             // char[5] at sp+0x18
    add   x3, x19, #0x8             // char[13] at sp+0x20
    bl    sscanf
    ldr   x19, [sp, #0x10]
    ldp   x29, x30, [sp], #0x40
    ret
    .size val_spcopy, .-val_spcopy
sfmt: .asciz "%s %s"
"""


def test_widthless_dest_through_sp_copy(tmp_path):
    (tmp_path / "t.s").write_text(_SPCOPY_ASM)
    so = tmp_path / "t.so"
    r = subprocess.run(["clang", "--target=aarch64-linux-gnu", "-shared", "-nostdlib", "-fuse-ld=lld",
                        str(tmp_path / "t.s"), "-o", str(so)], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(r.stderr[-200:])
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx = BinaryContext.load_or_build(str(so), force_rebuild=True)
    (f,) = [f for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.provenance == "unbounded_s"]
    assert "2 width-less conversions" in f.description and "[sp+0x18]" in f.description
