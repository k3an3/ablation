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
void h_mid(void) { char t[32]; get_name(t); log_event(1, t); }
void h_ext(const char *p) { log_event(2, p); }
void h_run(int fd) { char c[64]; recv(fd, c, 63, 0); c[63] = 0; run(c); }
void h_put(int fd, char *d) { unsigned char b[64]; recv(fd, b, 64, 0); put(d, (char *)b, b[0]); }
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
