"""ARM64FormatStringScanner and ARM64LengthUnderflowScanner (and the shared tag base).

Fixtures are cross-built with clang + ld.lld at -O0 and -O2; tests skip when the
toolchain is missing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
from ablation.analyzers.length_underflow_arm64 import ARM64LengthUnderflowScanner

from a64_fixtures import build_variants

_OPTS = {"O0": ["-O0", "-Wno-format-security"], "O2": ["-O2", "-Wno-format-security"]}

# ── format strings ───────────────────────────────────────────────────────────

_FMT_SRC = r'''
typedef unsigned long size_t;
typedef struct FILE FILE;
extern FILE *stderr;
extern long recv(int, void *, size_t, int);
extern int printf(const char *, ...);
extern int fprintf(FILE *, const char *, ...);
extern int snprintf(char *, size_t, const char *, ...);
extern void syslog(int, const char *, ...);
extern const char *get_msg(void);
void lit(int x) { printf("value=%d\n", x); fprintf(stderr, "err %d\n", x); syslog(3, "x=%d", x); }
void passthru(const char *fmt) { printf(fmt); }
void wire(int fd) { char b[256]; recv(fd, b, sizeof b - 1, 0); b[255] = 0; syslog(3, b); }
void viaret(void) { printf(get_msg()); }
void argonly(int fd) { char b[256]; recv(fd, b, sizeof b, 0); char o[300]; snprintf(o, sizeof o, "got %s", b); printf("%s\n", o); }
'''


@pytest.fixture(scope="module")
def fmt_bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_FMT_SRC, variants=_OPTS)


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_format_string_verdicts(fmt_bins, opt):
    s = ARM64FormatStringScanner.from_path_full(str(fmt_bins[opt]))
    got = {s._syms.get(f.func_va): (f.callee, f.provenance, f.severity) for f in s.scan()}
    assert got == {
        "passthru": ("printf", "arg_passthrough", "LOW"),    # exported wrapper, no caller here
        "wire": ("syslog", "tainted", "HIGH"),
        "viaret": ("printf", "call_result", "MEDIUM"),
    }


_WRAP_SRC = r"""
typedef unsigned long size_t;
extern long recv(int, void *, size_t, int);
extern int printf(const char *, ...);
extern int sprintf(char *, const char *, ...);
extern int vsprintf(char *, const char *, __builtin_va_list);
extern int sscanf(const char *, const char *, ...);
extern int scanf(const char *, ...);
extern void sink(char *);
__attribute__((noinline)) static void log_lit(const char *fmt) { printf(fmt); }
__attribute__((noinline)) static void log_any(const char *fmt) { printf(fmt); }
__attribute__((noinline)) static void log_mid(int lvl, const char *f) { log_any(f); }
void log_api(const char *fmt) { printf(fmt); }
void use_lit(void) { log_lit("hello\n"); log_lit("bye\n"); }
void use_wire(int fd) { char b[64]; recv(fd, b, 63, 0); b[63] = 0; log_mid(1, b); }
void sp_name(const char *user) { char o[32]; sprintf(o, "user=%s", user); sink(o); }
void sp_bounded(const char *user) { char o[32]; sprintf(o, "user=%.16s", user); sink(o); }
void sp_wire(int fd) { char b[128]; char o[32]; recv(fd, b, 127, 0); b[127] = 0; sprintf(o, "%d:%s", 5, b); sink(o); }
void ss_wire(int fd) { char b[128]; char n[16]; int x; recv(fd, b, 127, 0); b[127] = 0; sscanf(b, "%d %s", &x, n); sink(n); }
void ss_width(int fd) { char b[128]; char n[16]; recv(fd, b, 127, 0); b[127] = 0; sscanf(b, "%15s", n); sink(n); }
void sc_stdin(void) { char n[16]; scanf("%*d %[^\n]", n); sink(n); }
void vs(char *d, const char *f, __builtin_va_list ap) { char o[16]; vsprintf(o, "%s", ap); sink(o); }
"""


@pytest.fixture(scope="module")
def wrap_bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_WRAP_SRC, variants=_OPTS)


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_wrappers_judged_by_callers_and_unbounded_s(wrap_bins, opt):
    s = ARM64FormatStringScanner.from_path_full(str(wrap_bins[opt]))
    got = {s._syms.get(f.func_va): (f.callee, f.provenance, f.severity) for f in s.scan()}
    assert got == {
        "log_api": ("printf", "arg_passthrough", "LOW"),      # exported, no caller in binary
        "log_any": ("printf", "tainted", "HIGH"),             # recv data via log_mid
        "sp_name": ("sprintf", "unbounded_s", "MEDIUM"),
        "sp_wire": ("sprintf", "unbounded_s", "HIGH"),
        "ss_wire": ("sscanf", "unbounded_s", "HIGH"),
        "sc_stdin": ("scanf", "unbounded_s", "HIGH"),
        "vs": ("vsprintf", "unbounded_s", "MEDIUM"),
    }                                                         # log_lit, sp_bounded, ss_width: silent


def test_unbounded_conversions():
    from ablation.analyzers.format_string_scanner_arm64 import unbounded_conversions as uc
    assert uc("%d %s", "print") == [(1, "%s")]
    assert uc("%.16s %*s %.*s %s", "print") == [(2, "%*s"), (5, "%s")]   # "*" consumes an arg
    assert uc("100%% %ls", "print") == [(0, "%ls")]
    assert uc("%d %s", "scan") == [(1, "%s")]
    assert uc("%15s %*s %ms %[a-z] %3[^]x]", "scan") == [(2, "%[a-z]")]
    assert uc("%[]x]", "scan") == [(0, "%[]x]")]


def test_format_string_needs_context(fmt_bins):
    from ablation.analyzers.taint_tracker_arm64 import _load_elf
    data, base, ts, te, syms = _load_elf(str(fmt_bins["O2"]))
    with pytest.raises(ValueError):
        ARM64FormatStringScanner(data, base, ts, te, syms).scan()


# ── length underflow ─────────────────────────────────────────────────────────

_UF_SRC = r'''
typedef unsigned long size_t;
typedef unsigned short u16;
extern long recv(int, void *, size_t, int);
extern void *memcpy(void *, const void *, size_t);
extern int parse_field(const unsigned char *, u16);
extern unsigned char out[4096];
void wrap32(int fd) { unsigned char b[512]; recv(fd, b, 512, 0); u16 len = *(u16 *)(b + 2); memcpy(out, b + 16, (unsigned)(len - 16)); }
int trunc16(int fd) { unsigned char b[512]; recv(fd, b, 512, 0); u16 len = *(u16 *)(b + 2); return parse_field(b + 16, (u16)(len - 16)); }
void trunc_len(int fd) { unsigned char b[512]; recv(fd, b, 512, 0); u16 len = *(u16 *)(b + 2); unsigned n = (u16)(len - 8); memcpy(out, b + 8, n); }
void guarded(int fd) { unsigned char b[512]; recv(fd, b, 512, 0); u16 len = *(u16 *)(b + 2); if (len < 16) return; memcpy(out, b + 16, len - 16); }
void consts(int fd) { unsigned char b[512]; recv(fd, b, 512, 0); memcpy(out, b + 16, 64 - 16); }
void notwire(unsigned n) { memcpy(out, out + 16, n - 16); }
'''


@pytest.fixture(scope="module")
def uf_bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_UF_SRC, variants=_OPTS)


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_underflow_verdicts(uf_bins, opt):
    s = ARM64LengthUnderflowScanner.from_path_full(str(uf_bins[opt]))
    got = {s._syms.get(f.func_va): (f.call_label, f.arg, f.severity, f.guarded) for f in s.scan()}
    assert got["wrap32"] == ("memcpy", "x2", "HIGH", False)
    assert got["trunc16"] == ("parse_field", "x1", "MEDIUM", False)
    assert got["trunc_len"][:3] == ("memcpy", "x2", "HIGH") and not got["trunc_len"][3]
    assert got["guarded"][3] is True
    assert "consts" not in got and "notwire" not in got


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_underflow_truncation_recorded(uf_bins, opt):
    s = ARM64LengthUnderflowScanner.from_path_full(str(uf_bins[opt]))
    f = next(f for f in s.scan() if s._syms.get(f.func_va) == "trunc_len")
    assert f.sub_const == 8 and f.truncate_va > f.sub_va
    assert "uxth" in f.fmt()


def test_unconsumed_subs_is_not_a_guard(uf_bins):
    # clang -O0 emits `subs w1, w9, #0x10` whose flags nothing reads (trunc16)
    s = ARM64LengthUnderflowScanner.from_path_full(str(uf_bins["O0"]))
    f = next(f for f in s.scan() if s._syms.get(f.func_va) == "trunc16")
    assert not f.guarded


# ── ARM64RegAnnotator stack slots ────────────────────────────────────────────

def test_annotator_spilled_arg_keeps_provenance():
    from ablation.analyzers.arm64_reg_annotator import ARM64RegAnnotator
    from ablation.analyzers.insn_arm64 import make_insn
    from ablation.analyzers.reg_annotator import AnnotationResult, RegVal
    from types import SimpleNamespace
    ra = ARM64RegAnnotator(SimpleNamespace(segments=[], relocations=[]), {}, {0x2000: "printf"})
    regs = {"x0": RegVal.arg(0, "x0")}
    res = AnnotationResult(func_va=0x1000, func_end_va=0x1100)
    lines = ["str x0, [sp, #8]", "mov x0, #0", "ldr x0, [sp, #8]", "bl #0x2000",
             "str x0, [sp, #16]", "sub sp, sp, #0x20", "ldr x1, [sp, #16]", "bl #0x2000"]
    for i, line in enumerate(lines):
        mn, _, ops = line.partition(" ")
        ra._step(make_insn(0x1000 + 4 * i, mn, ops), regs, res)
    assert res.calls[0].args["x0"].display().startswith("arg0_entry")
    assert res.calls[1].args["x1"].kind != "ret"        # sp moved: stale slot dropped
