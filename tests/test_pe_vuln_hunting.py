"""PE vuln hunting: x86-64 (lifted) and AArch64 Windows images through the Insn-model passes
(taint, fmtstr, lengths, wrappers, heap, overflow, egress, reach), Windows API tables, the
UCRT's __stdio_common_* shapes, and the x86-64 lifter itself.

Fixtures: clang + lld-link + llvm-dlltool (import libraries from .def files, no SDK);
skip without them. Every function is exported so tests can name it.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.binary_context import BinaryContext

_TARGETS = {"x64": ("x86_64-pc-windows-msvc", "i386:x86-64"), "arm64": ("aarch64-pc-windows-msvc", "arm64")}

_DEFS = {
    "msvcrt": "LIBRARY msvcrt.dll\nEXPORTS\nsprintf\nvsnprintf\nsscanf\nmemcpy\nmalloc\nstrcpy\nsystem\nprintf\n",
    "ws2_32": "LIBRARY ws2_32.dll\nEXPORTS\nrecv\nsend\n",
    "kernel32": ("LIBRARY kernel32.dll\nEXPORTS\nReadFile\nCreateProcessA\nHeapAlloc\nHeapFree\n"
                 "GetProcessHeap\nlstrcpyA\nLoadLibraryA\nCreateFileA\nWriteFile\nCloseHandle\nWinExec\n"),
    "wininet": "LIBRARY wininet.dll\nEXPORTS\nInternetReadFile\n",
    "user32": "LIBRARY user32.dll\nEXPORTS\nwsprintfA\n",
    "ucrt": ("LIBRARY api-ms-win-crt-stdio-l1-1-0.dll\nEXPORTS\n__stdio_common_vsprintf\n"
             "__stdio_common_vsscanf\n"),
}

_HDR = r'''
typedef unsigned long long size_t;
typedef long long ssize_t;
typedef __builtin_va_list va_list;
typedef void *HANDLE;
#define IMP __declspec(dllimport)
#define N __declspec(noinline) __declspec(dllexport)
'''

# the abl_repro shapes on Windows: lengths through a vendor body_read over recv, nine
# width-less %s, a varargs format wrapper, plus direct recv -> sink paths
_CORE_SRC = _HDR + r'''
IMP int recv(unsigned long long, char *, int, int);
IMP int send(unsigned long long, const char *, int, int);
IMP int sprintf(char *, const char *, ...);
IMP int vsnprintf(char *, size_t, const char *, va_list);
IMP int sscanf(const char *, const char *, ...);
IMP void *memcpy(void *, const void *, size_t);
IMP void *malloc(size_t);
IMP char *strcpy(char *, const char *);
IMP int system(const char *);
IMP int printf(const char *, ...);
IMP int WinExec(const char *, unsigned);

N int body_read(unsigned long long s, char *dst, size_t len)
{ if (len > 0x7fffffff) len = 0x7fffffff; if (!len) return 0; return recv(s, dst, (int)len, 0); }
struct req { long long content_length; unsigned long long s; };
static int g_truncated_len, g_consumed;
N int case1(struct req *r, char *buf) { int n = (int)r->content_length; if (!n) return -1;
  if (n > 6000) return -1; return body_read(r->s, buf, (size_t)(ssize_t)n); }
N void case2a(struct req *r) { g_truncated_len = (int)r->content_length; }
N int case3(unsigned long long s, char *buf, int want) { int rem = g_truncated_len - g_consumed;
  int n = rem < want ? rem : want; if (n == 0) return 0; return body_read(s, buf, (size_t)(ssize_t)n); }
N int case4(struct req *r, char *buf) { int n = (int)r->content_length; if (n <= 0) return -1;
  if (n > 6000) n = 6000; return body_read(r->s, buf, (size_t)n); }
N int case5(const char *q) { char d1[32], d2[32], d3[32], d4[32], d5[32], d6[32], d7[8], d8[5], d9[4];
  int n = sscanf(q, "%s %s %s %s %s %s %s %s %s", d1, d2, d3, d4, d5, d6, d7, d8, d9);
  if (n != 9) return -1; return d9[0] + d1[0]; }
N void access_log(int kind, const char *fmt, ...) { char line[300]; va_list ap; __builtin_va_start(ap, fmt);
  vsnprintf(line, sizeof line, fmt, ap); __builtin_va_end(ap); send(0, line, 300, kind); }
N void case6a(void) { access_log(1, "fixed literal"); }
N void case6b(const char *p) { char copy[256]; memcpy(copy, p, 255); copy[255] = 0; access_log(2, copy); }
N void case6c(const char **slot) { access_log(2, *slot); }
N void net_handler(unsigned long long s) { char pkt[512]; char out[64]; char cmd[128];
  int n = recv(s, pkt, 511, 0); pkt[n] = 0; printf(pkt); sprintf(out, "user=%s", pkt);
  strcpy(cmd, pkt); system(cmd); WinExec(pkt, 0); }
N void *alloc_n(unsigned long long s) { unsigned int cnt; recv(s, (char *)&cnt, 4, 0); return malloc(cnt * 16); }
N int under(unsigned long long s, char *dst) { char hdr[8]; recv(s, hdr, 8, 0); unsigned len = *(unsigned *)hdr;
  memcpy(dst, hdr, len - 4); return 0; }
N void entry(unsigned long long s) { static struct req r; static char buf[8192]; static const char *slot = "x";
  case1(&r, buf); case2a(&r); case3(r.s, buf, 1); case4(&r, buf); case5(buf);
  case6a(); case6b(buf); case6c(&slot); net_handler(s); alloc_n(s); under(s, buf); }
'''

# Windows API sources / sinks and what <stdio.h> compiles into an image under the UCRT
_WIN_SRC = _HDR + r'''
IMP int __stdio_common_vsprintf(unsigned long long, char *, size_t, const char *, void *, va_list);
IMP int __stdio_common_vsscanf(unsigned long long, const char *, size_t, const char *, void *, va_list);
IMP int ReadFile(HANDLE, void *, unsigned, unsigned *, void *);
IMP int CreateProcessA(const char *, char *, void *, void *, int, unsigned, void *, const char *, void *, void *);
IMP void *HeapAlloc(HANDLE, unsigned, size_t);
IMP int HeapFree(HANDLE, unsigned, void *);
IMP HANDLE GetProcessHeap(void);
IMP char *lstrcpyA(char *, const char *);
IMP void *LoadLibraryA(const char *);
IMP HANDLE CreateFileA(const char *, unsigned, unsigned, void *, unsigned, unsigned, HANDLE);
IMP int WriteFile(HANDLE, const void *, unsigned, unsigned *, void *);
IMP int CloseHandle(HANDLE);
IMP int InternetReadFile(HANDLE, void *, unsigned, unsigned *);
IMP int wsprintfA(char *, const char *, ...);
IMP int recv(unsigned long long, char *, int, int);
IMP int send(unsigned long long, const char *, int, int);

N int u_sprintf(char *b, const char *f, ...) { va_list a; __builtin_va_start(a, f);
  int r = __stdio_common_vsprintf(1, b, (size_t)-1, f, 0, a); __builtin_va_end(a); return r; }
N int u_snprintf(char *b, size_t n, const char *f, ...) { va_list a; __builtin_va_start(a, f);
  int r = __stdio_common_vsprintf(2, b, n, f, 0, a); __builtin_va_end(a); return r; }
N int u_sscanf(const char *s, const char *f, ...) { va_list a; __builtin_va_start(a, f);
  int r = __stdio_common_vsscanf(1, s, (size_t)-1, f, 0, a); __builtin_va_end(a); return r; }
N void file_to_copy(HANDLE h) { char in[512], out[32]; unsigned got; ReadFile(h, in, 511, &got, 0); in[got] = 0;
  lstrcpyA(out, in); u_sprintf(out, "%s", in); }
N void bounded(HANDLE h) { char in[512], out[32]; unsigned got; ReadFile(h, in, 511, &got, 0); in[got] = 0;
  u_snprintf(out, sizeof out, "%s", in); }
N void net_to_exec(HANDLE h) { char cmd[256]; unsigned got; InternetReadFile(h, cmd, 255, &got); cmd[got] = 0;
  CreateProcessA(0, cmd, 0, 0, 0, 0, 0, 0, 0, 0); LoadLibraryA(cmd); }
N int scan_fields(const char *q) { char a[16], b[8]; return u_sscanf(q, "%s %s", a, b) + a[0] + b[0]; }
N void user_fmt(unsigned long long s) { char pkt[256], out[64]; int n = recv(s, pkt, 255, 0); pkt[n] = 0;
  wsprintfA(out, pkt); u_sprintf(out, pkt); }
N void *heap_ovf(unsigned long long s) { unsigned cnt; recv(s, (char *)&cnt, 4, 0);
  return HeapAlloc(GetProcessHeap(), 0, cnt * 24); }
N int heap_uaf(void) { char *p = HeapAlloc(GetProcessHeap(), 0, 64); HeapFree(GetProcessHeap(), 0, p); return p[3]; }
N void log_req(const char *line) { unsigned w; HANDLE f = CreateFileA("C:\\ProgramData\\app\\req.log", 4, 0, 0, 4, 0, 0);
  WriteFile(f, line, 64, &w, 0); CloseHandle(f); }
N void serve_log(unsigned long long s) { char buf[512]; unsigned got;
  HANDLE f = CreateFileA("C:\\ProgramData\\app\\req.log", 0x80000000, 1, 0, 3, 0, 0);
  ReadFile(f, buf, 512, &got, 0); send(s, buf, (int)got, 0); }
N void entry(unsigned long long s, HANDLE h) { file_to_copy(h); bounded(h); net_to_exec(h);
  scan_fields("a b"); user_fmt(s); heap_ovf(s); heap_uaf(); log_req("x"); serve_log(s); }
'''


def _build(tmp_path_factory, name: str, src: str):
    if not all(shutil.which(t) for t in ("clang", "lld-link", "llvm-dlltool")):
        pytest.skip("clang + lld-link + llvm-dlltool required for PE fixtures")
    d = tmp_path_factory.mktemp(name)
    (d / "c.c").write_text(src)
    out = {}
    for arch, (triple, mach) in _TARGETS.items():
        libs = []
        for lib, text in _DEFS.items():
            (d / f"{lib}.def").write_text(text)
            lp = d / f"{lib}_{arch}.lib"
            subprocess.run(["llvm-dlltool", "-m", mach, "-d", str(d / f"{lib}.def"), "-l", str(lp)], check=True)
            libs.append(str(lp))
        obj, dll = d / f"c_{arch}.obj", d / f"c_{arch}.dll"
        r = subprocess.run(["clang", f"--target={triple}", "-O2", "-Wno-format-security", "-c", str(d / "c.c"),
                            "-o", str(obj)], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(r.stderr[-300:])
        r = subprocess.run(["lld-link", "/dll", "/noentry", "/nodefaultlib", f"/machine:{arch}", str(obj),
                            *libs, f"/out:{dll}"], capture_output=True, text=True)
        if r.returncode:
            pytest.skip(r.stdout[-300:] + r.stderr[-300:])
        out[arch] = dll
    return out


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    return _build(tmp_path_factory, "pecore", _CORE_SRC)


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    return _build(tmp_path_factory, "pewin", _WIN_SRC)


def _ctx(path):
    ctx = BinaryContext.load_or_build(str(path), force_rebuild=True)
    ctx.infer_wrappers(save=False)
    return ctx


def _cli(*a):
    return subprocess.run([sys.executable, "-m", "ablation.cli", *map(str, a)], capture_output=True,
                          text=True, cwd=Path(__file__).parent.parent)


ARCHES = ["x64", "arm64"]

# ── loader ─────────────────────────────────────────────────────────────────────


def test_x64_pe_function_starts_and_iat_edges(core):
    ctx = _ctx(core["x64"])
    e = ctx.exports
    starts = set(ctx.func_starts)
    assert set(e.values()) <= starts                     # leaf functions (no .pdata) included
    inner = [s for s in starts if any(s != v and v < s < v + 8 for v in e.values())]
    assert not inner, [hex(s) for s in inner]            # no starts inside a function (after ret)
    labels = {(a, lbl) for a, _b, lbl in ctx.call_edges}
    assert (e["net_handler"], "WinExec") in labels      # call [rip+__imp_WinExec]
    assert (e["body_read"], "recv") in labels           # tail jmp [rip+__imp_recv]
    assert "memcpy" in ctx.plt.values() and any(va in starts for va, n in ctx.plt.items() if n == "memcpy")


# ── taint ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", ARCHES)
def test_taint_windows_and_libc_sinks(core, win, arch):
    from ablation.analyzers.taint_tracker_arm64 import ARM64TaintTracker
    ctx = _ctx(core[arch])
    got = {(f.func_va, f.sink_name) for f in ARM64TaintTracker.from_context(ctx).run_interprocedural(depth=4)}
    e = ctx.exports
    for s in ("sprintf", "strcpy", "system", "WinExec"):
        assert (e["net_handler"], s) in got
    assert (e["under"], "memcpy") in got
    ctx = _ctx(win[arch])
    found = ARM64TaintTracker.from_context(ctx).run_interprocedural(depth=4)
    got = {(ctx.name(f.func_va), f.sink_name, f.source_name) for f in found}
    assert ("file_to_copy", "lstrcpyA", "ReadFile") in got
    assert ("net_to_exec", "CreateProcessA", "InternetReadFile") in got
    assert ("net_to_exec", "LoadLibraryA", "InternetReadFile") in got
    assert ("user_fmt", "wsprintfA", "recv") in got


# ── lengths / wrappers ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", ARCHES)
def test_lengths_through_vendor_wrapper(core, arch):
    from ablation.analyzers.conversion_length_arm64 import ARM64ConversionLengthScanner
    from ablation.analyzers.length_underflow_arm64 import ARM64LengthUnderflowScanner
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx = _ctx(core[arch])
    e = ctx.exports
    assert wrappers_of(ctx)[e["body_read"]].kinds["length"] == ["x2"]
    got = {(f.func_va, f.kind) for f in ARM64ConversionLengthScanner.from_context(ctx).scan()}
    assert (e["case1"], "sext_signed_check") in got     # test / cbz is not a lower bound
    assert (e["case3"], "signed_clamp") in got          # signed min keeps a negative
    assert not any(fv == e["case4"] for fv, _k in got)  # n <= 0 rejected, then clamped
    assert not any(fv == e["body_read"] for fv, _k in got)
    under = ARM64LengthUnderflowScanner.from_context(ctx).scan()
    assert any(f.func_va == e["under"] and f.call_label == "memcpy" and not f.guarded for f in under)


@pytest.mark.parametrize("arch", ARCHES)
def test_nine_widthless_dests_win64_stack_args(core, arch):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    ctx = _ctx(core[arch])
    e = ctx.exports
    found = ARM64FormatStringScanner.from_context(ctx).scan()
    (f,) = [f for f in found if f.provenance == "unbounded_s" and f.func_va == e["case5"]]
    assert "9 width-less conversions" in f.description
    for k in range(5 if arch == "x64" else 2):            # Win64: 2 in r8/r9, 7 past [rsp+0x20]
        assert f"-> stack+{8 * k:#x} [sp+" in f.details["dest"]
    assert f.details["smallest"].endswith("4 bytes before the next stack object")
    prov = {f.func_va: f.provenance for f in found if f.provenance != "unbounded_s"}
    assert prov.get(e["case6b"]) == "runtime_buffer"
    assert prov.get(e["case6c"]) == "memory_load"
    assert e["case6a"] not in prov


def test_win64_varargs_slot_map(core):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx = _ctx(core["x64"])
    e = ctx.exports
    w = wrappers_of(ctx)[e["access_log"]]
    assert w.kinds["format"] == ["x1"] and w.varargs and w.gr_offs == -16   # r8 / r9 home-spilled
    f = next(f for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.func_va == e["case6b"])
    slots = f.details["slots"]
    assert "%1$..%2$: stale argument registers" in slots                     # r8, r9
    assert any("saved return address: return address into the caller" in s for s in slots)
    assert "%3$" in slots[2]                                                 # [rsp+0x20] next


# ── Windows / UCRT shapes ──────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", ARCHES)
def test_ucrt_wrappers_and_formats(win, arch):
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx = _ctx(win[arch])
    e = ctx.exports
    w = wrappers_of(ctx)
    assert w[e["u_sprintf"]].kinds.get("dest") == ["x0"]       # count -1: unbounded
    assert "dest" not in w[e["u_snprintf"]].kinds              # real count: bounded
    assert w[e["u_sscanf"]].kinds.get("scanfmt") == ["x1"]
    found = ARM64FormatStringScanner.from_context(ctx).scan()
    unb = {ctx.name(f.func_va) for f in found if f.provenance == "unbounded_s"}
    assert unb == {"file_to_copy", "scan_fields"}              # not bounded()
    fmt = {(ctx.name(f.func_va), f.callee) for f in found if f.provenance != "unbounded_s"}
    assert fmt == {("user_fmt", "wsprintfA"), ("user_fmt", "u_sprintf")}   # "%s" in .rdata is a literal
    notes = {f.callee: f.details["fortify"] for f in found if f.provenance != "unbounded_s"}
    assert notes["wsprintfA"].startswith("%n: not supported by wsprintfA")
    assert notes["u_sprintf"].startswith("%n: blocked by default (UCRT")


@pytest.mark.parametrize("arch", ARCHES)
def test_heap_overflow_egress_windows(win, arch):
    from ablation.analyzers.egress_arm64 import EgressMap
    from ablation.analyzers.heap_scanner_arm64 import ARM64HeapScanner
    from ablation.analyzers.intoverflow_scanner_arm64 import ARM64IntOverflowScanner
    ctx = _ctx(win[arch])
    e = ctx.exports
    heap = ARM64HeapScanner.from_context(ctx).scan()
    assert any(f.kind == "uaf" and f.func_va == e["heap_uaf"] and f.alloc_sym == "HeapAlloc" for f in heap)
    ovf = ARM64IntOverflowScanner.from_context(ctx).scan()
    assert any(f.func_va == e["heap_ovf"] for f in ovf)
    em = EgressMap(ctx).build()
    path = "C:\\ProgramData\\app\\req.log"
    assert em.writers[path] == {e["log_req"]}                 # GENERIC write access flag 4
    assert em.readers[path] == {e["serve_log"]}


def test_cli_x64_pe_names_x86_registers(core):
    r = _cli("wrappers", core["x64"])
    assert "body_read" in r.stdout and "[length r8 (derived)]" in r.stdout
    r = _cli("lengths", core["x64"])
    assert "x86-64 conversion-length scan" in r.stdout and " x2" not in r.stdout
    for c in ("reach", "egress", "overflow", "heap", "taint", "fmtstr"):
        r = _cli(c, core["x64"])
        assert r.returncode == 0 and "supports" not in r.stdout, (c, r.stdout, r.stderr)


# ── x86-64 SysV (ELF) through the same passes ────────────────────────────────


def test_sysv_x64_pushed_stack_args(tmp_path):
    if not shutil.which("clang"):
        pytest.skip("clang required")
    from test_arm64_audit_gaps import _REPRO_SRC
    (tmp_path / "r.c").write_text(_REPRO_SRC)
    out = tmp_path / "r"
    r = subprocess.run(["clang", "--target=x86_64-linux-gnu", "-nostdlib", "-static", "-O2", "-fno-stack-protector",
                        "-fomit-frame-pointer", "-fuse-ld=lld", str(tmp_path / "r.c"), "-o", str(out)],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.skip(r.stderr[-200:])
    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    from ablation.analyzers.sink_wrappers import wrappers_of
    ctx = _ctx(out)
    e = ctx.exports
    assert wrappers_of(ctx)[e["body_read"]].kinds["length"] == ["x2"]       # rdx
    (f,) = [f for f in ARM64FormatStringScanner.from_context(ctx).scan() if f.provenance == "unbounded_s"]
    assert "9 width-less conversions" in f.description                      # 5 pushed before the call
    assert f.details["smallest"] == "%s (stack+0x20): 4 bytes before the next stack object"


# ── the lifter ──────────────────────────────────────────────────────────────────


def _lift(hexcode: str, abi="win64", iat=()):
    from ablation.analyzers.lift_x86_64 import X64Lifter
    return [f"{i.mnemonic} {', '.join(map(str, i.ops))}".strip()
            for i in X64Lifter(abi, iat).lift(bytes.fromhex(hexcode), 0x1000)][1:]


def test_lifter_idioms():
    assert _lift("85c07400") == ["cmp w9, #0", "b.eq #0x1004"]                        # test eax,eax; je
    assert _lift("4183c0fc") == ["sub w2, w2, #4"]                                   # add r8d, -4
    assert _lift("0f4cc1") == ["csel w9, w0, w9, lt"]                                # cmovl eax, ecx
    assert _lift("4863c1") == ["sxtw x9, w0"]                                        # movsxd rax, ecx
    assert _lift("31c0") == ["mov w9, #0"]                                           # xor eax, eax
    assert _lift("ff15fa0f0000", iat=[0x2000]) == ["bl #0x2000", "mov x9, x0"]       # call [rip+IAT]
    assert _lift("c3") == ["mov x0, x9", "ldr x30, [sp], #0x8", "ret"]
    assert _lift("b838120000e800000000482be0") [-1] == "sub sp, sp, #0x1238"          # __chkstk frame
    assert _lift("4889d7", abi="sysv") == ["mov x0, x2"]                             # mov rdi, rdx (SysV)
    from ablation.analyzers.lift_x86_64 import X64Lifter
    first = X64Lifter().lift(bytes.fromhex("c3"), 0x1000)[0]
    assert (first.mnemonic, first.size, str(first.ops[1])) == ("str", 0, "[sp, #-0x8]!")
