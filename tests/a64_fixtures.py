"""Shared AArch64 ELF fixtures: cross-built with clang + ld.lld, no libc needed.

Tests that use these skip when no AArch64-capable clang/lld is on PATH.
"""

import shutil
import subprocess

import pytest

SRC = r'''
extern int system(const char *);
extern char *strcpy(char *, const char *);
extern int puts(const char *);
__attribute__((visibility("hidden"))) const char *gmsg = "global-pointer-string";
__attribute__((noinline)) int helper(char *d, const char *s) { strcpy(d, s); return puts("helper-called"); }
__attribute__((noinline)) static int tail(const char *c) { return system(c); }
__attribute__((noinline)) static int framed(int x) { char b[300]; b[x & 255] = 1; puts(b); return b[3]; }
int handler(char *buf) {
    char tmp[64];
    helper(tmp, buf);
    puts("handler-string-xref");
    puts(gmsg);
    framed(7);
    return tail("/bin/sh -c reboot");
}
'''

NOEH = ["-fno-asynchronous-unwind-tables", "-fno-unwind-tables"]
PAC = ["-mbranch-protection=standard", "-Wl,-z,force-bti"]
VARIANTS = {
    "plain": [],
    "bti_pac": PAC,          # 24-byte PLT stubs, paciasp / bti c prologues
    "noeh": NOEH,            # no .eh_frame: func starts come from the scanner
    "noeh_bti_pac": NOEH + PAC,
}


def have_toolchain() -> bool:
    return bool(shutil.which("clang") and shutil.which("ld.lld"))


def build_variants(tmp_path_factory, src_text=None, variants=None):
    if not have_toolchain():
        pytest.skip("clang + ld.lld required to build AArch64 fixtures")
    d = tmp_path_factory.mktemp("a64")
    src = d / "t.c"
    src.write_text(src_text or SRC)
    out = {}
    for name, flags in (variants or VARIANTS).items():
        so = d / f"{name}.so"
        cmd = ["clang", "--target=aarch64-linux-gnu", "-O2", "-fPIC", "-shared",
               "-nostdlib", "-fuse-ld=lld", *flags, str(src), "-o", str(so)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"AArch64 cross-build failed: {r.stderr.strip()[:200]}")
        out[name] = so
    return out


def symbols(path):
    """Ground truth from the unstripped fixture's .symtab: {name: va}."""
    import lief
    b = lief.parse(str(path))
    return {s.name: s.value for s in b.symbols if s.name and s.value}
