"""PE (Windows) ARM64 support in BinaryContext, FuncProfiler, ARM64PathSolver;
PE machine detection; GOT-indirect call edges on ELF arm64.

PE fixtures are built with clang + lld-link + llvm-dlltool (an import library for
msvcrt from a .def, no Windows SDK needed) and skip when those are missing.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.binary_context import BinaryContext

_DEF = "LIBRARY msvcrt.dll\nEXPORTS\nsystem\nstrcpy\nputs\n"

# strcpy: dllimport -> inline `adrp x8 ; ldr x8, [IAT] ; blr x8`
# puts:   plain extern -> import thunk `adrp x16 ; ldr x16, [IAT] ; br x16`, reached by bl
# system: dllimport, tail-called from a leaf -> `ldr x1, [IAT] ; br x1`
_SRC = r'''
__declspec(dllimport) int system(const char *);
__declspec(dllimport) char *strcpy(char *, const char *);
int puts(const char *);
static const char *gmsg = "global-pointer-string";
__declspec(noinline) int helper(char *d, const char *s) { strcpy(d, s); return puts("helper-called"); }
__declspec(noinline) static int tail(const char *c) { return system(c); }
__declspec(dllexport) int handler(char *buf) {
    char tmp[64];
    helper(tmp, buf);
    puts("handler-string-xref");
    puts(gmsg);
    return tail("/bin/sh -c reboot");
}
'''

_TARGETS = {
    "arm64": ("aarch64-pc-windows-msvc", "arm64", "arm64"),
    "x86_64": ("x86_64-pc-windows-msvc", "x64", "i386:x86-64"),
    "x86_32": ("i686-pc-windows-msvc", "x86", "i386"),
}


@pytest.fixture(scope="module")
def dlls(tmp_path_factory):
    if not all(shutil.which(t) for t in ("clang", "lld-link", "llvm-dlltool")):
        pytest.skip("clang + lld-link + llvm-dlltool required for PE fixtures")
    d = tmp_path_factory.mktemp("pe")
    (d / "imp.def").write_text(_DEF)
    (d / "w.c").write_text(_SRC)
    out = {}
    for arch, (triple, machine, dlltool_m) in _TARGETS.items():
        lib, obj, dll = d / f"msvcrt_{arch}.lib", d / f"w_{arch}.obj", d / f"w_{arch}.dll"
        steps = [
            ["llvm-dlltool", "-m", dlltool_m, "-d", str(d / "imp.def"), "-l", str(lib)],
            ["clang", f"--target={triple}", "-O2", "-c", str(d / "w.c"), "-o", str(obj)],
            ["lld-link", "/dll", "/noentry", f"/machine:{machine}", str(obj), str(lib), f"/out:{dll}"],
        ]
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                pytest.skip(f"PE build failed: {r.stderr.strip()[:200]}")
        out[arch] = dll
    return out


def _local_callees(ctx, va):
    """Functions va calls inside the image (import thunks excluded), in VA order."""
    return sorted({t for t, _ in ctx.callees_of(va) if t in ctx.func_starts and t not in ctx.plt})


@pytest.mark.parametrize("arch", ["arm64", "x86_64", "x86_32"])
def test_pe_machine_detected(dlls, arch):
    # lief >= 0.15 moved MACHINE_TYPES under Header; every PE used to read as x86_64
    assert BinaryContext.build(str(dlls[arch])).arch == arch


def test_pe_arm64_func_starts_pdata_plus_leaf(dlls):
    ctx = BinaryContext.build(str(dlls["arm64"]))
    handler = ctx.exports["handler"]
    assert handler in ctx.func_starts
    # helper (non-leaf, .pdata) and tail (leaf, BL target only) are both found
    assert len(_local_callees(ctx, handler)) == 2


def test_pe_arm64_import_thunk_named(dlls):
    ctx = BinaryContext.build(str(dlls["arm64"]))
    text_lo = ctx.base_va + 0x1000
    thunks = {v for va, v in ctx.plt.items() if text_lo <= va < text_lo + 0x1000}
    assert thunks == {"puts"}


def test_pe_arm64_call_graph_imports(dlls):
    ctx = BinaryContext.build(str(dlls["arm64"]))
    handler = ctx.exports["handler"]
    helper, tail = _local_callees(ctx, handler)
    assert [c for c, _ in ctx.callers_of("strcpy")] == [helper]       # blr via IAT
    assert [c for c, _ in ctx.callers_of("system")] == [tail]         # br via IAT (tail)
    assert sorted(c for c, _ in ctx.callers_of("puts")) == [helper, handler, handler]


def test_pe_arm64_string_xrefs(dlls):
    ctx = BinaryContext.build(str(dlls["arm64"]))
    by_text = {s: va for va, s in ctx.strings.items()}
    handler = ctx.exports["handler"]
    assert ctx.funcs_referencing_string(by_text["handler-string-xref"]) == [handler]
    assert ctx.funcs_referencing_string(by_text["global-pointer-string"]) == [handler]
    assert len(ctx.funcs_referencing_string(by_text["/bin/sh -c reboot"])) == 1
    assert len(ctx.funcs_referencing_string(by_text["helper-called"])) == 1


def test_pe_arm64_func_profiler(dlls):
    from ablation.analyzers.func_profiler import FuncProfiler
    ctx = BinaryContext.build(str(dlls["arm64"]))
    fp = FuncProfiler.from_context(ctx)
    handler = ctx.exports["handler"]
    helper, tail = _local_callees(ctx, handler)

    h = fp.profile(helper)
    assert [c.site.target_name for c in h.sink_calls] == ["strcpy"]
    assert any(c.site.target_name == "puts" and c.site.args["x0"].str_val == "helper-called"
               for c in h.calls)
    putss = {c.site.args["x0"].str_val for c in fp.profile(handler).calls
             if c.site.target_name == "puts"}
    assert putss == {"handler-string-xref", "global-pointer-string"}
    t = fp.profile(tail)
    assert [(c.site.target_name, c.site.args["x0"].str_val) for c in t.sink_calls] == [
        ("system", "/bin/sh -c reboot")]


def test_pe_arm64_path_solver(dlls):
    pytest.importorskip("z3")
    from ablation.analyzers.path_solver_arm64 import ARM64PathSolver
    ctx = BinaryContext.build(str(dlls["arm64"]))
    handler = ctx.exports["handler"]
    tail = _local_callees(ctx, handler)[1]
    ps = ARM64PathSolver.from_context(ctx, handler)
    bl_tail = next(i.address for b in ps.cfg.blocks.values() for i in b.insns
                   if i.mnemonic == "bl" and i.imm(0) == tail)
    assert ps.solve_first_sat(bl_tail) is not None


# ── ELF arm64: call through a GOT slot (-fno-plt shape) ─────────────────────

_GOT_SRC = r'''
extern int system(const char *);
__attribute__((naked)) int via_got(const char *c) {
    __asm__(
        "stp x29, x30, [sp, #-16]!\n"
        "adrp x8, :got:system\n"
        "ldr x8, [x8, :got_lo12:system]\n"
        "blr x8\n"
        "ldp x29, x30, [sp], #16\n"
        "ret\n");
}
'''


def test_elf_got_indirect_call_edge(tmp_path_factory):
    from a64_fixtures import build_variants
    so = build_variants(tmp_path_factory, src_text=_GOT_SRC, variants={"got": []})["got"]
    ctx = BinaryContext.build(str(so))
    assert [c for c, _ in ctx.callers_of("system")] == [ctx.exports["via_got"]]
