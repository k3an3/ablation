"""Raw (headerless) firmware images: BinaryContext.from_raw + raw-aware scanners.

Fixtures: clang + ld.lld with a linker script, flattened with llvm-objcopy -O binary.
Skip without the toolchain.
"""

import shutil
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ablation.analyzers import raw_image as ri
from ablation.analyzers.binary_context import BinaryContext

pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("clang", "ld.lld", "llvm-objcopy", "llvm-nm")),
    reason="needs clang, ld.lld, llvm-objcopy, llvm-nm")

# ── fixtures ─────────────────────────────────────────────────────────────────

_CM_SRC = r'''
typedef unsigned int u32; typedef unsigned long size_t;
extern u32 _estack;
void Reset_Handler(void); void Default_Handler(void); void USART1_IRQHandler(void);
__attribute__((section(".isr_vector"), used))
const void *vectors[16 + 40] = {
    &_estack, Reset_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, 0, 0, 0, 0, Default_Handler, Default_Handler, 0,
    Default_Handler, Default_Handler,
    [16 + 37] = USART1_IRQHandler,
};
__attribute__((noinline)) void *memcpy(void *d, const void *s, size_t n) { char *a = d; const char *b = s; while (n--) *a++ = *b++; return d; }
__attribute__((noinline)) char *strcpy(char *d, const char *s) { char *r = d; while ((*d++ = *s++) != 0) {} return r; }
volatile u32 *const UART_DR = (u32 *)0x40011004;
__attribute__((noinline)) long recv(int port, void *buf, size_t n, int fl) {
    unsigned char *p = buf; for (size_t i = 0; i < n; i++) p[i] = (unsigned char)*UART_DR;
    return (long)*UART_DR; }
__attribute__((noinline)) void log_msg(const char *s) { while (*s) *UART_DR = *s++; }
__attribute__((noinline)) void handle_cmd(void) {
    unsigned char pkt[64]; char name[16];
    long n = recv(1, pkt, sizeof pkt, 0);
    log_msg("cmd: rx packet");
    strcpy(name, (char *)pkt + 4);
    log_msg(name);
    memcpy(name, pkt, n);                    /* length from the wire */
}
void USART1_IRQHandler(void) { handle_cmd(); }
void Default_Handler(void) { for (;;) {} }
void Reset_Handler(void) { log_msg("boot: rtos firmware v1.2"); for (;;) handle_cmd(); }
'''

_CM_LD = '''
MEMORY { FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K  RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 128K }
_estack = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS {
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text : { *(.text*) } > FLASH
  .rodata : { *(.rodata*) } > FLASH
  .data : { *(.data*) } > RAM AT > FLASH
  .bss : { *(.bss*) } > RAM
}
'''

_A64_SRC = r'''
typedef unsigned long size_t;
volatile unsigned int *const UART = (unsigned int *)0x09000000;
__attribute__((noinline)) void *memcpy(void *d, const void *s, size_t n) { char *a = d; const char *b = s; while (n--) *a++ = *b++; return d; }
__attribute__((noinline)) char *strcpy(char *d, const char *s) { char *r = d; while ((*d++ = *s++) != 0) {} return r; }
__attribute__((noinline)) long recv(int fd, void *buf, size_t n, int fl) { unsigned char *p = buf; for (size_t i = 0; i < n; i++) p[i] = (unsigned char)*UART; return (long)n; }
__attribute__((noinline)) int printf(const char *f, ...) { while (*f) *UART = *f++; return 0; }
__attribute__((noinline)) void handle_cmd(int fd) {
    char pkt[128]; char name[16];
    recv(fd, pkt, sizeof pkt, 0);
    printf("cmd: rx packet\n");
    strcpy(name, pkt + 4);
    printf(pkt);
}
void _start(void) { printf("boot: a64 rtos v3\n"); for (;;) handle_cmd(0); }
'''

_A64_LD = '''
ENTRY(_start)
SECTIONS { . = 0x40080000; .text : { *(.text._start) *(.text*) } .rodata : { *(.rodata*) }
           .data : { *(.data*) } .bss : { *(.bss*) } }
'''

A64_BASE = 0x40080000


def _build(tmp: Path, name: str, src: str, ld: str, cflags):
    (tmp / f"{name}.c").write_text(src)
    (tmp / f"{name}.ld").write_text(ld)
    elf, binf, nm = tmp / f"{name}.elf", tmp / f"{name}.bin", tmp / f"{name}.nm"
    r = subprocess.run(["clang", *cflags, "-O1", "-ffreestanding", "-fno-builtin", "-nostdlib",
                        "-fuse-ld=lld", f"-Wl,-T,{tmp / f'{name}.ld'}", "-w",
                        "-o", str(elf), str(tmp / f"{name}.c")], capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"cross build failed: {r.stderr[-300:]}")
    subprocess.run(["llvm-objcopy", "-O", "binary", str(elf), str(binf)], check=True)
    nm.write_text(subprocess.run(["llvm-nm", str(elf)], check=True, capture_output=True,
                                 text=True).stdout)
    truth = {}
    for line in nm.read_text().splitlines():
        addr, typ, sym = line.split()
        if typ in "tT":
            truth[sym] = int(addr, 16) & ~1
    return binf, nm, truth


@pytest.fixture(scope="module")
def cm(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cm"), "fw", _CM_SRC, _CM_LD,
                  ["--target=thumbv7em-none-eabi", "-mcpu=cortex-m4"])


@pytest.fixture(scope="module")
def a64(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("a64"), "a64", _A64_SRC, _A64_LD,
                  ["--target=aarch64-none-elf", "-ffunction-sections"])


# ── raw_image helpers ────────────────────────────────────────────────────────

def test_parse_symbols_formats(tmp_path):
    f = tmp_path / "syms"
    f.write_text("08000100 T memcpy\n"
                 "08000200 D some_table\n"               # nm data symbol: not a function
                 "0x08000300 uart_rx   # comment\n"
                 "handle_cmd = 0x08000401\n"
                 "garbage line here\n")
    assert ri.parse_symbols(str(f)) == {"memcpy": 0x08000100, "uart_rx": 0x08000300,
                                        "handle_cmd": 0x08000401}


def test_thumb_bl_decode():
    # bl from 0x1000 to 0x1100: offset 0xFC -> hw1 = F000, hw2 = F87E
    data = b"\x00" * 0x10 + struct.pack("<HH", 0xF000, 0xF87E) + b"\x00" * 0x200
    assert (0x1010, 0x1110) in ri.thumb_bl_targets(data, 0x1000)


def test_infer_cortexm_base(cm):
    binf, _nm, truth = cm
    data = binf.read_bytes()
    assert ri.infer_cortexm_base(data) == 0x08000000
    vec = ri.cortexm_vectors(data, 0x08000000)
    assert vec["Reset_Handler"] & ~1 == truth["Reset_Handler"]
    assert vec["Default_Handler"] & ~1 == truth["Default_Handler"]
    assert vec["IRQ37_Handler"] & ~1 == truth["USART1_IRQHandler"]


# ── Cortex-M (Thumb-2) ───────────────────────────────────────────────────────

def test_cortexm_context_stripped(cm):
    binf, _nm, truth = cm
    ctx = BinaryContext.from_raw(str(binf), "cortex-m", force_rebuild=True)
    assert ctx.base_va == 0x08000000 and ctx.arch == "arm32" and ctx.raw["arch"] == "thumb"
    assert ctx.name(truth["Reset_Handler"]) == "Reset_Handler"
    assert ctx.name(truth["USART1_IRQHandler"]) == "IRQ37_Handler"
    for fn in ("strcpy", "recv", "log_msg", "handle_cmd"):      # all called: found by BL scan
        assert truth[fn] in ctx.func_starts, fn
    assert set(ctx.func_starts) <= set(truth.values())
    assert truth["handle_cmd"] in ctx.thumb_funcs
    # MOVW/MOVT string xrefs, no '@boot...' glued variant
    assert "boot: rtos firmware v1.2" in ctx.strings.values()
    assert not any(s.endswith("boot: rtos firmware v1.2") and s[0] != "b" for s in ctx.strings.values())
    xr = {ctx.strings[s]: {ctx.func_containing(v) for v in vs} for s, vs in ctx._str_xref_idx.items()}
    assert xr["boot: rtos firmware v1.2"] == {truth["Reset_Handler"]}
    assert xr["cmd: rx packet"] == {truth["handle_cmd"]}
    edges = {(a, b) for a, b, _ in ctx.call_edges}
    assert (truth["handle_cmd"], truth["recv"]) in edges
    assert (truth["USART1_IRQHandler"], truth["handle_cmd"]) in edges


def test_cortexm_symbols_and_scanners(cm):
    binf, nm, truth = cm
    ctx = BinaryContext.from_raw(str(binf), "thumb", base=0x08000000, symbols=str(nm),
                                 force_rebuild=True)
    assert set(ctx.func_starts) == set(truth.values())
    assert ctx.name(truth["memcpy"]) == "memcpy"
    assert ctx.image_regions() == [(0x08000000, binf.read_bytes())]

    from ablation.analyzers.func_profiler import FuncProfiler
    prof = FuncProfiler.from_context(ctx).profile(va=truth["handle_cmd"])
    names = [c.site.target_name for c in prof.calls]
    assert names == ["recv", "log_msg", "strcpy", "log_msg", "memcpy"]
    assert any(c.site.target_name == "log_msg" and "cmd: rx packet" in c.fmt() for c in prof.calls)

    from ablation.analyzers.taint_tracker_arm32 import ARM32TaintTracker
    tr = ARM32TaintTracker.from_context(ctx)
    hits = {(f.sink_name, f.func_va) for f in tr.run_interprocedural()}
    assert ("memcpy", truth["handle_cmd"]) in hits


# ── AArch64 ──────────────────────────────────────────────────────────────────

def test_arm64_context_stripped(a64):
    binf, _nm, truth = a64
    ctx = BinaryContext.from_raw(str(binf), "aarch64", base=A64_BASE, force_rebuild=True)
    assert ctx.arch == "arm64" and ctx.base_va == A64_BASE
    for fn in ("_start", "strcpy", "recv", "printf", "handle_cmd"):
        assert truth[fn] in ctx.func_starts, fn
    xr = {ctx.strings[s]: {ctx.func_containing(v) for v in vs} for s, vs in ctx._str_xref_idx.items()}
    assert xr["boot: a64 rtos v3\n"] == {truth["_start"]}
    assert xr["cmd: rx packet\n"] == {truth["handle_cmd"]}
    assert (truth["handle_cmd"], truth["strcpy"]) in {(a, b) for a, b, _ in ctx.call_edges}


def test_arm64_symbols_and_scanners(a64):
    binf, nm, truth = a64
    ctx = BinaryContext.from_raw(str(binf), "arm64", base=A64_BASE, symbols=str(nm),
                                 force_rebuild=True)
    assert set(ctx.func_starts) == set(truth.values())        # UART (data) is not a function
    h = truth["handle_cmd"]

    from ablation.analyzers.taint_tracker_arm64 import ARM64TaintTracker
    tr = ARM64TaintTracker.from_context(ctx)
    assert any(f.sink_name == "strcpy" for f in tr.run_interprocedural())

    from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
    fs = ARM64FormatStringScanner.from_context(ctx)
    bad = [f for f in fs.scan() if f.verdict != "SAFE"]
    assert len(bad) == 1 and bad[0].func_va == h               # printf(pkt), not the literals

    from ablation.analyzers.func_profiler import FuncProfiler
    prof = FuncProfiler.from_context(ctx).profile(va=h)
    assert [c.site.target_name for c in prof.calls] == ["recv", "printf", "strcpy", "printf"]
    assert "cmd: rx packet" in prof.calls[1].fmt()

    from ablation.analyzers.path_solver_arm64 import ARM64PathSolver
    assert ARM64PathSolver.from_context(ctx, h) is not None


def test_base_changes_cache_key(a64):
    binf, _nm, _truth = a64
    c1 = BinaryContext.from_raw(str(binf), "arm64", base=A64_BASE)
    c2 = BinaryContext.from_raw(str(binf), "arm64", base=0)
    assert c1.base_va == A64_BASE and c2.base_va == 0
    assert BinaryContext.from_raw(str(binf), "arm64", base=A64_BASE).raw == c1.raw   # cache hit


def test_bad_arch_rejected(a64):
    with pytest.raises(ValueError):
        BinaryContext.from_raw(str(a64[0]), "sparc")


def test_unknown_format_warns(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x13\x37" * 512)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctx = BinaryContext.load_or_build(str(f), force_rebuild=True)
    assert ctx.arch == "unknown"
    assert any("--raw" in str(x.message) for x in w)


def test_cli_raw(cm):
    binf, nm, _truth = cm
    r = subprocess.run([sys.executable, "-m", "ablation.cli", "analyze", str(binf),
                        "--raw", "thumb", "--symbols", str(nm)],
                       capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode == 0, r.stderr[-500:]
    assert "arm32" in r.stdout
