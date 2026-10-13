"""Tests for ARM64RegAnnotator and its FuncProfiler / IPRegAnnotator wiring.

Unit tests drive _step() with hand-built instructions and a stub binary, so they
run without a toolchain. Fixture tests cross-build AArch64 .so files with clang +
ld.lld and skip when that is unavailable.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.arm64_reg_annotator import ARM64RegAnnotator
from ablation.analyzers.binary_context import BinaryContext
from ablation.analyzers.func_profiler import FuncProfiler, ProfiledCall
from ablation.analyzers.insn_arm64 import make_insn
from ablation.analyzers.ipreg_annotator import IPRegAnnotator
from ablation.analyzers.reg_annotator import AnnotationResult, CallSite, RegVal

from a64_fixtures import build_variants, symbols

# ── unit tests: no toolchain ─────────────────────────────────────────────────

_STR_VA = 0x3C0
_PLT = {0x10520: "strcpy", 0x10540: "system"}
_GOT_SLOT = 0x30600


def _stub_binary():
    rel = SimpleNamespace(
        type="TYPE.AARCH64_GLOB_DAT", address=_GOT_SLOT, addend=0, has_symbol=True,
        symbol=SimpleNamespace(value=0, name="popen"),
    )
    return SimpleNamespace(segments=[], relocations=[rel])


def _annotator():
    return ARM64RegAnnotator(_stub_binary(), {_STR_VA: "reboot"}, dict(_PLT),
                             exports={"helper": 0x10494}, func_starts=[0x10000, 0x10494])


def _run(lines, func_va=0x10000, end_va=0x10100, seed=True):
    ra = _annotator()
    regs = {f"x{i}": RegVal.arg(i, f"x{i}") for i in range(8)} if seed else {}
    res = AnnotationResult(func_va=func_va, func_end_va=end_va)
    va = func_va
    for line in lines:
        mn, _, ops = line.partition(" ")
        ra._step(make_insn(va, mn, ops), regs, res)
        va += 4
    return res, regs


def test_adrp_add_resolves_string():
    res, _ = _run(["adrp x0, #0", f"add x0, x0, #{_STR_VA:#x}", "bl #0x10540"])
    assert res.calls[0].target_name == "system"
    assert res.calls[0].args["x0"].kind == "string"
    assert res.calls[0].args["x0"].str_val == "reboot"


def test_adrp_page_is_not_a_string():
    # a string sitting at a page start must not swallow the page: the following
    # add / ldr resolve against the page constant
    ra = ARM64RegAnnotator(_stub_binary(), {0x3000: "page-start", 0x3030: "target"}, dict(_PLT))
    regs = {}
    res = AnnotationResult(func_va=0x10000, func_end_va=0x10100)
    for va, line in enumerate(["adrp x0, #0x3000", "add x0, x0, #0x30", "bl #0x10540"]):
        mn, _, ops = line.partition(" ")
        ra._step(make_insn(0x10000 + va * 4, mn, ops), regs, res)
    assert res.calls[0].args["x0"].str_val == "target"


def test_br_through_got_slot_is_tail_call():
    page = _GOT_SLOT & ~0xFFF
    res, _ = _run([f"adrp x1, #{page:#x}", f"ldr x1, [x1, #{_GOT_SLOT - page:#x}]", "br x1"])
    assert [c.target_name for c in res.calls] == ["popen"]


def test_movz_movk_chain():
    res, _ = _run(["mov x1, #0xabcd", "movk x1, #0x1234, lsl #16",
                   "movk x1, #0xc0de, lsl #32", "bl #0x10520"])
    assert res.calls[0].args["x1"].int_val == 0xC0DE1234ABCD


def test_callee_saved_survives_call_caller_saved_does_not():
    res, regs = _run(["mov x19, x1", "mov x9, x2", "bl #0x10520",
                      "mov x0, x19", "mov x1, x9", "bl #0x10540"])
    second = res.calls[1]
    assert second.args["x0"].display().startswith("arg1_entry")
    assert second.args["x1"].kind == "unknown"


def test_return_value_and_zero():
    res, _ = _run(["bl #0x10520", "mov x1, x0", "mov x2, xzr", "bl #0x10540"])
    assert res.calls[1].args["x1"].display() == "ret(strcpy)  (via x0)"
    assert res.calls[1].args["x2"].kind == "zero"


def test_w_register_write_truncates():
    res, _ = _run(["mov x3, #-1", "mov w0, w3", "bl #0x10540"])
    assert res.calls[0].args["x0"].int_val == 0xFFFFFFFF


def test_stack_buffer_address():
    res, _ = _run(["add x0, sp, #0x10", "mov x1, sp", "bl #0x10520"])
    assert res.calls[0].args["x0"].display() == "[sp+0x10]"
    assert res.calls[0].args["x1"].display() == "[sp]"


def test_tail_call_to_plt_is_recorded():
    # b system@plt sits inside the (over-long) window: still a call
    res, _ = _run(["adrp x0, #0", f"add x0, x0, #{_STR_VA:#x}", "b #0x10540"],
                  end_va=0x20000)
    assert [c.target_name for c in res.calls] == ["system"]


def test_branch_within_function_is_not_a_call():
    res, _ = _run(["cbz x0, #0x10008", "b #0x10004"])
    assert res.calls == []


def test_blr_through_import_got_slot():
    page = _GOT_SLOT & ~0xFFF
    res, _ = _run([f"adrp x8, #{page:#x}", f"ldr x8, [x8, #{_GOT_SLOT - page:#x}]", "blr x8"])
    assert res.calls[0].target_name == "popen"


def test_blr_unknown_register_named_by_register():
    res, _ = _run(["ldr x8, [x0, #0x18]", "blr x8"])
    assert res.calls[0].target_name == "blr x8"
    assert res.calls[0].target_va == 0


def test_entry_regs_seed_overrides_args():
    ra = _annotator()
    res = ra.annotate_calls(0x10000, 0x10010, entry_regs={"x0": RegVal.string("seeded")})
    assert res.calls == []  # stub binary has no bytes: empty, but no crash


# ── arch-neutral formatting (shared types) ───────────────────────────────────

def test_profiled_call_fmt_uses_site_arg_registers():
    """ARM args are keyed x0../r0..; fmt previously only printed rdi..r9."""
    site = CallSite(site_va=0x1000, target_va=0x2000, target_name="system",
                    args={"x0": RegVal.string("id"), "x1": RegVal.unknown()})
    out = ProfiledCall(site=site, sink_tag="cmd-exec").fmt()
    assert "x0='id'" in out and "x1=" not in out
    assert "x0 = 'id'" in site.fmt()


def test_x86_profile_still_shows_sysv_args():
    if not Path("/usr/bin/ls").exists():
        pytest.skip("/usr/bin/ls not present")
    ctx = BinaryContext.load_or_build("/usr/bin/ls")
    if ctx.arch != "x86_64":
        pytest.skip("host /usr/bin/ls is not x86_64")
    fp = FuncProfiler.from_context(ctx)
    prof = next(p for p in (fp.profile(v) for v in ctx.func_starts[:200]) if p.calls)
    assert any(k in ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
               for c in prof.calls for k in c.site.args)


# ── fixture tests ────────────────────────────────────────────────────────────

_EXTRA_SRC = r'''
extern int system(const char *);
extern int ioctl(int, unsigned long, ...);
int dispatch(int fd, const char *cmd) {
    ioctl(fd, 0xC0DE1234ABCDull, 0);
    return system(cmd);
}
__attribute__((naked)) int via_got(const char *c) {
    __asm__(
        "stp x29, x30, [sp, #-16]!\n"
        "ldr x1, 1f\n"
        "adrp x8, :got:system\n"
        "ldr x8, [x8, :got_lo12:system]\n"
        "blr x8\n"
        "ldp x29, x30, [sp], #16\n"
        "ret\n"
        ".p2align 3\n"
        "1: .quad 0x4142434445464748\n");
}
'''


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    return build_variants(tmp_path_factory, variants={"plain": [], "noeh": ["-fno-asynchronous-unwind-tables", "-fno-unwind-tables"]})


@pytest.fixture(scope="module")
def extra(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_EXTRA_SRC, variants={"extra": []})["extra"]


@pytest.mark.parametrize("variant", ["plain", "noeh"])
def test_func_profiler_uses_arm64_annotator(fixtures, variant):
    ctx = BinaryContext.build(str(fixtures[variant]))
    fp = FuncProfiler.from_context(ctx)
    assert isinstance(fp._ra, ARM64RegAnnotator)
    prof = fp.profile(ctx.exports["handler"])
    by_name = {}
    for c in prof.calls:
        by_name.setdefault(c.site.target_name, []).append(c.site)
    assert by_name["helper"][0].args["x0"].display() == "[sp]"
    assert by_name["helper"][0].args["x1"].display().startswith("arg0_entry (x0@entry)")
    putss = {a.args["x0"].str_val for a in by_name["puts"]}
    assert {"handler-string-xref", "global-pointer-string"} <= putss
    assert "handler-string-xref" in [s for _, s in prof.strings]
    assert "x0='handler-string-xref'" in prof.fmt()


@pytest.mark.parametrize("variant", ["plain", "noeh"])
def test_tail_called_sink_with_string(fixtures, variant):
    ctx = BinaryContext.build(str(fixtures[variant]))
    tail_va = ctx.callers_of("system")[0][0]
    prof = FuncProfiler.from_context(ctx).profile(tail_va)
    sinks = prof.sink_calls
    assert [s.site.target_name for s in sinks] == ["system"]
    assert sinks[0].site.args["x0"].str_val == "/bin/sh -c reboot"


@pytest.mark.parametrize("variant", ["plain", "noeh"])
def test_ipreg_chain_traces_entry_arg_into_sink(fixtures, variant):
    ctx = BinaryContext.build(str(fixtures[variant]))
    chain = IPRegAnnotator.from_context(ctx).annotate_chain(ctx.exports["handler"], max_hops=2)
    hits = {cs.target_name: cs for _, _, cs in chain.all_sinks({"strcpy", "system"})}
    # helper is reached through helper@plt and followed to its own export
    assert hits["strcpy"].args["x0"].display() == "[sp]"
    assert hits["strcpy"].args["x1"].display().startswith("arg0_entry (x0@entry)")
    assert hits["system"].args["x0"].str_val == "/bin/sh -c reboot"
    assert "x0='/bin/sh -c reboot'" in chain.sink_report({"system"})


def test_callee_saved_arg_reaches_tail_called_system(extra):
    ctx = BinaryContext.build(str(extra))
    prof = FuncProfiler.from_context(ctx).profile(ctx.exports["dispatch"])
    calls = {c.site.target_name: c.site for c in prof.calls}
    assert calls["ioctl"].args["x1"].int_val == 0xC0DE1234ABCD
    assert calls["ioctl"].args["x2"].kind == "zero"
    assert calls["system"].args["x0"].display().startswith("arg1_entry (x1@entry)")


def test_blr_via_got_and_literal_pool(extra):
    ctx = BinaryContext.build(str(extra))
    va = symbols(extra)["via_got"]
    prof = FuncProfiler.from_context(ctx).profile(va, va + 28)
    assert [c.site.target_name for c in prof.sink_calls] == ["system"]
    assert prof.sink_calls[0].site.args["x1"].int_val == 0x4142434445464748
