"""ARM64 taint: pointer spills, alias classes, return-address copies, opt-in findings,
and the binary path's buffer-filling sources / pointee sinks.

Engine tests use hand-written listings. Binary tests cross-build with clang + ld.lld
and skip when the toolchain is missing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.analysis_arm64 import analyze_cfg
from ablation.analyzers.insn_arm64 import from_listing
from ablation.analyzers.taint_tracker_arm64 import (
    ARM64TaintTracker, Policy, TaintEngine, TaintState,
)

from a64_fixtures import build_variants


def run(listing, taints=(("x0", "in"),), policy=None):
    eng = TaintEngine(policy)
    for r, lbl in taints:
        eng.taint_register(r, lbl)
    eng.run(list(from_listing(listing)))
    return eng


def kinds(eng):
    return [f.kind for f in eng.findings]


# ── pointer spills (mem_ptrs) ────────────────────────────────────────────────

def test_spilled_frame_pointer_is_restored():
    e = run("""
        add x1, sp, #0x10
        str x1, [sp]
        mov x1, #0
        ldr x2, [sp]
    """, taints=())
    assert e.state.frame_ptrs["x2"] == ("sp", 0x10)


def test_read_through_spilled_pointer_taints_buffer_only():
    e = run("""
        add x1, sp, #0x20
        str x1, [sp, #8]
        mov x2, #16
        mov x8, #63
        svc #0
        ldr x3, [sp, #8]
        ldrb w4, [x3]
        ldrb w5, [x3, #20]
    """, taints=())
    assert e.state.get("x4").tainted
    assert not e.state.get("x5").tainted     # past the 16 bytes read


def test_constant_survives_spill():
    e = run("""
        mov x2, #0x40
        str x2, [sp, #8]
        mov x2, #0
        ldr x2, [sp, #8]
        mov w3, #7
        str w3, [sp, #4]
        ldr w6, [sp, #4]
    """, taints=())
    assert e.state.consts["x2"] == 0x40
    assert e.state.consts["x6"] == 7


# ── alias classes (eq) ───────────────────────────────────────────────────────

def test_buffer_filled_through_one_alias_is_seen_through_another():
    # heap pointer kept in x19, copied to x1 for read(): the region is shared
    e = run("""
        mov x19, x1
        mov x2, #8
        mov x8, #63
        svc #0
        ldr x5, [x19]
    """, taints=())
    assert e.state.get("x5").tainted


def test_alias_through_stack_slot():
    # -O0 shape: pointer spilled, reloaded into x1 for read, reloaded again later
    e = run("""
        str x0, [sp, #0x10]
        ldr x1, [sp, #0x10]
        mov x2, #8
        mov x8, #63
        svc #0
        ldr x9, [sp, #0x10]
        ldr x5, [x9]
    """, taints=())
    assert e.state.get("x5").tainted


def test_overwritten_alias_is_dropped():
    e = run("""
        mov x19, x1
        mov x19, #0
        mov x2, #8
        mov x8, #63
        svc #0
        ldr x5, [x19]
    """, taints=())
    assert not e.state.get("x5").tainted


def test_cfg_join_keeps_alias_only_when_both_paths_have_it():
    both = """
        0x1000: cbz x3, 0x100c
        0x1004: mov x19, x1
        0x1008: b 0x1010
        0x100c: mov x19, x1
        0x1010: mov x2, #8
        0x1014: mov x8, #63
        0x1018: svc #0
        0x101c: ldr x5, [x19]
    """
    one = both.replace("0x100c: mov x19, x1", "0x100c: mov x19, #0")
    r_both = analyze_cfg(list(from_listing(both)))
    r_one = analyze_cfg(list(from_listing(one)))
    assert r_both.exit_states[0x1010].get("x5").tainted
    assert not r_one.exit_states[0x1010].get("x5").tainted


def test_state_copy_preserves_shared_groups():
    st = TaintState()
    st.alias("x19", "x1")
    st.alias(("sp", 8), "x1")
    cp = st.copy()
    cp.point_at("x1", "@r", 0)
    assert cp.frame_ptrs["x19"] == ("@r", 0) and cp.mem_ptrs[("sp", 8)] == ("@r", 0)
    assert "x19" not in st.frame_ptrs         # original untouched


# ── return-address copies (ra_regs) ──────────────────────────────────────────

def test_br_through_lr_copy_is_tainted_return():
    e = run("""
        mov x9, x30
        br x9
    """, taints=(("x30", "in"),))
    assert kinds(e) == ["tainted-return-address"]


def test_br_through_unrelated_reg_is_indirect_jump():
    e = run("br x0")
    assert kinds(e) == ["tainted-indirect-jump"]


def test_lr_copy_spill_then_tainted_overwrite():
    e = run("""
        mov x9, x30
        str x9, [sp, #8]
        str x0, [sp, #8]
    """)
    assert "tainted-overwrite-of-saved-lr" in kinds(e)


def test_lr_reload_restores_role():
    e = run("""
        str x30, [sp, #8]
        bl 0x2000
        ldr x10, [sp, #8]
        add x10, x10, x0
        mov x11, x30
    """)
    assert "x10" not in e.state.ra_regs       # arithmetic result is not the RA
    e2 = run("""
        str x30, [sp, #8]
        bl 0x2000
        ldr x10, [sp, #8]
    """, taints=())
    assert "x10" in e2.state.ra_regs and "x30" not in e2.state.ra_regs


# ── opt-in findings (Policy) ─────────────────────────────────────────────────

def test_tainted_branch_condition_is_opt_in():
    listing = """
        cmp x0, #5
        b.eq 0x2000
        cbz x0, 0x2000
    """
    assert kinds(run(listing)) == []
    assert kinds(run(listing, policy=Policy(report_tainted_branch=True))) == [
        "tainted-branch-condition", "tainted-branch-condition"]


def test_bounded_index_is_opt_in():
    listing = """
        and x1, x0, #0xff
        ldr x2, [x3, x1]
    """
    assert kinds(run(listing)) == []
    assert kinds(run(listing, policy=Policy(report_bounded=True))) == ["bounded-tainted-load-address"]


# ── binary path ──────────────────────────────────────────────────────────────

_SRC = r'''
typedef unsigned long size_t;
typedef struct FILE FILE;
extern long read(int, void *, size_t);
extern long recv(int, void *, size_t, int);
extern char *fgets(char *, int, FILE *);
extern int system(const char *);
extern char *strcpy(char *, const char *);
extern void *memcpy(void *, const void *, size_t);
extern void *malloc(size_t);
int stack_buf(int fd) { char buf[128]; read(fd, buf, sizeof buf); return system(buf); }
int heap_buf(int fd) { char *p = malloc(256); recv(fd, p, 256, 0); return system(p); }
int clean_ctrl(int fd) { char buf[64]; read(fd, buf, sizeof buf); return system("id"); }
__attribute__((noinline)) int run_cmd(const char *c) { return system(c); }
int via_helper(int fd) { char b[64]; recv(fd, b, sizeof b, 0); return run_cmd(b); }
int via_strcpy(int fd) { char in[64], cmd[64]; recv(fd, in, sizeof in, 0); strcpy(cmd, in); return system(cmd); }
int via_fgets(FILE *f) { char line[80]; fgets(line, sizeof line, f); return system(line); }
int copy_clean(int fd) { char in[64], other[64] = "uptime"; char dst[64]; recv(fd, in, sizeof in, 0); memcpy(dst, in, 64); return system(other); }
'''

_TRUE = {"stack_buf", "heap_buf", "run_cmd", "via_strcpy", "via_fgets"}


@pytest.fixture(scope="module")
def bins(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_SRC,
                          variants={"O0": ["-O0"], "O2": ["-O2"]})


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_source_buffer_reaches_system(bins, opt):
    t = ARM64TaintTracker.from_path_full(str(bins[opt]))
    hits = {f.func_name for f in t.run_interprocedural() if f.sink_name == "system"}
    assert hits == _TRUE          # and no clean_ctrl / copy_clean false positives


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_system_finding_names_the_pointee(bins, opt):
    t = ARM64TaintTracker.from_path_full(str(bins[opt]))
    f = next(f for f in t.run_interprocedural()
             if f.func_name == "stack_buf" and f.sink_name == "system")
    assert f.tainted_args == ["*x0"] and f.source_name == "read" and f.severity == "CRITICAL"


def test_from_context_resolves_plt_after_text(bins):
    """lld puts .plt after .text; names must still resolve."""
    from ablation.analyzers.binary_context import BinaryContext
    ctx = BinaryContext.load_or_build(str(bins["O0"]))
    t = ARM64TaintTracker.from_context(ctx)
    assert {"read", "recv", "system"} <= set(t._syms.values())
    assert {f.func_name for f in t.run_interprocedural() if f.sink_name == "system"} == _TRUE
