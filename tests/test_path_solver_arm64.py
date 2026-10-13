"""Tests for ARM64PathSolver (and the shared PathResult.fields addition).

Listing tests need only z3. Binary tests cross-build with clang + ld.lld (arm64)
or gcc (x86_64) and skip when the toolchain is missing.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

z3 = pytest.importorskip("z3")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from ablation.analyzers.path_solver import MemoryConstraint as MC
from ablation.analyzers.path_solver_arm64 import ARM64PathSolver

from a64_fixtures import build_variants

_M32 = 0xFFFFFFFF


def _solve(listing, target, **kw):
    return ARM64PathSolver.from_listing(listing, base=0x1000, **kw).solve_first_sat(target)


# ── listing tests ────────────────────────────────────────────────────────────

def test_compare_chain_sat():
    r = _solve("""
        cmp w0, #0x41
        b.ne 0x1014
        cmp x1, #5
        b.ls 0x1014
        bl 0x2000
        ret
    """, 0x1010)
    assert r and r.model["x0"] & _M32 == 0x41 and r.model["x1"] > 5


def test_contradiction_unsat():
    ps = ARM64PathSolver.from_listing("""
        cmp x0, #3
        b.ne 0x1014
        cmp x0, #4
        b.ne 0x1014
        bl 0x2000
        ret
    """, base=0x1000)
    assert ps.solve_first_sat(0x1010) is None
    assert all(not r.sat and r.note == "unsat" for r in ps.solve(0x1010))


def test_stack_spill_forwarding():
    r = _solve("""
        sub sp, sp, #0x20
        str w0, [sp, #0xc]
        mov w0, #0
        ldr w8, [sp, #0xc]
        cmp w8, #7
        b.ne 0x1020
        bl 0x2000
        ret
    """, 0x1018)
    assert r and r.model["x0"] & _M32 == 7


def test_stp_ldp_pair_forwarding():
    r = _solve("""
        stp x0, x1, [sp, #-0x10]!
        mov x0, #0
        mov x1, #0
        ldp x2, x3, [sp], #0x10
        cmp x3, #9
        b.ne 0x1020
        bl 0x2000
        ret
    """, 0x1018)
    assert r and r.model["x1"] == 9


def test_ccmp_conjunction():
    r = _solve("""
        cmp x0, #1
        ccmp x1, #2, #0, eq
        b.ne 0x1010
        bl 0x2000
        ret
    """, 0x100c)
    assert r and r.model["x0"] == 1 and r.model["x1"] == 2


def test_tbz_cbz():
    r = _solve("""
        tbz w2, #3, 0x100c
        cbz x3, 0x100c
        bl 0x2000
        ret
    """, 0x1008)
    assert r and r.model["x2"] & 8 and r.model["x3"] != 0


def test_cset_signed_lt():
    r = _solve("""
        cmp w4, #-5
        cset w9, lt
        cbnz w9, 0x1010
        ret
        bl 0x2000
        ret
    """, 0x1010)
    assert r
    w4 = r.model["x4"] & _M32
    assert (w4 - (1 << 32) if w4 >> 31 else w4) < -5


def test_unsigned_range_idiom():
    # clang's `16 <= len <= 64`:  sub w8, w1, #0x41 ; cmn w8, #0x31 ; b.hs ok
    r = _solve("""
        sub w8, w1, #0x41
        cmn w8, #0x31
        b.hs 0x1010
        ret
        bl 0x2000
        ret
    """, 0x1010)
    assert r
    ln = r.model["x1"] & _M32
    assert 16 <= ln <= 64


def test_arith_and_shift():
    r = _solve("""
        lsl x5, x0, #2
        add x5, x5, #8
        cmp x5, #0x30
        b.ne 0x1014
        bl 0x2000
        ret
    """, 0x1010)
    assert r and ((r.model["x0"] << 2) + 8) & (2**64 - 1) == 0x30


def test_call_clobbers_caller_saved():
    # x0 after the call is the callee's return value, not entry x0: without the
    # clobber this path would need x0 == 3 and x0 == 4 at once (UNSAT)
    r = _solve("""
        mov x19, x0
        bl 0x2000
        cmp x19, #3
        b.ne 0x101c
        cmp x0, #4
        b.ne 0x101c
        bl 0x3000
        ret
    """, 0x1018)
    assert r and r.model["x0"] == 3


def test_target_mid_block_and_seed_fields():
    seed = {("x0", 0): MC("magic", 8), ("x0", 4): MC("kind", 8, lo=0, hi=250)}
    listing = """
        ldrb w8, [x0]
        cmp w8, #0x7f
        b.ne 0x1020
        ldrb w8, [x0, #4]
        cmp w8, #0xc9
        b.lo 0x1020
        adr x0, 0x3000
        bl 0x2000
        ret
    """
    r = _solve(listing, 0x101c, struct_seed=seed)
    assert r and r.seeded_fields == ["kind", "magic"]
    assert r.fields["magic"] == 0x7F and 201 <= r.fields["kind"] <= 250
    assert "fields=(" in str(r)
    seed[("x0", 4)] = MC("kind", 8, lo=0, hi=100)
    assert _solve(listing, 0x101c, struct_seed=seed) is None


# ── compiled arm64 fixtures ──────────────────────────────────────────────────

_GATE_SRC = r'''
extern int system(const char *);
int gate(unsigned int cmd, int len, unsigned char flags, long key) {
    if (cmd != 0x1337) return -1;
    if (len < 16 || len > 64) return -2;
    if (!(flags & 0x4)) return -3;
    if ((key ^ 0x5a5a) * 3 != 0x1e) return -4;
    return system("gate-open");
}
int unreachable(int a) {
    if (a > 10 && a < 5) return system("never");
    return 0;
}
'''


def _gate_ok(m):
    cmd = m["x0"] & _M32
    ln = m["x1"] & _M32
    ln = ln - (1 << 32) if ln >> 31 else ln
    return (cmd == 0x1337 and 16 <= ln <= 64 and m["x2"] & 4
            and ((m["x3"] ^ 0x5A5A) * 3) & (2**64 - 1) == 0x1E)


@pytest.fixture(scope="module")
def gates(tmp_path_factory):
    return build_variants(tmp_path_factory, src_text=_GATE_SRC,
                          variants={"O0": ["-O0"], "O2": ["-O2"]})


def _system_sites(ctx, fva):
    from ablation.analyzers.arm64_reg_annotator import ARM64RegAnnotator
    calls = ARM64RegAnnotator.from_context(ctx).annotate_calls(fva).calls
    return [c.site_va for c in calls if c.target_name == "system"]


@pytest.mark.parametrize("opt", ["O0", "O2"])
def test_gate_model_satisfies_source_predicate(gates, opt):
    from ablation.analyzers.binary_context import BinaryContext
    ctx = BinaryContext.build(str(gates[opt]))
    fva = ctx.exports["gate"]
    r = ARM64PathSolver.from_context(ctx, fva).solve_first_sat(_system_sites(ctx, fva)[0])
    assert r is not None and _gate_ok(r.model), r


def test_dead_sink_is_unsat_at_O0(gates):
    from ablation.analyzers.binary_context import BinaryContext
    ctx = BinaryContext.build(str(gates["O0"]))
    fva = ctx.exports["unreachable"]
    sites = _system_sites(ctx, fva)
    assert sites, "-O0 keeps the dead call"
    assert ARM64PathSolver.from_context(ctx, fva).solve_first_sat(sites[0]) is None


# ── x86-64: PathResult.fields on the shared solver ───────────────────────────

_PARSE_SRC = r'''
extern int system(const char *);
int parse(const unsigned char *pkt) {
    if (pkt[0] == 0x7f && pkt[4] > 200) return system("pkt");
    return 0;
}
'''


def test_x86_seeded_fields_reported(tmp_path):
    import platform
    if platform.machine() not in ("x86_64", "AMD64") or not shutil.which("gcc"):
        pytest.skip("needs a native x86_64 gcc")
    import capstone
    import lief
    from ablation.analyzers.cfg_builder import CFGBuilder
    from ablation.analyzers.path_solver import PathSolver

    src, so = tmp_path / "p.c", tmp_path / "p.so"
    src.write_text(_PARSE_SRC)
    subprocess.run(["gcc", "-O1", "-fPIC", "-shared", "-nostdlib", str(src), "-o", str(so)], check=True)
    b = lief.parse(str(so))
    fva = next(s.value for s in b.exported_functions if s.name == "parse")
    cfg = CFGBuilder(str(so)).build_function(fva)
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.detail = True
    off = b.virtual_address_to_offset(fva)
    data = so.read_bytes()[off: off + 0x100]
    sink_block = max(k for k, v in cfg.blocks.items()
                     if any(m == "call" for _, m, _ in v.insns))
    seed = {("rdi", 0): MC("magic", 8), ("rdi", 4): MC("kind", 8, lo=0, hi=250)}
    r = PathSolver(data, fva, fva + 0x100, {}, cfg, md, struct_seed=seed).solve_first_sat(sink_block)
    assert r is not None and r.fields.get("magic") == 0x7F
    assert 201 <= r.fields.get("kind", 0) <= 250
