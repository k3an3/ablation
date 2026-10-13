"""Gaps found auditing a multi-binary AArch64 firmware: relocations, sinks without imports,
taint sources / sinks overrides, conversion-driven lengths, finding quality.

clang + ld.lld fixtures; skip without the toolchain.
"""

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
