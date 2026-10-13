# ARM64PathSolver

**File:** `ablation/analyzers/path_solver_arm64.py`
**Requires:** `z3-solver` (`pip install -e '.[full]'`)

---

## Why this exists

4 things that were not possible before in Ablation:

**1. No path feasibility on arm64.** `PathSolver` decodes only x86-64 capstone instruction IDs.
The "confirm feasibility" step of the RE workflow therefore had no arm64 backend. An arm64
sink found by `ARM64TaintTracker` or `FuncProfiler` could not be checked for reachability, and
the inputs that reach it could not be solved for.

**2. Flag semantics that compilers rely on.** AArch64 code sets flags with `subs`/`adds`/`ands`
and chains them with `ccmp`. Compilers also turn range checks into unsigned-wraparound idioms:
clang emits `16 <= len <= 64` as `sub w8, w1, #0x41; cmn w8, #0x31; b.hs`. Getting those
right needs real N/Z/C/V flags, not "operands of the last compare".

**3. `-O0` firmware spills every argument.** At `-O0`, an argument is stored to the frame and
reloaded before each compare. A solver that treats every load as a fresh symbol loses every
constraint on the inputs. Store-to-load forwarding keeps them.

**4. Seeded struct fields came back without values.** `PathResult` listed which seeded fields
constrained a path but not the values Z3 picked, which are the packet bytes a PoC needs.
`PathResult.fields` now carries them, for both the x86-64 and arm64 solvers.

---

## Usage

```python
from ablation.analyzers.binary_context import BinaryContext
from ablation.analyzers.path_solver_arm64 import ARM64PathSolver
from ablation.analyzers.path_solver import MemoryConstraint as MC

ctx = BinaryContext.load_or_build('/path/to/target_a64.so')

# Target = any instruction VA, typically the sink call FuncProfiler reported
ps = ARM64PathSolver.from_context(ctx, func_va=0x103ac)
r = ps.solve_first_sat(0x103f4)          # the `b system@plt`
print(r)
# [SAT] path=0x103ac -> 0x103b8 -> 0x103d4 -> 0x103e0 -> 0x103ec clauses=4
#       model=(x0=0x1337, x1=0x40, x2=0x4, x3=0x5a50)
# None => no path within bounds reaches the target (UNSAT / dead code)

# Name and bound struct fields read through a base register
seed = {("x0", 0): MC("magic", 8), ("x0", 4): MC("kind", 8, lo=0, hi=250)}
r = ARM64PathSolver.from_context(ctx, parse_va, struct_seed=seed).solve_first_sat(sink_va)
r.fields      # {'kind': 0xe0, 'magic': 0x7f}

# Snippets / objdump text (no binary needed)
ARM64PathSolver.from_listing(asm_text, base=0x1000).solve(target_va)
```

`solve()` returns one `PathResult` per enumerated path, and `solve_first_sat()` returns the
first SAT result or `None`. The tuning parameters are the same as `PathSolver`:
`max_path_blocks`, `max_paths` and `z3_timeout_ms`.

---

## Semantics

| Construct | Model |
|---|---|
| Registers | `x0..x30`, `sp` as BitVec(64) `xN_0` at entry. W writes zero-extend; `xzr` reads 0 |
| `cmp` `cmn` `tst` `subs` `adds` `ands` `bics` `negs` | Exact N/Z/C/V, 32- or 64-bit by operand width |
| `ccmp` / `ccmn` | Compare if cond holds on the incoming flags, else the immediate NZCV |
| `b.cond` | Architectural condition (`hi`, `ge`, `le`, ...) over the flags |
| `cbz` `cbnz` `tbz` `tbnz` | Constraint on the register / bit directly |
| `csel` `csinc` `csinv` `csneg` `cset` `csetm` `cinc` `cinv` `cneg` | `z3.If` on the condition |
| ALU | add/sub/and/orr/eor/bic/orn/eon, shifts, mul/madd/msub/mneg, udiv/sdiv (x/0 = 0), umulh/smulh |
| Operand modifiers | `lsl/lsr/asr/ror #n`, `uxt*/sxt* #n` |
| Moves | mov/movz/movn/movk, mvn, neg, adr/adrp, uxt*/sxt*, ubfx/sbfx/ubfiz |
| Loads / stores | Forwarded when the address expression and width match a store on the path; else struct seed; else a fresh symbol. Pairs, pre/post-index writeback |
| `bl` / `blr` | `x0..x18`, `x30` become fresh `xN_callK` |
| Anything else writing a GPR | Fresh symbol (sound over-approximation) |

Validated on clang-built `gate()` at `-O0`, `-O1`, `-O2` and `-Os`: every model satisfies the C
predicate. A dead `if (a > 10 && a < 5) system()` at `-O0` is UNSAT.

---

## Limits

- Simple paths only (each block at most once). Loop bodies are not unrolled.
- No aliasing: a store through one base does not invalidate a slot reached through another.
- Memory not written on the path is unconstrained, apart from seeded fields.
- Vector/FP registers are not tracked. Values moved from them into GPRs are fresh symbols.
- Callees are abstracted (no interprocedural constraints).
