# ARM64HeapScanner

**File:** `ablation/analyzers/heap_scanner_arm64.py`

---

## Why this exists

3 things that were not possible before in Ablation:

**1. No heap-corruption scan for arm64.** `HeapVulnScanner` and `HeapUAFScanner` decode
x86-64 only.

**2. Freed pointers were followed through registers only.** The x86 scanners track the register
that was passed to `free()`. At `-O0` that pointer lives in a stack slot and is reloaded for each
use; at `-O2` it is often copied to a callee-saved register. Here each allocation gets a region in
the ARM64 taint state, so its identity follows copies, spills, reloads and alias classes.

**3. Error-path frees were reported as use-after-free.** A linear pass treats
`if (err) { free(p); return; } use(p);` as UAF. This scanner runs a must-analysis over the CFG: a
region stays freed at a merge only if every incoming path freed it.

---

## Findings (`heap_vuln_scanner.HeapFinding`)

| kind | severity | when |
|---|---|---|
| `uaf` | HIGH | load / store through a freed pointer |
| `uaf` | MEDIUM | freed pointer passed to another call |
| `double_free` | HIGH | `free` / `kfree` / `operator delete` of a freed pointer (also on both arms of a branch) |
| `off_by_one_alloc` | HIGH | `malloc(strlen(s))` then `strcpy`/`strcat`/`sprintf` into it, or `memcpy(..., strlen(s)+k)` |
| `heap_overflow` | HIGH | attacker length (`read`/`recv`/`memcpy` …) into a constant-size allocation without a bound ≤ size; constant copy larger than the allocation; constant-offset store past the end (inlined `memcpy` / `memset`) |
| `heap_overflow` | MEDIUM | constant-offset read past the end |
| `int_overflow_before_alloc` | HIGH / MEDIUM | from `ARM64IntOverflowScanner` (MEDIUM when a check exists) |

Allocators: `malloc` `calloc` `realloc` `x*alloc` `g_malloc*` `_Znwm`/`_Znam` `kmalloc` `kzalloc`
`kcalloc` `kvmalloc` `vmalloc` `krealloc` `strdup`. Frees: `free` `kfree` `vfree` `kvfree`
`g_free` `_ZdlPv`/`_ZdaPv` (+ sized). `realloc` frees its input.

```python
from ablation.analyzers.heap_scanner_arm64 import ARM64HeapScanner
s = ARM64HeapScanner.from_path_full(elf)
print(s.report(s.scan()))
#   HIGH    uaf             0x10774+0x1c  read through x19 after free (pointer from malloc@0x10784)  alloc=malloc  freed_at=0x1078c
```
CLI: `ablation heap <bin>` dispatches on arch.

Validated on clang `-O0` and `-O2` fixtures. Nine true cases are found at both levels. Five
controls stay silent: an error-path free, reassignment after free, an alloc/free loop,
`strlen+1`, and a masked length. See `tests/test_heap_scanner_arm64.py`.

---

**Fixpoint reporting.** Findings are reported from each block's final dataflow state, not
while the dataflow is still iterating. A pointer walked in a loop (`while (*p) p++`,
`ldrb w8, [x0, #1]!`) has a constant offset only on the first trip; the loop-head join drops
it. So a NUL-terminated walk over a small buffer is not reported as a "read past N-byte
alloc".

**realloc growth.** An object passed to `realloc` (or any object, when the pointer handed to
`realloc` is unknown) has no constant size afterwards. Constant-offset accesses to it are not
reported.

## Limits

- Intraprocedural for UAF / double-free (like the x86 scanners): a pointer freed in a callee
  is not seen.
- Pointers stored into heap objects (`obj->buf = malloc(..)`) are tracked only while they
  stay in registers / stack slots.
- Type confusion (`HeapUAFScanner`'s fourth class) is not ported.
