# ARM64BssTaintTracker / ARM64VtableDispatchScanner

**Files:** `ablation/analyzers/bss_taint_tracker_arm64.py`, `ablation/analyzers/vtable_dispatch_arm64.py`

---

## Why this exists

3 things that were not possible before in Ablation:

**1. No BSS range-walk detection for arm64.** `BssTaintTracker` (initrd decrypt-loop finder)
decodes x86 `mov reg, [abs]` / `[rip+disp]` loads. AArch64 reaches globals with
`adrp xB, page` + `ldr xD, [xB, #lo12]` (or `ldp` for both halves of a pair), often with the
`adrp` in a different block from the load. That needs a constant dataflow, not a byte pattern.

**2. No vtable dispatch scan for arm64.** `VtableDispatchScanner` matches x86
`call [reg+disp]` encodings. An AArch64 virtual call is `ldr xT, [xV, #slot]` then `blr xT`
(`br` for tail dispatch, `blraa`/`blraaz` with pointer authentication).

**3. The x86 BSS tracker crashed on every real pair.** `BSSSymbol` was an unhashable dataclass
placed into sets, and `build_function` read capstone's `.addr` instead of `.address`. Both are
fixed. The x86 tracker still matches only `mov reg, [global]` loads and does not split blocks at
jump targets, so it misses gcc's `cmp [rip+x], reg` loops.

---

## ARM64BssTaintTracker

Same `BSSSymbol` input and report dict as the x86 tracker:

```python
from ablation.analyzers.bss_taint_tracker_arm64 import (
    ARM64BssTaintTracker, build_function_arm64, bss_symbols_from_elf)

syms = bss_symbols_from_elf(vmlinux)             # .symtab objects, or every 8-byte .bss slot if stripped
tracker = ARM64BssTaintTracker(syms)
report = tracker.analyze(build_function_arm64(data, fn_va, fn_size, load_base), name="fn")
# {'fn_start': ..., 'bss_pairs': [(start_addr, end_addr)], 'loops': [{'head', 'tail', 'bss_pairs'}]}
```

- Constants (`adrp`/`adr`/`add #imm`/`mov`) flow across the CFG (intersection at merges), so a
  loop-head reload of the end pointer resolves.
- A pair is two symbols 8 bytes apart loaded anywhere in the function.
- Taint follows `mov`/`add`/`sub` and `sp`/`x29` stack slots (`-O0` spills).
- A loop is reported when a tainted register is a memory base and a tainted register is compared
  before a `b.cond` / `csel` in the loop.

Validated on clang `-O0` and `-O2`: the `[initrd_start, initrd_end)` XOR loop is found, while a
pair read without a loop and a loop over one unrelated global are not.

## ARM64VtableDispatchScanner

Subclass of `VtableDispatchScanner`: same `scan(slot_map)` / `scan_offset` / `report` and
`DispatchSite` (`encoding` = `ldr+blr`, `ldr+br`, `ldr+blra`, `ldur+...`).

```python
from ablation.analyzers.vtable_dispatch_arm64 import ARM64VtableDispatchScanner
r = ARM64VtableDispatchScanner.from_path(lib).scan({"Shape::area": 0x0, "Shape::draw": 0x8})
print(ARM64VtableDispatchScanner.report(r))
```

A slot load counts when an indirect branch through the same register follows within 8
instructions and the register is not rewritten in between. Validated on clang++ `-O0` and `-O2`:
live slots, the doubled call, the `-O2` tail dispatch, and the dead slot all come out right.
`VtableResolver.resolve_blr_sites` remains the tool for resolving *which* function a `blr`
reaches. This scanner answers which slots are dispatched at all.
