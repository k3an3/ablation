# ARM64FuncStartScanner

**File:** `ablation/analyzers/arm64_func_start_scanner.py`

---

## Why this exists

2 things that were not possible before in Ablation:

**1. No function starts for stripped AArch64 ELF without `.eh_frame`.** `BinaryContext` seeds
`func_starts` from `.eh_frame` FDEs plus exports. AArch64 firmware built with
`-fno-asynchronous-unwind-tables`, or with `.eh_frame` stripped, left only the exports. Then
`func_containing()` folded whole regions of `.text` into one function, and every call edge and
string xref was attributed to the wrong owner. ARM32, PPC and x86_32 already had prologue-scan
fallbacks for this case; AArch64 did not.

**2. No awareness of PAC/BTI entry sequences.** Hardened AArch64 code opens functions with
`paciasp` or `bti c` before the frame push. A plain `stp x29, x30` scan reports the start one
instruction late, which is not the address callers branch to.

---

## What it does

Scans 4-byte aligned words in executable, non-PLT sections (executable LOAD segments when the
section table is gone) for:

| Signal | Encoding |
|---|---|
| BL target inside scanned code | `(w & 0xFC000000) == 0x94000000` |
| `paciasp` / `pacibsp` | `0xD503233F` / `0xD503237F` |
| `bti c` / `bti jc` | `0xD503245F` / `0xD50324DF` |
| `stp x29, x30, [sp, #-N]!` | `(w & 0xFFC07FFF) == 0xA9807BFD` |
| `sub sp, sp, #N` ; `stp xA, xB, [sp, #M]` | `(w & 0xFF8003FF) == 0xD10003FF` then `(w & 0xFFC003E0) == 0xA90003E0` |

A start that directly follows a `paciasp`/`bti` is moved back onto the landing pad.

Measured against `.eh_frame` on two stripped clang/Rust AArch64 binaries (7.2k and 8.2k FDEs):
**92–98% precision, ~58% recall.** Most misses are frameless leaf functions reached only
indirectly or by tail call. The "first instruction after `ret`/`b`" heuristic was tried and
rejected because only 13% of its hits were real starts.

---

## Usage

`BinaryContext` runs it automatically for `arch == 'arm64'` when `.eh_frame` yields no FDEs.
Standalone:

```python
from ablation.analyzers.arm64_func_start_scanner import ARM64FuncStartScanner

sc = ARM64FuncStartScanner.from_path('/path/to/stripped_a64.elf')
res = sc.scan_full()
print(res.report())          # counts per signal
starts = res.starts          # sorted VAs

# merge into an existing context (e.g. one loaded from an old cache)
sc.inject_into_context(ctx)
```

`code_regions(lief_binary)` is shared with `BinaryContext`'s AArch64 call-graph and string-xref
passes.

---

## Related: BinaryContext AArch64 paths

The same change gave `BinaryContext` native AArch64 handling. These paths had been falling
through to the x86-64 code, which returned 0 PLT entries, 0 call edges and coincidental string
xrefs on real AArch64 binaries:

- **PLT:** decodes `[bti c;] adrp x16 / ldr x17, [x16,#off]` stubs (16- or 24-byte) to GOT
  slot to `.rela.plt` name.
- **Call graph:** numpy BL scan, plus `b` tail calls that leave the owning function and land on
  a PLT stub or function start.
- **String xrefs:** `adr`, `adrp`+`add`, and `adrp`+`ldr` through a pointer slot (RELATIVE/ABS64
  relocation or on-disk value). Completions may sit up to 16 instructions after the ADRP.
  The recorded site is the completing `add`/`ldr`.
