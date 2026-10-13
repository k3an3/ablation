# Raw firmware images (`BinaryContext.from_raw`)

**Files:** `ablation/analyzers/raw_image.py`, `BinaryContext.from_raw` / `load_or_build(raw=...)`

---

## Why this exists

Before this, Ablation could not do 4 things with raw images:

**1. A headerless image gave an empty context.** RTOS firmware (FreeRTOS, Zephyr, ThreadX,
bare-metal Cortex-M) usually ships as a flat flash dump: no ELF, no PE, no sections, no
symbols. `load_or_build()` parsed it as nothing and returned zero functions, strings and
edges. Every tool that builds on the context then reported nothing. Only the ECU decoders and
`FlatBinaryFuncStartScanner` took a `base_va`.

**2. No load address meant no xrefs.** Absolute pointers (literal pools, MOVW/MOVT pairs,
vector tables) only resolve when the base is known. A Cortex-M image declares its base in its
vector table, but nothing read it.

**3. No way to bring names.** A user with a partial map file or an `nm` listing from a sister
build had no way to name `memcpy` / `recv` / `uart_rx`. So the taint trackers' sink and
source names never matched.

**4. The scanners assumed a file format.** `ARM64TaintTracker`, `ARM64RegAnnotator`,
`ARM64PathSolver` and the format-string scanner all re-parsed the ELF. `ARM32TaintTracker`
read code with base 0.

---

## Usage

```python
from ablation.analyzers.binary_context import BinaryContext

# Cortex-M: base inferred from the vector table; handlers named from it
ctx = BinaryContext.from_raw('fw.bin', 'cortex-m')
ctx.name(0x80000e0)          # 'Reset_Handler'  (IRQn -> 'IRQ37_Handler')

# any arch, explicit base, optional names (nm / "addr name" / "name = addr" / ld map)
ctx = BinaryContext.from_raw('flash.bin', 'arm64', base=0x40080000, symbols='fw.nm')
```

CLI: `analyze`, `taint`, `overflow`, `profile`, `fmtstr` and `heap` take
`--raw {thumb|cortex-m|arm32|arm64} [--base 0x08000000] [--symbols FILE]`:

```
ablation taint flash.bin --raw arm64 --base 0x40080000 --symbols fw.nm
ablation profile fw.bin 0x8000174 --raw cortex-m --symbols fw.nm
```

The cache key includes arch, base and the symbols file hash, so building with a different
base does not reuse a stale context. `ctx.raw` holds `{"arch", "base", "symbols"}`, and
`ctx.image_regions()` returns `[(base, bytes)]` for analyzers that read code by VA.

---

## What the raw build populates

| Field | thumb / arm32 | arm64 |
|---|---|---|
| `exports` | symbols file + Cortex-M vector table (`Reset_Handler`, `NMI_Handler`, ..., `IRQn_Handler`; a shared handler is `Default_Handler`) | symbols file |
| `func_starts` | names + `FlatBinaryFuncStartScanner` (Thumb) + BL targets called from ≥ 2 sites or opening with `push {.., lr}` | names + `ARM64FuncStartScanner` |
| `call_edges` | BL, plus Thumb-2 `b.w` tail calls to a known function | BL / B (same pass as ELF) |
| `strings` | NUL-terminated runs, plus every referenced address that holds a string | NUL-terminated runs |
| string xrefs | literal-pool `ldr` + MOVW/MOVT pairs | ADRP+ADD / ADR / literal |
| `thumb_funcs` | every start (Thumb mode) | n/a |

Base inference (`infer_cortexm_base`): it takes the reset vector and the other handlers in the
first 16 words. It returns the highest aligned base (2^8 .. 2^24) for which every handler
lands inside the image, past the vector table.

## Raw-aware consumers

`ARM64TaintTracker` (and its subclasses: format string, length underflow, heap,
int overflow), `ARM64RegAnnotator`, `ARM64PathSolver`, `FuncProfiler`, `ARM32TaintTracker`,
`ARM32RegAnnotator` and `ARM32IntOverflowScanner`. For arm32 consumers, names from the
symbols file act as call targets, as PLT names do in an ELF.

---

## Loader hygiene (v2.87.0)

- **Symbols files** (`raw_image.SYMBOL_FORMATS`, also in `--help`). Accepted formats:
  - `VA name`, `name VA`, `name = VA`;
  - `VA size name`;
  - `nm` output (code types only);
  - `readelf -sW` output (`FUNC` only);
  - `VA file func line`. Here the VA can be anywhere inside the function, for example a
    log-call site harvested from a logging helper's arguments. The name goes to the
    function containing it (majority vote when sites disagree).

  `ctx.raw["symbols_stats"]`, the summary and the CLI report `loaded N, attached M (k by
  containing function), outside image X`. A file where no line parses is an error that lists
  the formats. `attached 0` warns to check `--base`.
- **Code ranges.** Inferred from exact `RET` / frame-push / PAC / BTI density in 4 KB
  windows (Thumb: `bx lr` / `pop {pc}` with `push {lr}`). Each range ends at its last `RET`.
  `--code-range LO-HI[,LO-HI]` overrides the inference. BL, ADRP, literal-pool and MOVW/MOVT
  scans and function starts only use code ranges, so data words do not invent call sites.
- **Entry filtering.** `ARM64FuncStartScanner(strict=True)` is used for raw images. A BL
  target or prologue counts as a function start only if:
  - the word before it ends the previous function (`ret` / `b` / `br` / `brk` / `udf` /
    padding), or
  - it opens with a landing pad or frame push.

  A `bl` before it does not count, because every return site follows one. This stops
  data-derived targets from splitting a 600-byte function into a 24-byte stub.
- `--base` / `--symbols` / `--code-range` without `--raw` is an error.
- `--sink-map sinks.json` (`{"0x419a44e0": "sprintf"}`, ELF or raw) and
  `LibcRoleInferer` (see [libc-roles.md](libc-roles.md)) name sinks when no symbols exist.

## Limits

- One region only. Images with code at several load addresses (e.g. code copied to RAM at
  boot) need one context per region.
- Without a symbols file, `LibcRoleInferer` names the libc-shaped sinks (arm64). Sources
  (`recv`, framework accessors) cannot be inferred: pass `--sources`. The engines warn when
  no sink or source resolves.
- An uncalled function with no prologue the scanner knows is not found (dead code,
  functions reached only through a table).
- arm64 without `base`: base 0. ADRP-relative xrefs still resolve for any page-aligned base.
  Absolute pointers in data do not.
- `ARM32TaintTracker` taints only `r0` after a source (the return value), not the received
  buffer. This limit exists for ELF too.
