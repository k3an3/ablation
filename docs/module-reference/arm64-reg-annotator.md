# ARM64RegAnnotator

**File:** `ablation/analyzers/arm64_reg_annotator.py`

---

## Why this exists

4 things that were not possible before in Ablation:

**1. No call-site argument values on arm64.** `FuncProfiler` and `IPRegAnnotator` only had
the x86-64 `RegAnnotator` (`FuncProfiler` also had ARM32). On an arm64 binary they
disassembled AArch64 code as x86, so "what is passed to `system()`?" had no answer on arm64.

**2. ARM argument registers were invisible in reports.** `ProfiledCall.fmt`, `CallSite.fmt`
and `ChainCallSite.fmt` iterated a hard-coded `rdi..r9` list. ARM32 call args (keyed
`r0..r3`) were collected but printed as empty `name()`. The formatters now iterate the
call's own args in calling-convention order, so x86 output is unchanged and ARM args appear.

**3. Tail calls and `-fno-plt` calls hid sinks.** AArch64 compilers end functions with
`b system@plt`, and hardened builds call imports through `adrp; ldr x8, [GOT]; blr x8`.
Neither is a `bl`, so a `bl`-only pass misses both sinks.

**4. Interprocedural chains stopped at a `.so`'s own exports.** A `-fPIC` shared object
calls its own preemptible functions through its PLT. `IPRegAnnotator` treated those calls as
external leaves and stopped there. It now follows a PLT symbol into the same binary's export
when no `lib_graph` is in play. This applies to every arch.

---

## What it tracks

One linear pass per function (no branch or loop following), with AAPCS64 state:

| Pattern | Result |
|---|---|
| `adr xD, L` / `adrp xD, P ; add xD, xD, #lo` | const, or the string at that VA |
| `adrp xB, P ; ldr xD, [xB, #lo]` | pointer at the slot: string / const / `&import` |
| `ldr xD, <literal>` | 4/8-byte literal-pool value |
| `mov` / `movz` / `movn` / `movk` | const (movk patches one 16-bit lane) |
| `mov xD, xN` / `mov xD, xzr` / W-form | copy / zero / 32-bit truncation |
| `add/sub xD, sp\|x29, #imm`, `mov xD, sp` | stack address, `[sp+0x10]` |
| `ldr xD, [xN, #off]` (other) | `mem` description, tagged with an entry arg base |
| `bl T` | call: PLT name, export name, or hex |
| `blr xN` | call: named from an import GOT slot or a known function VA, else `blr xN` |
| `b T` | tail call when T is a PLT stub, another function start, or outside the window |

After each call `x0 = ret(callee)`, the rest of the caller-saved set (`x1..x17`, `x30`) is
unknown, and `x19..x28` survive. That is how `mov x19, x1; bl ioctl; mov x0, x19;
b system` reports `system(x0=arg1_entry)`.

---

## Usage

Automatic: `FuncProfiler` and `IPRegAnnotator` pick it when `ctx.arch == 'arm64'`.

```python
from ablation.analyzers.binary_context import BinaryContext
from ablation.analyzers.arm64_reg_annotator import ARM64RegAnnotator

ctx = BinaryContext.load_or_build('/path/to/target_a64.so')
ra = ARM64RegAnnotator.from_context(ctx)
print(ra.annotate_calls(0x10408, 0x1044c).fmt())
#   0x1041c: call helper
#     x0 = [sp]
#     x1 = arg0_entry (x0@entry)
#   0x10428: call puts
#     x0 = 'handler-string-xref'

# Seed a callee with explicit entry state (what IPRegAnnotator does per hop)
ra.annotate_calls(callee_va, entry_regs=call_site.args)
```

Measured on a stripped 7.2k-function AArch64 `.so`: all functions profiled in 12.5s
(~1.7 ms per function), with no failures.

---

## Limits

- Linear pass: values set on only one side of a branch can be wrong at a join.
- No stack-slot tracking: a value spilled with `str` and reloaded is reported as `mem`.
- `BinaryContext.call_edges` still has no edges for `blr` through the GOT. Only the
  annotator resolves those.
