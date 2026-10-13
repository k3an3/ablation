# ARM64FormatStringScanner / ARM64LengthUnderflowScanner

**Files:** `ablation/analyzers/format_string_scanner_arm64.py`,
`ablation/analyzers/length_underflow_arm64.py`, `ablation/analyzers/arm64_tag_scanner.py`

---

## Why this exists

4 things that were not possible before in Ablation:

**1. No format-string scan for arm64.** `FormatStringScanner` decodes x86-64 only (RDI/RSI
format registers, `lea [rip+...]` literals).

**2. No length-underflow scan for arm64.** `LengthUnderflowScanner` matches the x86
`lea -N(%reg); movzwl; call` sequence. On AArch64 the W-register subtraction wraps by itself
and usually has no truncation, so that byte pattern does not apply.

**3. Neither x86 scanner knows whether the value is attacker data.** Both classify by local
instruction shape. The ARM64 scanners also run the ARM64 taint pass. A format string built from
`recv` bytes is HIGH (`tainted`), and only attacker-derived lengths are reported as underflows.

**4. Unconsumed compares counted as guards.** `clang -O0` emits `subs w1, w9, #16` with nothing
reading the flags. The shared base (`ARM64TaggingScanner`) counts a flag-setting compare as a
guard only when a `b.cond` / `csel` / `ccmp` consumes its flags.

---

## ARM64FormatStringScanner

Output is the x86 scanner's `FmtStringFinding`. The format register comes from
`ARM64RegAnnotator` (literal, entry argument through spills, stack/memory load, call result),
covering tail calls and GOT calls.

| Provenance | Severity |
|---|---|
| string literal / `.rodata` address | not reported |
| `tainted` (register or pointee holds recv/read data) | HIGH |
| `arg_passthrough` (caller's argument, also via `-O0` spills) | HIGH |
| `stack_load`, `memory_load`, `call_result`, `register`, `unknown` | MEDIUM |

Covers printf / fprintf / sprintf / snprintf / dprintf / asprintf (+ `v*`), syslog, err / warn
families, the `__*_chk` fortify variants, and printk / panic. `puts` is not a format function
and is not checked.

```python
from ablation.analyzers.format_string_scanner_arm64 import ARM64FormatStringScanner
s = ARM64FormatStringScanner.from_path_full(elf)        # needs a BinaryContext
print(s.report(s.scan()))
#   HIGH    fmt_string  0x106b0+0x2c  syslog(x1=user?)  format string in x1 carries received data (recv)
```
CLI: `ablation fmtstr <bin>` dispatches on arch.

## ARM64LengthUnderflowScanner

`attacker_len - N` (`sub`/`subs`, 4 ≤ N ≤ 256) reaching any call argument. A finding is HIGH
when the argument is a length (`memcpy` / `memmove` / `memset` / `strncpy` / `read` / `recv`
`x2`, `snprintf` `x1`, allocator sizes, `copy_from_user`), MEDIUM otherwise. A later `uxth` /
`and #0xffff` is recorded as `truncate_va` (x86's `movzwl`). `guarded` means a compare / `cbz`
on the length before the subtraction, on the result after it, or a flag-consumed `subs`.

```python
from ablation.analyzers.length_underflow_arm64 import ARM64LengthUnderflowScanner
for f in ARM64LengthUnderflowScanner.from_path_full(elf).scan_unguarded():
    print(f.fmt())
```

## ARM64TaggingScanner (base)

Shared by these scanners and `ARM64IntOverflowScanner`:
- `_site()` picks operations and tags the result (`%<kind>@<va>`), and the tag travels as taint.
- `_on_call()` checks sinks with `sites_in(taint)`.
- `guarded(label, site)` applies the guard rules above.

Validated on clang `-O0` and `-O2` fixtures (see `tests/test_arm64_fmt_underflow.py`).
