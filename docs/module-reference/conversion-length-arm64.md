# ARM64ConversionLengthScanner

**File:** `ablation/analyzers/conversion_length_arm64.py` · CLI: `ablation lengths <bin> [--libc glibc|musl|other]`

---

## Why this exists

Ablation had 3 gaps here:

**1. Width and signedness conversions in lengths were unmodelled.** Four real bugs on one
AArch64 target shared one shape: checked at one width or signedness, used at another.
`ARM64IntOverflowScanner` looks at multiplies and `ARM64LengthUnderflowScanner` at `len - N`.
Neither sees a conversion.

**2. The AArch64 markers are exact, but nothing read them.** The markers are:
- `sxtw` / `sxth` / `sxtb` / `ldrsw` / `ldrsh` / `ldrsb` (and `, sxtw` operand extends);
- `and wT, wN, #0xffff` / `uxth` before a check while `wN` is used afterwards;
- `uxtb` counters in loop compares.

**3. snprintf's verdict depends on the libc.** Some builds reject size > INT_MAX and write
nothing, while others trust the size. One finding flipped from critical to non-existent on
that alone.

---

## Kinds

| Kind | Shape | Severity |
|---|---|---|
| `sext_signed_check` | value bounded only by a signed compare (b.gt / b.le ...) is sign-extended into a size: a negative passes | HIGH attacker / MEDIUM |
| `sext_unchecked` | attacker value sign-extended into a size, no check | HIGH |
| `sext_unchecked` | signed 32/16/8-bit field (`ldrsw` ...) straight into a size, e.g. a 64-bit length stored to an `int` global | LOW |
| `masked_check` | bound checked on `len & 0xffff` / `uxth`; the size uses full-width `len` | HIGH attacker / MEDIUM |
| `signed_clamp` | `min(len, MAX)` with a signed `csel` (lt / le / gt / ge) reaching a size: a negative passes the clamp | HIGH attacker / MEDIUM |
| `narrowing_store` | a 64-bit length (`strlen` / `read` / `recv` result) stored with a 32-bit `str` outside the frame (LOW), or attacker data stored narrow anywhere (HIGH) | HIGH / LOW |
| `byte_counter_loop` | loop compares a byte-width counter with a 16-bit / > 255 / attacker limit: never ends | HIGH attacker / MEDIUM |

**Guards.** These make a sign extension safe:
- an unsigned compare of the value (`b.hi` / `b.hs` / `b.lo` / `b.ls`);
- a signed upper bound together with a lower bound (`cmp #0` + `b.lt`, `tbnz #31`).

**Sinks** include inferred or sink-map *length wrappers*: a vendor `net_read(fd, b, n)` →
`read` is a size sink. `ablation lengths --json PATH` writes both scanners' findings. The
sinks are the length arguments of `memcpy` / `memmove` / `memset` / `read` / `recv` /
`snprintf` / `malloc` / `calloc` / `realloc` and similar calls.

**snprintf.** A converted or underflowed size reaching `snprintf` / `vsnprintf` x1 is rated by
libc. The same rule applies in `ARM64LengthUnderflowScanner`.
- `snprintf_rejects_huge=True` (`--libc musl`): LOW.
- `False` (`--libc glibc`): severity unchanged.
- `None`: MEDIUM, with a note.

`LIBC_SNPRINTF_REJECTS_HUGE` only lists libcs whose source was checked.

```
  HIGH    sext_signed_check   0x10b68  site 0x10b8c -> memcpy(x2) @0x10ba0  source=recv
      ldrsw at 0x10b8c after a signed bound check (b.gt at 0x10b90): a negative value passes
      and becomes a huge memcpy size
```

Validated on clang `-O1` / `-O2` fixtures: six bugs are found at both levels, and four
controls stay silent (`n > 64 || n < 0`, an unsigned compare, mask-then-check-then-use, a
16-bit counter). See `tests/test_arm64_audit_gaps.py`.

**Not reported** (guards that are modelled):
- A mask of a value already zero-extended (`ldrh` / `ldurh` / `ldrb` / `uxth`, kept through
  `rev16` and copies): the masked and copied lengths are the same value.
- `capacity - (used + ret)` after the `*printf` return was checked against the remaining
  capacity by a signed compare. This covers gcc's `cmp w0, #0; ccmp wR, w0, #4, ge; b.le`
  truncation idiom.

## Limits

- The guard search is linear and per function, like the other tag scanners. A check in the
  caller with the conversion in the callee is not linked.
- Structural tags (on values with no attacker label) stay inside the function.
- A narrowing store (`str w`) is only caught when the narrow value is reloaded with a
  sign-extending load.
