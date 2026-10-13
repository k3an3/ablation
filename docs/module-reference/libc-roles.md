# LibcRoleInferer (sinks without imports)

**File:** `ablation/analyzers/libc_roles.py`

---

## Why this exists

Ablation had 3 gaps here:

**1. Name-keyed sink oracles found nothing in static or raw images.** A headerless RTOS image
or a statically linked ELF has no PLT and no imports. `ablation taint`, `fmtstr`, `heap` and
`overflow` match calls against `memcpy` / `sprintf` / `malloc`..., found nothing to match,
and reported "no findings". On a 92 MB image that read as a cleared surface. In fact it held
an unbounded `sprintf("%s")` into a stack buffer.

**2. Nothing said the pass was vacuous.** An engine with zero resolved sinks printed the same
"no findings" as a clean binary.

**3. Names had to come from somewhere.** Before this, the only option was a hand-written
symbols file.

---

## What it does

`LibcRoleInferer(ctx).infer()` returns `{va: role}` and stores it in `ctx.roles`, which is
cached. `ctx.name(va)` and every engine that resolves call names use it: ARM64 / ARM32 taint,
the annotators, fmtstr, heap and overflow.

| Evidence | Role |
|---|---|
| leaf; loop stores the value it loaded through another base; not NUL-terminated; reads x0, x1, x2 | `memcpy` |
| leaf; loop stores without loading, value from x1 / xzr | `memset` |
| leaf; NUL-terminated loop that copies | `strcpy` |
| leaf; NUL-terminated loop that only loads | `strlen` |
| ≥ 50% of call sites pass a `%` literal in x0 | `printf` |
| same in x1, x0 a stack buffer | `sprintf` |
| same in x1, every conversion's argument a stack address | `sscanf` |
| same in x1, otherwise | `fprintf` (stream / fd / level first) |
| same in x2, x1 not a string | `snprintf` |
| other index k | `__fmt_x<k>`: a logger; `ARM64FormatStringScanner` checks xk |

Call-site evidence comes from `ARM64RegAnnotator`. It runs over the functions that reference
a format string, and then over the other callers of anything that received one, because short
formats like `"%s"` are not in the string table. Formats shorter than the string table's
minimum are read from the image. `max_callers` (4000) caps the annotation work on huge images.

## CLI

`taint` / `overflow` / `fmtstr` / `heap` / `lengths` infer roles automatically on arm64 when
no sink name resolves. `--infer-sinks` forces inference and `--no-infer-sinks` disables it.
If nothing resolves even after inference:

```
WARNING: 0 of 40 fmtstr sink oracles resolved (no PLT, symbol, sink-map or inferred name
for memcpy / sprintf / ...); results are not meaningful. Use --sink-map, --symbols or
--infer-sinks.
```

Sources get the same warning, since recv-like inputs cannot be inferred. `--sink-map
sinks.json` names functions by address directly: `{"0x419a44e0": "sprintf"}`.

## Limits

- arm64 only.
- No `malloc` / `free` / `recv` inference. Use `--sink-map` / `--sources`.
- Signature sets for specific libcs (newlib, musl, uClibc, glibc, Bionic) are not shipped. The
  shapes above cover them without per-version signatures.
- An optimised `memcpy` that only copies through NEON pairs with size classes but no loop is
  missed. So is a `printf` with fewer than 3 call sites.
- A copy-until-NUL helper is named `strcpy`. It behaves like one as a sink.
