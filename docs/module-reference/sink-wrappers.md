# SinkWrapperInferer (vendor wrappers of sinks)

**File:** `ablation/analyzers/sink_wrappers.py` · CLI: `ablation wrappers <bin>`, `--sink-wrappers`

---

## Why this exists

Ablation had 3 gaps here:

**1. Vendor wrappers hid every call site.** A vendor `log_event(int kind, const char *fmt, ...)`
that forwards to `vsnprintf` was invisible to name-keyed sinks, and so was each of its call
sites. Four of those sites passed request data as the format, and `fmtstr` reported 82 other
findings but not these. Only the `vsnprintf` inside the helper was seen, and there the format
is legitimately the caller's problem.

**2. Firmware sinks sit one or two levels above libc.** Logging helpers, `run(cmd)` →
`system`, `buf_put(b, p, n)` → `memcpy`: the unsanitised data reaches the wrapper, not libc.

**3. `lengths` keyed only on libc sinks.** It missed conversion bugs whose sink was a vendor
read wrapper.

---

## What it infers

A function that passes one of its own parameters, unchanged, to a sink argument is a wrapper
of that kind. Copies and `-O0` spills are followed, and so are `-fPIC` PLT calls to the
binary's own exports. Inference runs to a fixpoint, so wrappers of wrappers are found too
(`dbg(msg)` → `log_event` → `vsnprintf`).

| Kind | Sink argument reached |
|---|---|
| `format` | format of printf / `v*printf` / `__*printf_chk` (or of a format wrapper) |
| `exec` | `system` / `popen` / `exec*` command arguments |
| `length` | length arguments: `memcpy` x2, `read` x2, `malloc` x0, `snprintf` x1 ... |
| `copy` | pointer arguments whose pointee must not be attacker data |

For varargs wrappers, the AAPCS64 register save area is recorded:
- `gr_offs = -8 × GP slots saved` (6 slots → `-0x30`), from `stp x(n)..x7` to consecutive
  slots;
- `vr_offs` from the q-register saves.

A wrapper whose chain ends in a `__*printf_chk` records its FORTIFY flag.

```
sink wrappers: 5 (param -> sink argument; override with --sink-map '{"0xVA": "format:x1"}')
  0x10744  log_event   -> vsnprintf  [format x1, copy x1]  varargs (6 GP, 8 FP slots saved)
  0x107f4  fn_107f4    -> vsnprintf via log_event  [format x0, copy x0]
  0x108d0  fn_108d0    -> system  [exec x0]
  0x10918  fn_10918    -> memcpy  [length x2, copy x1]
```

## Consumers

- **fmtstr:** each call site of a format wrapper is checked with the caller's provenance, and
  `recv` data passed there is HIGH. The forwarding inside the wrapper is not a finding. A
  wrapper with no caller in the binary is LOW.
- **taint:** exec and format wrappers are sinks; exec wrappers are CRITICAL.
- **lengths:** length wrappers are size sinks.

Wrappers are inferred automatically on arm64 and cached in `ctx.wrappers`. `--no-infer-sinks`
disables inference. A `--sink-map` value of `"format:x1"` / `"exec:x0"` / `"length:x2"` /
`"copy:x1+x2"` adds or overrides a wrapper by address.

## Limits

- A parameter that is transformed before reaching the sink (`fmt + 1`, a struct field) is
  not a passthrough.
- Indirect calls (`blr`) inside a wrapper are not followed.
