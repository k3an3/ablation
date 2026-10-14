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
| `runtime_buffer`: format is a stack buffer (e.g. `sprintf`'d) or `std::string::c_str()` / `data()` | MEDIUM |
| `stack_load`, `memory_load`, `call_result`, `register`, `unknown` | MEDIUM |
| `unbounded_s` (see below) | HIGH / MEDIUM |

**Wrappers.** A format that is the function's own argument (also through `-O0` spills) makes
the function a printf wrapper. The bug, if any, is at the caller, so the verdict comes from
the wrapper's call sites in the binary, following nested wrappers (`log_mid(lvl, f)` →
`log_any(f)` → `printf(f)`):

| Callers | Result |
|---|---|
| a caller passes recv/read data (register or pointee) | HIGH `tainted`; that caller and site are named |
| a caller passes a non-literal | MEDIUM `arg_passthrough`; that caller's provenance |
| no caller in the binary (exported API) | LOW `arg_passthrough` |
| every caller passes a literal | not reported |

**What a controllable format reads (`details.slots`, `details.leak`).** The map is computed at
the variadic boundary: a direct printf-family call, or the call site of a varargs wrapper.
Non-variadic wrappers are followed inwards, and a frameless wrapper's tail call is mapped
against its caller's frame. `frame_layout_arm64.FrameLayout` supplies the frame size, the
saved `x29` / `x30` / callee-saved slots and the address-taken objects. Each `%i$`
(sequential or positional) maps to one of:
- stale GP argument registers (stack / image pointers among them are flagged);
- local slots;
- slots holding attacker bytes at the call (`%n` / `%s` targets);
- the caller's saved `x29`: stack disclosure;
- the caller's saved `x30`: code pointer, `base+K`, where K is the return site into the
  caller;
- the caller's caller frame (unknown).

How the map drives severity:
- A tainted format that renders a code or stack pointer is HIGH ("defeats PIE").
- One whose map reaches the saved `x30` without any such pointer is MEDIUM.

```
leak: %7$ reads attacker-controlled bytes; %39$p renders a stack address; %40$p renders the saved return address (base+0x...)
slots: h_bad frame 0x120 bytes; %1$..%6$: stale argument registers; %7$..%38$: attacker bytes (recv) ...;
       %39$: [sp+0x100] saved x29 ...; %40$: [sp+0x108] saved x30: return address into the caller ...
```

**Three buckets.** `ARM64RegAnnotator.annotate_calls_cfg` is path-sensitive: a register or
slot with different definitions on different paths is `unresolved` with the reason (e.g.
`ret(calloc) | 'lit %d'`). It is never silently taken as a literal.
- literal: not reported;
- externally-influenced: a finding;
- `unresolved (reason)`: LOW, listed separately as a to-do (`details.bucket`).

**FORTIFY (`details.fortify`).**
- `%n: blocked by FORTIFY (__vsnprintf_chk flag=1 ...)`;
- `%n: not blocked (... with flag 0)`;
- `%n: not blocked (plain vsnprintf ...)`.

This works through wrappers too. All `__*printf_chk` variants are format sinks.

**Egress (`details.egress`).** See [reachability-egress.md](reachability-egress.md).

**Writable formats.** A format in writable memory (`.data` / `.bss`) is `writable_format`
MEDIUM, with its initial text: code can change it.

**`--explain VA`.** For one call site, `ablation fmtstr --explain 0xSITE` prints:
- the callee, and whether it is a printf name or a format wrapper;
- the format register's value;
- the classification, or why the site is skipped;
- the taint at the call;
- reachability and registration.

**Unbounded `%s`.** For a literal format, `%s` with no precision (print family) or
`%s` / `%[` with no width (scan family, skipping `%*s` and `%ms`) is checked. It is flagged
when it writes into a stack buffer:

| Call | Checked target | HIGH when |
|---|---|---|
| `sprintf`, `vsprintf` | `x0` on the stack | a `%s` source arg's pointee holds recv/read data |
| `sscanf` (+ `__isoc99_`) | the `%s` / `%[` arg on the stack | the input string (`x0`) holds recv/read data |
| `scanf` | same | always (stdin) |
| `fscanf` | same | never (MEDIUM) |

The other cases are MEDIUM. Every width-less conversion with a stack destination is listed,
and vendor wrappers of the scan family are checked too. `ablation lengths` shows them as well
(under `"unbounded"` with `--json`). `details.budget` gives the bytes from the nearest
conversion to a saved `x30`. When the frame record sits below the buffer, the budget continues
into each caller's frame. `details.dest` gives the per-destination byte budget: the bytes to the next stack
object or saved register, and to this frame's saved `x30`. A note is added when the frame
record sits below the buffer, where an overflow runs into the caller's frame. `__sprintf_chk` is not flagged, because it aborts on overflow.
`unbounded_conversions(fmt, family)` exposes the parser.

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
