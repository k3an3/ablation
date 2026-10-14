# Reachability, config gates, egress coupling

**Files:** `ablation/analyzers/reachability_arm64.py`, `ablation/analyzers/egress_arm64.py` ·
CLI: `ablation reach <bin>`, `ablation egress <bin>`, `--show-suppressed`

---

## Why this exists

Ablation had 4 gaps here:

**1. Dead code was reported like live code.** Findings sat in a handler that is never
registered or called: no caller, and its address in no relocation, data word or constant.

**2. A live route was believed dead.** Its registration had not been found, which suppressed
re-testing of a real bug for months. Registrations have to be resolved and shown.

**3. Config-gated paths need a label, not a verdict.** An error path gated on a configuration
slot this build never sets is dead *in this build only*. It must be labelled, never silently
dropped.

**4. A leak and the route that serves it were reviewed separately.** A format-string leak
wrote to a log file that another handler served back verbatim. Each half was dismissed on its
own: one looked like local logging, the other like static file serving.

---

## Reachability (`Reachability`)

- **address-taken:** a function start appears as an 8-byte word anywhere in the relocated
  image (GOT, `.data.rel.ro`, `init_array`, tables), or is materialised by ADR / ADRP+ADD.
- **live:** reachable through call edges from the ELF entry point, dynamic exports (not
  `.symtab` names) and address-taken functions. For a raw image, the roots are its named
  functions.
- **unreachable** reasons:
  - `no callers, address never taken`;
  - `only called from unreachable code (...)`.
- **registrations:**
  - `setter(key, handler)`: a handler address and a string literal passed in one call;
  - `{key, handler}` data tables.

  Findings in a registered handler say `registered as '/live' via setter http_register (0x..)`.
- **gates:** the nearest branch that every path to the site takes, testing a global or a field
  of `*global` (`cbz` / `cbnz` / `b.eq`|`ne` after `cmp #0`). The global can be reached
  directly or through its GOT slot; imported globals are reported by name. Writers are then
  searched across the whole binary:

  | Gate | Label | Suppressed |
  |---|---|---|
  | written somewhere | `gated: reached only when [g_debug] is non-zero ...; written at 0x..` | no |
  | needs non-zero, never written (defined here) | `... never written in this binary: dead unless set from outside` | yes |
  | imported global | `... an imported global never written here: its own module may set it` | no |

`fmtstr` / `taint` / `heap` / `lengths` print `N finding(s) suppressed (1 unreachable, 1 gated
on a never-written global)` on every run. `--show-suppressed` lists those findings, marked
`SUPPRESSED` and with their reasons. Nothing is dropped silently.

**Roots and address-taken.**
- A program's `.dynsym` exports are entry points only for a shared library, or a program
  importing `dlopen` / `dlsym` (`Reachability.exports_are_roots`). An executable built with
  `-rdynamic` is not.
- Symbol tables, relocation records and hash / version / unwind / note / debug sections are
  not scanned for address words: a function's own `st_value` is not a reference.

## Egress coupling (`EgressMap`)

- **writers / readers per path literal.** Classified from the `fopen` mode, the `open` flags
  (`O_WRONLY|O_RDWR|O_CREAT|O_APPEND`) or `creat`.
- **Global handles.** A `FILE*` / fd that a write-open stores into a global links every
  function writing through that global to the path.
- **emitters.** `send` / `sendto` / `sendmsg` / `sendfile` / `SSL_write` / `writev` / `write`
  and `--egress` names, within two call levels of a reader.
- **fmtstr.** A finding whose function, or the wrapper chain it calls, writes such a path gets
  `details.egress`:

```
egress: write in log_event (0x10754) -> "/var/log/httpd.log" -> read in serve_log (0x107b8) -> send (0x107f8)
```

Vendor open helpers (`open_log(path, mode)` → `fopen`) are inferred as `open` wrappers with
their mode. Calls through `-fPIC` PLT stubs are followed to the binary's own exports.

**Coarse fallback.** It is used when nothing pairs exactly (a runtime-built path, an
unmodelled helper). It pairs two functions that reference the same path literal: one near the
finding that writes or opens within two calls, and one that reaches a send. Such chains are
labelled `(coarse: shared path literal)`.

`ablation egress <bin>` lists every path with its writers and readers, and marks `EGRESS` where
both exist.

## Limits

- Addresses computed at runtime are not followed: function pointers built with `movz`/`movk`,
  and handler tables filled in by code.
- Gates are matched on `cbz`/`cbnz` or a compare with 0, on a global or one field level.
- Egress needs path literals. A path built at runtime, or output shared through a global
  buffer, is not followed.
