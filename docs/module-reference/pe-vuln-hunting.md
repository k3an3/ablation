# PE vuln hunting: x86-64 lift, Windows APIs, UCRT shapes

Modules: `lift_x86_64.py` (X64Lifter, x86_names), `decode.py` (decoder_for), `winapi.py`
(Windows API tables, MSVC CRT notes, pe_sections), plus the x86-64 PE loader in
`binary_context.py` (`_x64_pe_sweep`).

## Why this exists

1. **x86-64 PE had none of the vuln-hunting passes.** `taint`, `fmtstr` and `heap` fell
   back to the older SysV-only engines and found nothing on a DLL with `recv -> system`
   paths. `lengths`, `wrappers`, `reach`, `egress` and `overflow` refused to run.
2. **The x86-64 PE loader was a prologue byte scan.** It missed leaf functions (no
   `.pdata`), invented starts mid-function on `push rbx` patterns, and left `call
   [rip+__imp_X]` edges unlabelled, so nothing downstream knew a call reached `recv`.
3. **AArch64 PE ran the passes but saw no sources or sinks.** The taint tracker read the
   PE through an ELF loader (section RVAs taken as VAs), and dllimport calls (`adrp ;
   ldr ; blr` through the IAT) were opaque indirect calls.
4. **Every table was libc-only.** `ReadFile`, `InternetReadFile`, `WinExec`,
   `CreateProcessA`, `LoadLibraryA`, `lstrcpyA`, `wsprintfA`, `HeapAlloc` / `HeapFree`,
   `CreateFileA` were unknown names.
5. **Modern MSVC binaries never import `printf` / `sscanf`.** The UCRT headers compile
   `sprintf` / `sscanf` into the image as small varargs functions that call
   `__stdio_common_vsprintf` / `__stdio_common_vsscanf`. A scanner keyed on libc names sees
   neither the format nor the destination.
6. **Win64 is not AAPCS64.** Four register arguments, stack arguments behind a 0x20-byte
   home area, varargs spilled into that home area, the return address on the stack
   instead of in x30: slot maps, width-less `%s` destinations and varargs detection all
   assumed AAPCS64.

## Design: one set of passes, two front ends

`decoder_for(ctx)` returns the instruction source for a context:

| image | decoder |
|---|---|
| AArch64 ELF / raw | Capstone ARM64 (unchanged) |
| AArch64 PE | Capstone ARM64 + IAT peephole: `adrp ; ldr xT, [IAT slot] ; blr xT` -> `bl <slot>` |
| x86-64 PE | `X64Lifter("win64")` |
| x86-64 ELF | `X64Lifter("sysv")` (lengths / wrappers / reach / egress / overflow; taint / fmtstr / heap keep the older x86 engines on ELF) |

Every pass that decoded with Capstone ARM64 (taint engine and its tag scanners, fmtstr,
lengths, frame layout, call annotation, wrappers, reachability, egress, libc roles, path
solver) now asks the context's decoder. A fix to a pass applies to both architectures.

## The lift (X64Lifter)

Register map (lifted names are what the passes see; reports print the x86 names):

| ABI | args | return | scratch | callee-saved |
|---|---|---|---|---|
| Win64 | rcx x0, rdx x1, r8 x2, r9 x3 | rax x9 | r10 x10, r11 x11 | rbx x19, rsi x20, rdi x21, r12-r15 x22-x25, rbp x29 |
| SysV | rdi x0, rsi x1, rdx x2, rcx x3, r8 x4, r9 x5 | rax x9 | r10, r11 | rbx x19, r12-r15 x22-x25, rbp x29 |

- Entry: `str x30, [sp, #-8]!` (size 0) models the return address the call pushed, so the
  frame layout sees a real "saved x30" slot: a stack overflow's budget to the return
  address is computed by the same code as on AArch64. `ret` reloads it after `mov x0, x9`.
- After every call: `mov x9, x0` (rax holds the result).
- `test r, r` -> `cmp r, #0` (`je` is a zero test; `jle` / `js` are signed lower bounds);
  `add r, -4` -> `sub r, #4` (header-size underflow sites); `cmovcc` -> `csel`;
  `movsxd` / `movsx` / `movzx` -> `sxtw` / `sxt*` / `uxt*` or `ldrs*` / `ldr{b,h}`;
  `lea` -> `add` / `sub` / `adrp+add`.
- Flags: add / sub / and become flag-setting only when the next instruction reads flags.
- Memory: `[b + i*s + d]` -> `add x16, b, i, lsl #s` + `[x16, #d]`; rip-relative ->
  `adrp x16` + `[x16, #lo]` (globals and strings resolve like on AArch64); `gs:` -> TLS.
- `call [rip+IAT]` -> `bl <slot>`; `jmp [rip+IAT]` (tail call / thunk) -> `b <slot>`;
  conditional tail calls (`jne other_function`) are treated as calls on one path.
- `mov eax, N ; call __chkstk ; sub rsp, rax` -> `sub sp, sp, #N`.
- Micro-ops share the x86 instruction's address and size: every reported VA is a real,
  byte-verifiable instruction address. `cfg_arm64.build_cfg` keeps them in one block.
- Pushes of outgoing arguments (SysV) move sp in the body: the annotator re-keys stack
  slots and keeps stack addresses relative to the post-prologue sp.
- Unmodelled instructions lift to `x86.<mnemonic> rd, ...`: the destination is clobbered
  with the union of the sources' taint, never silently cleaned.

## ABI awareness (abi_of)

`abi_of(ctx)` gives arg registers and the stack-argument base (AAPCS64: 8 regs, base 0;
Win64: 4 regs, base 0x20; SysV: 6 regs, base 0). Used by:

- call annotation: `stack+0x8k` is the k-th stack argument (Win64: `[rsp+0x20+8k]` at the call);
- width-less `%s` destination mapping (argument n past the registers);
- slot maps: `%1$..` read the remaining argument registers, then the caller's stack from the
  stack-argument base; the return address is "saved return address" on x86-64;
- varargs detection: the Win64 home-area spill of the last argument registers (r8 / r9 to
  `[rsp+0x18] / [rsp+0x20]`) or the SysV register save area.

## Windows API tables (winapi.py)

Merged into the passes' tables at load (`merge()`, idempotent). Highlights:

| class | names |
|---|---|
| sources (buffer, length) | ReadFile, InternetReadFile, WinHttpReadData, _read, recvfrom, HttpQueryInfo, GetEnvironmentVariable, DeviceIoControl, PeekNamedPipe; results: GetCommandLine, getenv |
| exec / load | WinExec, CreateProcess{A,W}, CreateProcessAsUser, ShellExecute{,Ex}, LoadLibrary{,Ex}{A,W}, _wsystem, _popen |
| copies | lstrcpy / lstrcat / StrCpy / StrCat (A/W), wcscpy, wcscat, _mbscpy, RtlMoveMemory, RtlCopyMemory |
| formats | wsprintf / wvsprintf (A/W), _snprintf, _snwprintf, swprintf, _swprintf, *_s, __stdio_common_v* (UCRT) |
| width-less `%s` | wsprintf, _swprintf, swscanf, __stdio_common_vsprintf (count -1 only), __stdio_common_vsscanf |
| allocators / frees | HeapAlloc / HeapReAlloc / HeapFree (pointer in the third argument), LocalAlloc, GlobalAlloc, VirtualAlloc, CoTaskMemAlloc, SysAllocStringLen, MSVC operator new / delete |
| egress | CreateFile{A,W} (dwDesiredAccess write bits), _open (MSVC _O_* values), _wfopen; WriteFile; WSASend, InternetWriteFile, HttpSendRequest, WinHttpWriteData |

UCRT: the header's inline `sprintf` / `sscanf` are inferred as wrappers of
`__stdio_common_vsprintf` / `__stdio_common_vsscanf`. A wrapper gets the `dest` kind (its
parameter that becomes the destination) only when its inner call passes count `-1`, so
`snprintf` through the UCRT is not a width-less finding; sscanf wrappers get `scanfmt`.

`%n` notes on PE: `blocked by default (UCRT ...)` when the image links the UCRT; `may be
enabled` when it imports `_set_printf_count_output`; `not supported by wsprintfA`
(user32 formatter, 1024-character cap).

Literals: `.rdata` (non-writable, non-executable sections at imagebase + RVA) holds
literals; writable sections are "format in writable memory". Wide literals (`L"%s"`) are
read as UTF-16LE.

## Loader (x86-64 PE)

`.pdata` RUNTIME_FUNCTIONs (chained-unwind fragments dropped) + exports + direct call
targets + leaf functions: the first instruction of each stretch `.pdata` does not cover,
and after a `ret` / `jmp` in such a stretch when padding or 16-byte alignment follows and
no earlier branch targets it. One linear sweep labels `call / jmp [rip+IAT]` edges with
the import and names `jmp [rip+IAT]` thunks. The prologue byte scan remains only for
images without `.pdata`. Cache version 5 (x86-64 caches from v4 are rebuilt).

## Output

On x86-64 the CLI prints x86 register names (`rcx`, `r8`, `eax`), "saved return address"
and "x86-64 ..." headings. Python APIs and `--json` keep the lifted names (`x0` = first
argument); `ablation.analyzers.lift_x86_64.x86_names(text, abi_of(ctx))` converts.

## Known limits

- 32-bit x86 PE is not lifted (the older engines still run on it).
- `rep movs` is modelled as a qword copy of taint, not a sized copy.
- Win64 arguments past the fourth are stack slots; table entries naming x4+ apply to
  AArch64 PE only (RegQueryValueEx's data buffer, CreateFile's later parameters).
- The 64->32 truncating-store case (`narrowing_store`) depends on a 64-bit load being
  stored as 32 bits; x86-64 compilers often load only the low 32 bits, which reads as a
  32-bit field (reported at the use as `sext_unchecked`).
