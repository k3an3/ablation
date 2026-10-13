# ARM64IntOverflowScanner

**File:** `ablation/analyzers/intoverflow_scanner_arm64.py`

---

## Why this exists

3 things that were not possible before in Ablation:

**1. No integer-overflow-in-allocation scan for arm64.** `ARM32IntOverflowScanner` covers
ARM32. `ablation overflow` refused every other arch, so "wire length × element size →
`malloc`" bugs in arm64 firmware and Android libraries had no scanner.

**2. "Wire-controlled" meant only the source's return value.** The ARM32 scanner treats an
operand as attacker-controlled when its reaching definition is the `bl recv` itself, i.e. the
byte count. The usual bug multiplies a *field read out of the received buffer*. This scanner
runs the ARM64 taint tracker, so buffer contents, spills, register aliases and callees count.

**3. Overflow was not judged by width or bounds.** A 16-bit field × 8, or a `u32` zero-extended
to 64 bits and then shifted by 4, cannot overflow. The same multiply on a 32-bit `w` register
can. Operand bounds and the operation's width decide this, so only products that can actually
wrap are reported.

---

## What it reports

A finding is a multiply-like instruction with an attacker-derived operand that can overflow
its width, whose result reaches an allocator's size argument:

| | |
|---|---|
| Multiplies | `mul` `madd` `msub` `mneg`, `lsl #k` / `lsl xR`, `add/sub xD, xN, xM, lsl #k` |
| Ignored | `umull` `smull` `umaddl` `smaddl` ... (32×32→64 cannot overflow); products bounded below `2**width` |
| Allocators | `malloc` `calloc` `realloc` `aligned_alloc` `memalign` `posix_memalign` `valloc` `_Znwm` `_Znam` `xmalloc` `g_malloc*` `CRYPTO_*alloc` `kmalloc` `kzalloc` `kvmalloc` `vmalloc` ... |
| Reach | same function, or through calls (the product is a taint label, so it crosses calls and spills) |
| `bounds_checked` | a compare / `cbz` / `tbz` tests the product after the multiply, or tests a tainted operand register before it, or a `umulh`/`smulh` overflow check is tested |

Compilers split a constant multiply (`n*12` → `(n + n<<1) << 2`), so steps feeding the same
allocation are merged into one finding: `add+lsl#1 -> lsl#2/32`.

---

## Usage

```python
from ablation.analyzers.intoverflow_scanner_arm64 import ARM64IntOverflowScanner

s = ARM64IntOverflowScanner.from_path_full('/path/to/target_a64.so')
findings = s.scan()            # interprocedural, depth 3
print(s.report(findings))
# INT_OVERFLOW  func=0x1068c  mul@0x106ac  add+lsl#1 -> lsl#2/32(x8, x8) -> malloc(x0)@0x106b4  wire=[recv] [UNGUARDED]
```

CLI: `ablation overflow <binary>` dispatches on arch (ARM32 or ARM64).

Validated on clang `-O0` and `-O2` fixtures. These are flagged unguarded: 32-bit `u32 * 12`,
64-bit `u64 * 24`, and a size computed in a callee. `__builtin_mul_overflow` is flagged guarded.
These are silent: `u16 * 8`, `(size_t)u32 * 16`, `calloc(n, 24)`, and an untainted size. On a
stripped 7.2k-function Rust `.so` it reports nothing, in 19s.

---

## Limits

- A bounds check counts if it tests the operand or the product. Whether the bound it enforces
  is tight enough is not evaluated; use `ARM64PathSolver` on the allocation site for that.
- Linux libc / kernel sources and allocators only; Windows (`ReadFile`, `HeapAlloc`) are not
  modelled.
