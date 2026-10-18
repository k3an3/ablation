"""
winapi.py: Windows API knowledge for the Insn-model passes (PE, x86-64 and AArch64).

Why this exists
  1. Every source / sink / length / allocator / format table in the AArch64 passes was
     libc-only: recv was a source, but ReadFile, InternetReadFile and WinHttpReadData were
     not; system was an exec sink, WinExec / CreateProcess / ShellExecute / LoadLibrary
     were not; sprintf was a copy, lstrcpyA / wcscpy / wsprintfA / _snprintf were not.
     A PE could only ever show findings that went through the C runtime.
  2. The MSVC CRT differs where it matters for triage: %n is rejected by default
     (_set_printf_count_output), _snprintf / _snwprintf do not NUL-terminate on
     truncation, wsprintf caps output at 1024 characters and has no %n, and the *_s
     functions take a destination size.

Arguments use the lifted names (x0 = first argument), which are the same on AArch64
Windows (AAPCS64) and on x86-64 through lift_x86_64 (Win64: rcx, rdx, r8, r9). An
argument past the fourth on x86-64 lives on the stack and is not listed here.

merge() folds these into the module tables at import time; nothing here is PE-only in
effect: a name is a name, and ELF binaries do not import these.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

_W = ("A", "W")


def _aw(*names: str):
    """'lstrcpy' -> ('lstrcpyA', 'lstrcpyW')."""
    return tuple(n + s for n in names for s in _W)


# ── taint: attacker input ───────────────────────────────────────────────────────

# name -> (buffer arg, length arg or None)
SOURCE_BUFFERS: Dict[str, Tuple[str, Optional[str]]] = {
    "ReadFile": ("x1", "x2"), "ReadFileEx": ("x1", "x2"),
    "InternetReadFile": ("x1", "x2"), "WinHttpReadData": ("x1", "x2"),
    "_read": ("x1", "x2"), "fgetws": ("x0", "x1"), "_getws": ("x0", None),
    "ReadConsoleA": ("x1", "x2"), "ReadConsoleW": ("x1", "x2"),
    "GetEnvironmentVariableA": ("x1", "x2"), "GetEnvironmentVariableW": ("x1", "x2"),
    "HttpQueryInfoA": ("x2", None), "HttpQueryInfoW": ("x2", None),
    "WinHttpQueryHeaders": ("x3", None),
    "DeviceIoControl": ("x2", "x3"),               # input buffer of an IOCTL client stub
    "PeekNamedPipe": ("x1", "x2"),
    "recvfrom": ("x1", "x2"),
}
# sources whose result (pointer and what it points at) is attacker data
SOURCE_RETURNS = frozenset({
    "GetCommandLineA", "GetCommandLineW", "_wgetenv", "getenv", "CommandLineToArgvW",
})
SOURCES = frozenset(SOURCE_BUFFERS) | SOURCE_RETURNS | frozenset({
    "WSARecv", "WSARecvFrom", "recv", "InternetReadFileExA", "InternetReadFileExW",
    "HttpReceiveHttpRequest", "HttpReceiveRequestEntityBody",
})

# ── sinks ─────────────────────────────────────────────────────────────────────────

EXEC_ARGS: Dict[str, Tuple[str, ...]] = {
    "WinExec": ("x0",), "_wsystem": ("x0",), "_popen": ("x0",), "_wpopen": ("x0",),
    "CreateProcessA": ("x0", "x1"), "CreateProcessW": ("x0", "x1"),
    "CreateProcessAsUserA": ("x1", "x2"), "CreateProcessAsUserW": ("x1", "x2"),
    "CreateProcessWithLogonW": (), "ShellExecuteA": ("x2", "x3"), "ShellExecuteW": ("x2", "x3"),
    "ShellExecuteExA": ("x0",), "ShellExecuteExW": ("x0",),
    "LoadLibraryA": ("x0",), "LoadLibraryW": ("x0",), "LoadLibraryExA": ("x0",),
    "LoadLibraryExW": ("x0",), "_execv": ("x0", "x1"), "_execvp": ("x0", "x1"),
    "_wexecv": ("x0", "x1"), "_spawnv": ("x1", "x2"), "_wspawnv": ("x1", "x2"),
}
EXEC_SINKS = frozenset(EXEC_ARGS)

# copies: name -> (dst, src args, length arg or None)
COPY_FUNCS: Dict[str, Tuple[str, Tuple[str, ...], Optional[str]]] = {
    **{n: ("x0", ("x1",), None) for n in _aw("lstrcpy", "lstrcat", "StrCpy", "StrCat")},
    **{n: ("x0", ("x1",), "x2") for n in _aw("lstrcpyn", "StrCpyN", "StrNCat")},
    "wcscpy": ("x0", ("x1",), None), "wcscat": ("x0", ("x1",), None),
    "_mbscpy": ("x0", ("x1",), None), "_mbscat": ("x0", ("x1",), None),
    "wcsncpy": ("x0", ("x1",), "x2"), "wcsncat": ("x0", ("x1",), "x2"),
    "RtlMoveMemory": ("x0", ("x1",), "x2"), "RtlCopyMemory": ("x0", ("x1",), "x2"),
    "CopyMemory": ("x0", ("x1",), "x2"), "MoveMemory": ("x0", ("x1",), "x2"),
    "wsprintfA": ("x0", ("x1", "x2", "x3"), None), "wsprintfW": ("x0", ("x1", "x2", "x3"), None),
    "wvsprintfA": ("x0", ("x1", "x2"), None), "wvsprintfW": ("x0", ("x1", "x2"), None),
    "_swprintf": ("x0", ("x1", "x2", "x3"), None), "swprintf": ("x0", ("x2", "x3"), "x1"),
    "_snprintf": ("x0", ("x2", "x3"), "x1"), "_snwprintf": ("x0", ("x2", "x3"), "x1"),
    "_vsnprintf": ("x0", ("x2", "x3"), "x1"), "_vsnwprintf": ("x0", ("x2", "x3"), "x1"),
    "vswprintf": ("x0", ("x2", "x3"), "x1"),
    "__stdio_common_vsprintf": ("x1", ("x3",), "x2"), "__stdio_common_vswprintf": ("x1", ("x3",), "x2"),
}
# sink args whose pointee carries the attacker data
SINK_POINTER_ARGS: Dict[str, Tuple[str, ...]] = {
    **{n: a for n, a in EXEC_ARGS.items() if a},
    **{n: src for n, (_d, src, _l) in COPY_FUNCS.items()},
    "swscanf": ("x0",), "sscanf_s": ("x0",), "swscanf_s": ("x0",),
    "__stdio_common_vsscanf": ("x1",), "__stdio_common_vswscanf": ("x1",),
}
SINKS = frozenset(SINK_POINTER_ARGS) | EXEC_SINKS

# ── formats ───────────────────────────────────────────────────────────────────────

FMT_ARG: Dict[str, str] = {
    "wprintf": "x0", "vwprintf": "x0", "printf_s": "x0", "wprintf_s": "x0",
    "fwprintf": "x1", "vfwprintf": "x1", "fprintf_s": "x1", "fwprintf_s": "x1",
    "wsprintfA": "x1", "wsprintfW": "x1", "wvsprintfA": "x1", "wvsprintfW": "x1",
    "_swprintf": "x1", "swprintf": "x2", "vswprintf": "x2",
    "_snprintf": "x2", "_snwprintf": "x2", "_vsnprintf": "x2", "_vsnwprintf": "x2",
    "sprintf_s": "x2", "swprintf_s": "x2", "vsprintf_s": "x2", "vswprintf_s": "x2",
    "_snprintf_s": "x3", "_snwprintf_s": "x3", "_vsnprintf_s": "x3",
    "_printf_p": "x0", "_sprintf_p": "x2", "_fprintf_p": "x1",
    # UCRT: every printf-family call in a modern MSVC binary lands here, through the
    # header's inline sprintf / printf (compiled into the image as small varargs wrappers)
    "__stdio_common_vfprintf": "x2", "__stdio_common_vfprintf_s": "x2", "__stdio_common_vfprintf_p": "x2",
    "__stdio_common_vfwprintf": "x2", "__stdio_common_vfwprintf_s": "x2",
    "__stdio_common_vsprintf": "x3", "__stdio_common_vsprintf_s": "x3", "__stdio_common_vsprintf_p": "x3",
    "__stdio_common_vswprintf": "x3", "__stdio_common_vswprintf_s": "x3",
    "OutputDebugStringA": None, "OutputDebugStringW": None,
}
FMT_ARG = {k: v for k, v in FMT_ARG.items() if v}
# width-less %s destinations: name -> (fmt reg, first vararg index, dst reg, family)
UNBOUNDED: Dict[str, Tuple[str, Optional[int], Optional[str], str]] = {
    "wsprintfA": ("x1", 2, "x0", "print"), "wsprintfW": ("x1", 2, "x0", "print"),
    "wvsprintfA": ("x1", None, "x0", "print"), "wvsprintfW": ("x1", None, "x0", "print"),
    "_swprintf": ("x1", 2, "x0", "print"),
    "swscanf": ("x1", 2, "x0", "scan"), "wscanf": ("x0", 1, None, "scan"),
    "fwscanf": ("x1", 2, None, "scan"),
    # UCRT (options, buffer, count, format, locale, va_list): unbounded only when the
    # count is -1, which is what the header's sprintf / vsprintf pass (COUNT_ARG)
    "__stdio_common_vsprintf": ("x3", None, "x1", "print"),
    "__stdio_common_vswprintf": ("x3", None, "x1", "print"),
    "__stdio_common_vsscanf": ("x3", None, "x1", "scan"),
    "__stdio_common_vswscanf": ("x3", None, "x1", "scan"),
}
# UNBOUNDED entries that are bounded unless this argument is the constant -1
COUNT_ARG = {"__stdio_common_vsprintf": "x2", "__stdio_common_vswprintf": "x2"}
STDIN_SCANS = frozenset({"wscanf", "scanf_s", "wscanf_s"})
# format sinks whose output is capped regardless of the destination
OUTPUT_CAP = {"wsprintfA": 1024, "wsprintfW": 1024, "wvsprintfA": 1024, "wvsprintfW": 1024}
# no %n at all in this formatter
NO_PERCENT_N = frozenset({"wsprintfA", "wsprintfW", "wvsprintfA", "wvsprintfW"})
# truncation leaves the destination without a NUL
NO_NUL_ON_TRUNC = frozenset({"_snprintf", "_snwprintf", "_vsnprintf", "_vsnwprintf", "strncpy",
                             "wcsncpy", "lstrcpynA"})

# ── lengths and allocators ───────────────────────────────────────────────────────

LENGTH_ARGS: Dict[str, Tuple[str, ...]] = {
    "ReadFile": ("x2",), "InternetReadFile": ("x2",), "WinHttpReadData": ("x2",), "_read": ("x2",),
    "WriteFile": ("x2",), "_write": ("x2",), "RtlMoveMemory": ("x2",), "RtlCopyMemory": ("x2",),
    "CopyMemory": ("x2",), "MoveMemory": ("x2",), "RtlZeroMemory": ("x1",), "ZeroMemory": ("x1",),
    "RtlFillMemory": ("x1",), "wcsncpy": ("x2",), "wcsncat": ("x2",), "lstrcpynA": ("x2",),
    "lstrcpynW": ("x2",), "memcpy_s": ("x1", "x3"), "memmove_s": ("x1", "x3"),
    "strncpy_s": ("x1", "x3"), "wcsncpy_s": ("x1", "x3"),
    "_snprintf": ("x1",), "_snwprintf": ("x1",), "_vsnprintf": ("x1",), "_vsnwprintf": ("x1",),
    "swprintf": ("x1",), "vswprintf": ("x1",),
    "MultiByteToWideChar": ("x3",), "WideCharToMultiByte": ("x3",),
    "HeapAlloc": ("x2",), "HeapReAlloc": ("x3",), "LocalAlloc": ("x1",), "GlobalAlloc": ("x1",),
    "VirtualAlloc": ("x1",), "CoTaskMemAlloc": ("x0",), "SysAllocStringLen": ("x1",),
    "SysAllocStringByteLen": ("x1",), "_malloc_base": ("x0",), "_calloc_base": ("x0", "x1"),
    "_aligned_malloc": ("x0",), "send": ("x2",), "WSASend": (),
}
LENGTH_ARGS = {k: v for k, v in LENGTH_ARGS.items() if v}
ALLOC_SIZE_ARGS: Dict[str, Tuple[str, ...]] = {
    "HeapAlloc": ("x2",), "HeapReAlloc": ("x3",), "RtlAllocateHeap": ("x2",),
    "LocalAlloc": ("x1",), "LocalReAlloc": ("x1",), "GlobalAlloc": ("x1",), "GlobalReAlloc": ("x1",),
    "VirtualAlloc": ("x1",), "VirtualAllocEx": ("x2",), "CoTaskMemAlloc": ("x0",),
    "CoTaskMemRealloc": ("x1",), "SysAllocStringLen": ("x1",), "SysAllocStringByteLen": ("x1",),
    "_malloc_base": ("x0",), "_calloc_base": ("x0", "x1"), "_realloc_base": ("x1",),
    "_aligned_malloc": ("x0",), "_recalloc": ("x1", "x2"),
    "??2@YAPEAX_K@Z": ("x0",), "??_U@YAPEAX_K@Z": ("x0",),     # operator new / new[] (MSVC x64)
}
# heap_scanner_arm64._ALLOCS shape: name -> (size args, realloc'd pointer arg or None)
ALLOCS: Dict[str, Tuple[Tuple[str, ...], Optional[str]]] = {
    **{n: (a, None) for n, a in ALLOC_SIZE_ARGS.items()},
    "HeapReAlloc": (("x3",), "x2"), "LocalReAlloc": (("x1",), "x0"), "GlobalReAlloc": (("x1",), "x0"),
    "CoTaskMemRealloc": (("x1",), "x0"), "_realloc_base": (("x1",), "x0"),
}
FREES = frozenset({"HeapFree", "RtlFreeHeap", "LocalFree", "GlobalFree", "VirtualFree",
                   "CoTaskMemFree", "SysFreeString", "_free_base", "_aligned_free",
                   "??3@YAXPEAX@Z", "??_V@YAXPEAX@Z", "??3@YAXPEAX_K@Z", "??_V@YAXPEAX_K@Z"})
# free whose pointer is not x0: name -> pointer arg
FREE_PTR_ARG = {"HeapFree": "x2", "RtlFreeHeap": "x2", "VirtualFree": "x0"}
CXX_NEW = frozenset({"??2@YAPEAX_K@Z", "??_U@YAPEAX_K@Z"})
STR_COPIES = frozenset({"lstrcpyA", "lstrcpyW", "lstrcatA", "lstrcatW", "wcscpy", "wcscat",
                        "StrCpyA", "StrCpyW", "wsprintfA", "wsprintfW", "_swprintf"})
SIZED_COPIES: Dict[str, Tuple[str, str]] = {
    "RtlMoveMemory": ("x0", "x2"), "RtlCopyMemory": ("x0", "x2"), "CopyMemory": ("x0", "x2"),
    "wcsncpy": ("x0", "x2"), "ReadFile": ("x1", "x2"), "InternetReadFile": ("x1", "x2"),
    "WinHttpReadData": ("x1", "x2"), "_read": ("x1", "x2"), "RtlZeroMemory": ("x0", "x1"),
}
STRLENS = frozenset({"lstrlenA", "lstrlenW", "wcslen", "_mbslen", "StrLenA"})

# ── egress (files written / read back to a peer) ─────────────────────────────────

# name -> (path arg, mode / flags arg, how to read it)
OPEN_FUNCS: Dict[str, Tuple[str, Optional[str], str]] = {
    "_wfopen": ("x0", "x1", "wmode"), "_fsopen": ("x0", "x1", "mode"), "_wfsopen": ("x0", "x1", "wmode"),
    "_open": ("x0", "x1", "msflags"), "_wopen": ("x0", "x1", "msflags"),
    "CreateFileA": ("x0", "x1", "access"), "CreateFileW": ("x0", "x1", "access"),
}
# write intent: CreateFile dwDesiredAccess (GENERIC_WRITE / GENERIC_ALL / FILE_WRITE_DATA /
# FILE_APPEND_DATA), MSVC _open flags (_O_WRONLY / _O_RDWR / _O_APPEND / _O_CREAT / _O_TRUNC)
WRITE_MASK = {"access": 0x40000000 | 0x10000000 | 0x2 | 0x4, "msflags": 0x1 | 0x2 | 0x8 | 0x100 | 0x200}
STREAM_OUT = {"WriteFile": "x0", "fwprintf": "x0", "fputws": "x1", "_write": "x0"}
EGRESS = frozenset({"WSASend", "WSASendTo", "InternetWriteFile", "HttpSendRequestA",
                    "HttpSendRequestW", "WinHttpWriteData", "HttpSendResponseEntityBody",
                    "HttpSendHttpResponse", "TransmitFile"})

# ── MSVC CRT behaviour ───────────────────────────────────────────────────────────


def percent_n_note(sink: str, imports) -> str:
    """%n behaviour of a format sink in a PE image (MSVC CRT / user32)."""
    if sink in NO_PERCENT_N:
        return f"%n: not supported by {sink} (user32 formatter; output capped at {OUTPUT_CAP[sink]} characters)"
    if "_set_printf_count_output" in imports:
        return (f"%n: may be enabled: the binary imports _set_printf_count_output (the MSVC CRT "
                f"rejects %n in {sink} unless it was set non-zero)")
    if is_ucrt(imports):
        return (f"%n: blocked by default (UCRT: {sink} calls the invalid-parameter handler, which "
                f"terminates the process; enabling it takes _set_printf_count_output(1))")
    return (f"%n: depends on the CRT: blocked by default in the UCRT and msvcr80+ (process "
            f"terminates); the image does not show which CRT {sink} comes from")


def is_ucrt(imports) -> bool:
    """The image links the Universal CRT (its printf / scanf family are __stdio_common_*)."""
    return any(n.startswith(("__stdio_common_", "__acrt_")) for n in imports)


def merge() -> None:
    """Fold the Windows names into the Insn-model passes' tables (idempotent)."""
    from . import (egress_arm64 as eg, format_string_scanner_arm64 as fs, heap_scanner_arm64 as hp,
                   intoverflow_scanner_arm64 as io, length_underflow_arm64 as lu,
                   sink_wrappers as sw, taint_tracker_arm64 as tt)
    tt._SOURCE_NAMES = tt._SOURCE_NAMES | SOURCES
    tt._SINK_NAMES = tt._SINK_NAMES | SINKS
    for k, v in SOURCE_BUFFERS.items():
        tt._SOURCE_BUFFERS.setdefault(k, v)
    for k, v in SINK_POINTER_ARGS.items():
        tt._SINK_POINTER_ARGS.setdefault(k, v)
    for k, v in COPY_FUNCS.items():
        tt._COPY_FUNCS.setdefault(k, v)
    tt._EXEC_SINKS = tt._EXEC_SINKS | EXEC_SINKS
    tt._SCAN_SINKS = tt._SCAN_SINKS | frozenset({"swscanf", "__stdio_common_vsscanf",
                                                 "__stdio_common_vswscanf"})
    for k, v in FMT_ARG.items():
        fs.FMT_ARG.setdefault(k, v)
    for k, v in UNBOUNDED.items():
        fs._UNBOUNDED.setdefault(k, v)
    fs._STDIN_SCANS = fs._STDIN_SCANS | STDIN_SCANS
    for k, v in LENGTH_ARGS.items():
        lu._LENGTH_ARGS.setdefault(k, v)
    for k, v in ALLOC_SIZE_ARGS.items():
        io._ALLOC_SIZE_ARGS.setdefault(k, v)
    for k, v in ALLOCS.items():
        hp._ALLOCS.setdefault(k, v)
    hp._FREES = hp._FREES | FREES
    hp._CXX_NEW = hp._CXX_NEW | CXX_NEW
    hp._STR_COPIES = hp._STR_COPIES | STR_COPIES
    hp._STRLENS = hp._STRLENS | STRLENS
    for k, v in SIZED_COPIES.items():
        hp._SIZED_COPIES.setdefault(k, v)
    for k, v in OPEN_FUNCS.items():
        eg.OPEN_FUNCS.setdefault(k, v)
    for k, v in STREAM_OUT.items():
        eg.STREAM_OUT.setdefault(k, v)
    eg.EGRESS = eg.EGRESS | EGRESS
    for k, v in EXEC_ARGS.items():
        if v:
            sw._EXEC_ARGS.setdefault(k, v)


def pe_sections(path: str):
    """[(lo, hi, writable, executable)] of a PE's sections at imagebase + RVA (virtual
    size), for 'is this format a literal' and 'is this in writable memory'."""
    import lief
    out = []
    try:
        binary = lief.parse(path)
        base = int(binary.optional_header.imagebase)
        for sec in binary.sections:
            ch = int(sec.characteristics)
            lo = base + int(sec.virtual_address)
            size = max(int(sec.virtual_size), int(sec.size))
            if size:
                out.append((lo, lo + size, bool(ch & 0x80000000), bool(ch & 0x20000000)))
    except Exception:
        pass
    return out
