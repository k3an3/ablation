"""
fortify.py: glibc FORTIFY (_FORTIFY_SOURCE) variants of sinks, and what they change.

Why this exists
  1. A FORTIFY build calls __memcpy_chk / __strcpy_chk / __read_chk / __vsnprintf_chk ...
     instead of the plain functions. Sink tables keyed on plain names missed every one.
  2. The variant decides exploitability: a __*printf_chk with flag > 0 aborts on %n in a
     writable format (a read primitive stays a read primitive), and a __memcpy_chk with a
     known destination size aborts instead of overflowing. A plain vsnprintf blocks
     nothing. Findings should say which one they are looking at.

CHK_BASE           __x_chk -> x for variants whose leading arguments match the plain
                   function (callers can treat them as x)
CHK_DEST_SIZE      __x_chk -> register holding the destination object size (-1: unknown)
FMT_CHK_FLAG       __*printf_chk -> register holding the FORTIFY flag
"""

from __future__ import annotations

from typing import Dict, Optional

CHK_BASE: Dict[str, str] = {
    "__memcpy_chk": "memcpy", "__memmove_chk": "memmove", "__memset_chk": "memset",
    "__mempcpy_chk": "memcpy", "__strcpy_chk": "strcpy", "__stpcpy_chk": "strcpy",
    "__strcat_chk": "strcat", "__strncpy_chk": "strncpy", "__stpncpy_chk": "strncpy",
    "__strncat_chk": "strncat", "__read_chk": "read", "__pread_chk": "pread",
    "__recv_chk": "recv", "__recvfrom_chk": "recvfrom", "__gets_chk": "gets",
    "__fread_chk": "fread", "__strlen_chk": "strlen",
}
CHK_DEST_SIZE: Dict[str, str] = {
    "__memcpy_chk": "x3", "__memmove_chk": "x3", "__memset_chk": "x3", "__mempcpy_chk": "x3",
    "__strcpy_chk": "x2", "__stpcpy_chk": "x2", "__strcat_chk": "x2", "__strncpy_chk": "x3",
    "__stpncpy_chk": "x3", "__strncat_chk": "x3", "__read_chk": "x3", "__pread_chk": "x4",
    "__recv_chk": "x3", "__recvfrom_chk": "x3", "__gets_chk": "x1", "__fread_chk": "x1",
    "__sprintf_chk": "x2", "__vsprintf_chk": "x2", "__snprintf_chk": "x3", "__vsnprintf_chk": "x3",
}
FMT_CHK_FLAG: Dict[str, str] = {
    "__printf_chk": "x0", "__vprintf_chk": "x0", "__fprintf_chk": "x1", "__vfprintf_chk": "x1",
    "__sprintf_chk": "x1", "__vsprintf_chk": "x1", "__snprintf_chk": "x2", "__vsnprintf_chk": "x2",
    "__dprintf_chk": "x1", "__vdprintf_chk": "x1", "__syslog_chk": "x1", "__vsyslog_chk": "x1",
    "__asprintf_chk": "x1", "__vasprintf_chk": "x1",
}
_UNKNOWN_SIZE = (1 << 64) - 1


def percent_n_note(sink: str, flag: Optional[int]) -> str:
    """What happens to %n in a writable format passed to `sink`."""
    if sink in FMT_CHK_FLAG:
        if flag is None:
            return f"%n: blocked only if {sink}'s FORTIFY flag > 0 (flag not a constant)"
        if flag > 0:
            return f"%n: blocked by FORTIFY ({sink} flag={flag}: %n in a writable format aborts)"
        return f"%n: not blocked ({sink} with flag 0)"
    return (f"%n: not blocked (plain {sink}; a libc that rejects %n outright, e.g. Bionic, "
            f"aborts instead)")


def dest_size_note(sink: str, size: Optional[int]) -> str:
    """FORTIFY effect of a __*_chk copy on an overflow."""
    if sink not in CHK_DEST_SIZE:
        return ""
    if size is None:
        return f"FORTIFY {sink}: destination size not a constant"
    if size in (_UNKNOWN_SIZE, -1):
        return f"FORTIFY {sink}: destination size unknown (-1), not checked"
    return f"FORTIFY {sink}: aborts when the copy exceeds {size:#x} bytes (DoS, not corruption)"
