"""Access to the optional compiled fast-loop extension.

``lib`` is the compiled module, or ``None`` when it was not built (in which case
callers fall back to pure-Python loops). ``addr(fn)`` returns the raw C address
of a ctypes CDLL function, which the extension casts to a typed pointer.
"""

from __future__ import annotations

import ctypes as C

try:
    from . import _fastloop as lib
except ImportError:  # extension not compiled -> pure-Python fallback
    lib = None


def available() -> bool:
    return lib is not None


def addr(fn) -> int:
    """Raw address of a ctypes CDLL function pointer."""
    return C.cast(fn, C.c_void_p).value
