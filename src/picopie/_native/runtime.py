"""Load the native runtime once and bind all function prototypes to it."""

from __future__ import annotations

import contextlib
import ctypes as C

from . import prototypes
from .loader import find_runtime, load_runtime

_BOUND: C.CDLL | None = None


def lib() -> C.CDLL:
    """Return the loaded + prototype-bound native runtime (cached)."""
    global _BOUND
    if _BOUND is None:
        cdll = load_runtime()
        missing = prototypes.apply(cdll)
        if missing:
            # Non-fatal: a version mismatch may drop a few symbols. Warn loudly.
            import warnings
            warnings.warn(
                f"{len(missing)} PicoGK functions missing from the runtime "
                f"(version mismatch?): {missing[:5]}...", RuntimeWarning, stacklevel=2)
        _install_error_guard(cdll)
        _BOUND = cdll
    return _BOUND


def _install_error_guard(cdll: C.CDLL) -> None:
    """Route the runtime's captured errors into ``PicoPieError`` after every call.

    When the runtime is built with the never-abort guard (scripts/patch_runtime.py),
    a C++/OpenVDB exception is caught at the C ABI -- it sets ``g_pkLastErrorFlag``
    and a message instead of aborting. We attach a ctypes ``errcheck`` to every
    bound function that reads the flag (a cheap memory read via ``in_dll`` -- no
    extra native call) and raises. The compiled fast loop calls native function
    pointers directly, so it bypasses this and pays nothing per element.

    On an unpatched runtime (no flag symbol) this is a no-op: the binding still
    works, just without abort-to-exception conversion.
    """
    try:
        flag = C.c_int.in_dll(cdll, "g_pkLastErrorFlag")
        get_err = cdll.PicoGK_nGetLastError
    except (ValueError, AttributeError):
        return  # unpatched runtime
    get_err.restype = C.c_int
    get_err.argtypes = [C.c_char_p, C.c_int]

    from .._errors import PicoPieError

    def _last_error() -> str:
        buf = C.create_string_buffer(1024)
        get_err(buf, len(buf))
        return buf.value.decode(errors="replace")

    def _errcheck(result, func, args):
        if flag.value:
            flag.value = 0
            raise PicoPieError(_last_error() or "PicoGK: native error")
        return result

    for name in prototypes.FUNCTIONS:
        with contextlib.suppress(AttributeError):
            getattr(cdll, name).errcheck = _errcheck
    cdll._picopie_guarded = True  # type: ignore[attr-defined]


__all__ = ["find_runtime", "lib"]
