"""Load the native runtime once and bind all function prototypes to it."""

from __future__ import annotations

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
                f"(version mismatch?): {missing[:5]}...", RuntimeWarning)
        _BOUND = cdll
    return _BOUND


__all__ = ["lib", "find_runtime"]
