"""Exception types for PicoPie."""

from __future__ import annotations


class PicoGKError(RuntimeError):
    """Base class for all PicoPie errors."""


class NotInitializedError(PicoGKError):
    """Raised when the PicoGK session is used before :func:`picogk.init`."""


class InvalidHandleError(PicoGKError):
    """Raised when a native object is used after being closed/destroyed."""
