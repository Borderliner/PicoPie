"""Exception types for PicoPie."""

from __future__ import annotations


class PicoPieError(RuntimeError):
    """Base class for all PicoPie errors."""


class NotInitializedError(PicoPieError):
    """Raised when the PicoGK session is used before :func:`picopie.init`."""


class InvalidHandleError(PicoPieError):
    """Raised when a native object is used after being closed/destroyed."""
