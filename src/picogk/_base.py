"""Common base for handle-backed native objects (lifetime management)."""

from __future__ import annotations

import contextlib

from . import library
from ._errors import InvalidHandleError


class NativeObject:
    """Wraps a ``uint64`` native handle and ties its lifetime to Python's.

    Subclasses set ``_destroy_fn`` to the name of the native destructor, which
    takes ``(instance, handle)``.
    """

    _destroy_fn: str = ""

    def __init__(self, handle: int):
        self._lib = library.lib()
        self._inst = library.instance()
        self._h = int(handle)
        self._closed = False
        if not self._h:
            raise InvalidHandleError(
                f"{type(self).__name__} created with a null handle")
        library._register(self)   # so shutdown() can invalidate us safely

    @property
    def handle(self) -> int:
        if self._closed:
            raise InvalidHandleError(
                f"{type(self).__name__} has been closed")
        return self._h

    def close(self) -> None:
        """Destroy the native object. Safe to call more than once."""
        if not self._closed and self._h:
            if self._destroy_fn:
                with contextlib.suppress(Exception):
                    getattr(self._lib, self._destroy_fn)(
                        self._inst, self._h)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # Guard: interpreter teardown may have already collected library state.
        with contextlib.suppress(Exception):
            self.close()
