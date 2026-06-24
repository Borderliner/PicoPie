"""The PicoGK session: a single native library *instance*.

PicoGK is built around one library instance that owns a fixed voxel size (the
fundamental resolution, in millimetres). Almost every native call takes that
instance handle as its first argument; we keep it in a module-global session so
the Pythonic wrappers can supply it implicitly -- mirroring the static
``Library`` class in the C# original.

Usage::

    import picogk
    picogk.init(voxel_size_mm=0.2)
    ...
    picogk.shutdown()          # optional; also runs atexit

or as a context manager::

    with picogk.session(voxel_size_mm=0.2):
        ...
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes as C
import weakref

from ._errors import NotInitializedError, PicoGKError
from ._native.runtime import lib as _lib

_STRLEN = 255

# Live handle-backed wrapper objects, so shutdown() can invalidate them before
# destroying the instance (otherwise a later op on a stale handle aborts the
# process via an uncaught native exception).
_live_objects: weakref.WeakSet = weakref.WeakSet()


def _register(obj) -> None:
    """Register a NativeObject so it can be invalidated on shutdown()."""
    _live_objects.add(obj)


class _Session:
    __slots__ = ("instance", "lib", "voxel_size_mm")

    def __init__(self, cdll: C.CDLL, instance: int, voxel_size_mm: float):
        self.lib = cdll
        self.instance = instance
        self.voxel_size_mm = voxel_size_mm


_session: _Session | None = None


# --- lifecycle ---------------------------------------------------------------
def init(voxel_size_mm: float = 0.1) -> None:
    """Create the global library instance with the given voxel size (mm).

    Calling again with the same voxel size is a no-op; with a different size it
    raises (the voxel size is fixed for the lifetime of an instance).
    """
    global _session
    if voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be > 0")
    if _session is not None:
        if abs(_session.voxel_size_mm - voxel_size_mm) > 1e-9:
            raise PicoGKError(
                f"PicoGK already initialised with voxel size "
                f"{_session.voxel_size_mm} mm; cannot change to {voxel_size_mm} "
                f"mm. Call picogk.shutdown() first.")
        return
    cdll = _lib()
    handle = cdll.Library_hCreateInstance(voxel_size_mm)
    if not handle:
        raise PicoGKError("Library_hCreateInstance returned a null handle")
    _session = _Session(cdll, int(handle), float(voxel_size_mm))


def shutdown() -> None:
    """Destroy the global library instance, if any.

    Live wrapper objects are invalidated first so that any later use raises
    :class:`InvalidHandleError` instead of aborting the process on a stale
    native handle.
    """
    global _session
    if _session is not None:
        for obj in list(_live_objects):
            obj._closed = True            # ops now raise InvalidHandleError; close() is a no-op
        _live_objects.clear()
        with contextlib.suppress(Exception):
            _session.lib.Library_DestroyInstance(_session.instance)
        _session = None


@contextlib.contextmanager
def session(voxel_size_mm: float = 0.1):
    """Context manager wrapping :func:`init` / :func:`shutdown`."""
    init(voxel_size_mm)
    try:
        yield
    finally:
        shutdown()


atexit.register(shutdown)


# --- accessors used by the wrapper classes -----------------------------------
def _active() -> _Session:
    if _session is None:
        raise NotInitializedError(
            "PicoGK is not initialised. Call picogk.init(voxel_size_mm=...) "
            "first.")
    return _session


def lib() -> C.CDLL:
    return _active().lib


def instance() -> int:
    return _active().instance


def voxel_size() -> float:
    """The active voxel size in millimetres."""
    return _active().voxel_size_mm


def is_initialized() -> bool:
    return _session is not None


# --- info / diagnostics ------------------------------------------------------
def _info_string(fn_name: str) -> str:
    cdll = _lib()
    buf = C.create_string_buffer(_STRLEN)
    getattr(cdll, fn_name)(buf)
    return buf.value.decode(errors="replace")


def name() -> str:
    """Native library name, e.g. 'PicoGK Core Library'."""
    return _info_string("Library_GetName")


def version() -> str:
    """Native runtime version, e.g. '26.2.0'."""
    return _info_string("Library_GetVersion")


def build_info() -> str:
    """Native runtime build info (timestamp + name)."""
    return _info_string("Library_GetBuildInfo")


def total_memory_bytes() -> int:
    """Total memory the native runtime currently attributes to this instance."""
    s = _active()
    return int(s.lib.Library_nTotalMemUsage(s.instance))
