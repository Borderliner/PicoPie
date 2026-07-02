"""Lightweight conversions between Python/NumPy values and the packed C structs.

A *vector* argument may be given as a ``PKVector3``, a 3-tuple/list, or a NumPy
array of shape (3,). Returned vectors are NumPy ``float32`` arrays.
"""

from __future__ import annotations

import ctypes as C
import math

import numpy as np

from ._native.ctypes_types import PKBBox3, PKColorFloat, PKVector2, PKVector3


def require_finite(name: str, *values: float) -> None:
    """Reject NaN/inf numeric geometry inputs before they reach native code.

    Non-finite coordinates/sizes don't raise a catchable C++ exception in the
    runtime -- they cause a hard ``SIGSEGV`` or an unbounded loop, neither of
    which the never-abort guard can intercept. Found by Phase-11b fuzzing (e.g.
    a NaN capsule endpoint segfaults; an inf offset hangs)."""
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")


def read_voxel_dimensions(lib, fn_name: str, inst: int, handle: int):
    """Read a voxel-space bounding box via a native ``*_GetVoxelDimensions``
    function. Returns ``(origin, size)`` as int32 arrays of shape (3,)."""
    ints = [C.c_int32() for _ in range(6)]
    getattr(lib, fn_name)(inst, handle, *[C.byref(i) for i in ints])
    origin = np.array([ints[i].value for i in range(3)], dtype=np.int32)
    size = np.array([ints[i].value for i in range(3, 6)], dtype=np.int32)
    return origin, size


def to_vec3(v) -> PKVector3:
    """Coerce a vector-like value into a ``PKVector3`` (must be finite)."""
    if isinstance(v, PKVector3):
        return v
    x, y, z = v  # works for tuple/list/ndarray; raises clearly otherwise
    x, y, z = float(x), float(y), float(z)
    require_finite("vector component", x, y, z)
    return PKVector3(x, y, z)


def vec3_to_np(v: PKVector3) -> np.ndarray:
    return np.array((v.X, v.Y, v.Z), dtype=np.float32)


def to_vec2(v) -> PKVector2:
    if isinstance(v, PKVector2):
        return v
    x, y = v
    return PKVector2(float(x), float(y))


def to_color(c) -> PKColorFloat:
    """Accept ``PKColorFloat`` or an RGB/RGBA tuple (0..1)."""
    if isinstance(c, PKColorFloat):
        return c
    vals = list(c)
    if len(vals) == 3:
        vals.append(1.0)
    r, g, b, a = vals
    return PKColorFloat(float(r), float(g), float(b), float(a))


class BBox3:
    """An axis-aligned bounding box in millimetres."""

    __slots__ = ("max", "min")

    def __init__(self, vmin, vmax):
        self.min = np.asarray(vmin, dtype=np.float32)
        self.max = np.asarray(vmax, dtype=np.float32)

    @classmethod
    def _from_pk(cls, box: PKBBox3) -> BBox3:
        return cls(vec3_to_np(box.vecMin), vec3_to_np(box.vecMax))

    @property
    def size(self) -> np.ndarray:
        # float64 avoids overflow for the degenerate "empty" box (FLT_MAX/-FLT_MAX)
        return self.max.astype(np.float64) - self.min.astype(np.float64)

    @property
    def center(self) -> np.ndarray:
        return (self.max + self.min) * 0.5

    def is_empty(self) -> bool:
        return bool(np.any(self.max < self.min))

    def __repr__(self) -> str:
        return (f"BBox3(min={self.min.tolist()}, max={self.max.tolist()}, "
                f"size={self.size.tolist()})")
