"""Lightweight conversions between Python/NumPy values and the packed C structs.

A *vector* argument may be given as a ``PKVector3``, a 3-tuple/list, or a NumPy
array of shape (3,). Returned vectors are NumPy ``float32`` arrays.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ._native.ctypes_types import (
    PKBBox3, PKColorFloat, PKTriangle, PKVector2, PKVector3,
)

Vec3Like = "PKVector3 | Iterable[float] | np.ndarray"


def to_vec3(v) -> PKVector3:
    """Coerce a vector-like value into a ``PKVector3``."""
    if isinstance(v, PKVector3):
        return v
    x, y, z = v  # works for tuple/list/ndarray; raises clearly otherwise
    return PKVector3(float(x), float(y), float(z))


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

    __slots__ = ("min", "max")

    def __init__(self, vmin, vmax):
        self.min = np.asarray(vmin, dtype=np.float32)
        self.max = np.asarray(vmax, dtype=np.float32)

    @classmethod
    def _from_pk(cls, box: PKBBox3) -> "BBox3":
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
