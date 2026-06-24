"""``PolyLine``: an ordered set of 3D points with a color (used for paths,
debug overlays, toolpaths, and viewer annotations)."""

from __future__ import annotations

import ctypes as C

import numpy as np

from . import library
from ._base import NativeObject
from ._native.ctypes_types import PKBBox3, PKColorFloat, PKVector3
from .types import BBox3, to_color, to_vec3


class PolyLine(NativeObject):
    _destroy_fn = "PolyLine_Destroy"

    def __init__(self, color=(1.0, 1.0, 1.0, 1.0), handle: int | None = None):
        if handle is None:
            clr = to_color(color)
            handle = library.lib().PolyLine_hCreate(library.instance(), C.byref(clr))
        super().__init__(handle)

    @classmethod
    def from_points(cls, points, color=(1.0, 1.0, 1.0, 1.0)) -> PolyLine:
        pl = cls(color=color)
        for p in points:
            pl.add_vertex(p)
        return pl

    def add_vertex(self, point) -> int:
        v = to_vec3(point)
        return int(self._lib.PolyLine_nAddVertex(self._inst, self.handle, C.byref(v)))

    def vertex_count(self) -> int:
        return int(self._lib.PolyLine_nVertexCount(self._inst, self.handle))

    @property
    def vertices(self) -> np.ndarray:
        n = self.vertex_count()
        out = np.empty((n, 3), dtype=np.float32)
        v = PKVector3()
        for i in range(n):
            self._lib.PolyLine_GetVertex(self._inst, self.handle, i, C.byref(v))
            out[i] = (v.X, v.Y, v.Z)
        return out

    def color(self) -> np.ndarray:
        clr = PKColorFloat()
        self._lib.PolyLine_GetColor(self._inst, self.handle, C.byref(clr))
        return np.array([clr.R, clr.G, clr.B, clr.A], dtype=np.float32)

    def bounding_box(self) -> BBox3:
        box = PKBBox3()
        self._lib.PolyLine_GetBoundingBox(self._inst, self.handle, C.byref(box))
        return BBox3._from_pk(box)

    def is_valid(self) -> bool:
        return bool(self._lib.PolyLine_bIsValid(self._inst, self.handle))

    def memory_bytes(self) -> int:
        return int(self._lib.PolyLine_nMemUsage(self._inst, self.handle))

    def __repr__(self) -> str:
        return "PolyLine(<closed>)" if self._closed else \
            f"PolyLine(handle={self._h}, vertices={self.vertex_count()})"
