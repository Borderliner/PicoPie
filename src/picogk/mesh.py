"""Triangle ``Mesh`` wrapper, with NumPy access and STL export.

The native API exposes vertices/triangles one element at a time. For now we
loop in Python (correct, but O(n) ctypes calls); a compiled fast path is
planned. See ``vertices`` / ``triangles``.
"""

from __future__ import annotations

import ctypes as C
import struct

import numpy as np

from . import library
from ._base import NativeObject
from ._native.ctypes_types import PKBBox3, PKTriangle, PKVector3
from .types import BBox3, to_vec3


class Mesh(NativeObject):
    _destroy_fn = "Mesh_Destroy"

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().Mesh_hCreate(C.c_uint64(library.instance()))
        super().__init__(handle)

    @classmethod
    def from_voxels(cls, voxels) -> "Mesh":
        lib, inst = library.lib(), library.instance()
        h = lib.Mesh_hCreateFromVoxels(C.c_uint64(inst), C.c_uint64(voxels.handle))
        return cls(h)

    # --- counts --------------------------------------------------------------
    def vertex_count(self) -> int:
        return int(self._lib.Mesh_nVertexCount(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    def triangle_count(self) -> int:
        return int(self._lib.Mesh_nTriangleCount(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    # --- building ------------------------------------------------------------
    def add_vertex(self, point) -> int:
        v = to_vec3(point)
        return int(self._lib.Mesh_nAddVertex(
            C.c_uint64(self._inst), C.c_uint64(self.handle), C.byref(v)))

    def add_triangle(self, a: int, b: int, c: int) -> int:
        tri = PKTriangle(int(a), int(b), int(c))
        return int(self._lib.Mesh_nAddTriangle(
            C.c_uint64(self._inst), C.c_uint64(self.handle), C.byref(tri)))

    # --- bulk access (Python loop for now) -----------------------------------
    @property
    def vertices(self) -> np.ndarray:
        """(N, 3) float32 array of vertex positions in mm."""
        n = self.vertex_count()
        out = np.empty((n, 3), dtype=np.float32)
        v = PKVector3()
        get = self._lib.Mesh_GetVertex
        inst, h = C.c_uint64(self._inst), C.c_uint64(self.handle)
        ref = C.byref(v)
        for i in range(n):
            get(inst, h, C.c_int32(i), ref)
            out[i, 0], out[i, 1], out[i, 2] = v.X, v.Y, v.Z
        return out

    @property
    def triangles(self) -> np.ndarray:
        """(M, 3) int32 array of vertex indices."""
        n = self.triangle_count()
        out = np.empty((n, 3), dtype=np.int32)
        t = PKTriangle()
        get = self._lib.Mesh_GetTriangle
        inst, h = C.c_uint64(self._inst), C.c_uint64(self.handle)
        ref = C.byref(t)
        for i in range(n):
            get(inst, h, C.c_int32(i), ref)
            out[i, 0], out[i, 1], out[i, 2] = t.A, t.B, t.C
        return out

    def bounding_box(self) -> BBox3:
        box = PKBBox3()
        self._lib.Mesh_GetBoundingBox(
            C.c_uint64(self._inst), C.c_uint64(self.handle), C.byref(box))
        return BBox3._from_pk(box)

    # --- export --------------------------------------------------------------
    def save_stl(self, path: str) -> None:
        """Write a binary STL file."""
        verts = self.vertices
        tris = self.triangles
        tv = verts[tris]                          # (M, 3, 3)
        n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.divide(n, norms, out=np.zeros_like(n), where=norms > 0)
        m = tris.shape[0]
        with open(path, "wb") as f:
            f.write(b"PicoPie binary STL".ljust(80, b"\0"))
            f.write(struct.pack("<I", m))
            rec = np.zeros((m, 12), dtype=np.float32)
            rec[:, 0:3] = n
            rec[:, 3:6] = tv[:, 0]
            rec[:, 6:9] = tv[:, 1]
            rec[:, 9:12] = tv[:, 2]
            buf = bytearray()
            for i in range(m):
                buf += rec[i].tobytes()
                buf += b"\0\0"                      # attribute byte count
            f.write(buf)

    def save_obj(self, path: str) -> None:
        verts = self.vertices
        tris = self.triangles + 1                  # OBJ is 1-indexed
        with open(path, "w") as f:
            f.write("# PicoPie OBJ export\n")
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, tris, fmt="f %d %d %d")

    def __repr__(self) -> str:
        if self._closed:
            return "Mesh(<closed>)"
        return f"Mesh(handle={self._h}, vertices={self.vertex_count()}, " \
               f"triangles={self.triangle_count()})"
