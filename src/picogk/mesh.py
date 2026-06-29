"""Triangle ``Mesh`` wrapper, with NumPy access and STL export.

The native API exposes vertices/triangles one element at a time. For now we
loop in Python (correct, but O(n) ctypes calls); a compiled fast path is
planned. See ``vertices`` / ``triangles``.
"""

from __future__ import annotations

import ctypes as C
import struct

import numpy as np

from . import _fast, library
from ._base import NativeObject, require_type
from ._native.ctypes_types import PKBBox3, PKTriangle, PKVector3
from .types import BBox3, to_vec3


class Mesh(NativeObject):
    _destroy_fn = "Mesh_Destroy"

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().Mesh_hCreate(library.instance())
        super().__init__(handle)

    @classmethod
    def from_voxels(cls, voxels) -> Mesh:
        from .voxels import Voxels
        require_type(voxels, Voxels, "voxels")
        lib, inst = library.lib(), library.instance()
        h = lib.Mesh_hCreateFromVoxels(inst, voxels.handle)
        return cls(h)

    @classmethod
    def from_arrays(cls, vertices, triangles) -> Mesh:
        """Build a mesh from an (N,3) vertex array and an (M,3) index array."""
        verts = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
        tris = np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1, 3)
        if tris.size and (tris.max() >= len(verts) or tris.min() < 0):
            raise ValueError("triangle index out of range")
        # Reject non-finite vertices here (both the fast and fallback paths) so
        # NaN/inf never reaches the native rasteriser -- this also validates the
        # output of every shape, whose dimensions/modulations flow through here.
        if verts.size and not np.isfinite(verts).all():
            raise ValueError("mesh vertices must be finite (got NaN/inf)")
        m = cls()
        if _fast.lib is not None:
            if len(verts):
                _fast.lib.add_vertices(_fast.addr(m._lib.Mesh_nAddVertex),
                                       m._inst, m.handle, verts)
            if len(tris):
                _fast.lib.add_triangles(_fast.addr(m._lib.Mesh_nAddTriangle),
                                        m._inst, m.handle, tris)
        else:
            for v in verts:
                m.add_vertex((v[0], v[1], v[2]))
            for t in tris:
                m.add_triangle(int(t[0]), int(t[1]), int(t[2]))
        return m

    @classmethod
    def load_stl(cls, path: str) -> Mesh:
        """Load a binary or ASCII STL file (vertices are de-duplicated)."""
        return cls.from_arrays(*_read_stl(path))

    @classmethod
    def load_obj(cls, path: str) -> Mesh:
        """Load a Wavefront OBJ file (polygons are triangulated as a fan)."""
        return cls.from_arrays(*_read_obj(path))

    def is_valid(self) -> bool:
        return bool(self._lib.Mesh_bIsValid(
            self._inst, self.handle))

    def memory_bytes(self) -> int:
        return int(self._lib.Mesh_nMemUsage(
            self._inst, self.handle))

    # --- counts --------------------------------------------------------------
    def vertex_count(self) -> int:
        return int(self._lib.Mesh_nVertexCount(
            self._inst, self.handle))

    def triangle_count(self) -> int:
        return int(self._lib.Mesh_nTriangleCount(
            self._inst, self.handle))

    # --- building ------------------------------------------------------------
    def add_vertex(self, point) -> int:
        v = to_vec3(point)
        return int(self._lib.Mesh_nAddVertex(
            self._inst, self.handle, C.byref(v)))

    def add_triangle(self, a: int, b: int, c: int) -> int:
        # indices must reference already-added vertices; an out-of-range index is
        # a deferred OOB read at to_voxels/render time (from_arrays validates the
        # same way up front).
        a, b, c = int(a), int(b), int(c)
        n = self.vertex_count()
        if not all(0 <= i < n for i in (a, b, c)):
            raise IndexError(f"triangle indices {(a, b, c)} out of range [0, {n})")
        tri = PKTriangle(a, b, c)
        return int(self._lib.Mesh_nAddTriangle(
            self._inst, self.handle, C.byref(tri)))

    # --- bulk access (Python loop for now) -----------------------------------
    @property
    def vertices(self) -> np.ndarray:
        """(N, 3) float32 array of vertex positions in mm."""
        n = self.vertex_count()
        if _fast.lib is not None and n:
            return _fast.lib.read_vertices(
                _fast.addr(self._lib.Mesh_GetVertex), self._inst, self.handle, n)
        return self._vertices_py(n)

    def _vertices_py(self, n: int) -> np.ndarray:
        out = np.empty((n, 3), dtype=np.float32)
        v = PKVector3()
        get = self._lib.Mesh_GetVertex
        inst, h, ref = self._inst, self.handle, C.byref(v)
        for i in range(n):
            get(inst, h, i, ref)
            out[i, 0], out[i, 1], out[i, 2] = v.X, v.Y, v.Z
        return out

    @property
    def triangles(self) -> np.ndarray:
        """(M, 3) int32 array of vertex indices."""
        n = self.triangle_count()
        if _fast.lib is not None and n:
            return _fast.lib.read_triangles(
                _fast.addr(self._lib.Mesh_GetTriangle), self._inst, self.handle, n)
        return self._triangles_py(n)

    def _triangles_py(self, n: int) -> np.ndarray:
        out = np.empty((n, 3), dtype=np.int32)
        t = PKTriangle()
        get = self._lib.Mesh_GetTriangle
        inst, h, ref = self._inst, self.handle, C.byref(t)
        for i in range(n):
            get(inst, h, i, ref)
            out[i, 0], out[i, 1], out[i, 2] = t.A, t.B, t.C
        return out

    def bounding_box(self) -> BBox3:
        box = PKBBox3()
        self._lib.Mesh_GetBoundingBox(
            self._inst, self.handle, C.byref(box))
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


# --- STL / OBJ readers -------------------------------------------------------
def _dedup(triangle_verts: np.ndarray):
    """Given (3M, 3) per-corner vertices, return (unique_verts, (M,3) indices)."""
    n_tri = triangle_verts.shape[0] // 3
    uniq, inv = np.unique(triangle_verts, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)            # numpy 1.x/2.x shape-safe
    return uniq.astype(np.float32), inv.reshape(n_tri, 3).astype(np.int64)


def _read_stl(path: str):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + n * 50:            # binary STL
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            arr = np.frombuffer(data, dtype=dt, count=n, offset=84)
            return _dedup(arr["v"].reshape(-1, 3))
    # ASCII STL
    verts = []
    for line in data.decode("ascii", errors="replace").splitlines():
        s = line.split()
        if s and s[0] == "vertex":
            verts.append((float(s[1]), float(s[2]), float(s[3])))
    return _dedup(np.array(verts, dtype=np.float32).reshape(-1, 3))


def _read_obj(path: str):
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0] == "v":
                verts.append((float(s[1]), float(s[2]), float(s[3])))
            elif s[0] == "f":
                idx = []
                for tok in s[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):     # fan triangulation
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return (np.array(verts, dtype=np.float32).reshape(-1, 3),
            np.array(faces, dtype=np.int64).reshape(-1, 3))
