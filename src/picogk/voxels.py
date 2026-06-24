"""The ``Voxels`` object: a signed-distance / level-set volume.

This is the heart of PicoGK. A ``Voxels`` holds an OpenVDB narrow-band level
set. Boolean operations, offsets and rendering all mutate the volume in the
native engine; this wrapper exposes both **in-place** methods (trailing ``_``,
fast, no copy) and **functional** operators (``+ - &``, return a new volume).
"""

from __future__ import annotations

import ctypes as C

import numpy as np

from . import library
from ._base import NativeObject
from ._native.ctypes_types import PKBBox3, PKPFnfSdf, PKVector3
from .types import BBox3, to_vec3, vec3_to_np


def _to_pkbbox(box) -> PKBBox3:
    if isinstance(box, BBox3):
        lo, hi = box.min, box.max
    else:
        lo, hi = box  # ((x,y,z),(x,y,z))
    return PKBBox3(to_vec3(lo), to_vec3(hi))


class Voxels(NativeObject):
    _destroy_fn = "Voxels_Destroy"

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().Voxels_hCreate(C.c_uint64(library.instance()))
        super().__init__(handle)

    # --- constructors --------------------------------------------------------
    @classmethod
    def _wrap(cls, handle: int) -> "Voxels":
        return cls(handle)

    @classmethod
    def sphere(cls, center=(0.0, 0.0, 0.0), radius: float = 1.0) -> "Voxels":
        lib, inst = library.lib(), library.instance()
        c = to_vec3(center)
        h = lib.Voxels_hCreateSphere(C.c_uint64(inst), C.byref(c), C.c_float(radius))
        return cls(h)

    @classmethod
    def capsule(cls, start, end, radius: float, radius2: float | None = None) -> "Voxels":
        lib, inst = library.lib(), library.instance()
        a, b = to_vec3(start), to_vec3(end)
        r2 = radius if radius2 is None else radius2
        h = lib.Voxels_hCreateCapsule(C.c_uint64(inst), C.byref(a), C.byref(b),
                                      C.c_float(radius), C.c_float(r2))
        return cls(h)

    @classmethod
    def mesh_shell(cls, mesh, radius: float) -> "Voxels":
        """A hollow shell of the given thickness around a mesh surface."""
        lib, inst = library.lib(), library.instance()
        h = lib.Voxels_hCreateMeshShell(C.c_uint64(inst),
                                        C.c_uint64(mesh.handle), C.c_float(radius))
        return cls(h)

    @classmethod
    def from_mesh(cls, mesh) -> "Voxels":
        """Voxelize a closed mesh into a solid volume."""
        v = cls()
        v.render_mesh_(mesh)
        return v

    @classmethod
    def from_lattice(cls, lattice) -> "Voxels":
        v = cls()
        v.render_lattice_(lattice)
        return v

    def copy(self) -> "Voxels":
        h = self._lib.Voxels_hCreateCopy(C.c_uint64(self._inst), C.c_uint64(self.handle))
        return Voxels(h)

    # --- boolean ops (in-place, return self) ---------------------------------
    def bool_add_(self, other: "Voxels") -> "Voxels":
        self._lib.Voxels_BoolAdd(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                 C.c_uint64(other.handle))
        return self

    def bool_subtract_(self, other: "Voxels") -> "Voxels":
        self._lib.Voxels_BoolSubtract(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                      C.c_uint64(other.handle))
        return self

    def bool_intersect_(self, other: "Voxels") -> "Voxels":
        self._lib.Voxels_BoolIntersect(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                       C.c_uint64(other.handle))
        return self

    # --- boolean ops (functional, return new) --------------------------------
    def __add__(self, other: "Voxels") -> "Voxels":
        return self.copy().bool_add_(other)

    def __or__(self, other: "Voxels") -> "Voxels":
        return self.copy().bool_add_(other)

    def __sub__(self, other: "Voxels") -> "Voxels":
        return self.copy().bool_subtract_(other)

    def __and__(self, other: "Voxels") -> "Voxels":
        return self.copy().bool_intersect_(other)

    def __iadd__(self, other: "Voxels"):
        return self.bool_add_(other)

    def __isub__(self, other: "Voxels"):
        return self.bool_subtract_(other)

    def __iand__(self, other: "Voxels"):
        return self.bool_intersect_(other)

    # --- offsets -------------------------------------------------------------
    def offset_(self, dist_mm: float) -> "Voxels":
        """Grow (positive) / shrink (negative) the surface, in mm. In place."""
        self._lib.Voxels_Offset(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                C.c_float(dist_mm))
        return self

    def offset(self, dist_mm: float) -> "Voxels":
        return self.copy().offset_(dist_mm)

    def double_offset_(self, dist1_mm: float, dist2_mm: float) -> "Voxels":
        self._lib.Voxels_DoubleOffset(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                      C.c_float(dist1_mm), C.c_float(dist2_mm))
        return self

    def shell_(self, thickness_mm: float) -> "Voxels":
        """Hollow the volume to a wall of ``thickness_mm`` (inward)."""
        return self.double_offset_(-thickness_mm, thickness_mm)

    def triple_offset_(self, dist_mm: float) -> "Voxels":
        self._lib.Voxels_TripleOffset(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                      C.c_float(dist_mm))
        return self

    # --- rendering into this volume ------------------------------------------
    def render_mesh_(self, mesh) -> "Voxels":
        self._lib.Voxels_RenderMesh(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                    C.c_uint64(mesh.handle))
        return self

    def render_lattice_(self, lattice) -> "Voxels":
        self._lib.Voxels_RenderLattice(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                       C.c_uint64(lattice.handle))
        return self

    def project_z_slice_(self, z_start_mm: float, z_end_mm: float) -> "Voxels":
        self._lib.Voxels_ProjectZSlice(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                       C.c_float(z_start_mm), C.c_float(z_end_mm))
        return self

    def render_implicit_(self, sdf, bbox) -> "Voxels":
        """Render a signed-distance function ``sdf(x, y, z) -> float`` within
        ``bbox`` (a :class:`BBox3` or ``((xmin,ymin,zmin),(xmax,ymax,zmax))``).

        NOTE: ``sdf`` is invoked once per voxel from native code; a pure-Python
        callback is correspondingly slow. Prefer primitives + booleans, or a
        compiled callback, for large volumes.
        """
        @PKPFnfSdf
        def _cb(pcoord):
            v = pcoord[0]
            return float(sdf(v.X, v.Y, v.Z))

        box = _to_pkbbox(bbox)
        self._lib.Voxels_RenderImplicit(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                        C.byref(box), _cb)
        return self

    def intersect_implicit_(self, sdf) -> "Voxels":
        """Intersect this volume with the region where ``sdf(x,y,z) <= 0``.

        WARNING: the resulting grid may not be a valid level set, and a
        subsequent boolean op can raise an OpenVDB error that the native C
        layer does not catch -- aborting the process. For clipping, prefer
        composing fields inside a single :meth:`render_implicit_` callback
        (e.g. ``max(feature_sdf, clip_sdf)``).
        """
        @PKPFnfSdf
        def _cb(pcoord):
            v = pcoord[0]
            return float(sdf(v.X, v.Y, v.Z))

        self._lib.Voxels_IntersectImplicit(C.c_uint64(self._inst),
                                           C.c_uint64(self.handle), _cb)
        return self

    # --- queries -------------------------------------------------------------
    def volume_mm3(self) -> float:
        return float(self._lib.Voxels_fCalculateVolume(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    def is_empty(self) -> bool:
        return bool(self._lib.Voxels_bIsEmpty(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    def is_valid(self) -> bool:
        return bool(self._lib.Voxels_bIsValid(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    def voxel_size_mm(self) -> float:
        return float(self._lib.Voxels_fVoxelSize(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    def memory_bytes(self) -> int:
        return int(self._lib.Voxels_nMemUsage(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    def voxel_dimensions(self):
        """Return ``(origin, size)`` as integer NumPy arrays in voxel space."""
        ints = [C.c_int32() for _ in range(6)]
        self._lib.Voxels_GetVoxelDimensions(
            C.c_uint64(self._inst), C.c_uint64(self.handle),
            *[C.byref(i) for i in ints])
        origin = np.array([ints[0].value, ints[1].value, ints[2].value], dtype=np.int32)
        size = np.array([ints[3].value, ints[4].value, ints[5].value], dtype=np.int32)
        return origin, size

    def bounding_box(self) -> BBox3:
        """Bounding box in millimetres (derived from the voxel extent)."""
        origin, size = self.voxel_dimensions()
        lib, inst = self._lib, self._inst
        lo_v = PKVector3(float(origin[0]), float(origin[1]), float(origin[2]))
        hi_v = PKVector3(float(origin[0] + size[0]), float(origin[1] + size[1]),
                         float(origin[2] + size[2]))
        lo_mm, hi_mm = PKVector3(), PKVector3()
        lib.Library_VoxelsToMm(C.c_uint64(inst), C.byref(lo_v), C.byref(lo_mm))
        lib.Library_VoxelsToMm(C.c_uint64(inst), C.byref(hi_v), C.byref(hi_mm))
        return BBox3(vec3_to_np(lo_mm), vec3_to_np(hi_mm))

    def is_inside(self, point) -> bool:
        p = to_vec3(point)
        return bool(self._lib.Voxels_bIsInside(
            C.c_uint64(self._inst), C.c_uint64(self.handle), C.byref(p)))

    def surface_normal(self, point) -> np.ndarray:
        p = to_vec3(point)
        out = PKVector3()
        self._lib.Voxels_GetSurfaceNormal(
            C.c_uint64(self._inst), C.c_uint64(self.handle), C.byref(p), C.byref(out))
        return vec3_to_np(out)

    def closest_point(self, point):
        p = to_vec3(point)
        out = PKVector3()
        ok = self._lib.Voxels_bClosestPointOnSurface(
            C.c_uint64(self._inst), C.c_uint64(self.handle), C.byref(p), C.byref(out))
        return vec3_to_np(out) if ok else None

    def ray_cast(self, origin, direction):
        o, d = to_vec3(origin), to_vec3(direction)
        out = PKVector3()
        ok = self._lib.Voxels_bRayCastToSurface(
            C.c_uint64(self._inst), C.c_uint64(self.handle),
            C.byref(o), C.byref(d), C.byref(out))
        return vec3_to_np(out) if ok else None

    def equals(self, other: "Voxels") -> bool:
        return bool(self._lib.Voxels_bIsEqual(
            C.c_uint64(self._inst), C.c_uint64(self.handle), C.c_uint64(other.handle)))

    def diagnose(self) -> str:
        buf = C.create_string_buffer(255)
        self._lib.Voxels_bDiagnose(C.c_uint64(self._inst), C.c_uint64(self.handle), buf)
        return buf.value.decode(errors="replace")

    # --- conversion ----------------------------------------------------------
    def to_mesh(self):
        from .mesh import Mesh
        return Mesh.from_voxels(self)

    def __repr__(self) -> str:
        if self._closed:
            return "Voxels(<closed>)"
        try:
            o, s = self.voxel_dimensions()
            return f"Voxels(handle={self._h}, voxels={s[0]}x{s[1]}x{s[2]})"
        except Exception:
            return f"Voxels(handle={self._h})"
