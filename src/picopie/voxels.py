"""The ``Voxels`` object: a signed-distance / level-set volume.

This is the heart of PicoGK. A ``Voxels`` holds an OpenVDB narrow-band level
set. Boolean operations, offsets and rendering all mutate the volume in the
native engine; this wrapper exposes both **in-place** methods (trailing ``_``,
fast, no copy) and **functional** operators (``+ - &``, return a new volume).
"""

from __future__ import annotations

import ctypes as C
import math
from typing import TYPE_CHECKING

import numpy as np

from . import library
from ._base import NativeObject, require_type
from ._errors import InvalidHandleError, PicoPieError
from ._native.ctypes_types import PKBBox3, PKPFnfSdf, PKVector3
from .types import BBox3, read_voxel_dimensions, require_finite, to_vec3, vec3_to_np

if TYPE_CHECKING:
    from .lattice import Lattice
    from .mesh import Mesh


def _to_pkbbox(box) -> PKBBox3:
    if isinstance(box, BBox3):
        lo, hi = box.min, box.max
    else:
        lo, hi = box  # ((x,y,z),(x,y,z))
    return PKBBox3(to_vec3(lo), to_vec3(hi))


class Voxels(NativeObject):
    _destroy_fn = "Voxels_Destroy"

    # intersect_implicit_ can leave the grid as a non-level-set; a *second*
    # implicit intersect then hard-aborts the process inside OpenVDB (and
    # is_valid()/is_empty() don't reveal the bad state). We track it to raise
    # a catchable error instead. See intersect_implicit_().
    _implicit_intersected: bool = False

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().Voxels_hCreate(library.instance())
        super().__init__(handle)

    # --- constructors --------------------------------------------------------
    @classmethod
    def _wrap(cls, handle: int) -> Voxels:
        return cls(handle)

    @classmethod
    def sphere(cls, center=(0.0, 0.0, 0.0), radius: float = 1.0) -> Voxels:
        require_finite("radius", radius)
        lib, inst = library.lib(), library.instance()
        c = to_vec3(center)
        h = lib.Voxels_hCreateSphere(inst, C.byref(c), radius)
        return cls(h)

    @classmethod
    def capsule(cls, start, end, radius: float, radius2: float | None = None) -> Voxels:
        r2 = radius if radius2 is None else radius2
        require_finite("radius", radius, r2)
        lib, inst = library.lib(), library.instance()
        a, b = to_vec3(start), to_vec3(end)
        h = lib.Voxels_hCreateCapsule(inst, C.byref(a), C.byref(b),
                                      radius, r2)
        return cls(h)

    @classmethod
    def mesh_shell(cls, mesh: Mesh, radius: float) -> Voxels:
        """A hollow shell of the given thickness around a mesh surface."""
        from .mesh import Mesh as _Mesh
        require_type(mesh, _Mesh, "mesh")
        require_finite("radius", radius)        # consistent with sphere/capsule/offsets
        lib, inst = library.lib(), library.instance()
        h = lib.Voxels_hCreateMeshShell(inst,
                                        mesh.handle, radius)
        return cls(h)

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> Voxels:
        """Voxelize a closed mesh into a solid volume."""
        v = cls()
        v.render_mesh_(mesh)
        return v

    @classmethod
    def from_lattice(cls, lattice: Lattice) -> Voxels:
        v = cls()
        v.render_lattice_(lattice)
        return v

    def copy(self) -> Voxels:
        h = self._lib.Voxels_hCreateCopy(self._inst, self.handle)
        c = Voxels(h)
        c._implicit_intersected = self._implicit_intersected  # bad CSG state copies too
        return c

    # --- boolean ops (in-place, return self) ---------------------------------
    def _check_operand(self, other: Voxels) -> None:
        """Guard CSG inputs. Uncaught OpenVDB errors abort the process, so we
        reject obvious misuse (wrong type, closed handle, cross-session) before
        crossing into native code."""
        if not isinstance(other, Voxels):
            raise TypeError(f"expected Voxels, got {type(other).__name__}")
        if self._closed or other._closed:
            raise InvalidHandleError("boolean op on a closed Voxels")
        if other._inst != self._inst:
            raise PicoPieError("cannot combine Voxels from different sessions")

    def bool_add_(self, other: Voxels) -> Voxels:
        self._check_operand(other)
        self._lib.Voxels_BoolAdd(self._inst, self.handle,
                                 other.handle)
        return self

    def bool_subtract_(self, other: Voxels) -> Voxels:
        self._check_operand(other)
        self._lib.Voxels_BoolSubtract(self._inst, self.handle,
                                      other.handle)
        return self

    def bool_intersect_(self, other: Voxels) -> Voxels:
        self._check_operand(other)
        self._lib.Voxels_BoolIntersect(self._inst, self.handle,
                                       other.handle)
        return self

    # --- boolean ops (functional, return new) --------------------------------
    def __add__(self, other: Voxels) -> Voxels:
        return self.copy().bool_add_(other)

    def __or__(self, other: Voxels) -> Voxels:
        return self.copy().bool_add_(other)

    def __sub__(self, other: Voxels) -> Voxels:
        return self.copy().bool_subtract_(other)

    def __and__(self, other: Voxels) -> Voxels:
        return self.copy().bool_intersect_(other)

    def __iadd__(self, other: Voxels):
        return self.bool_add_(other)

    def __isub__(self, other: Voxels):
        return self.bool_subtract_(other)

    def __iand__(self, other: Voxels):
        return self.bool_intersect_(other)

    # --- offsets -------------------------------------------------------------
    def offset_(self, dist_mm: float) -> Voxels:
        """Grow (positive) / shrink (negative) the surface, in mm. In place."""
        require_finite("dist_mm", dist_mm)
        self._lib.Voxels_Offset(self._inst, self.handle,
                                dist_mm)
        return self

    def offset(self, dist_mm: float) -> Voxels:
        return self.copy().offset_(dist_mm)

    def double_offset_(self, dist1_mm: float, dist2_mm: float) -> Voxels:
        """Two sequential offsets (grow/shrink) of the surface -- a rounding /
        morphological op, NOT a hollow. Use :meth:`shell_` to hollow."""
        require_finite("dist_mm", dist1_mm, dist2_mm)
        self._lib.Voxels_DoubleOffset(self._inst, self.handle,
                                      dist1_mm, dist2_mm)
        return self

    def shell_(self, thickness_mm: float) -> Voxels:
        """Hollow the volume to a wall of ``thickness_mm`` (wall lies inside the
        current surface). Implemented as ``solid - erode(solid, thickness)``.

        A ``thickness_mm`` >= the object's half-extent leaves it solid (the core
        erodes away entirely)."""
        require_finite("thickness_mm", thickness_mm)
        if thickness_mm <= 0:
            raise ValueError("shell thickness must be > 0")
        inner = self.copy().offset_(-thickness_mm)        # erode the core inward
        self.bool_subtract_(inner)                        # solid minus core = wall
        inner.close()
        return self

    def triple_offset_(self, dist_mm: float) -> Voxels:
        require_finite("dist_mm", dist_mm)
        self._lib.Voxels_TripleOffset(self._inst, self.handle,
                                      dist_mm)
        return self

    # --- rendering into this volume ------------------------------------------
    def render_mesh_(self, mesh: Mesh) -> Voxels:
        from .mesh import Mesh as _Mesh
        require_type(mesh, _Mesh, "mesh")
        self._lib.Voxels_RenderMesh(self._inst, self.handle,
                                    mesh.handle)
        return self

    def render_lattice_(self, lattice: Lattice) -> Voxels:
        from .lattice import Lattice as _Lattice
        require_type(lattice, _Lattice, "lattice")
        self._lib.Voxels_RenderLattice(self._inst, self.handle,
                                       lattice.handle)
        return self

    def project_z_slice_(self, z_start_mm: float, z_end_mm: float) -> Voxels:
        require_finite("z_start_mm", z_start_mm, z_end_mm)
        self._lib.Voxels_ProjectZSlice(self._inst, self.handle,
                                       z_start_mm, z_end_mm)
        return self

    @staticmethod
    def _wrap_sdf(sdf, errors: list):
        """Wrap a user SDF as a native callback that never lets a Python
        exception or non-finite value reach the native loop (which would
        silently inject a 0.0 surface distance). Captured errors are re-raised
        by the caller after the native call returns."""
        @PKPFnfSdf
        def _cb(pcoord):
            v = pcoord[0]
            try:
                r = float(sdf(v.X, v.Y, v.Z))
            except BaseException as exc:
                if not errors:
                    errors.append(exc)
                return 1.0e30              # large positive -> voxel stays empty
            return r if math.isfinite(r) else 1.0e30
        return _cb

    @staticmethod
    def _as_native_sdf(sdf):
        """If ``sdf`` is already a *compiled* callback, return it rebuilt as a
        :data:`PKPFnfSdf` so it can be handed straight to the native loop --
        no per-voxel Python round-trip. Otherwise return ``None`` (the guarded
        Python path is used).

        Detected forms:

        * a ``numba.cfunc`` (exposes an integer ``.address``);
        * any ctypes function pointer -- a ``CFUNCTYPE`` instance, a ``CDLL``
          export, or a cffi callback.

        The compiled fn must have the ABI ``float(const PKVector3*)`` -- i.e.
        take a pointer to three contiguous ``float32`` and return a ``float``.
        For numba that is ``types.float32(types.CPointer(types.float32))``,
        reading ``p[0], p[1], p[2]``.

        This path deliberately **bypasses** the finite/exception guard of
        :meth:`_wrap_sdf`: a compiled callback runs with no interpreter and
        (for ``nopython``/``nogil`` code) no GIL, which is exactly what makes
        it fast, so it owns its own correctness -- returning NaN/inf injects a
        zero-distance surface, the same behaviour as upstream C# PicoGK. The
        caller keeps ``sdf`` referenced for the duration of the native call so
        the compiled code is not collected out from under the pointer.
        """
        addr = getattr(sdf, "address", None)          # numba.cfunc
        if isinstance(addr, int) and addr:
            return PKPFnfSdf(addr)
        if isinstance(sdf, C._CFuncPtr):              # ctypes / CDLL / cffi
            return PKPFnfSdf(C.cast(sdf, C.c_void_p).value)
        return None

    def render_implicit_(self, sdf, bbox) -> Voxels:
        """Render a signed-distance function ``sdf(x, y, z) -> float`` within
        ``bbox`` (a :class:`BBox3` or ``((xmin,ymin,zmin),(xmax,ymax,zmax))``).

        ``sdf`` may be a plain Python callable (invoked once per voxel from
        native code -- correspondingly slow; prefer primitives + booleans for
        bulk solids) or a **compiled callback** -- a ``numba.cfunc`` or any
        ctypes function pointer with the ABI ``float(const PKVector3*)``. A
        compiled callback runs entirely in native code (no per-voxel Python
        round-trip; ~30x faster in practice) but bypasses the finite/exception
        guard, so it must return finite values itself -- see
        :meth:`_as_native_sdf`.

        If a Python ``sdf`` raises, the exception is re-raised after the native
        pass (the volume is left partially built).
        """
        box = _to_pkbbox(bbox)
        native_cb = self._as_native_sdf(sdf)
        if native_cb is not None:
            self._lib.Voxels_RenderImplicit(self._inst, self.handle,
                                            C.byref(box), native_cb)
            return self                               # sdf kept alive by param
        errors: list = []
        cb = self._wrap_sdf(sdf, errors)
        self._lib.Voxels_RenderImplicit(self._inst, self.handle,
                                        C.byref(box), cb)
        if errors:
            raise errors[0]
        return self

    def intersect_implicit_(self, sdf) -> Voxels:
        """Intersect this volume with the region where ``sdf(x,y,z) <= 0``.

        ``sdf`` may be a plain Python callable or a **compiled callback**
        (``numba.cfunc`` / ctypes function pointer with the ABI
        ``float(const PKVector3*)``); the compiled form runs natively and
        bypasses the finite guard -- see :meth:`render_implicit_` and
        :meth:`_as_native_sdf`.

        WARNING: the resulting grid may not be a valid level set. Calling this a
        second time on the same volume (or a copy of it) raises
        :class:`PicoPieError` -- the underlying OpenVDB CSG would otherwise abort
        the whole process, uncatchably. A boolean op after one implicit
        intersect is usually fine but not guaranteed. For clipping, prefer
        composing fields inside a single :meth:`render_implicit_` callback
        (e.g. ``max(feature_sdf, clip_sdf)``), which has none of these caveats.
        """
        if self._implicit_intersected:
            raise PicoPieError(
                "intersect_implicit_() called twice on the same volume: the grid "
                "is no longer a valid level set and a second implicit intersect "
                "aborts the process. Compose the clip into one render_implicit_ "
                "callback instead, e.g. max(feature_sdf, clip_sdf).")
        native_cb = self._as_native_sdf(sdf)
        if native_cb is not None:
            self._lib.Voxels_IntersectImplicit(self._inst, self.handle,
                                               native_cb)
            self._implicit_intersected = True
            return self                               # sdf kept alive by param
        errors: list = []
        cb = self._wrap_sdf(sdf, errors)
        self._lib.Voxels_IntersectImplicit(self._inst,
                                           self.handle, cb)
        self._implicit_intersected = True
        if errors:
            raise errors[0]
        return self

    # --- queries -------------------------------------------------------------
    def volume_mm3(self) -> float:
        """Volume from a direct native measurement -- fast, but OpenVDB leaves
        zero-distance surface voxels after boolean ops that can skew it. For an
        accurate figure use :meth:`calculate_properties`."""
        return float(self._lib.Voxels_fCalculateVolume(
            self._inst, self.handle))

    def calculate_properties(self) -> tuple[float, BBox3]:
        """Return ``(volume_mm3, bbox)`` measured the accurate way -- via a mesh
        round-trip that drops spurious surface voxels -- matching C# PicoGK's
        ``CalculateProperties``. ``bbox`` is the surface (mesh) bounding box."""
        mesh = self.to_mesh()
        clean = Voxels.from_mesh(mesh)
        try:
            vol = float(self._lib.Voxels_fCalculateVolume(
                self._inst, clean.handle))
            return vol, mesh.bounding_box()
        finally:
            clean.close()
            mesh.close()

    def is_empty(self) -> bool:
        return bool(self._lib.Voxels_bIsEmpty(
            self._inst, self.handle))

    def is_valid(self) -> bool:
        return bool(self._lib.Voxels_bIsValid(
            self._inst, self.handle))

    def voxel_size_mm(self) -> float:
        return float(self._lib.Voxels_fVoxelSize(
            self._inst, self.handle))

    def memory_bytes(self) -> int:
        return int(self._lib.Voxels_nMemUsage(
            self._inst, self.handle))

    def voxel_dimensions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(origin, size)`` as integer NumPy arrays in voxel space."""
        return read_voxel_dimensions(
            self._lib, "Voxels_GetVoxelDimensions", self._inst, self.handle)

    def bounding_box(self) -> BBox3:
        """Bounding box in mm of the active *voxel grid* extent (includes the
        narrow-band padding, so it is looser than the surface).

        For the tight *surface* bounding box -- what C# PicoGK's
        ``oCalculateBoundingBox`` returns -- use :meth:`calculate_properties`,
        whose second return value is the mesh bounding box."""
        origin, size = self.voxel_dimensions()
        lib, inst = self._lib, self._inst
        lo_v = PKVector3(float(origin[0]), float(origin[1]), float(origin[2]))
        hi_v = PKVector3(float(origin[0] + size[0]), float(origin[1] + size[1]),
                         float(origin[2] + size[2]))
        lo_mm, hi_mm = PKVector3(), PKVector3()
        lib.Library_VoxelsToMm(inst, C.byref(lo_v), C.byref(lo_mm))
        lib.Library_VoxelsToMm(inst, C.byref(hi_v), C.byref(hi_mm))
        return BBox3(vec3_to_np(lo_mm), vec3_to_np(hi_mm))

    def is_inside(self, point) -> bool:
        p = to_vec3(point)
        return bool(self._lib.Voxels_bIsInside(
            self._inst, self.handle, C.byref(p)))

    def surface_normal(self, point) -> np.ndarray:
        p = to_vec3(point)
        out = PKVector3()
        self._lib.Voxels_GetSurfaceNormal(
            self._inst, self.handle, C.byref(p), C.byref(out))
        return vec3_to_np(out)

    def closest_point(self, point) -> np.ndarray | None:
        p = to_vec3(point)
        out = PKVector3()
        ok = self._lib.Voxels_bClosestPointOnSurface(
            self._inst, self.handle, C.byref(p), C.byref(out))
        return vec3_to_np(out) if ok else None

    def ray_cast(self, origin, direction) -> np.ndarray | None:
        o, d = to_vec3(origin), to_vec3(direction)
        out = PKVector3()
        ok = self._lib.Voxels_bRayCastToSurface(
            self._inst, self.handle,
            C.byref(o), C.byref(d), C.byref(out))
        return vec3_to_np(out) if ok else None

    def equals(self, other: Voxels) -> bool:
        return bool(self._lib.Voxels_bIsEqual(
            self._inst, self.handle, other.handle))

    def diagnose(self) -> str:
        buf = C.create_string_buffer(255)
        self._lib.Voxels_bDiagnose(self._inst, self.handle, buf)
        return buf.value.decode(errors="replace")

    # --- slices --------------------------------------------------------------
    # Each returns a 2D float32 array of signed distances in mm (<= 0 is inside),
    # filled by a single native call. Row 0 is the high end of the vertical axis
    # (image-style, top-down). Index is 0-based within the active bounding box.
    def _slice(self, fn_name: str, index: int, h: int, w: int) -> np.ndarray:
        buf = np.empty(h * w, dtype=np.float32)
        bg = C.c_float()
        getattr(self._lib, fn_name)(
            self._inst, self.handle, int(index),
            buf.ctypes.data_as(C.POINTER(C.c_float)), C.byref(bg))
        return buf.reshape(h, w)

    def slice_z(self, index: int) -> np.ndarray:
        """Z slice (XY plane) -> (size_y, size_x)."""
        _, s = self.voxel_dimensions()
        sx, sy, sz = (int(v) for v in s)
        if not 0 <= index < sz:
            raise IndexError(f"z index {index} out of range [0, {sz})")
        return self._slice("Voxels_GetZSlice", index, sy, sx)

    def slice_y(self, index: int) -> np.ndarray:
        """Y slice (XZ plane) -> (size_z, size_x)."""
        _, s = self.voxel_dimensions()
        sx, sy, sz = (int(v) for v in s)
        if not 0 <= index < sy:
            raise IndexError(f"y index {index} out of range [0, {sy})")
        return self._slice("Voxels_GetYSlice", index, sz, sx)

    def slice_x(self, index: int) -> np.ndarray:
        """X slice (YZ plane) -> (size_z, size_y)."""
        _, s = self.voxel_dimensions()
        sx, sy, sz = (int(v) for v in s)
        if not 0 <= index < sx:
            raise IndexError(f"x index {index} out of range [0, {sx})")
        return self._slice("Voxels_GetXSlice", index, sz, sy)

    def slice_z_interpolated(self, z_voxels: float) -> np.ndarray:
        """Z slice at a fractional voxel height (BoxSampler) -> (size_y, size_x)."""
        _, s = self.voxel_dimensions()
        sx, sy, sz = (int(v) for v in s)
        if not 0.0 <= z_voxels <= max(sz - 1, 0):
            raise IndexError(f"z {z_voxels} out of range [0, {sz - 1}]")
        buf = np.empty(sy * sx, dtype=np.float32)
        bg = C.c_float()
        self._lib.Voxels_GetInterpolatedZSlice(
            self._inst, self.handle, float(z_voxels),
            buf.ctypes.data_as(C.POINTER(C.c_float)), C.byref(bg))
        return buf.reshape(sy, sx)

    # --- conversion ----------------------------------------------------------
    def to_mesh(self) -> Mesh:
        from .mesh import Mesh
        return Mesh.from_voxels(self)

    def __repr__(self) -> str:
        if self._closed:
            return "Voxels(<closed>)"
        try:
            _, s = self.voxel_dimensions()
            return f"Voxels(handle={self._h}, voxels={s[0]}x{s[1]}x{s[2]})"
        except Exception:
            return f"Voxels(handle={self._h})"
