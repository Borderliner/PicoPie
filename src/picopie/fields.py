"""Scalar and vector fields stored on the same sparse voxel grid as ``Voxels``.

A ``ScalarField`` holds a float per active voxel (e.g. temperature, a weighting
mask); a ``VectorField`` holds a 3-vector per active voxel (e.g. a flow or
orientation field). Both can be built from a ``Voxels`` volume, sampled and
edited per point, and round-tripped through ``.vdb`` files (see ``picopie.vdb``).
"""

from __future__ import annotations

import ctypes as C

import numpy as np

from . import _fast, library
from ._base import NativeObject, require_type
from ._native.ctypes_types import PKFnTraverseActiveS, PKFnTraverseActiveV, PKVector3
from .types import read_voxel_dimensions, to_vec3, vec3_to_np


def _finite_positions(pos: np.ndarray) -> None:
    """Reject NaN/inf sample coordinates. The single-point set/get route through
    ``to_vec3`` (finite-checked), but the bulk fast path passes ``pos`` straight
    into a nogil native loop that turns coords into voxel indices -- where
    non-finite is silent garbage / UB. Keep the two paths consistent."""
    if pos.size and not np.isfinite(pos).all():
        raise ValueError("positions must be finite (got NaN/inf)")


class ScalarField(NativeObject):
    _destroy_fn = "ScalarField_Destroy"

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().ScalarField_hCreate(library.instance())
        super().__init__(handle)

    # --- constructors --------------------------------------------------------
    @classmethod
    def from_voxels(cls, voxels) -> ScalarField:
        """A field defined wherever ``voxels`` has data (values copied from the
        underlying signed-distance grid)."""
        from .voxels import Voxels
        require_type(voxels, Voxels, "voxels")
        h = library.lib().ScalarField_hCreateFromVoxels(
            library.instance(), voxels.handle)
        return cls(h)

    @classmethod
    def build_from_voxels(cls, voxels, value: float,
                          sd_threshold: float) -> ScalarField:
        """A field set to ``value`` for every voxel whose signed distance is
        below ``sd_threshold`` (i.e. inside / near the surface)."""
        from .voxels import Voxels
        require_type(voxels, Voxels, "voxels")
        h = library.lib().ScalarField_hBuildFromVoxels(
            library.instance(), voxels.handle, float(value), float(sd_threshold))
        return cls(h)

    def copy(self) -> ScalarField:
        return ScalarField(self._lib.ScalarField_hCreateCopy(self._inst, self.handle))

    # --- per-point access ----------------------------------------------------
    def set(self, position, value: float) -> ScalarField:
        p = to_vec3(position)
        self._lib.ScalarField_SetValue(self._inst, self.handle, C.byref(p), float(value))
        return self

    def get(self, position) -> float | None:
        """Return the value at ``position`` (mm), or ``None`` if inactive."""
        p = to_vec3(position)
        out = C.c_float()
        ok = self._lib.ScalarField_bGetValue(
            self._inst, self.handle, C.byref(p), C.byref(out))
        return out.value if ok else None

    def remove(self, position) -> ScalarField:
        p = to_vec3(position)
        self._lib.ScalarField_RemoveValue(self._inst, self.handle, C.byref(p))
        return self

    # --- bulk get/set (compiled fast path when available) --------------------
    def set_many(self, positions, values) -> ScalarField:
        """Set many values at once. ``positions`` is (N,3) mm, ``values`` (N,)."""
        pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        vals = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        if len(pos) != len(vals):
            raise ValueError("positions and values must have the same length")
        if not len(pos):
            return self
        _finite_positions(pos)
        if _fast.lib is not None:
            _fast.lib.scalar_set_many(_fast.addr(self._lib.ScalarField_SetValue),
                                      self._inst, self.handle, pos, vals)
        else:
            for p, v in zip(pos, vals, strict=True):
                self.set(p, float(v))
        return self

    def get_many(self, positions) -> tuple[np.ndarray, np.ndarray]:
        """Sample many positions. Returns ``(values, found)``: values is (N,)
        float32 (NaN where inactive), found is (N,) bool."""
        pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        n = len(pos)
        if not n:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)
        _finite_positions(pos)
        if _fast.lib is not None:
            return _fast.lib.scalar_get_many(
                _fast.addr(self._lib.ScalarField_bGetValue),
                self._inst, self.handle, pos)
        out = np.full(n, np.nan, dtype=np.float32)
        found = np.zeros(n, dtype=bool)
        for i, p in enumerate(pos):
            v = self.get(p)
            if v is not None:
                out[i], found[i] = v, True
        return out, found

    # --- bulk / introspection ------------------------------------------------
    def is_valid(self) -> bool:
        return bool(self._lib.ScalarField_bIsValid(self._inst, self.handle))

    def memory_bytes(self) -> int:
        return int(self._lib.ScalarField_nMemUsage(self._inst, self.handle))

    def voxel_dimensions(self) -> tuple[np.ndarray, np.ndarray]:
        return read_voxel_dimensions(
            self._lib, "ScalarField_GetVoxelDimensions", self._inst, self.handle)

    def slice(self, z_index: int) -> np.ndarray:
        """Return a (sy, sx) float32 array for Z slice ``z_index``.

        ``z_index`` is 0-based, measured from the field's active bounding-box
        minimum (i.e. 0 .. size_z - 1). Element [y, x] is the value at that
        voxel column.
        """
        _, size = self.voxel_dimensions()
        sx, sy, sz = int(size[0]), int(size[1]), int(size[2])
        if not 0 <= z_index < sz:                  # native does an unchecked OOB read
            raise IndexError(f"z_index {z_index} out of range [0, {sz})")
        buf = np.empty(sx * sy, dtype=np.float32)
        self._lib.ScalarField_GetSlice(
            self._inst, self.handle, int(z_index),
            buf.ctypes.data_as(C.POINTER(C.c_float)))
        return buf.reshape(sy, sx)

    def active_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(coords, values)`` over all active voxels.

        ``coords`` is (N, 3) float32 **in millimetres**, ``values`` is (N,)
        float32. Implemented via a native traversal calling back into Python --
        slow for large fields (see roadmap Phase 2 for the compiled path).
        """
        coords: list[tuple[float, float, float]] = []
        values: list[float] = []

        @PKFnTraverseActiveS
        def _cb(pcoord, value):
            c = pcoord[0]
            coords.append((c.X, c.Y, c.Z))
            values.append(value)

        self._lib.ScalarField_TraverseActive(self._inst, self.handle, _cb)
        return (np.array(coords, dtype=np.float32).reshape(-1, 3),
                np.array(values, dtype=np.float32))

    def __repr__(self) -> str:
        return "ScalarField(<closed>)" if self._closed else \
            f"ScalarField(handle={self._h})"


class VectorField(NativeObject):
    _destroy_fn = "VectorField_Destroy"

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().VectorField_hCreate(library.instance())
        super().__init__(handle)

    @classmethod
    def from_voxels(cls, voxels) -> VectorField:
        from .voxels import Voxels
        require_type(voxels, Voxels, "voxels")
        h = library.lib().VectorField_hCreateFromVoxels(
            library.instance(), voxels.handle)
        return cls(h)

    @classmethod
    def build_from_voxels(cls, voxels, value, sd_threshold: float) -> VectorField:
        from .voxels import Voxels
        require_type(voxels, Voxels, "voxels")
        v = to_vec3(value)
        h = library.lib().VectorField_hBuildFromVoxels(
            library.instance(), voxels.handle, C.byref(v), float(sd_threshold))
        return cls(h)

    def copy(self) -> VectorField:
        return VectorField(self._lib.VectorField_hCreateCopy(self._inst, self.handle))

    def set(self, position, value) -> VectorField:
        p, val = to_vec3(position), to_vec3(value)
        self._lib.VectorField_SetValue(self._inst, self.handle, C.byref(p), C.byref(val))
        return self

    def get(self, position) -> np.ndarray | None:
        """Return the 3-vector at ``position`` (mm) as float32, or ``None``."""
        p = to_vec3(position)
        out = PKVector3()
        ok = self._lib.VectorField_bGetValue(
            self._inst, self.handle, C.byref(p), C.byref(out))
        return vec3_to_np(out) if ok else None

    def remove(self, position) -> VectorField:
        p = to_vec3(position)
        self._lib.VectorField_RemoveValue(self._inst, self.handle, C.byref(p))
        return self

    # --- bulk get/set (compiled fast path when available) --------------------
    def set_many(self, positions, values) -> VectorField:
        """Set many vectors at once. ``positions`` and ``values`` are (N,3)."""
        pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        vals = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 3)
        if len(pos) != len(vals):
            raise ValueError("positions and values must have the same length")
        if not len(pos):
            return self
        _finite_positions(pos)
        if _fast.lib is not None:
            _fast.lib.vector_set_many(_fast.addr(self._lib.VectorField_SetValue),
                                      self._inst, self.handle, pos, vals)
        else:
            for p, v in zip(pos, vals, strict=True):
                self.set(p, v)
        return self

    def get_many(self, positions) -> tuple[np.ndarray, np.ndarray]:
        """Sample many positions. Returns ``(values, found)``: values is (N,3)
        float32 (NaN rows where inactive), found is (N,) bool."""
        pos = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        n = len(pos)
        if not n:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)
        _finite_positions(pos)
        if _fast.lib is not None:
            return _fast.lib.vector_get_many(
                _fast.addr(self._lib.VectorField_bGetValue),
                self._inst, self.handle, pos)
        out = np.full((n, 3), np.nan, dtype=np.float32)
        found = np.zeros(n, dtype=bool)
        for i, p in enumerate(pos):
            v = self.get(p)
            if v is not None:
                out[i], found[i] = v, True
        return out, found

    def is_valid(self) -> bool:
        return bool(self._lib.VectorField_bIsValid(self._inst, self.handle))

    def memory_bytes(self) -> int:
        return int(self._lib.VectorField_nMemUsage(self._inst, self.handle))

    def active_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(coords, values)``, both (N, 3) float32, over active voxels
        (``coords`` in millimetres)."""
        coords: list[tuple[float, float, float]] = []
        values: list[tuple[float, float, float]] = []

        @PKFnTraverseActiveV
        def _cb(pcoord, pvalue):
            c, v = pcoord[0], pvalue[0]
            coords.append((c.X, c.Y, c.Z))
            values.append((v.X, v.Y, v.Z))

        self._lib.VectorField_TraverseActive(self._inst, self.handle, _cb)
        return (np.array(coords, dtype=np.float32).reshape(-1, 3),
                np.array(values, dtype=np.float32).reshape(-1, 3))

    def __repr__(self) -> str:
        return "VectorField(<closed>)" if self._closed else \
            f"VectorField(handle={self._h})"
