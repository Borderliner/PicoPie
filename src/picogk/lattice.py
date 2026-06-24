"""``Lattice`` of beams and spheres, voxelized via ``Voxels.from_lattice``."""

from __future__ import annotations

import ctypes as C

from . import library
from ._base import NativeObject
from .types import to_vec3


class Lattice(NativeObject):
    _destroy_fn = "Lattice_Destroy"

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().Lattice_hCreate(C.c_uint64(library.instance()))
        super().__init__(handle)

    def add_sphere(self, center, radius: float) -> "Lattice":
        c = to_vec3(center)
        self._lib.Lattice_AddSphere(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                    C.byref(c), C.c_float(radius))
        return self

    def add_beam(self, start, end, radius_start: float,
                 radius_end: float | None = None, round_cap: bool = True) -> "Lattice":
        a, b = to_vec3(start), to_vec3(end)
        r2 = radius_start if radius_end is None else radius_end
        self._lib.Lattice_AddBeam(C.c_uint64(self._inst), C.c_uint64(self.handle),
                                  C.byref(a), C.byref(b),
                                  C.c_float(radius_start), C.c_float(r2),
                                  C.c_bool(round_cap))
        return self

    def is_valid(self) -> bool:
        return bool(self._lib.Lattice_bIsValid(
            C.c_uint64(self._inst), C.c_uint64(self.handle)))

    def to_voxels(self):
        from .voxels import Voxels
        return Voxels.from_lattice(self)

    def __repr__(self) -> str:
        return "Lattice(<closed>)" if self._closed else f"Lattice(handle={self._h})"
