"""OpenVDB file I/O: persist and load ``Voxels``, ``ScalarField`` and
``VectorField`` objects to/from ``.vdb`` files.

A ``.vdb`` holds one or more named grids. PicoGK maps grid kinds to types:

============  ================================  ===================
field type    OpenVDB grid                      PicoPie object
============  ================================  ===================
0  VOXELS     float, level set                  :class:`~picogk.Voxels`
1  SCALAR     float, fog volume                 :class:`~picogk.ScalarField`
2  VECTOR     Vec3                              :class:`~picogk.VectorField`
============  ================================  ===================

Convenience::

    picogk.save_vdb("part.vdb", body=voxels, heat=scalar_field)
    objs = picogk.load_vdb("part.vdb")    # -> {"body": Voxels, "heat": ScalarField}
"""

from __future__ import annotations

import ctypes as C
from enum import IntEnum

from . import library
from ._base import NativeObject
from ._errors import PicoGKError
from .fields import ScalarField, VectorField
from .voxels import Voxels

_STRLEN = 255


class FieldType(IntEnum):
    UNKNOWN = -1
    VOXELS = 0   # float level-set grid
    SCALAR = 1   # float fog-volume grid
    VECTOR = 2   # Vec3 grid


class VdbFile(NativeObject):
    _destroy_fn = "VdbFile_Destroy"

    def __init__(self, handle: int | None = None):
        if handle is None:
            handle = library.lib().VdbFile_hCreate(library.instance())
        super().__init__(handle)

    @classmethod
    def load(cls, path: str) -> "VdbFile":
        import os
        h = library.lib().VdbFile_hCreateFromFile(
            library.instance(), str(path).encode())
        if not h:
            hint = "file not found" if not os.path.exists(path) else "unreadable or not a valid .vdb"
            raise PicoGKError(f"failed to load VDB file ({hint}): {path}")
        return cls(h)

    def is_valid(self) -> bool:
        return bool(self._lib.VdbFile_bIsValid(self._inst, self.handle))

    def memory_bytes(self) -> int:
        return int(self._lib.VdbFile_nMemUsage(self._inst, self.handle))

    def save(self, path: str) -> None:
        ok = self._lib.VdbFile_bSaveToFile(self._inst, self.handle, str(path).encode())
        if not ok:
            raise PicoGKError(f"failed to save VDB file: {path}")

    # --- field introspection -------------------------------------------------
    def field_count(self) -> int:
        return int(self._lib.VdbFile_nFieldCount(self._inst, self.handle))

    def _check_index(self, index: int) -> int:
        # The native field accessors call std::vector::at(); an out-of-range
        # index throws std::out_of_range across the C ABI and ABORTS the process
        # (it cannot be caught in Python). Validate here and raise instead.
        n = self.field_count()
        if not 0 <= index < n:
            raise IndexError(f"field index {index} out of range [0, {n})")
        return index

    def field_name(self, index: int) -> str:
        self._check_index(index)
        buf = C.create_string_buffer(_STRLEN)
        self._lib.VdbFile_GetFieldName(self._inst, self.handle, int(index), buf)
        return buf.value.decode(errors="replace")

    def field_type(self, index: int) -> FieldType:
        self._check_index(index)
        return FieldType(int(self._lib.VdbFile_nFieldType(
            self._inst, self.handle, int(index))))

    def fields(self) -> list[tuple[str, FieldType]]:
        return [(self.field_name(i), self.field_type(i))
                for i in range(self.field_count())]

    # --- add -----------------------------------------------------------------
    def add_voxels(self, name: str, voxels: Voxels) -> int:
        return int(self._lib.VdbFile_nAddVoxels(
            self._inst, self.handle, name.encode(), voxels.handle))

    def add_scalar_field(self, name: str, field: ScalarField) -> int:
        return int(self._lib.VdbFile_nAddScalarField(
            self._inst, self.handle, name.encode(), field.handle))

    def add_vector_field(self, name: str, field: VectorField) -> int:
        return int(self._lib.VdbFile_nAddVectorField(
            self._inst, self.handle, name.encode(), field.handle))

    def add(self, name: str, obj) -> int:
        """Add any supported object, dispatching on its type."""
        if isinstance(obj, Voxels):
            return self.add_voxels(name, obj)
        if isinstance(obj, ScalarField):
            return self.add_scalar_field(name, obj)
        if isinstance(obj, VectorField):
            return self.add_vector_field(name, obj)
        raise TypeError(f"cannot add {type(obj).__name__} to a VDB file")

    # --- get -----------------------------------------------------------------
    def get_voxels(self, index: int) -> Voxels:
        self._check_index(index)
        return Voxels(self._lib.VdbFile_hGetVoxels(self._inst, self.handle, int(index)))

    def get_scalar_field(self, index: int) -> ScalarField:
        self._check_index(index)
        return ScalarField(self._lib.VdbFile_hGetScalarField(
            self._inst, self.handle, int(index)))

    def get_vector_field(self, index: int) -> VectorField:
        self._check_index(index)
        return VectorField(self._lib.VdbFile_hGetVectorField(
            self._inst, self.handle, int(index)))

    def get(self, index: int) -> "Voxels | ScalarField | VectorField":
        """Return the field at ``index`` as the correct typed object."""
        t = self.field_type(index)
        if t is FieldType.VOXELS:
            return self.get_voxels(index)
        if t is FieldType.SCALAR:
            return self.get_scalar_field(index)
        if t is FieldType.VECTOR:
            return self.get_vector_field(index)
        raise PicoGKError(f"field {index} has unknown type")

    def to_dict(self) -> dict:
        """Load every field into ``{name: typed object}``."""
        return {self.field_name(i): self.get(i) for i in range(self.field_count())}

    def __repr__(self) -> str:
        if self._closed:
            return "VdbFile(<closed>)"
        return f"VdbFile(handle={self._h}, fields={self.fields()})"


# --- module-level convenience ------------------------------------------------
def save_vdb(path: str, **named_objects) -> None:
    """Save named ``Voxels`` / ``ScalarField`` / ``VectorField`` to a ``.vdb``.

        save_vdb("part.vdb", body=voxels, heat=scalar_field)
    """
    if not named_objects:
        raise ValueError("save_vdb requires at least one named object")
    f = VdbFile()
    try:
        for name, obj in named_objects.items():
            f.add(name, obj)
        f.save(path)
    finally:
        f.close()


def load_vdb(path: str) -> dict:
    """Load all grids from a ``.vdb`` as ``{name: typed object}``."""
    f = VdbFile.load(path)
    try:
        return f.to_dict()
    finally:
        f.close()
