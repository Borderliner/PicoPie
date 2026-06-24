"""Key/value metadata attached to a ``Voxels`` / ``ScalarField`` / ``VectorField``.

Values are strings, floats, or 3-vectors. Metadata travels with the grid into
``.vdb`` files, so it's the place to stamp units, provenance, parameters, etc.

    md = picogk.Metadata.from_voxels(part)
    md["material"] = "Ti-6Al-4V"
    md["wall_mm"] = 1.0
    print(md.to_dict())
"""

from __future__ import annotations

import ctypes as C
from enum import IntEnum

import numpy as np

from . import library
from ._base import NativeObject
from ._native.ctypes_types import PKVector3
from .types import to_vec3, vec3_to_np


class MetaType(IntEnum):
    UNKNOWN = -1
    STRING = 0
    FLOAT = 1
    VECTOR = 2


class Metadata(NativeObject):
    _destroy_fn = "Metadata_Destroy"

    # Metadata is obtained from an owning object; never created bare.
    @classmethod
    def from_voxels(cls, voxels) -> Metadata:
        return cls(library.lib().Metadata_hFromVoxels(library.instance(), voxels.handle))

    @classmethod
    def from_scalar_field(cls, field) -> Metadata:
        return cls(library.lib().Metadata_hFromScalarField(
            library.instance(), field.handle))

    @classmethod
    def from_vector_field(cls, field) -> Metadata:
        return cls(library.lib().Metadata_hFromVectorField(
            library.instance(), field.handle))

    # --- introspection -------------------------------------------------------
    def count(self) -> int:
        return int(self._lib.Metadata_nCount(self._inst, self.handle))

    def names(self) -> list[str]:
        out = []
        for i in range(self.count()):
            n = int(self._lib.Metadata_nNameLengthAt(self._inst, self.handle, i))
            buf = C.create_string_buffer(max(n + 1, 1))
            if self._lib.Metadata_bGetNameAt(self._inst, self.handle, i, buf, len(buf)):
                out.append(buf.value.decode(errors="replace"))
        return out

    def type_of(self, name: str) -> MetaType:
        return MetaType(int(self._lib.Metadata_nTypeAt(
            self._inst, self.handle, name.encode())))

    # --- typed getters -------------------------------------------------------
    def get_string(self, name: str) -> str | None:
        n = int(self._lib.Metadata_nStringLengthAt(self._inst, self.handle, name.encode()))
        buf = C.create_string_buffer(max(n + 1, 1))
        ok = self._lib.Metadata_bGetStringAt(
            self._inst, self.handle, name.encode(), buf, len(buf))
        return buf.value.decode(errors="replace") if ok else None

    def get_float(self, name: str) -> float | None:
        out = C.c_float()
        ok = self._lib.Metadata_bGetFloatAt(
            self._inst, self.handle, name.encode(), C.byref(out))
        return out.value if ok else None

    def get_vector(self, name: str) -> np.ndarray | None:
        out = PKVector3()
        ok = self._lib.Metadata_bGetVectorAt(
            self._inst, self.handle, name.encode(), C.byref(out))
        return vec3_to_np(out) if ok else None

    def get(self, name: str) -> str | float | np.ndarray | None:
        """Return the value, typed automatically (str / float / np.ndarray)."""
        t = self.type_of(name)
        if t is MetaType.STRING:
            return self.get_string(name)
        if t is MetaType.FLOAT:
            return self.get_float(name)
        if t is MetaType.VECTOR:
            return self.get_vector(name)
        return None

    # --- setters -------------------------------------------------------------
    def set_string(self, name: str, value: str) -> Metadata:
        self._lib.Metadata_SetStringValue(
            self._inst, self.handle, name.encode(), str(value).encode())
        return self

    def set_float(self, name: str, value: float) -> Metadata:
        self._lib.Metadata_SetFloatValue(
            self._inst, self.handle, name.encode(), float(value))
        return self

    def set_vector(self, name: str, value) -> Metadata:
        v = to_vec3(value)
        self._lib.Metadata_SetVectorValue(
            self._inst, self.handle, name.encode(), C.byref(v))
        return self

    def set(self, name: str, value) -> Metadata:
        """Set a value, dispatching on the Python type."""
        if isinstance(value, str):
            return self.set_string(name, value)
        if isinstance(value, (int, float, np.floating, np.integer)):
            return self.set_float(name, float(value))
        # treat anything vector-like as a 3-vector
        return self.set_vector(name, value)

    def remove(self, name: str) -> Metadata:
        self._lib.MetaData_RemoveValue(self._inst, self.handle, name.encode())
        return self

    def to_dict(self) -> dict:
        return {n: self.get(n) for n in self.names()}

    # --- mapping sugar -------------------------------------------------------
    def __getitem__(self, name: str):
        v = self.get(name)
        if v is None and self.type_of(name) is MetaType.UNKNOWN:
            raise KeyError(name)
        return v

    def __setitem__(self, name: str, value):
        self.set(name, value)

    def __delitem__(self, name: str):
        self.remove(name)

    def __contains__(self, name: str) -> bool:
        return self.type_of(name) is not MetaType.UNKNOWN

    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        return iter(self.names())

    def __repr__(self) -> str:
        return "Metadata(<closed>)" if self._closed else f"Metadata({self.to_dict()})"
