"""ctypes mirrors of the C structs and callback typedefs in PicoGKApiTypes.h.

Memory layout must match the native side exactly. The C header uses
``#pragma pack(1)`` for the POD structs, so we set ``_pack_ = 1`` here.
All native objects are opaque ``uint64_t`` handles (PKHANDLE); the only
exception is the viewer, which is a raw ``void*`` (PKVIEWER).
"""

from __future__ import annotations

import ctypes as C

# --- Handles -----------------------------------------------------------------
# PKHANDLE / PKINSTANCE / PKMESH / ... are all uint64_t.
PKHandle = C.c_uint64
# PKVIEWER is `void*`
PKViewer = C.c_void_p


# --- POD structs (packed, matching PicoGKApiTypes.h) -------------------------
class PKVector2(C.Structure):
    _pack_ = 1
    _fields_ = [("X", C.c_float), ("Y", C.c_float)]


class PKVector3(C.Structure):
    _pack_ = 1
    _fields_ = [("X", C.c_float), ("Y", C.c_float), ("Z", C.c_float)]

    def __iter__(self):  # lets tuple(v) / np.array(v) work
        return iter((self.X, self.Y, self.Z))

    def __repr__(self):
        return f"PKVector3({self.X}, {self.Y}, {self.Z})"


class PKVector4(C.Structure):
    _pack_ = 1
    _fields_ = [("X", C.c_float), ("Y", C.c_float),
                ("Z", C.c_float), ("W", C.c_float)]


class PKCoord(C.Structure):
    _pack_ = 1
    _fields_ = [("X", C.c_int32), ("Y", C.c_int32), ("Z", C.c_int32)]


class PKTriangle(C.Structure):
    _pack_ = 1
    _fields_ = [("A", C.c_int32), ("B", C.c_int32), ("C", C.c_int32)]


class PKBBox3(C.Structure):
    _pack_ = 1
    _fields_ = [("vecMin", PKVector3), ("vecMax", PKVector3)]


class PKMatrix4x4(C.Structure):
    _pack_ = 1
    _fields_ = [("vec1", PKVector4), ("vec2", PKVector4),
                ("vec3", PKVector4), ("vec4", PKVector4)]


class PKColorFloat(C.Structure):
    _pack_ = 1
    _fields_ = [("R", C.c_float), ("G", C.c_float),
                ("B", C.c_float), ("A", C.c_float)]


# --- Callback function types -------------------------------------------------
# float (*PKPFnfSdf)(const PKVector3*)
PKPFnfSdf = C.CFUNCTYPE(C.c_float, C.POINTER(PKVector3))

# void (*PKFnTraverseActiveS)(const PKVector3*, float)
PKFnTraverseActiveS = C.CFUNCTYPE(None, C.POINTER(PKVector3), C.c_float)

# void (*PKFnTraverseActiveV)(const PKVector3*, const PKVector3*)
PKFnTraverseActiveV = C.CFUNCTYPE(None, C.POINTER(PKVector3), C.POINTER(PKVector3))

# void (*PKFInfo)(const char*, bool)
PKFInfo = C.CFUNCTYPE(None, C.c_char_p, C.c_bool)

# --- Viewer callbacks (PicoGKApiTypes.h) -------------------------------------
# void (*PKPFUpdateRequested)(void*, const PKVector2* viewport, PKColorFloat* bg,
#                             PKMatrix4x4* viewProjection, PKVector3* eye)  [out params]
PKPFUpdateRequested = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(PKVector2),
                                  C.POINTER(PKColorFloat), C.POINTER(PKMatrix4x4),
                                  C.POINTER(PKVector3))
# void (*PKPFKeyPressed)(void*, int key, int scancode, int action, int mods)
PKPFKeyPressed = C.CFUNCTYPE(None, C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32)
# void (*PKPFMouseMoved)(void*, const PKVector2* pos, bool shift, ctrl, alt, super)
PKPFMouseMoved = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(PKVector2),
                             C.c_bool, C.c_bool, C.c_bool, C.c_bool)
# void (*PKPFMouseButton)(void*, int button, int action, int mods, const PKVector2* pos)
PKPFMouseButton = C.CFUNCTYPE(None, C.c_void_p, C.c_int32, C.c_int32, C.c_int32,
                              C.POINTER(PKVector2))
# void (*PKPFScrollWheel)(void*, const PKVector2* offset, const PKVector2* pos, bools...)
PKPFScrollWheel = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(PKVector2), C.POINTER(PKVector2),
                              C.c_bool, C.c_bool, C.c_bool, C.c_bool)
# void (*PKPFWindowSize)(void*, const PKVector2* size)
PKPFWindowSize = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(PKVector2))


# Name used by the prototype generator to resolve typedef'd argument types.
CALLBACK_TYPES = {
    "PKPFnfSdf": PKPFnfSdf,
    "PKFnTraverseActiveS": PKFnTraverseActiveS,
    "PKFnTraverseActiveV": PKFnTraverseActiveV,
    "PKFInfo": PKFInfo,
    "PKPFUpdateRequested": PKPFUpdateRequested,
    "PKPFKeyPressed": PKPFKeyPressed,
    "PKPFMouseMoved": PKPFMouseMoved,
    "PKPFMouseButton": PKPFMouseButton,
    "PKPFScrollWheel": PKPFScrollWheel,
    "PKPFWindowSize": PKPFWindowSize,
}

STRUCT_TYPES = {
    "PKVector2": PKVector2,
    "PKVector3": PKVector3,
    "PKVector4": PKVector4,
    "PKCoord": PKCoord,
    "PKTriangle": PKTriangle,
    "PKBBox3": PKBBox3,
    "PKMatrix4x4": PKMatrix4x4,
    "PKColorFloat": PKColorFloat,
}


# --- layout self-check -------------------------------------------------------
# The native ABI uses #pragma pack(1); a wrong _pack_ or field list would
# silently corrupt every call that passes these by value/pointer. Verify the
# packed byte sizes at import (the enforced-in-release analog of the C# port's
# Debug.Assert(Marshal.SizeOf(...)) checks in Library's constructor).
_EXPECTED_SIZES = {
    PKVector2: 8,
    PKVector3: 12,
    PKVector4: 16,
    PKCoord: 12,
    PKTriangle: 12,
    PKBBox3: 24,
    PKMatrix4x4: 64,
    PKColorFloat: 16,
}


def _assert_struct_layout() -> None:
    for struct, expected in _EXPECTED_SIZES.items():
        actual = C.sizeof(struct)
        if actual != expected:
            raise RuntimeError(
                f"ctypes struct {struct.__name__} is {actual} bytes, expected "
                f"{expected} (packed). The native ABI mirror is broken; check "
                f"_pack_/_fields_ in ctypes_types.py.")


_assert_struct_layout()
