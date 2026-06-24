#!/usr/bin/env python3
"""Minimal end-to-end check of the native PicoGK runtime via ctypes.

Hand-wires a handful of functions (no codegen involved) so a failure here
points at the *runtime*, not at our binding layer. Validates:
  - the .so loads and exports the expected symbols
  - a library instance can be created headlessly (no OpenGL)
  - a sphere voxel field can be created and its volume matches 4/3 pi r^3
  - a triangle mesh can be extracted from the voxels
"""
import ctypes as C
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from picogk._native.ctypes_types import PKVector3  # noqa: E402
from picogk._native.loader import load_runtime  # noqa: E402

STRLEN = 255


def main() -> int:
    lib = load_runtime()
    print(f"[ok] loaded runtime: {lib._picogk_path}")

    # --- string info getters: void Library_GetX(char psz[255]) ---
    for fn in ("Library_GetName", "Library_GetVersion", "Library_GetBuildInfo"):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = [C.c_char_p]
        buf = C.create_string_buffer(STRLEN)
        f(buf)
        print(f"[ok] {fn}: {buf.value.decode(errors='replace')}")

    # --- instance lifecycle ---
    lib.Library_hCreateInstance.restype = C.c_uint64
    lib.Library_hCreateInstance.argtypes = [C.c_float]
    lib.Library_DestroyInstance.restype = None
    lib.Library_DestroyInstance.argtypes = [C.c_uint64]

    voxel_size_mm = 0.5
    inst = lib.Library_hCreateInstance(C.c_float(voxel_size_mm))
    print(f"[ok] created instance handle={inst} (voxel size {voxel_size_mm} mm)")
    if inst == 0:
        print("[FAIL] instance handle is 0")
        return 1

    # --- sphere voxels ---
    lib.Voxels_hCreateSphere.restype = C.c_uint64
    lib.Voxels_hCreateSphere.argtypes = [C.c_uint64, C.POINTER(PKVector3), C.c_float]
    lib.Voxels_bIsValid.restype = C.c_bool
    lib.Voxels_bIsValid.argtypes = [C.c_uint64, C.c_uint64]
    lib.Voxels_fCalculateVolume.restype = C.c_float
    lib.Voxels_fCalculateVolume.argtypes = [C.c_uint64, C.c_uint64]
    lib.Voxels_GetVoxelDimensions.restype = None
    lib.Voxels_GetVoxelDimensions.argtypes = [C.c_uint64, C.c_uint64] + [C.POINTER(C.c_int32)] * 6
    lib.Voxels_Destroy.restype = None
    lib.Voxels_Destroy.argtypes = [C.c_uint64, C.c_uint64]

    radius = 10.0
    center = PKVector3(0.0, 0.0, 0.0)
    vox = lib.Voxels_hCreateSphere(inst, C.byref(center), C.c_float(radius))
    assert lib.Voxels_bIsValid(inst, vox), "sphere voxels invalid"

    dims = [C.c_int32() for _ in range(6)]
    lib.Voxels_GetVoxelDimensions(inst, vox, *[C.byref(d) for d in dims])
    ox, oy, oz, sx, sy, sz = (d.value for d in dims)
    print(f"[ok] sphere voxel dims: origin=({ox},{oy},{oz}) size=({sx},{sy},{sz})")

    vol = lib.Voxels_fCalculateVolume(inst, vox)
    expected = 4.0 / 3.0 * math.pi * radius ** 3
    err = abs(vol - expected) / expected * 100.0
    print(f"[ok] sphere volume: {vol:.2f} mm^3  (ideal {expected:.2f}, err {err:.2f}%)")
    if err > 2.0:
        print("[FAIL] volume error too large")
        return 1

    # --- mesh extraction ---
    lib.Mesh_hCreateFromVoxels.restype = C.c_uint64
    lib.Mesh_hCreateFromVoxels.argtypes = [C.c_uint64, C.c_uint64]
    lib.Mesh_nVertexCount.restype = C.c_int32
    lib.Mesh_nVertexCount.argtypes = [C.c_uint64, C.c_uint64]
    lib.Mesh_nTriangleCount.restype = C.c_int32
    lib.Mesh_nTriangleCount.argtypes = [C.c_uint64, C.c_uint64]

    mesh = lib.Mesh_hCreateFromVoxels(inst, vox)
    nv = lib.Mesh_nVertexCount(inst, mesh)
    nt = lib.Mesh_nTriangleCount(inst, mesh)
    print(f"[ok] mesh from voxels: {nv} vertices, {nt} triangles")
    if nt < 100:
        print("[FAIL] mesh has too few triangles")
        return 1

    lib.Voxels_Destroy(inst, vox)
    lib.Library_DestroyInstance(inst)
    print("\n[PASS] native PicoGK runtime is fully functional headless. \U0001F389")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
