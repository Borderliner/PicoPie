"""Compiled (native) callback path for render_implicit_ / intersect_implicit_.

We don't depend on numba in CI, so these tests compile a tiny C SDF at test
time with the system C compiler and load it via ``ctypes.CDLL`` -- a real
machine-code function pointer, exercising the exact ``_as_native_sdf`` detection
and address-rebuild path a ``numba.cfunc`` would hit. Skipped where no C
compiler is available.
"""

from __future__ import annotations

import ctypes as C
import math
import shutil
import subprocess
import sysconfig

import pytest

from picopie import Voxels
from picopie.voxels import _to_pkbbox

# The SDF ABI PicoGK expects: float(const PKVector3*), i.e. a pointer to three
# contiguous float32. sphere of radius 8 at the origin.
_C_SRC = """
#include <math.h>
typedef struct { float X, Y, Z; } V3;
float sphere_sdf(const V3* p) {
    return sqrtf(p->X*p->X + p->Y*p->Y + p->Z*p->Z) - 8.0f;
}
float nan_sdf(const V3* p) { (void)p; return NAN; }
"""

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(_CC is None, reason="no C compiler to build the native SDF")

_BB = ((-10, -10, -10), (10, 10, 10))
_EXPECTED = 4 / 3 * math.pi * 8**3


@pytest.fixture(scope="module")
def native_lib(tmp_path_factory):
    d = tmp_path_factory.mktemp("nativesdf")
    src = d / "sdf.c"
    src.write_text(_C_SRC)
    ext = ".dll" if sysconfig.get_platform().startswith("win") else ".so"
    lib = d / f"libsdf{ext}"
    subprocess.run([_CC, "-O3", "-shared", "-fPIC", "-o", str(lib), str(src)], check=True)
    return C.CDLL(str(lib))


def test_render_implicit_native_matches_python(native_lib):
    # native (compiled) path
    a = Voxels()
    a.render_implicit_(native_lib.sphere_sdf, _BB)
    # guarded Python path, same surface
    b = Voxels()
    b.render_implicit_(lambda x, y, z: math.sqrt(x * x + y * y + z * z) - 8.0, _BB)

    assert a.volume_mm3() == pytest.approx(_EXPECTED, rel=0.02)
    # native vs python callback must agree to the voxel
    assert a.volume_mm3() == pytest.approx(b.volume_mm3(), rel=1e-6)


def test_as_native_sdf_detects_and_rebuilds(native_lib):
    from picopie._native.ctypes_types import PKPFnfSdf

    cb = Voxels._as_native_sdf(native_lib.sphere_sdf)
    assert isinstance(cb, PKPFnfSdf)
    # points at the same code as the CDLL export
    want = C.cast(native_lib.sphere_sdf, C.c_void_p).value
    got = C.cast(cb, C.c_void_p).value
    assert got == want


def test_as_native_sdf_passes_through_plain_callable():
    # a plain Python callable is NOT treated as native -> keeps the finite guard
    assert Voxels._as_native_sdf(lambda x, y, z: 0.0) is None


def test_as_native_sdf_detects_numba_style_address():
    # duck-typed numba.cfunc: exposes an integer .address (no ctypes involved)
    class _FakeCFunc:
        address = C.cast(C.CFUNCTYPE(C.c_float)(lambda: 0.0), C.c_void_p).value

    from picopie._native.ctypes_types import PKPFnfSdf

    assert isinstance(Voxels._as_native_sdf(_FakeCFunc()), PKPFnfSdf)


def test_intersect_implicit_native(native_lib):
    # keep the x <= 0 half of a big block via a compiled half-space SDF
    v = Voxels.sphere(radius=9)
    before = v.volume_mm3()
    # native sphere intersect keeps the r<=8 interior -> volume shrinks to ~sphere(8)
    v.intersect_implicit_(native_lib.sphere_sdf)
    assert v.volume_mm3() < before
    assert v.volume_mm3() == pytest.approx(_EXPECTED, rel=0.05)


def test_intersect_implicit_native_still_guards_double_call(native_lib):
    from picopie import PicoPieError

    v = Voxels.sphere(radius=9)
    v.intersect_implicit_(native_lib.sphere_sdf)
    with pytest.raises(PicoPieError):
        v.intersect_implicit_(native_lib.sphere_sdf)


def test_native_nan_return_bypasses_guard(native_lib):
    # A compiled callback that returns NaN is the caller's responsibility: unlike
    # the Python path (which clamps NaN -> empty), the native path passes it
    # straight through. We only assert it does not raise on our side -- geometry
    # is whatever the kernel makes of a NaN, and that's the documented contract.
    box = _to_pkbbox(_BB)
    cb = Voxels._as_native_sdf(native_lib.nan_sdf)
    v = Voxels()
    v._lib.Voxels_RenderImplicit(v._inst, v.handle, C.byref(box), cb)  # must not raise
