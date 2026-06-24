import ctypes as C
import gc
import math

import numpy as np
import pytest

import picogk
from picogk import Lattice, Mesh, Voxels


def test_version_and_name():
    assert picogk.version().count(".") >= 1
    assert "PicoGK" in picogk.name()


def test_sphere_volume():
    r = 10.0
    v = Voxels.sphere(radius=r)
    ideal = 4 / 3 * math.pi * r**3
    assert v.volume_mm3() == pytest.approx(ideal, rel=0.02)
    assert not v.is_empty()
    assert v.is_valid()


def test_boolean_subtract_reduces_volume():
    a = Voxels.sphere(radius=10)
    b = Voxels.sphere(center=(8, 0, 0), radius=6)
    diff = a - b
    assert diff.volume_mm3() < a.volume_mm3()
    # operands are not mutated by the functional operator
    assert a.volume_mm3() == pytest.approx(4 / 3 * math.pi * 10**3, rel=0.02)


def test_intersection_smaller_than_either():
    a = Voxels.sphere(radius=10)
    b = Voxels.sphere(center=(6, 0, 0), radius=10)
    inter = a & b
    assert 0 < inter.volume_mm3() < a.volume_mm3()


def test_offset_grows_volume():
    a = Voxels.sphere(radius=8)
    grown = a.offset(2.0)
    assert grown.volume_mm3() > a.volume_mm3()


def test_shell_hollows():
    # shell_ must hollow to a wall, not leave the part solid (regression: it was
    # implemented as a no-op double offset).
    r, t = 12.0, 1.5
    solid = Voxels.sphere(radius=r).volume_mm3()
    shelled = Voxels.sphere(radius=r)
    shelled.shell_(t)
    wall = shelled.volume_mm3()
    assert wall < solid * 0.6
    ideal = 4 / 3 * math.pi * r**3 - 4 / 3 * math.pi * (r - t) ** 3
    assert wall == pytest.approx(ideal, rel=0.1)


def test_shell_zero_thickness_raises():
    with pytest.raises(ValueError):
        Voxels.sphere(radius=8).shell_(0.0)
    with pytest.raises(ValueError):
        Voxels.sphere(radius=8).shell_(-1.0)


def test_calculate_properties_accurate_vs_raw():
    # accurate (mesh round-trip) volume is close to but distinct from the raw call
    v = Voxels.sphere(radius=10)
    acc, bb = v.calculate_properties()
    assert acc == pytest.approx(4 / 3 * math.pi * 10**3, rel=0.02)
    assert not bb.is_empty()


def test_calculate_properties_empty_no_overflow():
    import warnings
    v = Voxels()                       # empty
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any numpy overflow RuntimeWarning -> fail
        vol, bb = v.calculate_properties()
        _ = bb.size                     # must not overflow on the degenerate bbox
    assert vol == 0.0
    assert bb.is_empty()


def test_shell_thicker_than_object_stays_solid():
    # eroding the core away entirely leaves the part solid (not empty/invalid)
    v = Voxels.sphere(radius=8)
    solid = v.volume_mm3()
    v.shell_(20.0)
    assert not v.is_empty()
    assert v.volume_mm3() == pytest.approx(solid, rel=0.05)


def test_mesh_from_voxels_and_numpy():
    v = Voxels.sphere(radius=6)
    m = v.to_mesh()
    assert m.triangle_count() > 100
    verts = m.vertices
    tris = m.triangles
    assert verts.shape == (m.vertex_count(), 3)
    assert verts.dtype == np.float32
    assert tris.shape == (m.triangle_count(), 3)
    assert tris.max() < verts.shape[0]
    # bounding box of a sphere ~ [-6,6] in each axis
    bb = m.bounding_box()
    assert bb.size == pytest.approx(np.array([12, 12, 12]), abs=1.0)


def test_stl_roundtrip(tmp_path):
    import struct
    m = Voxels.sphere(radius=5).to_mesh()
    p = tmp_path / "s.stl"
    m.save_stl(str(p))
    with open(p, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
    assert n == m.triangle_count()
    assert p.stat().st_size == 84 + 50 * n


def test_lattice_voxelizes():
    lat = Lattice()
    lat.add_beam((-5, 0, 0), (5, 0, 0), 1.0)
    lat.add_sphere((0, 0, 0), 2.0)
    v = lat.to_voxels()
    assert v.volume_mm3() > 0


def test_implicit_sphere_matches_primitive():
    r = 8.0
    v = Voxels()
    v.render_implicit_(lambda x, y, z: math.sqrt(x*x+y*y+z*z) - r,
                       ((-r-2, -r-2, -r-2), (r+2, r+2, r+2)))
    assert v.volume_mm3() == pytest.approx(4 / 3 * math.pi * r**3, rel=0.02)


def test_is_inside():
    v = Voxels.sphere(radius=10)
    assert v.is_inside((0, 0, 0))
    assert not v.is_inside((100, 0, 0))


def test_no_handle_leak():
    lib, inst = picogk.library.lib(), picogk.library.instance()
    before = lib.Library_nVoxelsAllocated(C.c_uint64(inst))
    for _ in range(15):
        a = Voxels.sphere(radius=5)
        b = Voxels.sphere(center=(3, 0, 0), radius=4)
        c = a - b
        del a, b, c
    gc.collect()
    after = lib.Library_nVoxelsAllocated(C.c_uint64(inst))
    assert after <= before


def test_context_manager_closes():
    v = Voxels.sphere(radius=3)
    with v:
        assert v.is_valid()
    with pytest.raises(picogk.InvalidHandleError):
        _ = v.handle
