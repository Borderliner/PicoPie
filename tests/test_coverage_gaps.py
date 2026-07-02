"""Phase 5.5: cover public methods that previously had no direct tests."""

import math

import numpy as np
import pytest

import picopie
from picopie import Mesh, Voxels


# --- Voxels geometric queries ------------------------------------------------
def test_surface_normal_points_outward():
    v = Voxels.sphere(radius=10)
    n = v.surface_normal((10, 0, 0))           # on the +x surface
    assert n.shape == (3,)
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=0.3)
    assert n[0] > 0.5                          # roughly +x


def test_closest_point_on_surface():
    v = Voxels.sphere(radius=10)
    p = v.closest_point((15, 0, 0))
    assert p is not None
    assert np.linalg.norm(p) == pytest.approx(10.0, rel=0.1)


def test_ray_cast_hits_surface():
    v = Voxels.sphere(radius=10)
    hit = v.ray_cast((20, 0, 0), (-1, 0, 0))
    assert hit is not None
    assert hit[0] == pytest.approx(10.0, abs=1.0)


def test_ray_cast_miss_returns_none():
    v = Voxels.sphere(radius=10)
    assert v.ray_cast((20, 20, 20), (1, 0, 0)) is None   # pointing away


def test_equals():
    a = Voxels.sphere(radius=5)
    assert a.equals(a.copy())
    assert not a.equals(Voxels.sphere(radius=6))


def test_diagnose_returns_str():
    assert isinstance(Voxels.sphere(radius=5).diagnose(), str)


def test_memory_bytes_positive():
    assert Voxels.sphere(radius=5).memory_bytes() > 0


def test_copy_is_independent():
    a = Voxels.sphere(radius=8)
    v0 = a.volume_mm3()
    b = a.copy()
    b.offset_(3.0)
    assert a.volume_mm3() == pytest.approx(v0)   # original unchanged
    assert b.volume_mm3() != pytest.approx(v0)


def test_triple_offset_valid():
    v = Voxels.sphere(radius=10)
    v.triple_offset_(1.0)
    assert v.is_valid() and not v.is_empty()


def test_project_z_slice_valid():
    v = Voxels.sphere(radius=10)
    v.project_z_slice_(-2.0, 2.0)
    assert v.is_valid()


def test_intersect_implicit_clips():
    # keep the x<=0 hemisphere; volume should drop to ~half, no abort
    v = Voxels.sphere(radius=10)
    full = v.volume_mm3()
    v.intersect_implicit_(lambda x, y, z: x)
    clipped = v.volume_mm3()
    assert 0 < clipped < full
    assert clipped == pytest.approx(full / 2, rel=0.15)


def test_intersect_implicit_fine_voxel_size():
    # Regression: at voxel sizes below ~0.33 mm, upstream's IntersectImplicit
    # truncated the narrow band (mm float) into an int 0, producing a grid with
    # background 0 -> OpenVDB's csgIntersection aborted. We patch the runtime to
    # pass the source narrow band instead (see scripts/patch_runtime.py). This
    # must work at a fine voxel size, where the old build raised.
    prev = picopie.voxel_size()
    picopie.shutdown()
    try:
        picopie.init(voxel_size_mm=0.1)
        v = Voxels.sphere(radius=10)
        full = v.volume_mm3()
        v.intersect_implicit_(lambda x, y, z: x)   # keep the x<=0 hemisphere
        clipped = v.volume_mm3()
        assert 0 < clipped < full
    finally:
        picopie.shutdown()
        picopie.init(voxel_size_mm=prev)            # restore the session voxel size


# --- in-place operators ------------------------------------------------------
def test_inplace_add_mutates():
    v = Voxels.sphere(radius=8)
    before = v.volume_mm3()
    v += Voxels.sphere(center=(10, 0, 0), radius=8)
    assert v.volume_mm3() > before


def test_inplace_subtract_and_intersect():
    v = Voxels.sphere(radius=10)
    v -= Voxels.sphere(center=(6, 0, 0), radius=6)
    assert v.volume_mm3() < 4 / 3 * math.pi * 10**3
    w = Voxels.sphere(radius=10)
    w &= Voxels.sphere(center=(6, 0, 0), radius=10)
    assert w.volume_mm3() > 0


def test_or_operator_is_union():
    a = Voxels.sphere(radius=10)
    b = Voxels.sphere(center=(6, 0, 0), radius=10)
    assert (a | b).volume_mm3() > a.volume_mm3()


# --- Mesh manual building ----------------------------------------------------
def test_mesh_add_vertex_triangle():
    m = Mesh()
    i0 = m.add_vertex((0, 0, 0))
    i1 = m.add_vertex((1, 0, 0))
    i2 = m.add_vertex((0, 1, 0))
    m.add_triangle(i0, i1, i2)
    assert m.vertex_count() == 3
    assert m.triangle_count() == 1


# --- library.init guard ------------------------------------------------------
def test_init_same_voxel_size_is_noop():
    picopie.init(picopie.voxel_size())           # same -> no error


def test_init_different_voxel_size_raises():
    other = picopie.voxel_size() + 0.25
    with pytest.raises(picopie.PicoPieError):
        picopie.init(other)
