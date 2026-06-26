"""Unit tests for the Phase 12f lattice & implicit layer (need the native kernel).

Lattice shapes, implicit SDF primitives, lattice builders, z-axis helpers, and
the supershape/polygon radii. Exact C# parity lives in
``test_parity_shapekernel.py``.
"""

import numpy as np
import pytest

from picogk import Voxels
from picogk.shapes import (
    Frames,
    ImplicitGenus,
    ImplicitGyroid,
    ImplicitSphere,
    ImplicitSuperEllipsoid,
    LatticeManifold,
    LatticePipe,
    LocalFrame,
    formulas,
)
from picogk.shapes import (
    functions as fn,
)

F = LocalFrame((0, 0, 0))


def _vol(vox):
    return vox.calculate_properties()[0]


# --- lattice shapes ------------------------------------------------------------
def test_lattice_pipe_is_a_rounded_tube():
    vox = LatticePipe(F, 20, 5).to_voxels()
    vol, bbox = vox.calculate_properties()
    # cylinder body + rounded beam caps -> a bit more than pi r^2 L, and taller
    assert vol > np.pi * 25 * 20
    assert bbox.max[2] > 20


def test_lattice_pipe_on_spine():
    spine = Frames.aligned([[0, 0, 0], [0, 0, 10], [0, 0, 20]], "z", spacing=1.0)
    assert _vol(LatticePipe(frames=spine, radius=4).to_voxels()) > 0


def test_lattice_manifold_adds_tips():
    pipe = LatticePipe(F, 20, 5).to_voxels()
    man = LatticeManifold(F, 20, 5, max_overhang_angle=45).to_voxels()
    # tear-drop tips extend the bounding box upward beyond the plain pipe
    assert man.calculate_properties()[1].max[2] > pipe.calculate_properties()[1].max[2]


# --- implicit primitives -------------------------------------------------------
def test_implicit_sphere_render_volume():
    sph = ImplicitSphere((0, 0, 0), 5)
    vol = _vol(sph.render(((-6, -6, -6), (6, 6, 6))))
    assert vol == pytest.approx(4 / 3 * np.pi * 125, rel=0.05)


def test_implicit_sphere_sign():
    sph = ImplicitSphere((0, 0, 0), 5)
    assert sph(0, 0, 0) < 0          # inside
    assert sph(5, 0, 0) == pytest.approx(0.0, abs=1e-9)   # on surface
    assert sph(10, 0, 0) > 0         # outside


def test_implicit_sphere_intersect_clips():
    base = Voxels.sphere(radius=10)
    clipped = ImplicitSphere((0, 0, 0), 6).intersect(base)
    assert _vol(clipped) == pytest.approx(4 / 3 * np.pi * 216, rel=0.05)


def test_implicit_gyroid_thins_a_solid():
    base = Voxels.sphere(radius=10)
    gyroid = ImplicitGyroid(6, 0.4)
    vol = _vol(gyroid.intersect(base))
    assert 0 < vol < _vol(base)      # gyroid carves the solid into a shell network


def test_implicit_gyroid_thickness_ratio_helper():
    assert ImplicitGyroid.thickness_ratio_for(1.0, 10.0) == pytest.approx(1.0)


def test_implicit_superellipsoid_eps1_is_ellipsoid_ish():
    se = ImplicitSuperEllipsoid((0, 0, 0), 5, 5, 5, 1, 1)
    assert se(0, 0, 0) < 0
    assert se(5, 0, 0) == pytest.approx(0.0, abs=1e-6)


def test_implicit_genus_callable():
    assert ImplicitGenus(0.0)(0, 0, 0) == pytest.approx(1.0)


# --- lattice builders ----------------------------------------------------------
def test_lat_from_line_and_points():
    assert _vol(fn.lat_from_line([[0, 0, 0], [10, 0, 0], [10, 10, 0]], 2).to_voxels()) > 0
    assert _vol(fn.lat_from_points([[0, 0, 0], [10, 0, 0]], 2).to_voxels()) > 0
    assert _vol(fn.lat_from_point([0, 0, 0], 3).to_voxels()) == pytest.approx(
        4 / 3 * np.pi * 27, rel=0.05)


def test_lat_from_beam_and_edges_and_grid():
    assert _vol(fn.lat_from_beam([0, 0, 0], [10, 0, 0], 2).to_voxels()) > 0
    assert _vol(fn.lat_from_edges([[[0, 0, 0], [5, 0, 0]], [[0, 5, 0], [5, 5, 0]]], 1)
                .to_voxels()) > 0
    grid = np.array([[[x, y, 0.0] for y in range(3)] for x in range(3)])
    assert _vol(fn.lat_from_grid(grid, 1).to_voxels()) > 0


# --- z-axis helpers ------------------------------------------------------------
def test_z_axis_solid_helpers():
    assert _vol(fn.cylinder_between_z(0, 10, 5)) == pytest.approx(np.pi * 25 * 10, rel=0.03)
    assert _vol(fn.box_between_z(0, 10, 8, 6)) == pytest.approx(10 * 8 * 6, rel=0.03)
    assert _vol(fn.cone_between_z(0, 10, 5, 0)) == pytest.approx(np.pi * 25 * 10 / 3, rel=0.03)
    assert _vol(fn.pipe_between_z(0, 10, 3, 5)) == pytest.approx(
        np.pi * (25 - 9) * 10, rel=0.03)


def test_z_helper_respects_frame():
    f = LocalFrame((100, 0, 0))
    _, bbox = fn.cylinder_between_z(0, 10, 5, frame=f).calculate_properties()
    assert bbox.center[0] == pytest.approx(100, abs=0.5)


# --- supershape / polygon radii ------------------------------------------------
def test_supershape_round_is_unit():
    phis = np.linspace(0, 2 * np.pi, 50)
    assert np.allclose(formulas.super_shape_radius_preset(phis, "round"), 1.0)


def test_supershape_vectorised():
    phis = np.linspace(0, 2 * np.pi, 20)
    r = formulas.super_shape_radius(phis, 6, 2, 1.2, 1.2)
    assert r.shape == (20,) and np.all(r > 0)


def test_polygon_radius_corners_reach_unit_circle():
    # a regular polygon touches the unit circle at its vertices (max radius ~1)
    phis = np.linspace(0, 2 * np.pi, 360)
    r = formulas.polygon_radius(phis, 6)
    assert np.max(r) == pytest.approx(1.0, abs=1e-3)
    assert np.min(r) < 1.0
