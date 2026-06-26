"""Unit tests for the Phase 12c spline/frame layer (pure Python, offline).

Curves/surfaces, spline operations, decimation, and the frame field — analytic
identities and structural checks. Cross-checks vs C# live in
``test_parity_shapekernel.py``.
"""

import numpy as np
import pytest

from picogk.shapes import (
    ControlPointSpline,
    ControlPointSurface,
    CylindricalControlSpline,
    Frames,
    LocalFrame,
    TangentialControlSpline,
    decimation,
    formulas,
)
from picogk.shapes import (
    spline_ops as SO,
)


# --- ControlPointSpline --------------------------------------------------------
def test_open_spline_passes_through_endpoints():
    cps = ControlPointSpline([[0, 0, 0], [5, 10, 0], [10, 0, 0]])
    pts = cps.points(20)
    assert np.allclose(pts[0], [0, 0, 0])
    assert np.allclose(pts[-1], [10, 0, 0])
    assert len(pts) == 20


def test_closed_spline_wraps_control_points():
    # ShapeKernel's closed B-spline appends n-1 control points (wrap); parity vs
    # C# for the sampled curve is checked in test_parity_shapekernel.py.
    cps = ControlPointSpline([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], closed=True)
    assert len(cps.control_points) == 7
    assert len(cps.points(50)) == 50


def test_straight_spline_is_straight():
    cps = ControlPointSpline([[0, 0, 0], [5, 0, 0], [10, 0, 0]])
    pts = cps.points(11)
    assert np.allclose(pts[:, 1:], 0.0)             # no y/z deviation


# --- ControlPointSurface -------------------------------------------------------
def test_surface_corners_match_grid():
    grid = np.array([[[u * 5.0, v * 5.0, 0.0] for v in range(3)] for u in range(3)])
    s = ControlPointSurface(grid)
    assert np.allclose(s.point_at(0, 0), [0, 0, 0])
    assert np.allclose(s.point_at(1, 1), [10, 10, 0])
    assert s.grid(5, 6).shape == (5, 6, 3)


def test_surface_update_control_point():
    grid = np.zeros((3, 3, 3))
    s = ControlPointSurface(grid)
    s.update_control_point([1, 2, 3], 1, 1)
    assert np.allclose(s.control_point(1, 1), [1, 2, 3])


# --- Tangential / Cylindrical --------------------------------------------------
def test_tangential_symmetric_bulge():
    tcs = TangentialControlSpline([0, 0, 0], [10, 0, 0], [0, 1, 0], [0, -1, 0])
    assert np.allclose(tcs.point_at(0.5), [5, 3, 0])


def test_tangential_from_frames():
    a = LocalFrame((0, 0, 0), local_z=[0, 1, 0])
    b = LocalFrame((10, 0, 0), local_z=[0, -1, 0])
    tcs = TangentialControlSpline.from_frames(a, b)
    pts = tcs.points(5)
    assert np.allclose(pts[0], [0, 0, 0])
    assert np.allclose(pts[-1], [10, 0, 0])


def test_cylindrical_steps_build_curve():
    c = CylindricalControlSpline([5, 0, 0])
    c.add_relative_step("z", 10).add_relative_step("radial", 5)
    pts = c.points(20)
    assert len(pts) == 20
    assert np.allclose(pts[0], [5, 0, 0])


# --- spline_ops ----------------------------------------------------------------
def test_reparametrized_uniform_spacing_and_count():
    rep = SO.reparametrized([[0, 0, 0], [10, 0, 0], [10, 10, 0]], 8)
    assert len(rep) == 9                               # n + 1
    seg = np.linalg.norm(np.diff(rep, axis=0), axis=1)
    assert np.allclose(seg, seg[0], atol=1e-6)         # constant spacing
    assert SO.total_length(rep) == pytest.approx(20.0)


def test_reparametrized_by_spacing_min_samples():
    rep = SO.reparametrized_by_spacing([[0, 0, 0], [1, 0, 0]], 0.1)
    assert len(rep) >= 11


def test_linear_interpolation_and_lengths():
    li = SO.linear_interpolation([0, 0, 0], [9, 0, 0], 10)
    assert li.shape == (10, 3)
    assert np.allclose(SO.lengths_at_indices(li)[-1], 9.0)


def test_transforms_and_queries():
    pts = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert np.allclose(SO.translate(pts, [1, 1, 1]), np.array(pts) + 1)
    assert np.allclose(SO.scale(pts, 2.0), np.array(pts) * 2)
    assert np.allclose(SO.average(pts), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(SO.closest_point(pts, [0.9, 0, 0]), [1, 0, 0])
    assert SO.distance_to_closest(pts, [2, 0, 0]) == pytest.approx(1.0)


def test_rotate_and_clustered_and_combine():
    rot = SO.rotate_around_z([[1, 0, 0]], np.pi / 2)
    assert np.allclose(rot[0], [0, 1, 0], atol=1e-9)
    clus = SO.clustered([[0, 0, 0], [0.1, 0, 0], [5, 0, 0]], 1.0)
    assert len(clus) == 2
    assert len(SO.combine([[[0, 0, 0]], [[1, 1, 1]]])) == 2


def test_to_frame_roundtrip():
    f = LocalFrame((1, 2, 3), local_z=[0.3, 0.4, 0.5])
    pts = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    assert np.allclose(SO.express_in_frame(f, SO.to_frame(f, pts)), pts, atol=1e-9)


# --- decimation ----------------------------------------------------------------
def test_decimation_collapses_straight_and_keeps_kinks():
    straight = [[i, 0, 0] for i in range(6)]
    assert len(decimation.decimate(straight, 0.01)) == 2
    kinked = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 2, 0], [2, 4, 0]]
    assert len(decimation.decimate(kinked, 0.01)) == 3   # start, corner, end


# --- Frames --------------------------------------------------------------------
def test_frames_extrude():
    fr = Frames.extrude(20.0, LocalFrame((0, 0, 0)), spacing=1.0)
    assert np.allclose(fr.spine_at(0.0), [0, 0, 0])
    assert np.allclose(fr.spine_at(1.0), [0, 0, 20], atol=1e-6)
    assert np.allclose(fr.local_z_at(0.5), [0, 0, 1])
    assert np.allclose(fr.local_x_at(0.5), [1, 0, 0])


def test_frames_along_spline_constant_axes():
    fr = Frames.along_spline([[0, 0, 0], [10, 0, 0]], LocalFrame((0, 0, 0)), spacing=1.0)
    assert np.allclose(fr.local_z_at(0.3), fr.local_z_at(0.8))


def test_frames_aligned_tangent_z():
    fr = Frames.aligned([[0, 0, 0], [10, 0, 0], [20, 0, 0]], "z", spacing=1.0)
    assert np.allclose(fr.local_z_at(0.5), [1, 0, 0], atol=1e-6)   # tangent along +x


def test_frames_aligned_to_x_and_frame_at():
    fr = Frames.aligned_to_x([[0, 0, 0], [0, 0, 10], [0, 0, 20]], [1, 0, 0], spacing=1.0)
    f = fr.frame_at(0.5)
    assert isinstance(f, LocalFrame)
    assert np.linalg.norm(f.local_z) == pytest.approx(1.0)        # re-normalised
    assert abs(float(np.dot(f.local_x, f.local_z))) < 1e-6        # orthonormal


def test_frames_min_rotation_keeps_x_constant_on_straight_spine():
    # On a straight spine the tangent is constant, so min-rotation transport
    # holds local-X fixed (no twist).
    fr = Frames.aligned([[0, 0, 0], [10, 0, 0], [20, 0, 0]], "min_rotation", spacing=2.0)
    xs = fr._x
    dots = [float(np.dot(xs[i], xs[i + 1])) for i in range(len(xs) - 1)]
    assert min(dots) > 0.999


def test_frames_points_resampled():
    fr = Frames.extrude(10.0, LocalFrame((0, 0, 0)), spacing=1.0)
    assert len(fr.points_resampled(20)) == 21


# --- formulas trans_fixed ------------------------------------------------------
def test_trans_fixed_endpoints():
    assert formulas.trans_fixed(0, 10, 0) == pytest.approx(0.0)
    assert formulas.trans_fixed(0, 10, 1) == pytest.approx(10.0)
    assert np.allclose(formulas.vec_trans_fixed([0, 0, 0], [10, 20, 30], 1), [10, 20, 30])
