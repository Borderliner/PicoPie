"""Unit tests for the Phase 12b math/support core (pure Python, offline).

These exercise the support layer (vectors, modulations, bisection, lists,
grids, formulas) without the native kernel or the C# golden — analytic
identities, round-trips, and edge cases. Cross-checks vs C# live in
``test_parity_shapekernel.py``.
"""

import math
import random

import numpy as np
import pytest

from picopie.shapes import (
    Bisection,
    BisectionError,
    Distribution,
    GenericContour,
    LineModulation,
    LocalFrame,
    SurfaceModulation,
    formulas,
    grids,
    lists,
)
from picopie.shapes import (
    vectors as V,
)


# --- vectors -------------------------------------------------------------------
def test_cylindrical_roundtrip():
    pt = V.cyl_point(7.0, 0.9, -2.0)
    assert V.planar_radius(pt) == pytest.approx(7.0)
    assert V.phi(pt) == pytest.approx(0.9)
    assert pt[2] == pytest.approx(-2.0)


def test_with_setters_and_shifts():
    pt = np.array([3.0, 4.0, 5.0])
    assert V.planar_radius(V.with_radius(pt, 10.0)) == pytest.approx(10.0)
    assert V.phi(V.with_phi(pt, 1.0)) == pytest.approx(1.0)
    assert V.with_z(pt, 9.0)[2] == pytest.approx(9.0)
    assert V.planar_radius(V.shift_radius(pt, 5.0)) == pytest.approx(10.0)
    assert V.shift_z(pt, 2.0)[2] == pytest.approx(7.0)


def test_rotate_around_axis_is_orthonormal_and_periodic():
    v = np.array([1.0, 2.0, 3.0])
    axis = V.safe_normalized([0.0, 0.0, 1.0])
    full = V.rotate_around_axis(v, 2 * math.pi, axis)
    assert np.allclose(full, v, atol=1e-9)
    quarter = V.rotate_around_axis([1.0, 0.0, 0.0], math.pi / 2, [0, 0, 1])
    assert np.allclose(quarter, [0.0, 1.0, 0.0], atol=1e-9)


def test_orthogonal_dir_is_orthogonal_and_unit():
    for d in ([1, 0, 0], [0, 0, 1], [0.3, 0.4, 0.5], [1, 1, 1]):
        o = V.orthogonal_dir(d)
        assert abs(float(np.dot(o, V.safe_normalized(d)))) < 1e-6
        assert np.linalg.norm(o) == pytest.approx(1.0)


def test_angle_between_edges():
    assert V.angle_between([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert V.angle_between([1, 0, 0], [-1, 0, 0]) == pytest.approx(math.pi)
    assert V.angle_between([1, 0, 0], [0, 1, 0]) == pytest.approx(math.pi / 2)


def test_signed_angle_sign_flips_with_normal():
    a, b = [1, 0, 0], [0, 1, 0]
    assert V.signed_angle_between(a, b, [0, 0, 1]) == pytest.approx(-math.pi / 2)
    assert V.signed_angle_between(a, b, [0, 0, -1]) == pytest.approx(math.pi / 2)


def test_signed_angle_rejects_zero_vector():
    with pytest.raises(ValueError):
        V.signed_angle_between([0, 0, 0], [0, 1, 0], [0, 0, 1])


def test_interpolation_endpoints():
    a, b = [5.0, 0.0, 0.0], [0.0, 5.0, 2.0]
    assert np.allclose(V.lerp(a, b, 0.0), a)
    assert np.allclose(V.lerp(a, b, 1.0), b)
    assert np.allclose(V.cylindrical_interpolation(a, b, 0.0), a, atol=1e-5)
    assert np.allclose(V.cylindrical_interpolation(a, b, 1.0), b, atol=1e-5)
    assert np.allclose(V.spherical_interpolation(a, b, 0.0), a, atol=1e-5)
    assert np.allclose(V.spherical_interpolation(a, b, 1.0), b, atol=1e-5)


def test_frame_point_roundtrip():
    f = LocalFrame((1, 2, 3), local_z=[0.3, 0.4, 0.5])
    world = np.array([4.0, -1.0, 2.5])
    local = V.point_to_local(f, world)
    assert np.allclose(V.point_to_world(f, local), world, atol=1e-9)


# --- modulations ---------------------------------------------------------------
def test_line_modulation_forms():
    assert LineModulation(3.0)(0.5) == pytest.approx(3.0)
    assert LineModulation(3.0).const_value == pytest.approx(3.0)
    assert LineModulation(lambda r: 2 * r)(0.25) == pytest.approx(0.5)


def test_line_modulation_operators():
    a = LineModulation(2.0)
    b = LineModulation(lambda r: r)
    assert (a + b)(0.5) == pytest.approx(2.5)
    assert (a - b)(0.5) == pytest.approx(1.5)
    assert (3.0 * a)(0.9) == pytest.approx(6.0)


def test_line_modulation_from_points_clamps_and_interpolates():
    lm = LineModulation.from_points([[0.25, 1, 0], [0.75, 3, 0]], values="y", axis="x")
    assert lm(0.0) == pytest.approx(1.0)   # held flat below first knot
    assert lm(1.0) == pytest.approx(3.0)   # held flat above last knot
    assert lm(0.5) == pytest.approx(2.0)   # midpoint
    assert np.allclose(lm(np.array([0.25, 0.75])), [1.0, 3.0])


def test_surface_modulation_forms_and_from_line():
    assert SurfaceModulation(5.0)(0.1, 0.2) == pytest.approx(5.0)
    line = LineModulation(lambda r: r)
    assert SurfaceModulation(line, line="second")(0.9, 0.4) == pytest.approx(0.4)
    assert SurfaceModulation(line, line="first")(0.9, 0.4) == pytest.approx(0.9)


def test_surface_modulation_from_image_constant():
    img = np.full((4, 5), 0.5)
    sm = SurfaceModulation.from_image(img, lambda g: 10.0 * g)
    assert float(sm(0.3, 0.7)) == pytest.approx(5.0)


def test_distribution_wrappers():
    d = Distribution(12.0, 4.0)
    assert d.total_length == pytest.approx(12.0)
    assert d.modulation(0.5) == pytest.approx(4.0)
    assert isinstance(GenericContour(1.0, 2.0), Distribution)


# --- bisection -----------------------------------------------------------------
def test_bisection_solves():
    b = Bisection(lambda x: x * x, 0, 2, 2, epsilon=1e-5)
    assert b.solve() == pytest.approx(math.sqrt(2), abs=1e-4)
    assert b.iterations > 0


def test_bisection_invalid_bracket_raises():
    with pytest.raises(BisectionError):
        Bisection(lambda x: x * x, 1, 2, 100, epsilon=1e-3).solve()


# --- lists ---------------------------------------------------------------------
def test_oversample_preserves_endpoints_and_grows():
    out = lists.oversample([0.0, 1.0, 2.0], 4)
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(2.0)
    assert len(out) == 2 * 4 + 1


def test_subsample_and_indices():
    assert lists.subsample([0, 1, 2, 3, 4, 5], 2)[-1] == pytest.approx(5.0)
    assert lists.index_of_max([3, 9, 1]) == 1
    assert lists.index_of_min([3, 9, 1]) == 2


# --- grids ---------------------------------------------------------------------
def test_grid_inverse_is_transpose():
    g = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    inv = grids.inverse(g)
    assert inv.shape == (3, 2, 3)
    assert np.allclose(grids.inverse(inv), g)


def test_grid_row_col_add_remove():
    g = np.zeros((2, 3, 3))
    g2 = grids.add_row_x(g, np.ones((3, 3)))
    assert g2.shape == (3, 3, 3)
    assert np.allclose(grids.row_x(g2, 2), 1.0)
    g3 = grids.add_col_y(g, np.ones((2, 3)))
    assert g3.shape == (2, 4, 3)
    assert np.allclose(grids.col_y(g3, 3), 1.0)
    assert grids.remove_row_x(g2, 0).shape == (2, 3, 3)
    assert grids.remove_col_y(g3, 0).shape == (2, 3, 3)


# --- formulas ------------------------------------------------------------------
def test_trans_smooth_endpoints():
    # far below the transition -> value1; far above -> value2
    assert formulas.trans_smooth(0.0, 10.0, -100.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-6)
    assert formulas.trans_smooth(0.0, 10.0, 100.0, 0.0, 1.0) == pytest.approx(10.0, abs=1e-6)


def test_fibonacci_distributions():
    circle = formulas.fibonacci_circle_points(5.0, 50)
    assert circle.shape == (50, 3)
    assert np.all(np.hypot(circle[:, 0], circle[:, 1]) <= 5.0 + 1e-6)
    assert np.allclose(circle[:, 2], 0.0)
    sphere = formulas.fibonacci_sphere_points(3.0, 100)
    assert np.allclose(np.linalg.norm(sphere, axis=1), 3.0, atol=1e-5)


def test_random_helpers_reproducible_with_seed():
    r1, r2 = random.Random(7), random.Random(7)
    assert formulas.random_gaussian(0, 1, r1) == formulas.random_gaussian(0, 1, r2)
    assert formulas.random_linear(0, 1, random.Random(1)) == \
        formulas.random_linear(0, 1, random.Random(1))
