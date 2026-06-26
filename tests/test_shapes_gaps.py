"""Fill the test gaps the geometry-surface review flagged (Phase 12j).

Covers previously-untested public API: LocalFrame transforms, Sphere.surface_point,
SurfaceModulation operators, Bisection accessors, CylindricalControlSpline absolute
steps (incl. its intentional divergence), assorted spline_ops/vectors/formulas
helpers, implicit render of every primitive, per-shape transform=, spine
constructors, modulated dimensions, and degenerate-but-tolerated inputs. Offline
(analytic/structural); C# parity for the deterministic ones is in
test_parity_shapekernel.py.
"""

import math

import numpy as np
import pytest

from picogk.shapes import (
    Bisection,
    Box,
    Cone,
    Cylinder,
    CylindricalControlSpline,
    Frames,
    ImplicitGenus,
    ImplicitSuperEllipsoid,
    LatticeManifold,
    LatticePipe,
    Lens,
    LocalFrame,
    Pipe,
    PipeSegment,
    Revolve,
    Ring,
    Sphere,
    SurfaceModulation,
    formulas,
    functions,
)
from picogk.shapes import (
    spline_ops as SO,
)
from picogk.shapes import (
    vectors as V,
)


# --- LocalFrame transforms (were zero-test) ------------------------------------
def test_local_frame_translated_rotated_inverted():
    f = LocalFrame((0, 0, 0))
    assert np.allclose(f.translated((10, 2, 0)).position, [10, 2, 0])
    assert np.allclose(f.translated((10, 0, 0)).local_z, f.local_z)
    rot = f.rotated(math.pi / 2, (0, 0, 1))
    assert np.allclose(rot.local_x, [0, 1, 0], atol=1e-9)        # +x -> +y
    inv = f.inverted(mirror_z=True, mirror_x=False)
    assert np.allclose(inv.local_z, [0, 0, -1])
    assert np.allclose(LocalFrame.get_local_y((0, 0, 1), (1, 0, 0)), [0, 1, 0])


# --- Sphere.surface_point (was zero-test) --------------------------------------
def test_sphere_surface_point():
    s = Sphere(radius=10)
    assert np.allclose(s.surface_point(0.0, 0.5, 1.0), [10, 0, 0], atol=1e-6)   # equator
    assert np.allclose(s.surface_point(0.0, 0.5, 0.0), [0, 0, 0], atol=1e-6)    # centre


# --- SurfaceModulation operators (were zero-test) ------------------------------
def test_surface_modulation_operators():
    a, b = SurfaceModulation(2.0), SurfaceModulation(3.0)
    assert (a + b)(0, 0) == pytest.approx(5.0)
    assert (b - a)(0, 0) == pytest.approx(1.0)
    assert (3.0 * a)(0, 0) == pytest.approx(6.0)


# --- Bisection accessors (were zero-test) --------------------------------------
def test_bisection_accessors():
    b = Bisection(lambda x: x * x, 0, 2, 2, epsilon=1e-3)
    b.solve()
    assert b.remaining_diff < 1e-3
    assert b.best_guess == pytest.approx(math.sqrt(2), abs=1e-3)


# --- CylindricalControlSpline absolute steps + divergence ----------------------
def test_cylindrical_control_spline_absolute_steps():
    c = CylindricalControlSpline([5, 0, 0])
    c.add_absolute_step("z", 20).add_absolute_step("radial", 8)
    pts = c.points(20)
    assert pts[-1][2] == pytest.approx(20, abs=0.5)            # ends near z=20
    with pytest.raises(ValueError):                            # diverges from C# no-op
        CylindricalControlSpline([5, 0, 0]).add_absolute_step("tangential", 3)


# --- spline_ops helpers (were untested) ----------------------------------------
def test_spline_ops_misc():
    pts = [[0, 0, 0], [10, 0, 0], [10, 10, 0]]
    assert SO.average_spacing(pts) == pytest.approx(10.0)
    a, b = SO.split(pts, 2)
    assert len(a) == 2 and len(b) == 1
    rot = SO.rotate_around_axis([[1, 0, 0]], math.pi / 2, [0, 0, 1])
    assert np.allclose(rot[0], [0, 1, 0], atol=1e-9)
    assert np.allclose(SO.lengths_at_indices(pts), [0, 10, 20])


def test_spline_ops_snapped_to_sphere():
    s = Sphere(radius=10).to_voxels()
    snapped = SO.snapped([[20, 0, 0], [0, 15, 0]], s)
    assert np.allclose(np.linalg.norm(snapped, axis=1), 10, atol=0.5)


# --- vectors helpers (were untested) -------------------------------------------
def test_vectors_misc():
    assert np.allclose(V.planar_dir([3, 4, 0]), [0.6, 0.8, 0])
    assert V.theta([1, 0, 1]) == pytest.approx(math.atan2(1, 1))
    assert V.is_aligned([1, 0, 0], [2, 0, 0]) is True
    assert np.allclose(V.flip_for_alignment([1, 0, 0], [-1, 0, 0]), [-1, 0, 0])
    assert np.allclose(V.sph_point(5, 0, 0), [5, 0, 0])
    f = LocalFrame((1, 0, 0))
    # radius about the frame's local Z (planar, ignores the local-Z component)
    assert V.radius_to_axis(f, [1, 3, 4]) == pytest.approx(3.0)
    assert V.phi_to_axis(f, [1, 1, 0]) == pytest.approx(math.pi / 2)
    assert np.allclose(V.direction_to_axis(f, [1, 1, 0]), [0, 1, 0], atol=1e-9)


# --- formulas helpers (were untested) ------------------------------------------
def test_formulas_misc():
    assert np.allclose(formulas.vec_trans_smooth([0, 0, 0], [10, 10, 10], -100, 0, 1),
                       [0, 0, 0], atol=1e-6)
    import random
    assert formulas.random_bool(random.Random(1)) == formulas.random_bool(random.Random(1))


# --- implicit render of every primitive ----------------------------------------
def test_implicit_genus_and_superellipsoid_render():
    s = 6.0
    g = ImplicitGenus(0.0)
    from picogk import Voxels
    vox = Voxels().render_implicit_(lambda x, y, z: g(x / s, y / s, z / s),
                                    ((-18, -18, -10), (18, 18, 10)))
    assert vox.calculate_properties()[0] > 0
    se = ImplicitSuperEllipsoid((0, 0, 0), 8, 8, 8, 1.5, 1.5)
    assert se.render(((-8, -8, -8), (8, 8, 8))).calculate_properties()[0] > 0


# --- functions.add_line --------------------------------------------------------
def test_functions_add_line():
    from picogk import Lattice
    lat = Lattice()
    functions.add_line(lat, [[0, 0, 0], [10, 0, 0], [10, 10, 0]], 2)
    assert lat.to_voxels().calculate_properties()[0] > 0


# --- per-shape transform= (only Box was tested / golden) -----------------------
@pytest.mark.parametrize("make", [
    lambda t: Cylinder(LocalFrame((0, 0, 0)), 20, 8, transform=t),
    lambda t: Cone(LocalFrame((0, 0, 0)), 20, 8, 2, transform=t),
    lambda t: Sphere(radius=10, transform=t),
    lambda t: Ring(LocalFrame((0, 0, 0)), 20, 5, transform=t),
    lambda t: Lens(LocalFrame((0, 0, 0)), 6, 0, 12, transform=t),
    lambda t: PipeSegment(LocalFrame((0, 0, 0)), 20, 4, 8, start=0, end=np.pi, transform=t),
    lambda t: Revolve(LocalFrame((0, 0, 0)), Frames.extrude(20, LocalFrame((0, 0, 0))), 0, 6,
                      radial_steps=30, length_steps=40, transform=t),
    lambda t: LatticePipe(LocalFrame((0, 0, 0)), 20, 5, transform=t),
])
def test_transform_translates_each_shape(make):
    moved = make(lambda p: p + np.array([60.0, 0, 0]))
    _, bbox = moved.to_voxels().calculate_properties()
    assert bbox.center[0] > 40


# --- spine constructors (Box/PipeSegment/LatticeManifold were untested) ---------
def test_spine_constructors():
    spine = Frames.aligned([[0, 0, 0], [0, 0, 10], [0, 0, 20]], "z", spacing=1.0)
    assert Box(frames=spine, width=8, depth=6).to_voxels().calculate_properties()[0] > 0
    assert PipeSegment(frames=spine, inner_radius=3, outer_radius=6,
                       start=0, end=np.pi).to_voxels().calculate_properties()[0] > 0
    assert LatticeManifold(frames=spine, radius=4).to_voxels().calculate_properties()[0] > 0


def test_box_spine_modulated_bumps_steps():       # B2 fix
    b = Box(frames=Frames.extrude(20, LocalFrame()), width=lambda lr: 8 + 2 * lr, depth=6)
    assert b.width_steps == 500 and b.length_steps == 500


# --- modulated dimensions (Lens/Ring/Cone were untested) -----------------------
def test_modulated_dimensions():
    assert Ring(LocalFrame((0, 0, 0)), 20,
                radius=lambda phi, a: 5 + np.cos(3 * phi)).to_voxels().calculate_properties()[0] > 0
    assert Lens(LocalFrame((0, 0, 0)), 6, 0, 12,
                upper=lambda phi, rr: 4 + np.cos(4 * phi)).to_voxels().calculate_properties()[0] > 0
    assert Cone(LocalFrame((0, 0, 0)), 20, 10, 2).to_voxels().calculate_properties()[0] > 0


# --- degenerate-but-tolerated inputs (no crash) --------------------------------
def test_degenerate_inputs_do_not_crash():
    # inner >= outer pipe/lens -> empty-ish but must not abort
    Pipe(LocalFrame((0, 0, 0)), 20, inner_radius=8, outer_radius=3).to_voxels()
    Lens(LocalFrame((0, 0, 0)), 4, inner_radius=12, outer_radius=2).to_voxels()
    # tiny lattice
    LatticePipe(LocalFrame((0, 0, 0)), 20, 5, length_steps=2).to_voxels()
    # too-few control points for the degree -> a clean error (not ZeroDivisionError)
    from picogk.shapes import ControlPointSpline
    with pytest.raises(ValueError):
        ControlPointSpline([[0, 0, 0], [1, 0, 0]])              # 2 pts, degree 2
    assert len(ControlPointSpline([[0, 0, 0], [1, 0, 0], [2, 0, 0]]).points(10)) == 10
