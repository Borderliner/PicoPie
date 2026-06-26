"""Parity vs the reference C# ShapeKernel (pinned tag ShapeKernel-v2.1.0).

The golden values in ``tests/golden/shapekernel_parity.json`` are produced by
compiling the pinned C# ShapeKernel against our native runtime and running it
headlessly (see ``parity-shapekernel/`` / ``scripts/gen_shapekernel_golden.sh``).
Because PicoPie replicates ShapeKernel's exact parametric sampling, the meshes
-- and therefore the rendered voxels -- match to float precision.

A subset of tests are analytic/structural and run without the golden (so CI
stays green without dotnet); the golden-backed tests skip when it is absent.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import picogk
from picogk.shapes import Bisection, LineModulation, LocalFrame, Sphere
from picogk.shapes import vectors as V

GOLDEN = Path(__file__).parent / "golden" / "shapekernel_parity.json"
G = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
needs_golden = pytest.mark.skipif(
    not GOLDEN.exists(), reason="ShapeKernel golden not generated (needs dotnet)")

VOL_REL = 1e-4   # float32 precision; values agree to ~7 sig figs


# --- structural / analytic (always run) ----------------------------------------
def test_sphere_mesh_triangle_count():
    # Replicates ShapeKernel's (azimuthal-1) x polar quad grid, 2 tris per cell.
    m = Sphere(radius=10).to_mesh()
    assert m.triangle_count() == 2 * (360 - 1) * 180


def test_sphere_volume_is_near_ideal():
    vol, _ = Sphere(radius=10).to_voxels().calculate_properties()
    ideal = 4.0 / 3.0 * np.pi * 10.0 ** 3
    assert vol == pytest.approx(ideal, rel=0.02)


def test_radius_accepts_number_callable_and_modulation():
    from picogk.shapes import SurfaceModulation
    assert Sphere(radius=5).to_voxels().calculate_properties()[0] > 0
    assert Sphere(radius=lambda phi, theta: 8.0).to_voxels().calculate_properties()[0] > 0
    assert Sphere(radius=SurfaceModulation(6)).to_voxels().calculate_properties()[0] > 0


# --- C# golden parity (skip without the golden) --------------------------------
@needs_golden
def test_voxel_size_matches_golden():
    assert picogk.voxel_size() == pytest.approx(G["voxel_size_mm"])


@needs_golden
def test_sphere_mesh_tris_match_csharp():
    m = Sphere(LocalFrame(position=(0, 0, 0)), radius=10).to_mesh()
    assert m.triangle_count() == G["sphere_mesh_tris"]


@needs_golden
def test_sphere_volume_matches_csharp():
    vox = Sphere(LocalFrame(position=(0, 0, 0)), radius=10).to_voxels()
    vol, _ = vox.calculate_properties()
    assert vol == pytest.approx(G["sphere_volume"], rel=VOL_REL)


@needs_golden
def test_sphere_bbox_matches_csharp():
    vox = Sphere(LocalFrame(position=(0, 0, 0)), radius=10).to_voxels()
    _, bbox = vox.calculate_properties()
    got = [*bbox.min.tolist(), *bbox.max.tolist()]
    assert np.allclose(got, G["sphere_bbox"], rtol=VOL_REL, atol=1e-3)


# --- 12b math-core parity (float32 C# vs float64 Python; tight tolerance) -------
MATH_RTOL, MATH_ATOL = 1e-5, 1e-6


@needs_golden
def test_vec_rotate_around_axis_matches_csharp():
    got = V.rotate_around_axis([1, 0, 0], np.pi / 3, [0, 0, 1])
    assert np.allclose(got, G["vec_rotate_axis"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_vec_orthogonal_dir_matches_csharp():
    got = V.orthogonal_dir([0.3, 0.4, 0.5])
    assert np.allclose(got, G["vec_orthogonal_dir"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_vec_angles_match_csharp():
    assert V.angle_between([1, 0, 0], [1, 1, 0]) == pytest.approx(
        G["vec_angle_between"], abs=MATH_ATOL)
    assert V.signed_angle_between([1, 0, 0], [0, 1, 0], [0, 0, 1]) == pytest.approx(
        G["vec_signed_angle"], abs=MATH_ATOL)


@needs_golden
def test_vec_interpolations_match_csharp():
    assert np.allclose(V.cylindrical_interpolation([5, 0, 0], [0, 5, 2], 0.3),
                       G["vec_cyl_interp"], rtol=MATH_RTOL, atol=MATH_ATOL)
    assert np.allclose(V.spherical_interpolation([5, 0, 0], [0, 5, 2], 0.3),
                       G["vec_sph_interp"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_vec_coord_points_match_csharp():
    assert np.allclose(V.cyl_point(5, np.pi / 4, 2), G["vec_cyl_point"],
                       rtol=MATH_RTOL, atol=MATH_ATOL)
    assert np.allclose(V.sph_point(5, np.pi / 4, np.pi / 6), G["vec_sph_point"],
                       rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_local_frame_tilted_axes_match_csharp():
    f = LocalFrame((0, 0, 0), local_z=[0.3, 0.4, 0.5])
    assert np.allclose(f.local_x, G["frame_local_x"], rtol=MATH_RTOL, atol=MATH_ATOL)
    assert np.allclose(f.local_y, G["frame_local_y"], rtol=MATH_RTOL, atol=MATH_ATOL)
    assert np.allclose(f.local_z, G["frame_local_z"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_line_modulation_samples_match_csharp():
    lm = LineModulation.from_points([[0, 0, 0], [0.5, 2, 0], [1, 1, 0]], values="y", axis="x")
    got = [float(lm(t)) for t in (0, 0.25, 0.5, 0.75, 1)]
    assert np.allclose(got, G["line_mod_samples"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_bisection_matches_csharp():
    got = Bisection(lambda x: x * x, 0, 2, 2, epsilon=1e-4).solve()
    # both approximate sqrt(2); float32-vs-float64 paths agree within ~epsilon
    assert got == pytest.approx(G["bisection_sqrt2"], abs=2e-4)
