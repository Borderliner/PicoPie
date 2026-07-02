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

import picopie
from picopie import Voxels
from picopie.shapes import (
    Bisection,
    Box,
    Cone,
    ControlPointSpline,
    ControlPointSurface,
    Cylinder,
    CylindricalControlSpline,
    Frames,
    ImplicitGenus,
    ImplicitGyroid,
    ImplicitSphere,
    ImplicitSuperEllipsoid,
    LatticeManifold,
    LatticePipe,
    Lens,
    LineModulation,
    LocalFrame,
    Pipe,
    PipeSegment,
    Revolve,
    Ring,
    Sphere,
    TangentialControlSpline,
)
from picopie.shapes import formulas as FM
from picopie.shapes import measure as ME
from picopie.shapes import spline_ops as SO
from picopie.shapes import vectors as V

_IMP_PTS = [(0, 0, 0), (3, 1, 2), (5, 5, 5)]
_PHIS = [0.0, 0.5, 1.2, 3.0]

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
    from picopie.shapes import SurfaceModulation
    assert Sphere(radius=5).to_voxels().calculate_properties()[0] > 0
    assert Sphere(radius=lambda phi, theta: 8.0).to_voxels().calculate_properties()[0] > 0
    assert Sphere(radius=SurfaceModulation(6)).to_voxels().calculate_properties()[0] > 0


# --- C# golden parity (skip without the golden) --------------------------------
@needs_golden
def test_voxel_size_matches_golden():
    assert picopie.voxel_size() == pytest.approx(G["voxel_size_mm"])


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


# --- 12c spline / frames parity ------------------------------------------------
@needs_golden
def test_control_point_spline_matches_csharp():
    cps = ControlPointSpline([[0, 0, 0], [5, 10, 0], [10, 0, 0], [15, 10, 0]])
    got = [cps.point_at(t).tolist() for t in (0, 0.25, 0.5, 0.75, 1)]
    assert np.allclose(got, G["cps_samples"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_control_point_spline_closed_matches_csharp():
    cps = ControlPointSpline([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], closed=True)
    got = [cps.point_at(t).tolist() for t in (0, 0.25, 0.5, 0.75, 1)]
    assert np.allclose(got, G["cps_closed_samples"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_control_point_surface_matches_csharp():
    grid = np.array([[[u * 5.0, v * 5.0, u + v] for v in range(3)] for u in range(3)])
    assert np.allclose(ControlPointSurface(grid).point_at(0.3, 0.7),
                       G["surf_point"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_tangential_spline_matches_csharp():
    tcs = TangentialControlSpline([0, 0, 0], [10, 0, 0], [0, 1, 0], [0, -1, 0])
    assert np.allclose(tcs.points(5), G["tcs_samples"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_reparametrized_spline_matches_csharp():
    rep = SO.reparametrized([[0, 0, 0], [10, 0, 0], [10, 10, 0]], 8)
    assert len(rep) == G["reparam_len"]
    assert np.allclose(rep, G["reparam_pts"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_frames_extrude_matches_csharp():
    fr = Frames.extrude(20.0, LocalFrame((0, 0, 0)), spacing=1.0)
    assert np.allclose(fr.spine_at(0.5), G["frames_spine_05"], rtol=MATH_RTOL, atol=MATH_ATOL)
    assert np.allclose(fr.local_z_at(0.5), G["frames_z_05"], rtol=MATH_RTOL, atol=MATH_ATOL)
    assert np.allclose(fr.local_x_at(0.5), G["frames_x_05"], rtol=MATH_RTOL, atol=MATH_ATOL)


# --- 12d frame-based shapes: voxel volume + surface bbox parity -----------------
def _vol_bbox(shape):
    vol, bbox = shape.to_voxels().calculate_properties()
    return vol, [*bbox.min.tolist(), *bbox.max.tolist()]


def _assert_shape(shape, vol_key, bbox_key):
    vol, bbox = _vol_bbox(shape)
    assert vol == pytest.approx(G[vol_key], rel=VOL_REL)
    assert np.allclose(bbox, G[bbox_key], rtol=VOL_REL, atol=1e-3)


@needs_golden
def test_box_matches_csharp():
    _assert_shape(Box(LocalFrame((0, 0, 0)), 20, 10, 8), "box_volume", "box_bbox")


@needs_golden
def test_cylinder_matches_csharp():
    _assert_shape(Cylinder(LocalFrame((0, 0, 0)), 20, 10), "cyl_volume", "cyl_bbox")


@needs_golden
def test_cone_matches_csharp():
    _assert_shape(Cone(LocalFrame((0, 0, 0)), 20, 10, 0), "cone_volume", "cone_bbox")


@needs_golden
def test_ring_matches_csharp():
    _assert_shape(Ring(LocalFrame((0, 0, 0)), 30, 5), "ring_volume", "ring_bbox")


@needs_golden
def test_lens_matches_csharp():
    _assert_shape(Lens(LocalFrame((0, 0, 0)), 4, 0, 10), "lens_volume", "lens_bbox")


# --- 12e spined / revolve shapes -----------------------------------------------
@needs_golden
def test_pipe_matches_csharp():
    _assert_shape(Pipe(LocalFrame((0, 0, 0)), 20, 5, 10), "pipe_volume", "pipe_bbox")


@needs_golden
def test_pipe_segment_matches_csharp():
    seg = PipeSegment(LocalFrame((0, 0, 0)), 20, 5, 10, start=0, end=np.pi / 2,
                      method="start_end")
    _assert_shape(seg, "seg_volume", "seg_bbox")


@needs_golden
def test_revolve_matches_csharp():
    # rotation-heavy + a degenerate axis (inner=0): float32-vs-float64 paths
    # agree to ~1e-3 rather than the 1e-4 the other shapes hit.
    rev = Revolve(LocalFrame((0, 0, 0)), Frames.extrude(20, LocalFrame((0, 0, 0))), 0, 5)
    vol, bbox = rev.to_voxels().calculate_properties()
    assert vol == pytest.approx(G["revolve_volume"], rel=1e-3)
    got = [*bbox.min.tolist(), *bbox.max.tolist()]
    assert np.allclose(got, G["revolve_bbox"], rtol=1e-3, atol=5e-3)


# --- 12f lattice shapes --------------------------------------------------------
@needs_golden
def test_lattice_pipe_matches_csharp():
    vol, bbox = LatticePipe(LocalFrame((0, 0, 0)), 20, 5).to_voxels().calculate_properties()
    assert vol == pytest.approx(G["latpipe_volume"], rel=VOL_REL)
    # a voxelised lattice's surface bbox is sub-voxel sensitive to the
    # float32 (C#) vs float64 (us) beam endpoints; volume is the strict check.
    got = [*bbox.min.tolist(), *bbox.max.tolist()]
    assert np.allclose(got, G["latpipe_bbox"], atol=0.05)


@needs_golden
def test_lattice_manifold_matches_csharp():
    # lattice beams + tiny tear-drop tips -> slightly float-sensitive (~1e-3).
    vox = LatticeManifold(LocalFrame((0, 0, 0)), 20, 5, 45).to_voxels()
    vol, bbox = vox.calculate_properties()
    assert vol == pytest.approx(G["latman_volume"], rel=2e-3)
    got = [*bbox.min.tolist(), *bbox.max.tolist()]
    assert np.allclose(got, G["latman_bbox"], rtol=2e-3, atol=5e-3)


# --- 12f implicit signed-distance functions ------------------------------------
@needs_golden
def test_implicit_sdf_values_match_csharp():
    cases = [
        (ImplicitGyroid(5, 0.3), "imp_gyroid"),
        (ImplicitSphere((0, 0, 0), 8), "imp_sphere"),
        (ImplicitGenus(0.5), "imp_genus"),
        (ImplicitSuperEllipsoid((0, 0, 0), 5, 5, 5, 1, 1), "imp_superellipsoid"),
    ]
    for sdf, key in cases:
        got = [sdf(*p) for p in _IMP_PTS]
        assert np.allclose(got, G[key], rtol=1e-4, atol=1e-4), key


# --- 12f supershape / polygon radii --------------------------------------------
@needs_golden
def test_surface_area_matches_csharp():
    # same primitive sphere + same native mesh as C# -> areas match to float precision
    area = ME.surface_area(Voxels.sphere(radius=10))
    assert area == pytest.approx(G["sphere_surface_area"], rel=VOL_REL)


# --- 12j: previously-untested-vs-C# surfaces (frames alignment, transform, etc.) ---
_SPINE = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [20, 10, 0]]
FRAME_ATOL = 1e-2   # brute-force align search: float32-vs-float64 may pick an adjacent grid step


@needs_golden
def test_frames_cylindrical_matches_csharp():
    f = Frames.aligned(_SPINE, "cylindrical")
    assert np.allclose(f.spine_at(0.5), G["frames_cyl_spine05"], atol=FRAME_ATOL)
    assert np.allclose(f.local_z_at(0.5), G["frames_cyl_z05"], atol=FRAME_ATOL)
    assert np.allclose(f.local_x_at(0.5), G["frames_cyl_x05"], atol=FRAME_ATOL)


@needs_golden
def test_frames_min_rotation_matches_csharp():
    f = Frames.aligned(_SPINE, "min_rotation")
    assert np.allclose(f.local_x_at(0.5), G["frames_min_x05"], atol=FRAME_ATOL)


@needs_golden
def test_frames_aligned_to_x_matches_csharp():
    f = Frames.aligned_to_x(_SPINE, (0, 0, 1))
    assert np.allclose(f.spine_at(0.5), G["frames_tx_spine05"], atol=FRAME_ATOL)
    assert np.allclose(f.local_x_at(0.5), G["frames_tx_x05"], atol=FRAME_ATOL)


@needs_golden
def test_pipe_segment_mid_range_matches_csharp():
    seg = PipeSegment(LocalFrame((0, 0, 0)), 20, 5, 10,
                      start=0.5 * np.pi, end=0.5 * np.pi, method="mid_range")
    vol, bbox = seg.to_voxels().calculate_properties()
    assert vol == pytest.approx(G["seg_midrange_volume"], rel=VOL_REL)
    got = [*bbox.min.tolist(), *bbox.max.tolist()]
    assert np.allclose(got, G["seg_midrange_bbox"], rtol=VOL_REL, atol=1e-3)


@needs_golden
def test_cylindrical_control_spline_matches_csharp():
    ccs = CylindricalControlSpline([5, 0, 0])
    ccs.add_relative_step("z", 10).add_relative_step("radial", 5)
    assert np.allclose(ccs.points(5), G["ccs_samples"], rtol=MATH_RTOL, atol=MATH_ATOL)


@needs_golden
def test_shape_transform_matches_csharp():
    box = Box(LocalFrame((0, 0, 0)), 20, 10, 8,
              transform=lambda p: p + np.array([10.0, 0, 0]))
    vol, bbox = box.to_voxels().calculate_properties()
    assert vol == pytest.approx(G["box_xform_volume"], rel=VOL_REL)
    got = [*bbox.min.tolist(), *bbox.max.tolist()]
    assert np.allclose(got, G["box_xform_bbox"], rtol=VOL_REL, atol=1e-3)


@needs_golden
def test_modulated_cylinder_matches_csharp():
    cyl = Cylinder(LocalFrame((0, 0, 0)), 20, radius=lambda phi, lr: 12 + 3 * np.cos(5 * phi))
    vol, _ = cyl.to_voxels().calculate_properties()
    assert vol == pytest.approx(G["mod_cyl_volume"], rel=VOL_REL)


@needs_golden
def test_supershape_and_polygon_radii_match_csharp():
    assert np.allclose([FM.super_shape_radius(p, 6, 2, 1.2, 1.2) for p in _PHIS],
                       G["supershape_custom"], rtol=1e-5, atol=1e-6)
    assert np.allclose([FM.super_shape_radius_preset(p, "hex") for p in _PHIS],
                       G["supershape_hex"], rtol=1e-5, atol=1e-6)
    assert np.allclose([FM.polygon_radius(p, 6) for p in _PHIS],
                       G["polygon_custom"], rtol=1e-5, atol=1e-6)
    assert np.allclose([FM.polygon_radius_preset(p, "tri") for p in _PHIS],
                       G["polygon_tri"], rtol=1e-5, atol=1e-6)
