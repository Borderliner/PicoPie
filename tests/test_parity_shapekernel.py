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
from picogk.shapes import LocalFrame, Sphere

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
