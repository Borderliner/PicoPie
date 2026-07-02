"""Parity vs the reference C# PicoGK wrapper.

The golden values in ``tests/golden/csharp_parity.json`` are produced by running
the C# PicoGK wrapper against the SAME native runtime PicoPie binds (see
``parity/Program.cs`` / ``scripts/gen_csharp_golden.sh``). Because both wrappers
drive identical native calls, matching values should agree to float precision --
so a mismatch means a PicoPie wrapper diverges from the reference.

Volumes use the accurate mesh-round-trip measurement (``calculate_properties``),
matching C#'s ``CalculateProperties``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import picopie
from picopie import Lattice, Voxels

GOLDEN = Path(__file__).parent / "golden" / "csharp_parity.json"
pytestmark = pytest.mark.skipif(not GOLDEN.exists(), reason="C# golden not generated")
G = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}

VOL_REL = 1e-4   # float32 precision; values agree to ~7 sig figs


def test_session_voxel_size_matches_golden():
    assert picopie.voxel_size() == pytest.approx(G["voxel_size_mm"])


def test_sphere_volume():
    vol, _ = Voxels.sphere(radius=10).calculate_properties()
    assert vol == pytest.approx(G["sphere_volume"], rel=VOL_REL)


def test_sphere_voxel_dims_exact():
    origin, size = Voxels.sphere(radius=10).voxel_dimensions()
    got = [*origin.tolist(), *size.tolist()]
    assert got == G["sphere_voxeldims"]


def test_sphere_mesh_counts_exact():
    m = Voxels.sphere(radius=10).to_mesh()
    assert m.vertex_count() == G["sphere_mesh_verts"]
    assert m.triangle_count() == G["sphere_mesh_tris"]


def test_sphere_mesh_bbox():
    m = Voxels.sphere(radius=10).to_mesh()
    bb = m.bounding_box()
    got = [*bb.min.tolist(), *bb.max.tolist()]
    assert np.allclose(got, G["sphere_mesh_bbox"], rtol=VOL_REL, atol=1e-4)


def test_boolean_subtract_volume():
    a = Voxels.sphere(radius=10)
    b = Voxels.sphere(center=(6, 0, 0), radius=6)
    vol, _ = (a - b).calculate_properties()
    assert vol == pytest.approx(G["bool_subtract_volume"], rel=VOL_REL)


def test_boolean_intersect_volume():
    a = Voxels.sphere(radius=10)
    c = Voxels.sphere(center=(6, 0, 0), radius=10)
    vol, _ = (a & c).calculate_properties()
    assert vol == pytest.approx(G["bool_intersect_volume"], rel=VOL_REL)


def test_offset_volume():
    vol, _ = Voxels.sphere(radius=10).offset(2.0).calculate_properties()
    assert vol == pytest.approx(G["offset_volume"], rel=VOL_REL)


def test_capsule_volume():
    # C#: voxLatticeBeam(lib, start, r1, end, r2) == capsule(start, end, r1, r2)
    cap = Voxels.capsule((-10, 0, 0), (10, 0, 0), radius=3, radius2=3)
    vol, _ = cap.calculate_properties()
    assert vol == pytest.approx(G["capsule_volume"], rel=VOL_REL)


def test_lattice_volume():
    lat = Lattice()
    lat.add_sphere((-8, 0, 0), 2.0)
    lat.add_sphere((8, 0, 0), 2.0)
    lat.add_beam((-8, 0, 0), (8, 0, 0), 1.0, 1.0, round_cap=True)
    vol, _ = lat.to_voxels().calculate_properties()
    assert vol == pytest.approx(G["lattice_volume"], rel=VOL_REL)


def test_boolean_add_volume():
    a = Voxels.sphere(radius=10)
    c = Voxels.sphere(center=(6, 0, 0), radius=10)
    vol, _ = (a + c).calculate_properties()
    assert vol == pytest.approx(G["bool_add_volume"], rel=VOL_REL)


def test_double_offset_volume():
    v = Voxels.sphere(radius=10)
    v.double_offset_(2.0, -2.0)            # grow then shrink (rounding)
    vol, _ = v.calculate_properties()
    assert vol == pytest.approx(G["double_offset_volume"], rel=VOL_REL)


def test_negative_offset_volume():
    vol, _ = Voxels.sphere(radius=10).offset(-2.0).calculate_properties()
    assert vol == pytest.approx(G["offset_neg_volume"], rel=VOL_REL)


def test_from_mesh_volume():
    mesh = Voxels.sphere(radius=10).to_mesh()
    vol, _ = Voxels.from_mesh(mesh).calculate_properties()
    assert vol == pytest.approx(G["from_mesh_volume"], rel=VOL_REL)


def test_is_inside_points():
    sph = Voxels.sphere(radius=10)
    pts = [(0, 0, 0), (7, 0, 0), (100, 0, 0), (9.5, 0, 0)]
    got = [bool(sph.is_inside(p)) for p in pts]
    assert got == [bool(b) for b in G["is_inside"]]


def test_vdb_cross_read_from_csharp():
    """A .vdb written by C# PicoGK loads in PicoPie and yields the same geometry."""
    from picopie import load_vdb
    vdb = GOLDEN.parent / "csharp_sphere.vdb"
    if not vdb.exists():
        pytest.skip("C# reference .vdb not generated")
    objs = load_vdb(str(vdb))
    name = G["vdb_field_name"]
    assert name in objs
    assert isinstance(objs[name], Voxels)
    vol, _ = objs[name].calculate_properties()
    assert vol == pytest.approx(G["vdb_sphere_volume"], rel=VOL_REL)


# --- geometric queries (same probe points as parity/Program.cs) --------------
def test_closest_point_parity():
    v = Voxels.sphere(radius=10)
    got = [v.closest_point(p).tolist() for p in [(20, 0, 0), (0, 15, 0), (0, 0, 18)]]
    assert np.allclose(got, G["closest_point"], atol=1e-3)


def test_surface_normal_parity():
    v = Voxels.sphere(radius=10)
    got = [v.surface_normal(p).tolist() for p in [(10, 0, 0), (0, 10, 0), (0, 0, 10)]]
    assert np.allclose(got, G["surface_normal"], atol=1e-3)


def test_ray_cast_parity():
    v = Voxels.sphere(radius=10)
    rays = [((20, 0, 0), (-1, 0, 0)), ((0, 20, 0), (0, -1, 0))]
    got = [v.ray_cast(o, d).tolist() for o, d in rays]
    assert np.allclose(got, G["ray_cast"], atol=1e-3)


def test_surface_bbox_parity():
    # C#'s oCalculateBoundingBox() is the *surface* (mesh) bbox, which PicoPie
    # exposes via calculate_properties() -- not bounding_box() (that's the looser
    # voxel-grid extent, parity-verified via the exact voxel_dimensions match).
    _, bb = Voxels.sphere(radius=10).calculate_properties()
    got = [*bb.min.tolist(), *bb.max.tolist()]
    assert np.allclose(got, G["sphere_bbox"], rtol=VOL_REL, atol=1e-4)
