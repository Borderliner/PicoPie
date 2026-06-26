"""Unit tests for the Phase 12g utilities/IO layer (need the native kernel).

Measure (volume/area/CoG/inertia), MeshUtility (grid/quad/transform/append),
and io_utils (CSV + export). Surface-area C# parity lives in
test_parity_shapekernel.py; CoG/inertia are validated against analytic formulas
(they integrate the meshed solid, unlike C#'s voxel sampling).
"""

import numpy as np
import pytest

from picogk import Mesh, Voxels
from picogk.shapes import (
    Cylinder,
    LocalFrame,
    Sphere,
    io_utils,
)
from picogk.shapes import (
    measure as ME,
)
from picogk.shapes import (
    mesh_utils as MU,
)

F = LocalFrame((0, 0, 0))


# --- Measure -------------------------------------------------------------------
def test_volume_and_surface_area():
    vox = Sphere(radius=10).to_voxels()
    assert ME.volume(vox) == pytest.approx(4 / 3 * np.pi * 1000, rel=0.02)
    assert ME.surface_area(vox) == pytest.approx(4 * np.pi * 100, rel=0.02)


def test_triangle_area():
    assert ME.triangle_area((0, 0, 0), (3, 0, 0), (0, 4, 0)) == pytest.approx(6.0)


def test_surface_area_accepts_mesh_or_voxels():
    vox = Sphere(radius=8).to_voxels()
    assert ME.surface_area(vox) == pytest.approx(ME.surface_area(vox.to_mesh()))


def test_centre_of_gravity_offset_cylinder():
    cg = ME.centre_of_gravity(Cylinder(LocalFrame((5, 5, 10)), 20, 8).to_voxels())
    assert np.allclose(cg, [5, 5, 20], atol=0.2)


def test_moment_of_inertia_matches_analytic_cylinder():
    r, h, rho = 50.0, 5.0, 7000.0       # "chubby" cylinder along +z
    vox = Cylinder(LocalFrame((0, 0, 0)), h, r).to_voxels()
    cg = ME.centre_of_gravity(vox)
    inertia = ME.moment_of_inertia(vox, LocalFrame(tuple(cg)), rho)
    m = rho * np.pi * r * r * h * 1e-9
    assert inertia[0, 0] == pytest.approx(0.25 * m * r * r, rel=0.02)   # Ixx
    assert inertia[2, 2] == pytest.approx(0.5 * m * r * r, rel=0.02)    # Izz
    assert np.abs(inertia - np.diag(np.diag(inertia))).max() < 1e-3     # diagonal


# --- MeshUtility ---------------------------------------------------------------
def test_mesh_from_quad_and_grid():
    q = MU.mesh_from_quad((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    assert q.triangle_count() == 2
    grid = np.array([[[x, y, 0.0] for y in range(3)] for x in range(4)])
    g = MU.mesh_from_grid(grid)
    assert g.triangle_count() == 2 * (4 - 1) * (3 - 1)


def test_add_quad_appends():
    m = Mesh()
    MU.add_quad(m, (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    assert m.triangle_count() == 2


def test_apply_transformation_translates():
    base = Sphere(radius=5).to_mesh()
    moved = MU.apply_transformation(base, lambda p: p + np.array([100.0, 0, 0]))
    assert moved.bounding_box().center[0] == pytest.approx(100, abs=0.5)
    assert moved.triangle_count() == base.triangle_count()


def test_vox_apply_transformation_preserves_volume():
    vox = Sphere(radius=8).to_voxels()
    moved = MU.vox_apply_transformation(vox, lambda p: p + np.array([50.0, 0, 0]))
    assert moved.calculate_properties()[0] == pytest.approx(
        vox.calculate_properties()[0], rel=0.02)


def test_translate_mesh_onto_frame():
    base = Sphere(radius=5).to_mesh()
    out = MU.translate_mesh_onto_frame(base, LocalFrame((0, 0, 0)), LocalFrame((10, 20, 30)))
    assert np.allclose(out.bounding_box().center, [10, 20, 30], atol=0.5)


def test_append_combines_triangles():
    a = Sphere(LocalFrame((0, 0, 0)), radius=5).to_mesh()
    b = Sphere(LocalFrame((20, 0, 0)), radius=5).to_mesh()
    combined = MU.append(a, b)
    assert combined.triangle_count() == a.triangle_count() + b.triangle_count()
    vol = Voxels.from_mesh(combined).calculate_properties()[0]
    assert vol == pytest.approx(2 * 4 / 3 * np.pi * 125, rel=0.05)   # two disjoint spheres


# --- io_utils ------------------------------------------------------------------
def test_export_path():
    # os.path.join uses the platform separator, so don't hardcode "/" (Windows uses "\")
    p = io_utils.export_path("stl", "part", "/tmp")
    assert p.startswith("/tmp") and p.endswith("part.stl")
    assert io_utils.export_path("VDB", "x").endswith("x.vdb")


def test_csv_writer(tmp_path):
    p = tmp_path / "out.csv"
    with io_utils.CsvWriter(str(p)) as w:
        w.add_line("a,b,c")
        w.add_line("1,2,3")
    assert p.read_text().splitlines() == ["a,b,c", "1,2,3"]


def test_export_voxels_to_stl_and_vdb(tmp_path):
    vox = Sphere(radius=5).to_voxels()
    stl = tmp_path / "s.stl"
    io_utils.export_voxels_to_stl(vox, str(stl))
    assert stl.exists() and stl.stat().st_size > 0
    vdb = tmp_path / "s.vdb"
    io_utils.export_voxels_to_vdb(vox, str(vdb))
    assert vdb.exists() and vdb.stat().st_size > 0
