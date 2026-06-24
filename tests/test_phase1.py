"""Phase 1: fields, VDB I/O, metadata, polyline, mesh import."""

import numpy as np
import pytest

import picogk
from picogk import (
    FieldType,
    Mesh,
    Metadata,
    PolyLine,
    ScalarField,
    VectorField,
    Voxels,
    load_vdb,
    save_vdb,
)


@pytest.fixture
def sphere():
    return Voxels.sphere(radius=8)


# --- ScalarField -------------------------------------------------------------
def test_scalarfield_set_get(sphere):
    sf = ScalarField.from_voxels(sphere)
    sf.set((0, 0, 0), 42.0)
    assert sf.get((0, 0, 0)) == pytest.approx(42.0)
    assert sf.is_valid()


def test_scalarfield_active_values(sphere):
    sf = ScalarField.from_voxels(sphere)
    coords, vals = sf.active_values()
    assert coords.shape[0] == vals.shape[0] > 0
    assert coords.shape[1] == 3


def test_scalarfield_dims_and_slice(sphere):
    sf = ScalarField.from_voxels(sphere)
    _, size = sf.voxel_dimensions()
    assert size.prod() > 0
    sl = sf.slice(int(size[2] // 2))
    assert sl.shape == (int(size[1]), int(size[0]))


# --- VectorField -------------------------------------------------------------
def test_vectorfield_set_get(sphere):
    vf = VectorField.from_voxels(sphere)
    vf.set((0, 0, 0), (1, 2, 3))
    got = vf.get((0, 0, 0))
    assert np.allclose(got, [1, 2, 3])


def test_vectorfield_missing_returns_none(sphere):
    vf = VectorField()
    assert vf.get((100, 100, 100)) is None


# --- Metadata ----------------------------------------------------------------
def test_metadata_roundtrip(sphere):
    md = Metadata.from_voxels(sphere)
    md["material"] = "Ti-6Al-4V"
    md["wall_mm"] = 1.25
    md["dir"] = (0, 0, 1)
    assert md["material"] == "Ti-6Al-4V"
    assert md["wall_mm"] == pytest.approx(1.25)
    assert np.allclose(md["dir"], [0, 0, 1])
    assert "material" in md
    assert "nope" not in md
    with pytest.raises(KeyError):
        _ = md["nope"]


# --- PolyLine ----------------------------------------------------------------
def test_polyline():
    pl = PolyLine.from_points([(0, 0, 0), (10, 0, 0), (10, 10, 0)], color=(1, 0, 0))
    assert pl.vertex_count() == 3
    assert np.allclose(pl.color(), [1, 0, 0, 1])
    assert np.allclose(pl.vertices[1], [10, 0, 0])
    assert np.allclose(pl.bounding_box().size, [10, 10, 0])


# --- VDB I/O -----------------------------------------------------------------
def test_vdb_roundtrip(tmp_path, sphere):
    sf = ScalarField.from_voxels(sphere)
    vf = VectorField.from_voxels(sphere)
    p = tmp_path / "part.vdb"
    save_vdb(str(p), body=sphere, heat=sf, flow=vf)
    assert p.exists()

    objs = load_vdb(str(p))
    assert set(objs) == {"body", "heat", "flow"}
    assert isinstance(objs["body"], Voxels)
    assert isinstance(objs["heat"], ScalarField)
    assert isinstance(objs["flow"], VectorField)
    assert objs["body"].volume_mm3() == pytest.approx(sphere.volume_mm3(), rel=1e-4)


def test_vdb_field_types(tmp_path, sphere):
    from picogk import VdbFile
    p = tmp_path / "v.vdb"
    save_vdb(str(p), body=sphere)
    f = VdbFile.load(str(p))
    assert f.field_count() == 1
    assert f.field_name(0) == "body"
    assert f.field_type(0) == FieldType.VOXELS


# --- Mesh import -------------------------------------------------------------
def test_mesh_from_arrays_tetrahedron():
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tris = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    m = Mesh.from_arrays(verts, tris)
    assert m.vertex_count() == 4
    assert m.triangle_count() == 4


def test_mesh_from_arrays_rejects_bad_index():
    with pytest.raises(ValueError):
        Mesh.from_arrays([(0, 0, 0)], [(0, 1, 2)])


def test_stl_roundtrip_dedup(tmp_path, sphere):
    m = sphere.to_mesh()
    p = tmp_path / "s.stl"
    m.save_stl(str(p))
    m2 = Mesh.load_stl(str(p))
    assert m2.triangle_count() == m.triangle_count()
    # de-duplicated: far fewer vertices than 3x triangles
    assert m2.vertex_count() < m.triangle_count() * 3


def test_obj_roundtrip(tmp_path):
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tris = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    m = Mesh.from_arrays(verts, tris)
    p = tmp_path / "t.obj"
    m.save_obj(str(p))
    m2 = Mesh.load_obj(str(p))
    assert m2.vertex_count() == 4
    assert m2.triangle_count() == 4


def test_loaded_mesh_voxelizes(tmp_path, sphere):
    p = tmp_path / "s.stl"
    sphere.to_mesh().save_stl(str(p))
    m = Mesh.load_stl(str(p))
    vox = Voxels.from_mesh(m)
    assert vox.volume_mm3() == pytest.approx(sphere.volume_mm3(), rel=0.1)


# --- abort-hardening guard ---------------------------------------------------
def test_bool_guard_rejects_non_voxels(sphere):
    with pytest.raises(TypeError):
        _ = sphere - "not a voxels"


def test_bool_guard_rejects_closed(sphere):
    other = Voxels.sphere(radius=3)
    other.close()
    with pytest.raises(picogk.InvalidHandleError):
        sphere.bool_add_(other)
