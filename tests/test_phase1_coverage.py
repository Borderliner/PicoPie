"""Phase 1 coverage gaps surfaced by the audit: value/metadata round-trips,
slice values, copy/build/remove, active_values, VdbFile direct API, ASCII STL,
OBJ polygons, lifetime-after-close, and error paths."""

import ctypes as C
import gc

import numpy as np
import pytest

import picogk
from picogk import (
    FieldType,
    Mesh,
    Metadata,
    PolyLine,
    ScalarField,
    VdbFile,
    VectorField,
    Voxels,
    load_vdb,
    save_vdb,
)

VS = 0.5  # session voxel size (see conftest)


# --- field values survive a VDB round-trip -----------------------------------
def test_scalarfield_value_roundtrip_vdb(tmp_path):
    sf = ScalarField()
    sf.set((1, 2, 3), 5.5)
    p = tmp_path / "sf.vdb"
    save_vdb(str(p), f=sf)
    loaded = load_vdb(str(p))["f"]
    assert isinstance(loaded, ScalarField)
    assert loaded.get((1, 2, 3)) == pytest.approx(5.5, abs=1e-4)


def test_vectorfield_value_roundtrip_vdb(tmp_path):
    vf = VectorField()
    vf.set((1, 2, 3), (4, 5, 6))
    p = tmp_path / "vf.vdb"
    save_vdb(str(p), f=vf)
    loaded = load_vdb(str(p))["f"]
    assert isinstance(loaded, VectorField)
    assert np.allclose(loaded.get((1, 2, 3)), [4, 5, 6], atol=1e-4)


def test_metadata_survives_vdb_roundtrip(tmp_path):
    v = Voxels.sphere(radius=5)
    md = Metadata.from_voxels(v)
    md["material"] = "Inconel 718"
    md["wall_mm"] = 2.5
    md["build_dir"] = (0, 0, 1)
    p = tmp_path / "meta.vdb"
    save_vdb(str(p), body=v)
    loaded = load_vdb(str(p))["body"]
    md2 = Metadata.from_voxels(loaded)
    assert md2["material"] == "Inconel 718"
    assert md2["wall_mm"] == pytest.approx(2.5)
    assert np.allclose(md2["build_dir"], [0, 0, 1])


# --- slice value correctness -------------------------------------------------
def test_scalarfield_slice_values():
    sf = ScalarField()
    sf.set((0, 0, 0), 7.0)
    _, size = sf.voxel_dimensions()
    assert tuple(size) == (1, 1, 1)            # a single active voxel
    sl = sf.slice(0)
    assert sl.shape == (1, 1)
    assert sl[0, 0] == pytest.approx(7.0)


# --- copy / build_from_voxels / remove ---------------------------------------
def test_scalarfield_copy_is_independent():
    a = ScalarField()
    a.set((0, 0, 0), 1.0)
    b = a.copy()
    b.set((0, 0, 0), 9.0)
    assert a.get((0, 0, 0)) == pytest.approx(1.0)
    assert b.get((0, 0, 0)) == pytest.approx(9.0)


def test_scalarfield_build_from_voxels():
    # build_from_voxels assigns `value` to voxels within sd_threshold (and keeps
    # signed-distance-derived values elsewhere), so the chosen value is present.
    v = Voxels.sphere(radius=6)
    sf = ScalarField.build_from_voxels(v, value=3.0, sd_threshold=0.0)
    assert sf.is_valid()
    _, vals = sf.active_values()
    assert vals.size > 0
    assert np.any(np.isclose(vals, 3.0))


def test_scalarfield_remove():
    sf = ScalarField()
    sf.set((0, 0, 0), 1.0)
    assert sf.get((0, 0, 0)) is not None
    sf.remove((0, 0, 0))
    assert sf.get((0, 0, 0)) is None


def test_vectorfield_copy_build_remove():
    v = Voxels.sphere(radius=6)
    vf = VectorField.build_from_voxels(v, (1, 0, 0), sd_threshold=0.0)
    assert vf.is_valid()
    vf2 = vf.copy()
    vf2.set((0, 0, 0), (9, 9, 9))
    vf.set((0, 0, 0), (1, 1, 1))
    assert np.allclose(vf.get((0, 0, 0)), [1, 1, 1])
    assert np.allclose(vf2.get((0, 0, 0)), [9, 9, 9])
    vf.remove((0, 0, 0))
    assert vf.get((0, 0, 0)) is None


# --- active_values coordinates are in mm -------------------------------------
def test_active_values_coords_in_mm():
    sf = ScalarField()
    sf.set((1.0, 0.0, 0.0), 9.0)
    coords, vals = sf.active_values()
    assert coords.shape == (1, 3)
    assert vals[0] == pytest.approx(9.0)
    assert np.allclose(coords[0], [1.0, 0.0, 0.0], atol=VS)  # within one voxel, in mm


def test_vectorfield_active_values():
    vf = VectorField()
    vf.set((0, 0, 0), (1, 2, 3))
    coords, vals = vf.active_values()
    assert coords.shape == (1, 3) and vals.shape == (1, 3)
    assert np.allclose(vals[0], [1, 2, 3])


# --- VdbFile direct API ------------------------------------------------------
def test_vdbfile_direct_api(tmp_path):
    v = Voxels.sphere(radius=5)
    sf = ScalarField.from_voxels(v)
    vf = VectorField.from_voxels(v)
    f = VdbFile()
    f.add_voxels("a", v)
    f.add("b", sf)              # dispatch overload
    f.add_vector_field("c", vf)
    assert f.field_count() == 3
    types = dict(f.fields())
    assert types == {"a": FieldType.VOXELS, "b": FieldType.SCALAR, "c": FieldType.VECTOR}
    # get(i) returns the right typed object for each index
    by_type = {f.field_name(i): type(f.get(i)).__name__ for i in range(3)}
    assert by_type == {"a": "Voxels", "b": "ScalarField", "c": "VectorField"}
    assert f.memory_bytes() >= 0
    f.close()


def test_vdbfile_add_rejects_bad_type():
    f = VdbFile()
    with pytest.raises(TypeError):
        f.add("x", "not a grid")
    f.close()


# --- mesh import edge cases --------------------------------------------------
def test_ascii_stl(tmp_path):
    ascii_stl = (
        "solid t\n"
        "facet normal 0 0 1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
        " endloop\nendfacet\nendsolid t\n"
    )
    p = tmp_path / "a.stl"
    p.write_text(ascii_stl)
    m = Mesh.load_stl(str(p))
    assert m.triangle_count() == 1
    assert m.vertex_count() == 3


def test_obj_quad_is_triangulated(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    p = tmp_path / "q.obj"
    p.write_text(obj)
    m = Mesh.load_obj(str(p))
    assert m.vertex_count() == 4
    assert m.triangle_count() == 2      # quad -> 2 triangles (fan)


def test_obj_negative_indices(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
    p = tmp_path / "n.obj"
    p.write_text(obj)
    m = Mesh.load_obj(str(p))
    assert m.triangle_count() == 1
    assert np.allclose(np.sort(m.triangles[0]), [0, 1, 2])


def test_mesh_is_valid_and_memory(tmp_path):
    m = Voxels.sphere(radius=4).to_mesh()
    assert m.is_valid()
    assert m.memory_bytes() > 0


# --- metadata mapping protocol ----------------------------------------------
def test_metadata_mapping_protocol():
    v = Voxels.sphere(radius=4)
    md = Metadata.from_voxels(v)
    md["a"] = "x"
    md["b"] = 1.0
    assert "a" in md
    assert set(["a", "b"]).issubset(set(md.names()))
    assert len(md) >= 2
    assert "a" in list(iter(md))
    del md["a"]
    assert "a" not in md


def test_metadata_from_fields():
    v = Voxels.sphere(radius=4)
    sf = ScalarField.from_voxels(v)
    vf = VectorField.from_voxels(v)
    Metadata.from_scalar_field(sf)["k"] = "s"
    Metadata.from_vector_field(vf)["k"] = "v"
    assert Metadata.from_scalar_field(sf)["k"] == "s"
    assert Metadata.from_vector_field(vf)["k"] == "v"


# --- polyline extras ---------------------------------------------------------
def test_polyline_valid_and_memory():
    pl = PolyLine.from_points([(0, 0, 0), (1, 1, 1)])
    assert pl.is_valid()
    assert pl.memory_bytes() > 0
    assert np.allclose(pl.color(), [1, 1, 1, 1])  # default white


# --- lifetime & leaks --------------------------------------------------------
def test_loaded_objects_usable_after_file_closed(tmp_path):
    # load_vdb closes the VdbFile internally; returned objects must stay valid
    v = Voxels.sphere(radius=5)
    p = tmp_path / "x.vdb"
    save_vdb(str(p), body=v)
    objs = load_vdb(str(p))
    assert objs["body"].volume_mm3() == pytest.approx(v.volume_mm3(), rel=1e-4)


def test_vdb_load_no_leak(tmp_path):
    v = Voxels.sphere(radius=5)
    p = tmp_path / "x.vdb"
    save_vdb(str(p), body=v)
    lib, inst = picogk.library.lib(), picogk.library.instance()
    before = lib.Library_nVdbFilesAllocated(C.c_uint64(inst))
    for _ in range(10):
        d = load_vdb(str(p))
        del d
    gc.collect()
    after = lib.Library_nVdbFilesAllocated(C.c_uint64(inst))
    assert after <= before


# --- error paths -------------------------------------------------------------
def test_save_vdb_requires_objects(tmp_path):
    with pytest.raises(ValueError):
        save_vdb(str(tmp_path / "empty.vdb"))


def test_load_missing_vdb_raises_clearly(tmp_path):
    with pytest.raises(picogk.PicoGKError, match="file not found"):
        VdbFile.load(str(tmp_path / "nope.vdb"))
