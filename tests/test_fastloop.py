"""Phase 2: the compiled fast path must match the pure-Python path exactly,
and the public API must work whether or not the extension is compiled."""

import numpy as np
import pytest

import picogk
from picogk import Mesh, ScalarField, VectorField, Voxels, _fast


@pytest.fixture
def mesh():
    return Voxels.sphere(radius=6).to_mesh()


def test_fastloop_is_built():
    # In this dev/CI environment the extension should be compiled.
    assert _fast.available(), "compiled _fastloop extension not found"


def test_vertices_fast_equals_pure(mesh):
    assert np.array_equal(mesh.vertices, mesh._vertices_py(mesh.vertex_count()))


def test_triangles_fast_equals_pure(mesh):
    assert np.array_equal(mesh.triangles, mesh._triangles_py(mesh.triangle_count()))


def test_vertices_fallback_matches(monkeypatch, mesh):
    fast = mesh.vertices
    monkeypatch.setattr(picogk._fast, "lib", None)
    pure = mesh.vertices
    assert np.array_equal(fast, pure)


def test_from_arrays_fast_equals_pure(mesh):
    verts, tris = mesh.vertices, mesh.triangles
    fast = Mesh.from_arrays(verts, tris)
    assert fast.vertex_count() == len(verts)
    assert fast.triangle_count() == len(tris)
    assert np.array_equal(fast.vertices, verts)
    assert np.array_equal(fast.triangles, tris)


def test_from_arrays_fallback_matches(monkeypatch):
    verts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
    tris = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], dtype=np.int32)
    fast = Mesh.from_arrays(verts, tris)
    monkeypatch.setattr(picogk._fast, "lib", None)
    pure = Mesh.from_arrays(verts, tris)
    assert np.array_equal(fast.vertices, pure.vertices)
    assert np.array_equal(fast.triangles, pure.triangles)


def _distinct_voxel_grid():
    # spacing 1 mm >> 0.5 mm voxel -> every point lands in a distinct voxel
    return (np.mgrid[0:8, 0:8, 0:4].reshape(3, -1).T.astype(np.float32))


def test_scalar_set_get_many_roundtrip():
    pos = _distinct_voxel_grid()
    vals = (pos[:, 0] * 100 + pos[:, 1] * 10 + pos[:, 2]).astype(np.float32)
    sf = ScalarField().set_many(pos, vals)
    got, found = sf.get_many(pos)
    assert found.all()
    assert np.allclose(got, vals, atol=1e-3)


def test_scalar_set_get_many_fallback(monkeypatch):
    pos = _distinct_voxel_grid()
    vals = np.arange(len(pos), dtype=np.float32)
    monkeypatch.setattr(picogk._fast, "lib", None)
    sf = ScalarField().set_many(pos, vals)
    got, found = sf.get_many(pos)
    assert found.all()
    assert np.allclose(got, vals, atol=1e-3)


def test_scalar_get_many_marks_inactive():
    sf = ScalarField()
    sf.set((0, 0, 0), 5.0)
    pos = np.array([(0, 0, 0), (50, 50, 50)], dtype=np.float32)
    got, found = sf.get_many(pos)
    assert found[0] and not found[1]
    assert got[0] == pytest.approx(5.0)
    assert np.isnan(got[1])


def test_vector_set_get_many_roundtrip():
    pos = _distinct_voxel_grid()
    vals = np.column_stack([pos[:, 0], pos[:, 1], pos[:, 2]]).astype(np.float32)
    vf = VectorField().set_many(pos, vals)
    got, found = vf.get_many(pos)
    assert found.all()
    assert np.allclose(got, vals, atol=1e-3)


def test_set_many_length_mismatch():
    with pytest.raises(ValueError):
        ScalarField().set_many([(0, 0, 0), (1, 1, 1)], [1.0])


def test_get_many_empty():
    got, found = ScalarField().get_many(np.empty((0, 3), dtype=np.float32))
    assert got.shape == (0,) and found.shape == (0,)
    gv, fv = VectorField().get_many(np.empty((0, 3), dtype=np.float32))
    assert gv.shape == (0, 3) and fv.shape == (0,)


# --- edge cases surfaced by the Phase 2 audit --------------------------------
def test_vector_get_many_marks_inactive():
    vf = VectorField()
    vf.set((0, 0, 0), (1, 2, 3))
    got, found = vf.get_many([(0, 0, 0), (99, 99, 99)])
    assert found.tolist() == [True, False]
    assert np.allclose(got[0], [1, 2, 3])
    assert np.all(np.isnan(got[1]))


def test_vector_set_many_length_mismatch():
    with pytest.raises(ValueError):
        VectorField().set_many([(0, 0, 0), (1, 1, 1)], [(1, 1, 1)])


def test_bulk_accepts_noncontiguous_and_float64():
    # F-contiguous float64 input must be coerced, not crash the memoryview
    pos = np.asfortranarray(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64))
    assert not pos.flags["C_CONTIGUOUS"]
    sf = ScalarField().set_many(pos, [10.0, 20.0, 30.0])
    got, found = sf.get_many(pos)
    assert found.all()
    assert np.allclose(got, [10, 20, 30])


def test_bulk_accepts_python_list_input():
    sf = ScalarField().set_many([(5, 0, 0), (6, 0, 0)], [7.0, 8.0])
    got, _ = sf.get_many([(5, 0, 0), (6, 0, 0)])
    assert np.allclose(got, [7, 8])


def test_empty_mesh_arrays():
    m = Mesh()
    assert m.vertices.shape == (0, 3) and m.vertices.dtype == np.float32
    assert m.triangles.shape == (0, 3) and m.triangles.dtype == np.int32


def test_from_arrays_no_triangles():
    m = Mesh.from_arrays([(0, 0, 0), (1, 0, 0)], np.empty((0, 3), dtype=np.int32))
    assert m.vertex_count() == 2 and m.triangle_count() == 0


def test_get_many_output_dtypes():
    sf = ScalarField()
    sf.set((0, 0, 0), 1.0)
    got, found = sf.get_many([(0, 0, 0)])
    assert got.dtype == np.float32 and found.dtype == np.bool_
