"""Coverage for small public API surfaces not exercised elsewhere:
library info/diagnostics, type helpers, BBox3, and a couple of Voxels paths."""

import numpy as np
import pytest

import picopie
from picopie import BBox3, Voxels
from picopie._native.ctypes_types import PKColorFloat, PKVector2, PKVector3
from picopie.types import to_color, to_vec2, to_vec3


# --- library info / diagnostics ---------------------------------------------
def test_library_info_and_state():
    assert picopie.is_initialized()
    assert isinstance(picopie.build_info(), str) and picopie.build_info()
    assert isinstance(picopie.name(), str)
    # create something so memory usage is non-zero
    _v = Voxels.sphere(radius=5)
    assert picopie.total_memory_bytes() > 0


# --- type helpers ------------------------------------------------------------
def test_to_vec3_variants():
    assert isinstance(to_vec3((1, 2, 3)), PKVector3)
    assert isinstance(to_vec3(np.array([1.0, 2.0, 3.0])), PKVector3)
    v = PKVector3(4, 5, 6)
    assert to_vec3(v) is v                         # passthrough


def test_to_vec2():
    p = to_vec2((1.5, 2.5))
    assert isinstance(p, PKVector2)
    assert (p.X, p.Y) == (1.5, 2.5)


def test_to_color_rgb_rgba_passthrough():
    c = to_color((1.0, 0.0, 0.0))                  # rgb -> alpha defaults to 1
    assert (c.R, c.G, c.B, c.A) == (1.0, 0.0, 0.0, 1.0)
    c2 = to_color((0.1, 0.2, 0.3, 0.4))
    assert pytest.approx((0.1, 0.2, 0.3, 0.4)) == (c2.R, c2.G, c2.B, c2.A)
    src = PKColorFloat(0.0, 0.0, 0.0, 1.0)
    assert to_color(src) is src


# --- BBox3 -------------------------------------------------------------------
def test_bbox3_center_size_repr():
    bb = BBox3([0, 0, 0], [2, 4, 6])
    assert np.allclose(bb.center, [1, 2, 3])
    assert np.allclose(bb.size, [2, 4, 6])
    assert not bb.is_empty()
    assert "BBox3" in repr(bb)


# --- Voxels paths ------------------------------------------------------------
def test_mesh_shell_from_mesh():
    mesh = Voxels.sphere(radius=8).to_mesh()
    shell = Voxels.mesh_shell(mesh, radius=1.0)
    assert shell.is_valid() and not shell.is_empty()


def test_render_implicit_accepts_bbox3_object():
    # exercises the BBox3 branch of _to_pkbbox (vs the tuple branch)
    r = 6.0
    v = Voxels()
    v.render_implicit_(lambda x, y, z: (x*x + y*y + z*z) ** 0.5 - r,
                       BBox3([-r-2, -r-2, -r-2], [r+2, r+2, r+2]))
    assert not v.is_empty()


def test_voxels_repr():
    s = repr(Voxels.sphere(radius=4))
    assert "Voxels" in s
    v = Voxels.sphere(radius=4)
    v.close()
    assert "closed" in repr(v)


def test_slice_x_y_index_validation():
    v = Voxels.sphere(radius=8)
    _, s = v.voxel_dimensions()
    with pytest.raises(IndexError):
        v.slice_x(int(s[0]))
    with pytest.raises(IndexError):
        v.slice_y(int(s[1]))


def test_vectorfield_bulk_fallback(monkeypatch):
    from picopie import VectorField
    monkeypatch.setattr(picopie._fast, "lib", None)   # force pure-Python path
    pos = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    vals = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    vf = VectorField().set_many(pos, vals)
    got, found = vf.get_many(pos)
    assert found.all()
    assert np.allclose(got, vals, atol=1e-3)


def test_loader_missing_runtime_env(monkeypatch):
    from picopie._native import loader
    monkeypatch.setenv("PICOGK_RUNTIME", "/nonexistent/picogk.so")
    with pytest.raises(FileNotFoundError):
        loader.find_runtime()
