"""Phase 3: slice extraction and headless visualization."""

import math

import numpy as np
import pytest

import picogk
from picogk import Voxels, viz


@pytest.fixture
def sphere():
    return Voxels.sphere(radius=10)


@pytest.fixture
def capsule():
    # elongated along X so size_x != size_y == size_z -> distinguishes axes
    return Voxels.capsule((-15, 0, 0), (15, 0, 0), radius=3)


# --- slice geometry ----------------------------------------------------------
def test_slice_shapes_match_axes(capsule):
    _, s = capsule.voxel_dimensions()
    sx, sy, sz = (int(v) for v in s)
    assert sx > sy  # sanity: actually elongated
    assert capsule.slice_z(sz // 2).shape == (sy, sx)
    assert capsule.slice_y(sy // 2).shape == (sz, sx)
    assert capsule.slice_x(sx // 2).shape == (sz, sy)


def test_slice_dtype_is_float32(sphere):
    _, s = sphere.voxel_dimensions()
    assert sphere.slice_z(int(s[2]) // 2).dtype == np.float32


def test_midslice_inside_area_matches_circle(sphere):
    _, s = sphere.voxel_dimensions()
    sl = sphere.slice_z(int(s[2]) // 2)
    inside = int((sl <= 0).sum())
    r_vox = 10.0 / picogk.voxel_size()           # radius in voxels
    expected = math.pi * r_vox ** 2
    assert inside == pytest.approx(expected, rel=0.1)


def test_slice_index_validation(sphere):
    _, s = sphere.voxel_dimensions()
    with pytest.raises(IndexError):
        sphere.slice_z(int(s[2]))
    with pytest.raises(IndexError):
        sphere.slice_z(-1)


def test_interpolated_slice_shape(sphere):
    _, s = sphere.voxel_dimensions()
    sx, sy, sz = (int(v) for v in s)
    assert sphere.slice_z_interpolated(sz / 2.0).shape == (sy, sx)


# --- colorize ----------------------------------------------------------------
def test_colorize_mask():
    arr = np.array([[-1.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    img = viz.colorize(arr, "mask")
    assert img.shape == (2, 2) and img.dtype == np.uint8
    assert set(np.unique(img).tolist()) <= {0, 255}
    assert img[0, 0] == 255 and img[0, 1] == 0   # <=0 white, >0 black


def test_colorize_sdf_and_gray():
    arr = np.linspace(-2, 2, 9, dtype=np.float32).reshape(3, 3)
    assert viz.colorize(arr, "sdf").shape == (3, 3, 3)
    assert viz.colorize(arr, "gray").shape == (3, 3)


def test_colorize_bad_mode():
    with pytest.raises(ValueError):
        viz.colorize(np.zeros((2, 2), np.float32), "nope")


# --- PNG output (needs Pillow) ----------------------------------------------
def test_save_slice_png(tmp_path, sphere):
    Image = pytest.importorskip("PIL.Image")
    _, s = sphere.voxel_dimensions()
    sx, sy = int(s[0]), int(s[1])
    p = viz.save_slice_png(sphere, str(tmp_path / "z.png"), axis="z", mode="sdf")
    im = Image.open(p)
    assert im.size == (sx, sy) and im.mode == "RGB"


def test_save_slice_png_mask_mode(tmp_path, sphere):
    Image = pytest.importorskip("PIL.Image")
    p = viz.save_slice_png(sphere, str(tmp_path / "m.png"), mode="mask")
    assert Image.open(p).mode in ("L", "1")


def test_save_slice_sheet(tmp_path, sphere):
    Image = pytest.importorskip("PIL.Image")
    p = viz.save_slice_sheet(sphere, str(tmp_path / "sheet.png"),
                             axis="z", count=9, cols=3, mode="mask")
    assert Image.open(p).size[0] > 0


def test_save_slice_png_bad_axis(tmp_path, sphere):
    pytest.importorskip("PIL.Image")
    with pytest.raises(ValueError):
        viz.save_slice_png(sphere, str(tmp_path / "x.png"), axis="w")


def test_save_slice_empty_voxels(tmp_path):
    pytest.importorskip("PIL.Image")
    with pytest.raises(ValueError):
        viz.save_slice_png(Voxels(), str(tmp_path / "e.png"))


# --- mesh preview (needs matplotlib) ----------------------------------------
def test_mesh_preview(tmp_path, sphere):
    pytest.importorskip("matplotlib")
    Image = pytest.importorskip("PIL.Image")
    p = str(tmp_path / "mesh.png")
    fig = viz.mesh_preview(sphere.to_mesh(), p, size=400)
    assert Image.open(p).size == (400, 400) or Image.open(p).size[0] > 0
    import matplotlib.pyplot as plt
    plt.close(fig)
