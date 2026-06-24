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


def test_slice_z_orientation():
    # column index follows +x; row 0 is +y (top). Verify by removing material
    # from the +x side and the +y side and checking where it disappears.
    px = Voxels.sphere(radius=10) - Voxels.sphere(center=(12, 0, 0), radius=8)
    _, s = px.voxel_dimensions()
    sx, sy, sz = (int(v) for v in s)
    m = px.slice_z(sz // 2) <= 0
    assert m[:, :sx // 2].sum() > m[:, sx // 2:].sum()   # left (-x) more solid

    py = Voxels.sphere(radius=10) - Voxels.sphere(center=(0, 12, 0), radius=8)
    _, s2 = py.voxel_dimensions()
    sy2 = int(s2[1])
    m2 = py.slice_z(int(s2[2]) // 2) <= 0
    assert m2[sy2 // 2:, :].sum() > m2[:sy2 // 2, :].sum()  # bottom (-y) more solid


def test_slice_x_has_content():
    part = Voxels.sphere(radius=10) - Voxels.sphere(center=(12, 0, 0), radius=8)
    _, s = part.voxel_dimensions()
    sl = part.slice_x(int(s[0]) // 2)
    assert sl.shape == (int(s[2]), int(s[1]))
    assert (sl <= 0).any()


def test_interpolated_equals_integer_slice(sphere):
    _, s = sphere.voxel_dimensions()
    k = int(s[2]) // 2
    assert np.allclose(sphere.slice_z_interpolated(float(k)), sphere.slice_z(k))


def test_interpolated_index_validation(sphere):
    _, s = sphere.voxel_dimensions()
    with pytest.raises(IndexError):
        sphere.slice_z_interpolated(float(int(s[2])))
    with pytest.raises(IndexError):
        sphere.slice_z_interpolated(-0.5)


def test_interpolated_slice_shape(sphere):
    _, s = sphere.voxel_dimensions()
    sx, sy, sz = (int(v) for v in s)
    assert sphere.slice_z_interpolated(sz / 2.0).shape == (sy, sx)


def test_shelled_midslice_is_annular():
    # a hollow shell's mid slice has an inside wall but an outside (hollow) center
    v = Voxels.sphere(radius=12)
    v.shell_(1.5)
    _, s = v.voxel_dimensions()
    sl = v.slice_z(int(s[2]) // 2)
    cy, cx = sl.shape[0] // 2, sl.shape[1] // 2
    assert sl[cy, cx] > 0          # hollow center is outside the wall
    assert (sl <= 0).any()         # the wall itself is present


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


def test_colorize_sdf_colors():
    # inside (<0) -> blue-ish (B>R); outside (>0) -> red-ish (R>B); 0 -> white
    arr = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    rgb = viz.colorize(arr, "sdf")
    assert rgb[0, 0, 2] > rgb[0, 0, 0]          # inside blue
    assert rgb[0, 2, 0] > rgb[0, 2, 2]          # outside red
    assert tuple(rgb[0, 1]) == (255, 255, 255)  # surface white


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


def test_save_slice_sheet_count_exceeds_slices(tmp_path):
    # requesting more tiles than there are slices must clamp/dedup, not crash
    Image = pytest.importorskip("PIL.Image")
    v = Voxels.sphere(radius=3)        # few slices
    _, s = v.voxel_dimensions()
    p = viz.save_slice_sheet(v, str(tmp_path / "sheet.png"),
                             count=int(s[2]) + 50, cols=4, mode="mask")
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
    viz.mesh_preview(sphere.to_mesh(), p, size=400)
    assert Image.open(p).size[0] > 0


def test_mesh_preview_empty_raises():
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError):
        viz.mesh_preview(picogk.Mesh())   # no triangles


def test_mesh_preview_no_figure_leak(tmp_path, sphere):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    plt.close("all")
    mesh = sphere.to_mesh()
    for i in range(5):
        viz.mesh_preview(mesh, str(tmp_path / f"m{i}.png"), size=200)
    assert len(plt.get_fignums()) == 0   # closed when path given


def test_mesh_preview_keeps_figure_open_for_notebook(sphere):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    fig = viz.mesh_preview(sphere.to_mesh(), path=None, size=200)
    assert fig.number in plt.get_fignums()
    plt.close(fig)
