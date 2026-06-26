"""Unit tests for the Phase 12h visualization layer (colours + mesh painter).

Colour palette / spectra / value->colour scales, and MeshPainter's per-property
mesh splitting. All display-independent (the interactive ``painter.preview`` is a
thin viewer wrapper, exercised by the viewer tests). C# parity isn't applicable
(colours are subjective and excluded from the golden compile).
"""

import pytest

from picogk.shapes import (
    Box,
    LocalFrame,
    Sphere,
    colors,
    painter,
)
from picogk.shapes.colors import (
    ColorScale3D,
    CustomColorScale,
    LinearColorScale,
    Palette,
    SmoothColorScale,
    rainbow_spectrum,
)


# --- palette / spectrum --------------------------------------------------------
def test_palette_named_colors():
    assert Palette.BLACK == (0.0, 0.0, 0.0)
    assert Palette.RED == (1.0, 0.0, 0.0)
    assert pytest.approx((0.0, 0xB8 / 255, 0.0)) == Palette.GREEN


def test_palette_random_reproducible():
    assert Palette.random(5) == Palette.random(5)
    assert Palette.random(5) != Palette.random(6)
    for c in Palette.random(3):
        assert 0.0 <= c <= 1.0


def test_rainbow_spectrum():
    s = rainbow_spectrum()
    assert len(s) == 5
    assert s[0] == (0.0, 0.0, 1.0)        # blue
    assert s[-1] == (1.0, 0.0, 0.0)       # red


# --- colour scales -------------------------------------------------------------
def test_linear_scale_endpoints_and_midpoint():
    sc = LinearColorScale((0, 0, 0), (1, 1, 1), 0.0, 10.0)
    assert sc.color(0) == pytest.approx((0, 0, 0))
    assert sc.color(10) == pytest.approx((1, 1, 1))
    assert sc.color(5) == pytest.approx((0.5, 0.5, 0.5))


def test_linear_scale_clamps_out_of_range():
    sc = LinearColorScale((0, 0, 0), (1, 1, 1), 0.0, 10.0)
    assert sc.color(-5) == pytest.approx((0, 0, 0))
    assert sc.color(99) == pytest.approx((1, 1, 1))


def test_smooth_scale_endpoints():
    sc = SmoothColorScale((0, 0, 0), (1, 1, 1), 0.0, 1.0)
    assert sc.color(0) == pytest.approx((0, 0, 0), abs=1e-6)
    assert sc.color(1) == pytest.approx((1, 1, 1), abs=1e-6)
    mid = sc.color(0.5)
    assert all(0.0 <= c <= 1.0 for c in mid)


def test_custom_scale_builds_and_bounds():
    sc = CustomColorScale((0, 0, 0), (1, 1, 1), 0.0, 1.0, transition=0.5, smoothness=0.1)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert all(0.0 <= c <= 1.0 for c in sc.color(v))


def test_color_scale_3d_spectrum_endpoints():
    sc = ColorScale3D(rainbow_spectrum(), 0.0, 1.0)
    lo, hi = sc.color(0.0), sc.color(1.0)
    assert lo[2] > 0.5 and lo[0] < 0.5    # ~blue end
    assert hi[0] > 0.5 and hi[2] < 0.5    # ~red end


# --- mesh painter --------------------------------------------------------------
def _box_mesh():
    return Box(LocalFrame((0, 0, 0)), 10, 10, 10).to_mesh()


def test_split_by_overhang_partitions_all_triangles():
    mesh = _box_mesh()
    scale = LinearColorScale(Palette.BLUE, Palette.RED, 0.0, 90.0)
    groups = painter.split_by_overhang_angle(mesh, scale)
    assert len(groups) >= 2                                   # walls (0deg) + caps (90deg)
    total = sum(g[0].triangle_count() for g in groups)
    assert total == mesh.triangle_count()
    for sub, rgb in groups:
        assert sub.triangle_count() > 0
        assert len(rgb) == 3 and all(0.0 <= c <= 1.0 for c in rgb)


def test_split_by_property_partitions_all_triangles():
    mesh = Sphere(radius=8).to_mesh()
    scale = LinearColorScale((0, 0, 1), (1, 0, 0), -8.0, 8.0)
    # colour by triangle-centroid height (z)
    groups = painter.split_by_property(mesh, scale,
                                       lambda a, b, c: (a[:, 2] + b[:, 2] + c[:, 2]) / 3.0)
    assert len(groups) > 1
    assert sum(g[0].triangle_count() for g in groups) == mesh.triangle_count()


def test_painter_preview_is_callable():
    assert callable(painter.preview)         # execution needs a display (viewer tests)


def test_colors_submodule_exports():
    assert hasattr(colors, "Palette") and hasattr(colors, "rainbow_spectrum")
