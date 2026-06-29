"""Offline tests for the browser viewer (picogk.web).

The actual three.js rendering needs a browser, so these only exercise the
Python side: that objects serialize into the ``geometry`` (heavy buffers) and
``style`` (light per-id color/material/visibility/matrix) traits correctly, and
that style tweaks don't re-emit geometry. Marked ``web`` (needs the [web] extra).
"""

import numpy as np
import pytest

from picogk import PolyLine, Voxels

# Offline serialization only (no browser), so these run in CI now that anywidget
# is in the wheel test-requires; importorskip keeps a minimal (no-[web]) env green.
web = pytest.importorskip("picogk.web")
WebViewer, show = web.WebViewer, web.show


def _verts(entry) -> np.ndarray:
    return np.frombuffer(entry["positions"], dtype="<f4").reshape(-1, 3)


def _tris(entry) -> np.ndarray:
    return np.frombuffer(entry["indices"], dtype="<u4").reshape(-1, 3)


def test_esm_asset_present():
    assert web._JS.exists() and web._JS.stat().st_size > 0
    assert web._JS.suffix == ".js"


def test_add_mesh_serializes_buffers():
    mesh = Voxels.sphere(radius=8).to_mesh()
    v = WebViewer().add(mesh)
    assert len(v.geometry) == 1 and len(v.style) == 1
    g, s = v.geometry[0], v.style[0]
    assert g["kind"] == "mesh"
    assert np.allclose(_verts(g), mesh.vertices)
    assert np.array_equal(_tris(g), mesh.triangles)
    assert "color" not in g                       # color is in style, not geometry
    assert s["id"] == g["id"]
    assert s["color"] == list(web._PALETTE[0])     # group 0 -> palette[0]


def test_add_voxels_converts_to_mesh():
    g = WebViewer().add(Voxels.sphere(radius=6)).geometry[0]
    assert g["kind"] == "mesh" and len(g["positions"]) > 0 and len(g["indices"]) > 0


def test_add_polyline_uses_its_color():
    pl = PolyLine.from_points([(0, 0, 0), (10, 0, 0), (10, 10, 0)], color=(1.0, 0.0, 0.0))
    v = WebViewer().add(pl)
    g, s = v.geometry[0], v.style[0]
    assert g["kind"] == "line" and "indices" not in g
    assert np.allclose(_verts(g), pl.vertices)
    assert s["color"] == pytest.approx([1.0, 0.0, 0.0])


def test_style_change_does_not_re_emit_geometry():
    # the whole point of the geometry/style split: recoloring must NOT re-send buffers
    v = WebViewer().add(Voxels.sphere(radius=5), group=0)
    geom_before = v.geometry
    v.set_group_material(0, (0.1, 0.2, 0.3), metallic=0.7, roughness=0.2)
    assert v.geometry is geom_before               # geometry trait untouched
    s = v.style[0]
    assert s["color"] == pytest.approx([0.1, 0.2, 0.3])
    assert s["metallic"] == pytest.approx(0.7) and s["roughness"] == pytest.approx(0.2)


def test_per_object_color_override_beats_group():
    v = WebViewer()
    v.set_group_material(0, (0.1, 0.1, 0.1))
    v.add(Voxels.sphere(radius=4), group=0, color=(0.9, 0.8, 0.7))
    assert v.style[0]["color"] == pytest.approx([0.9, 0.8, 0.7])


def test_group_visibility_flag():
    v = WebViewer().add(Voxels.sphere(radius=4), group=1)
    assert v.style[0]["visible"] is True
    v.set_group_visible(1, False)
    assert v.style[0]["visible"] is False


def test_set_background():
    v = WebViewer().set_background((0.2, 0.3, 0.4))
    assert v.background == pytest.approx([0.2, 0.3, 0.4])


def test_show_assigns_palette_per_group():
    a = Voxels.sphere(center=(-10, 0, 0), radius=4)
    b = Voxels.sphere(center=(10, 0, 0), radius=4)
    v = show(a, b)
    assert len(v.geometry) == 2 and len(v.style) == 2
    assert v.style[0]["color"] == list(web._PALETTE[0])
    assert v.style[1]["color"] == list(web._PALETTE[1])


def test_add_rejects_unknown_type():
    with pytest.raises(TypeError):
        WebViewer().add(object())


def test_remove_drops_only_that_object():
    a = Voxels.sphere(center=(-8, 0, 0), radius=4)
    b = Voxels.sphere(center=(8, 0, 0), radius=4)
    v = WebViewer().add(a, group=0).add(b, group=1)
    assert len(v.geometry) == 2
    v.remove(a)
    assert len(v.geometry) == 1 and v.style[0]["color"] == list(web._PALETTE[1])


def test_entries_have_unique_ids_and_identity_matrix():
    v = WebViewer().add(Voxels.sphere(radius=4)).add(Voxels.sphere(radius=3))
    assert [g["id"] for g in v.geometry] == [0, 1]
    assert v.style[0]["matrix"] == web._IDENTITY16


def test_identity_matrix_is_a_copy_not_shared():
    v = WebViewer().add(Voxels.sphere(radius=4)).add(Voxels.sphere(radius=3))
    assert v.style[0]["matrix"] == v.style[1]["matrix"]
    assert v.style[0]["matrix"] is not v.style[1]["matrix"]      # no shared-mutable global


def test_mat16_transposes_desktop_to_threejs():
    # desktop System.Numerics puts a +x translation in the last ROW (A[3,0]); three.js
    # (column vectors) wants it in the last COLUMN -> flat index 3 after the transpose.
    A = np.eye(4)
    A[3, 0] = 5.0
    assert web._mat16(A)[3] == 5.0


def test_group_matrix_applies_to_group():
    T = np.eye(4)
    T[3, 0] = 5.0
    v = WebViewer().add(Voxels.sphere(radius=4), group=0)
    v.set_group_matrix(0, T)
    assert v.style[0]["matrix"] == web._mat16(T)


def test_object_matrix_overrides_group_matrix():
    obj = Voxels.sphere(radius=4)
    v = WebViewer().add(obj, group=0)
    v.set_group_matrix(0, np.eye(4) * 2)           # group scale
    T = np.eye(4)
    T[3, 1] = 9.0
    v.set_object_matrix(obj, T)
    assert v.style[0]["matrix"] == web._mat16(T)


def test_mat16_rejects_bad_shape():
    with pytest.raises(ValueError, match="4x4"):
        web._mat16([1, 2, 3])


def test_reset_view_and_wireframe_bump_traits():
    v = WebViewer().add(Voxels.sphere(radius=4))
    f, w = v._fit, v._wireframe
    v.reset_view()
    v.toggle_wireframe()
    assert v._fit == f + 1 and v._wireframe == w + 1


def test_screenshot_requests_and_writes_on_response(tmp_path):
    out = tmp_path / "shot.png"
    v = WebViewer().add(Voxels.sphere(radius=4))
    g = v._grab
    v.screenshot(str(out))
    assert v._grab == g + 1 and v._pending_screenshot == str(out)
    v._handle_msg(v, {"type": "screenshot"}, [b"\x89PNG-bytes"])
    assert out.read_bytes() == b"\x89PNG-bytes"
    assert v._pending_screenshot is None


def test_export_html_is_self_contained(tmp_path):
    out = tmp_path / "v.html"
    p = web.export_html(Voxels.sphere(radius=6), str(out), title="T")
    assert p == str(out)
    txt = out.read_text(encoding="utf-8")
    assert "<title>T</title>" in txt
    assert "createViewer" in txt and "esm.sh/three" in txt
    assert '"kind": "mesh"' in txt or '"kind":"mesh"' in txt
    assert '"style"' in txt                        # style embedded (carries the matrix)
    assert "__JS__" not in txt and "__DATA__" not in txt
    assert out.stat().st_size > 10_000
