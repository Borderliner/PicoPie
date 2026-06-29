"""Offline tests for the browser viewer (picogk.web).

The actual three.js rendering needs a browser, so these only exercise the
Python side: that objects serialize into the ``geometry`` trait as the right
binary buffers + colors. Marked ``web`` (needs the [web] extra; excluded from
per-wheel CI).
"""

import numpy as np
import pytest

from picogk import PolyLine, Voxels

pytestmark = pytest.mark.web

web = pytest.importorskip("picogk.web")
WebViewer, show = web.WebViewer, web.show


def _verts(entry) -> np.ndarray:
    return np.frombuffer(entry["positions"], dtype="<f4").reshape(-1, 3)


def _tris(entry) -> np.ndarray:
    return np.frombuffer(entry["indices"], dtype="<u4").reshape(-1, 3)


def test_esm_asset_present():
    # the JS front-end must be shipped as package data next to the runtime
    assert web._JS.exists() and web._JS.stat().st_size > 0
    assert web._JS.suffix == ".js"


def test_add_mesh_serializes_buffers():
    mesh = Voxels.sphere(radius=8).to_mesh()
    v = WebViewer().add(mesh)
    assert len(v.geometry) == 1
    e = v.geometry[0]
    assert e["kind"] == "mesh"
    # buffers round-trip back to the mesh arrays
    assert np.allclose(_verts(e), mesh.vertices)
    assert np.array_equal(_tris(e), mesh.triangles)
    assert e["color"] == list(web._PALETTE[0])     # group 0 -> palette[0]


def test_add_voxels_converts_to_mesh():
    v = WebViewer().add(Voxels.sphere(radius=6))
    e = v.geometry[0]
    assert e["kind"] == "mesh" and len(e["positions"]) > 0 and len(e["indices"]) > 0


def test_add_polyline_uses_its_color():
    pl = PolyLine.from_points([(0, 0, 0), (10, 0, 0), (10, 10, 0)], color=(1.0, 0.0, 0.0))
    e = WebViewer().add(pl).geometry[0]
    assert e["kind"] == "line" and "indices" not in e
    assert np.allclose(_verts(e), pl.vertices)
    assert e["color"] == pytest.approx([1.0, 0.0, 0.0])


def test_group_material_overrides_color_and_pbr():
    v = WebViewer().add(Voxels.sphere(radius=5), group=2)
    v.set_group_material(2, (0.1, 0.2, 0.3), metallic=0.7, roughness=0.2)
    e = v.geometry[0]
    assert e["color"] == pytest.approx([0.1, 0.2, 0.3])
    assert e["metallic"] == pytest.approx(0.7) and e["roughness"] == pytest.approx(0.2)


def test_per_object_color_override_beats_group():
    v = WebViewer()
    v.set_group_material(0, (0.1, 0.1, 0.1))
    v.add(Voxels.sphere(radius=4), group=0, color=(0.9, 0.8, 0.7))
    assert v.geometry[0]["color"] == pytest.approx([0.9, 0.8, 0.7])


def test_group_visibility_flag():
    v = WebViewer().add(Voxels.sphere(radius=4), group=1)
    assert v.geometry[0]["visible"] is True
    v.set_group_visible(1, False)
    assert v.geometry[0]["visible"] is False


def test_set_background():
    v = WebViewer().set_background((0.2, 0.3, 0.4))
    assert v.background == pytest.approx([0.2, 0.3, 0.4])


def test_show_assigns_palette_per_group():
    a = Voxels.sphere(center=(-10, 0, 0), radius=4)
    b = Voxels.sphere(center=(10, 0, 0), radius=4)
    v = show(a, b)
    assert len(v.geometry) == 2
    assert v.geometry[0]["color"] == list(web._PALETTE[0])
    assert v.geometry[1]["color"] == list(web._PALETTE[1])


def test_add_rejects_unknown_type():
    with pytest.raises(TypeError):
        WebViewer().add(object())


def test_remove_drops_only_that_object():
    a = Voxels.sphere(center=(-8, 0, 0), radius=4)
    b = Voxels.sphere(center=(8, 0, 0), radius=4)
    v = WebViewer().add(a, group=0).add(b, group=1)
    assert len(v.geometry) == 2
    v.remove(a)
    assert len(v.geometry) == 1 and v.geometry[0]["color"] == list(web._PALETTE[1])


def test_entries_have_unique_ids_and_identity_matrix():
    v = WebViewer().add(Voxels.sphere(radius=4)).add(Voxels.sphere(radius=3))
    ids = [e["id"] for e in v.geometry]
    assert ids == [0, 1]
    assert v.geometry[0]["matrix"] == web._IDENTITY16


def test_group_matrix_applies_to_group():
    T = np.eye(4)
    T[0, 3] = 5.0                                  # row-major translate +x
    v = WebViewer().add(Voxels.sphere(radius=4), group=0)
    v.set_group_matrix(0, T)
    assert v.geometry[0]["matrix"] == web._mat16(T)


def test_object_matrix_overrides_group_matrix():
    obj = Voxels.sphere(radius=4)
    v = WebViewer().add(obj, group=0)
    v.set_group_matrix(0, np.eye(4) * 2)           # group scale
    T = np.eye(4)
    T[1, 3] = 9.0
    v.set_object_matrix(obj, T)
    assert v.geometry[0]["matrix"] == web._mat16(T)


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
    # simulate the browser sending the rendered PNG back to the kernel
    v._handle_msg(v, {"type": "screenshot"}, [b"\x89PNG-bytes"])
    assert out.read_bytes() == b"\x89PNG-bytes"
    assert v._pending_screenshot is None


def test_export_html_is_self_contained(tmp_path):
    out = tmp_path / "v.html"
    p = web.export_html(Voxels.sphere(radius=6), str(out), title="T")
    assert p == str(out)
    txt = out.read_text(encoding="utf-8")
    assert "<title>T</title>" in txt
    assert "createViewer" in txt                       # the renderer is inlined
    assert "esm.sh/three" in txt                       # three.js import present
    assert '"kind": "mesh"' in txt or '"kind":"mesh"' in txt   # geometry embedded
    assert "__JS__" not in txt and "__DATA__" not in txt        # all tokens filled
    assert out.stat().st_size > 10_000
