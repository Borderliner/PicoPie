"""Phase 10: interactive viewer.

The render tests need a window system + OpenGL, which CI runners lack -- they
self-skip when a Viewer can't be created, and are also marked ``viewer`` so the
per-wheel CI excludes them (-m "not viewer"). They run locally where a display
exists. The import test runs everywhere (the module must import head-less).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import picopie
from picopie import Viewer, Voxels, library, render_png, show
from picopie._errors import InvalidHandleError, PicoPieError


def _viewer_or_skip(**kw) -> Viewer:
    try:
        return Viewer(**kw)
    except Exception as e:           # no display / no GL on this machine
        pytest.skip(f"no interactive display: {e}")


def test_viewer_importable():
    # must import + expose the class with no display (headless-safe import)
    assert hasattr(picopie, "Viewer")
    assert Viewer.__module__ == "picopie.viewer"


@pytest.mark.viewer
def test_viewer_renders_sphere(tmp_path):
    v = _viewer_or_skip(size=(640, 480))
    try:
        v.add(Voxels.sphere(radius=10))
        v.set_group_material(0, (0.4, 0.6, 0.9), metallic=0.1, roughness=0.4)
        out = v.screenshot(str(tmp_path / "sphere.png"))
        a = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
        # a lit sphere yields many shades; a blank/failed render is near-uniform
        assert len(np.unique(a.reshape(-1, 3), axis=0)) > 50
    finally:
        v.close()


@pytest.mark.viewer
def test_viewer_add_rejects_wrong_type(tmp_path):
    v = _viewer_or_skip(size=(320, 240))
    try:
        with pytest.raises(TypeError):
            v.add(object())
    finally:
        v.close()


@pytest.mark.viewer
def test_viewer_camera_change_changes_render(tmp_path):
    v = _viewer_or_skip(size=(400, 300))
    try:
        v.add(Voxels.sphere(radius=10))
        a = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                       .open(v.screenshot(str(tmp_path / "a.png"))).convert("RGB"))
        v._azimuth += 1.0          # orbit + tilt (what a mouse drag does)
        v._elevation += 0.3
        v._autofit = False
        b = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                       .open(v.screenshot(str(tmp_path / "b.png"))).convert("RGB"))
        assert a.shape == b.shape
        assert not np.array_equal(a, b)     # the camera move must change the image
    finally:
        v.close()


@pytest.mark.viewer
def test_show_non_blocking_multi(tmp_path):
    v = None
    try:
        v = show(Voxels.sphere(radius=8),
                 Voxels.sphere(center=(12, 0, 0), radius=5), block=False)
    except Exception as e:
        pytest.skip(f"no interactive display: {e}")
    try:
        a = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                       .open(v.screenshot(str(tmp_path / "s.png"))).convert("RGB"))
        assert len(np.unique(a.reshape(-1, 3), axis=0)) > 50
    finally:
        v.close()


@pytest.mark.viewer
def test_render_png_oneshot(tmp_path):
    try:
        out = render_png(Voxels.sphere(radius=10), str(tmp_path / "r.png"), size=(400, 300))
    except Exception as e:
        pytest.skip(f"no interactive display: {e}")
    a = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(out).convert("RGB"))
    assert len(np.unique(a.reshape(-1, 3), axis=0)) > 50


# --- Phase 10e: hardening (need a display -> viewer-marked) -------------------
@pytest.mark.viewer
def test_use_after_close_raises_not_segfaults():
    v = _viewer_or_skip(size=(320, 240))
    v.add(Voxels.sphere(radius=5))
    v.close()
    # every native-touching method must raise, not dereference a freed handle
    for op in (lambda: v.add(Voxels.sphere(radius=3)),
               lambda: v.poll(),
               lambda: v.set_background((0, 0, 0)),
               lambda: v.remove_all(),
               lambda: v.screenshot("x.png")):
        with pytest.raises(InvalidHandleError):
            op()


@pytest.mark.viewer
def test_viewer_registered_and_is_valid():
    v = _viewer_or_skip(size=(320, 240))
    try:
        assert v in library._live_objects      # so shutdown() can invalidate it
        assert v.is_valid()
    finally:
        v.close()
    assert not v.is_valid()                     # False once closed (no crash)


@pytest.mark.viewer
def test_interaction_updates_camera():
    v = _viewer_or_skip(size=(320, 240))
    try:
        az, el, tgt = v._azimuth, v._elevation, v._target.copy()
        v._apply_orbit(50.0, 20.0)
        assert v._azimuth != az and v._elevation != el and v._autofit is False
        z = v._zoom
        v._apply_zoom(1.0)                       # scroll up -> zoom in
        assert v._zoom < z
        v._apply_pan(40.0, 0.0)
        assert not np.array_equal(v._target, tgt)
    finally:
        v.close()


@pytest.mark.viewer
def test_screenshot_no_temp_leak(tmp_path):
    import glob
    v = _viewer_or_skip(size=(320, 240))
    try:
        v.add(Voxels.sphere(radius=5))
        before = set(glob.glob("/tmp/tmp*"))
        for i in range(3):
            v.screenshot(str(tmp_path / f"s{i}.png"))
        assert set(glob.glob("/tmp/tmp*")) == before    # no leaked temp dirs
    finally:
        v.close()


@pytest.mark.viewer
def test_remove_and_group_matrix(tmp_path):
    v = _viewer_or_skip(size=(400, 300))
    try:
        s = Voxels.sphere(radius=8)
        v.add(s)
        a = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                       .open(v.screenshot(str(tmp_path / "a.png"))).convert("RGB"))
        # translate the group far sideways -> the render must change
        mat = np.eye(4, dtype=np.float32)
        mat[3, 0] = 25.0                         # row-major translation (System.Numerics)
        v.set_group_matrix(0, mat)
        b = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                       .open(v.screenshot(str(tmp_path / "b.png"))).convert("RGB"))
        assert not np.array_equal(a, b)
        v.remove(s)                              # removal must not crash
        v.screenshot(str(tmp_path / "c.png"))
    finally:
        v.close()


# --- headless (no display needed): thread guard + camera math ----------------
def test_viewer_requires_main_thread():
    # the main-thread check fires before any GL, so this runs headless (CI too)
    import threading
    errs: list[BaseException] = []

    def worker():
        try:
            Viewer()
        except BaseException as e:
            errs.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errs and isinstance(errs[0], PicoPieError)
    assert "main thread" in str(errs[0])


def test_camera_math_view_is_orthonormal():
    from picopie.viewer import _look_at, _perspective
    eye = np.array([40.0, 30.0, 25.0], np.float32)
    m = _look_at(eye, np.zeros(3, np.float32), np.array([0, 0, 1], np.float32))
    rot = m[:3, :3]                          # the 3 basis columns must be orthonormal
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-4)
    p = _perspective(np.radians(35), 1.5, 1.0, 1000.0)
    assert p[2, 3] == -1.0 and p[0, 0] > 0 and p[1, 1] > 0


def test_perspective_degenerate_near_far_no_nan():
    from picopie.viewer import _perspective, _to_mat4
    p = _perspective(np.radians(35), 1.0, 5.0, 5.0)   # near == far -> guarded
    assert np.all(np.isfinite(p))
    m = _to_mat4(np.eye(4))                            # identity round-trips
    assert (m.vec1.X, m.vec2.Y, m.vec3.Z, m.vec4.W) == (1.0, 1.0, 1.0, 1.0)


def test_camera_state_math_headless():
    # _apply_orbit/_pan/_zoom + _basis/_view_projection are pure camera-state math
    # (no native calls), so they're testable without a display via object.__new__
    # -> these clamp assertions run in CI (test_interaction_updates_camera covers
    # the same on a real window, but is display-gated).
    v = object.__new__(Viewer)
    v._radius = 10.0
    v._zoom = 1.0
    v._azimuth, v._elevation = math.radians(45), math.radians(25)
    v._orbit_speed = 0.008
    v._target = np.zeros(3, np.float32)
    v._autofit = True

    v._apply_zoom(1e6)                        # zoom clamps to [0.05, 20]
    assert v._zoom == pytest.approx(0.05)
    v._zoom = 1.0
    v._apply_zoom(-1e6)
    assert v._zoom == pytest.approx(20.0)

    v._zoom, v._elevation = 1.0, 0.0
    v._apply_orbit(0.0, 1e9)                  # elevation clamps to (-pi/2, pi/2)
    assert abs(v._elevation) < math.pi / 2 and v._autofit is False
    v._apply_orbit(0.0, -1e9)
    assert abs(v._elevation) < math.pi / 2

    d, right, up = v._basis()                 # orthonormal frame
    assert abs(float(np.dot(d, right))) < 1e-5 and abs(float(np.dot(d, up))) < 1e-5
    vp, eye = v._view_projection(1.5)
    assert np.all(np.isfinite(vp)) and np.all(np.isfinite(eye))

    t0 = v._target.copy()
    v._apply_pan(40.0, 10.0)                  # pan moves the orbit target
    assert not np.array_equal(v._target, t0)
