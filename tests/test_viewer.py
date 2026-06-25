"""Phase 10: interactive viewer.

The render tests need a window system + OpenGL, which CI runners lack -- they
self-skip when a Viewer can't be created, and are also marked ``viewer`` so the
per-wheel CI excludes them (-m "not viewer"). They run locally where a display
exists. The import test runs everywhere (the module must import head-less).
"""
from __future__ import annotations

import numpy as np
import pytest

import picogk
from picogk import Viewer, Voxels


def _viewer_or_skip(**kw) -> Viewer:
    try:
        return Viewer(**kw)
    except Exception as e:           # no display / no GL on this machine
        pytest.skip(f"no interactive display: {e}")


def test_viewer_importable():
    # must import + expose the class with no display (headless-safe import)
    assert hasattr(picogk, "Viewer")
    assert Viewer.__module__ == "picogk.viewer"


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
