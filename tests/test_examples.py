"""Integration smoke test for the ShapeKernel example gallery (Phase 12i).

Builds every scene in ``examples/shapekernel/gallery.py`` headlessly (no
rendering) and asserts each object is non-empty with a valid colour. This
exercises the whole ported API end-to-end the way a user would.
"""

import importlib.util
import pathlib

import pytest

from picopie import Mesh, Voxels

pytestmark = pytest.mark.examples

_GALLERY = (pathlib.Path(__file__).parent.parent / "examples" / "shapekernel" / "gallery.py")
_spec = importlib.util.spec_from_file_location("sk_gallery", _GALLERY)
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)


def _nonempty(obj) -> bool:
    if isinstance(obj, Voxels):
        return obj.calculate_properties()[0] > 0
    if isinstance(obj, Mesh):
        return obj.triangle_count() > 0
    return False


@pytest.mark.parametrize("name", list(gallery.SCENES))
def test_scene_builds_nonempty(name):
    scene = gallery.SCENES[name]()
    assert scene, f"{name}: empty scene"
    for obj, rgb in scene:
        assert len(rgb) == 3 and all(0.0 <= c <= 1.0 for c in rgb), name
        assert _nonempty(obj), f"{name}: produced an empty object"
