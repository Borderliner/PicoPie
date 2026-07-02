"""Unit tests for the Phase 12d frame-based shapes (need the native kernel).

Analytic volume checks, bbox sanity, modulation/transform/spine behaviour.
Exact C# parity lives in ``test_parity_shapekernel.py``.
"""

import numpy as np
import pytest

from picopie.shapes import (
    Box,
    Cone,
    Cylinder,
    Frames,
    Lens,
    LocalFrame,
    LogoBox,
    Ring,
    Sphere,
)

F = LocalFrame((0, 0, 0))


def _vol(shape):
    return shape.to_voxels().calculate_properties()[0]


# --- analytic volumes ----------------------------------------------------------
def test_box_volume():
    assert _vol(Box(F, 20, 10, 8)) == pytest.approx(20 * 10 * 8, rel=0.02)


def test_cylinder_volume():
    assert _vol(Cylinder(F, 20, 10)) == pytest.approx(np.pi * 100 * 20, rel=0.02)


def test_cone_volume():
    assert _vol(Cone(F, 20, 10, 0)) == pytest.approx(np.pi * 100 * 20 / 3, rel=0.02)


def test_ring_volume():
    assert _vol(Ring(F, 30, 5)) == pytest.approx(2 * np.pi**2 * 30 * 25, rel=0.02)


def test_lens_flat_disc_volume():
    assert _vol(Lens(F, 4, 0, 10)) == pytest.approx(np.pi * 100 * 4, rel=0.02)


# --- bbox sanity ---------------------------------------------------------------
def test_box_bbox():
    _, bbox = Box(F, 20, 10, 8).to_voxels().calculate_properties()
    assert np.allclose(bbox.min, [-5, -4, 0], atol=0.1)
    assert np.allclose(bbox.max, [5, 4, 20], atol=0.1)


def test_cylinder_spans_length_in_z():
    _, bbox = Cylinder(F, 20, 10).to_voxels().calculate_properties()
    assert bbox.max[2] == pytest.approx(20, abs=0.1)
    assert bbox.min[2] == pytest.approx(0, abs=0.1)


# --- modulation / construction variants ----------------------------------------
def test_cylinder_modulated_radius_bumps_length_steps():
    cyl = Cylinder(F, 20, radius=lambda phi, lr: 10 + 2 * np.sin(2 * np.pi * lr))
    assert cyl.length_steps == 500
    assert _vol(cyl) > 0


def test_box_modulated_width():
    box = Box(F, 20, width=lambda lr: 10 + 4 * lr, depth=8)
    assert box.width_steps == 500 and box.length_steps == 500
    assert _vol(box) > 0


def test_box_from_bbox_roundtrips_volume():
    src = Cylinder(F, 20, 10).to_voxels()
    _, bbox = src.calculate_properties()
    box = Box.from_bbox(bbox)
    # the box encloses the cylinder, so its volume is larger but comparable
    assert _vol(box) > _vol(Cylinder(F, 20, 10)) * 0.9


def test_shape_on_spine():
    spine = Frames.aligned([[0, 0, 0], [0, 0, 10], [0, 0, 20]], "z", spacing=1.0)
    cyl = Cylinder(frames=spine, radius=5)
    assert cyl.length_steps == 500
    assert _vol(cyl) == pytest.approx(np.pi * 25 * 20, rel=0.05)


# --- transform -----------------------------------------------------------------
def test_transform_translates_shape():
    moved = Box(F, 10, 10, 10, transform=lambda p: p + np.array([100.0, 0, 0]))
    _, bbox = moved.to_voxels().calculate_properties()
    assert bbox.center[0] == pytest.approx(100, abs=0.5)


# --- LogoBox -------------------------------------------------------------------
def test_logobox_flat_image_matches_plain_box():
    img = np.zeros((16, 16))
    lb = LogoBox(F, length=10, ref_width=20, image=img, mapping=lambda g: 0.0 * g)
    assert _vol(lb) == pytest.approx(10 * 20 * 20, rel=0.02)


def test_logobox_emboss_adds_height():
    flat = np.zeros((16, 16))
    raised = np.ones((16, 16))
    base = LogoBox(F, 10, 20, flat, lambda g: 5.0 * g)
    embossed = LogoBox(F, 10, 20, raised, lambda g: 5.0 * g)
    # a uniformly raised top adds volume
    assert _vol(embossed) > _vol(base) + 100


def test_sphere_still_works():
    assert _vol(Sphere(radius=10)) == pytest.approx(4 / 3 * np.pi * 1000, rel=0.02)
