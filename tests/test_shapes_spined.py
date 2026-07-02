"""Unit tests for the Phase 12e spined/revolve shapes (need the native kernel).

Analytic volume checks and construction variants. Exact C# parity lives in
``test_parity_shapekernel.py``. Revolve uses reduced step counts here to stay
fast (full-resolution parity is covered by the golden test).
"""

import numpy as np
import pytest

from picopie.shapes import (
    Frames,
    LineModulation,
    LocalFrame,
    Pipe,
    PipeSegment,
    Revolve,
)
from picopie.shapes import spline_ops as SO

F = LocalFrame((0, 0, 0))


def _vol(shape):
    return shape.to_voxels().calculate_properties()[0]


# --- pipe ----------------------------------------------------------------------
def test_pipe_annulus_volume():
    assert _vol(Pipe(F, 20, 5, 10)) == pytest.approx(np.pi * (100 - 25) * 20, rel=0.02)


def test_pipe_is_hollow():
    _, bbox = Pipe(F, 20, 5, 10).to_voxels().calculate_properties()
    assert bbox.max[0] == pytest.approx(10, abs=0.1)   # outer radius
    assert bbox.max[2] == pytest.approx(20, abs=0.1)   # length


def test_pipe_modulated_radius_bumps_length_steps():
    pipe = Pipe(F, 20, inner_radius=5, outer_radius=lambda phi, lr: 10 + 2 * lr)
    assert pipe.length_steps == 500
    assert _vol(pipe) > 0


def test_pipe_on_spine():
    spine = Frames.aligned([[0, 0, 0], [0, 0, 10], [0, 0, 20]], "z", spacing=1.0)
    pipe = Pipe(frames=spine, inner_radius=4, outer_radius=8)
    assert pipe.length_steps == 500
    assert _vol(pipe) == pytest.approx(np.pi * (64 - 16) * 20, rel=0.05)


# --- pipe segment --------------------------------------------------------------
def test_segment_quarter_volume():
    seg = PipeSegment(F, 20, 5, 10, start=0, end=np.pi / 2)
    assert _vol(seg) == pytest.approx(0.25 * np.pi * (100 - 25) * 20, rel=0.03)


def test_segment_half_volume():
    seg = PipeSegment(F, 20, 5, 10, start=0, end=np.pi)
    assert _vol(seg) == pytest.approx(0.5 * np.pi * (100 - 25) * 20, rel=0.03)


def test_segment_mid_range_equals_start_end():
    a = PipeSegment(F, 20, 5, 10, start=0, end=np.pi / 2, method="start_end")
    b = PipeSegment(F, 20, 5, 10, start=np.pi / 4, end=np.pi / 2, method="mid_range")
    assert _vol(a) == pytest.approx(_vol(b), rel=0.02)


# --- revolve -------------------------------------------------------------------
def test_revolve_cylinder_volume():
    spine = Frames.extrude(20, F)
    rev = Revolve(F, spine, 0, 5, radial_steps=40, length_steps=60)
    assert _vol(rev) == pytest.approx(np.pi * 25 * 20, rel=0.03)


def test_revolve_offset_spine_makes_tube():
    # a profile offset from the axis revolves into a hollow cylinder
    spine = Frames.extrude(20, LocalFrame((10, 0, 0)))
    rev = Revolve(F, spine, inner_radius=2, outer_radius=2,
                  radial_steps=30, length_steps=60)
    assert _vol(rev) == pytest.approx(np.pi * (144 - 64) * 20, rel=0.05)


def test_revolve_frames_from_contour():
    from picopie.shapes import GenericContour
    contour = GenericContour(20.0, LineModulation(5.0))
    frames = Revolve.frames_from_contour(contour)
    assert isinstance(frames, Frames)
    assert SO.total_length(frames.points) == pytest.approx(20.0, rel=0.05)


def test_revolve_modulated_radius_vase():
    spine = Frames.extrude(20, F)
    rev = Revolve(F, spine, inner_radius=0,
                  outer_radius=lambda lr: 5 + 3 * np.sin(np.pi * lr),
                  radial_steps=40, length_steps=80)
    assert _vol(rev) > _vol(Revolve(F, spine, 0, 5, radial_steps=40, length_steps=80))


# --- transform -----------------------------------------------------------------
def test_pipe_transform_translates():
    moved = Pipe(F, 10, 4, 8, transform=lambda p: p + np.array([50.0, 0, 0]))
    _, bbox = moved.to_voxels().calculate_properties()
    assert bbox.center[0] == pytest.approx(50, abs=0.5)
