"""Lattice-based shapes (port of ShapeKernel ``LatticePipe`` / ``LatticeManifold``).

These build a :class:`~picopie.Lattice` of beams along a :class:`Frames` spine
(rather than a surface mesh) and voxelise it. ``LatticeManifold`` adds a
tear-drop "tip" per station for self-supporting (printable) overhangs.
"""

from __future__ import annotations

import math

import numpy as np

from ..lattice import Lattice
from ..voxels import Voxels
from ._base import BaseShape, VertexTransform
from .frames import Frames, LocalFrame
from .modulations import LineModulation


class LatticePipe(BaseShape):
    """A round pipe built from lattice beams along a length-spine."""

    def __init__(self, frame: LocalFrame | None = None, length: float = 20.0,
                 radius=10.0, *, frames: Frames | None = None,
                 length_steps: int | None = None,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        spine = frames is not None
        if frames is not None:
            self.frames = frames
        else:
            self.frames = Frames.extrude(length, frame if frame is not None else LocalFrame())
        self.radius = radius
        self.length_steps = (500 if spine else 100) if length_steps is None else length_steps

    @property
    def radius(self) -> LineModulation:
        return self._radius

    @radius.setter
    def radius(self, value) -> None:
        self._radius = LineModulation(value)

    def spine_point(self, length_ratio: float) -> np.ndarray:
        pt = self.frames.spine_at(length_ratio)
        return pt if self._transform is None else self._apply_transform(pt.reshape(1, 3))[0]

    def to_lattice(self) -> Lattice:
        lat = Lattice()
        n = self.length_steps
        for i in range(1, n):
            lr0, lr1 = (i - 1) / n, i / n        # note: /n (not n-1), matching C#
            lat.add_beam(self.spine_point(lr0), self.spine_point(lr1),
                         float(self._radius(lr0)), float(self._radius(lr1)))
        return lat

    def to_voxels(self) -> Voxels:
        return Voxels.from_lattice(self.to_lattice())

    def to_mesh(self):
        return self.to_voxels().to_mesh()


class LatticeManifold(LatticePipe):
    """A lattice pipe with a tear-drop tip per station (printable overhang)."""

    def __init__(self, frame: LocalFrame | None = None, length: float = 20.0,
                 radius=10.0, max_overhang_angle: float = 45.0,
                 extend_both_sides: bool = False, min_printable_radius: float = 0.1, *,
                 frames: Frames | None = None, length_steps: int | None = None,
                 transform: VertexTransform | None = None):
        super().__init__(frame, length, radius, frames=frames,
                         length_steps=length_steps, transform=transform)
        self.max_printable_radius = min_printable_radius   # name follows C#'s field
        self.limit_angle = max_overhang_angle
        self.extend_both_sides = extend_both_sides

    def to_lattice(self) -> Lattice:
        lat = Lattice()
        n = self.length_steps
        for i in range(n):
            lr = i / n
            pt = self.spine_point(lr)
            beam = float(self._radius(lr))
            lat.add_beam(pt, pt, beam, beam)             # round pipe station
            self._add_tip(lat, pt, beam, True)
            if self.extend_both_sides:
                self._add_tip(lat, pt, beam, False)
        return lat

    def _add_tip(self, lat: Lattice, pt: np.ndarray, beam: float, z_positive: bool) -> None:
        half_alpha = math.radians(90.0 - self.limit_angle)
        r = beam
        h = r * (1 - math.cos(half_alpha))
        s = 2 * r * math.sin(half_alpha)
        tip_length = math.tan(half_alpha) * (0.5 * s - self.max_printable_radius)
        z = np.array([0.0, 0.0, 1.0])
        sign = 1.0 if z_positive else -1.0
        mid = pt + sign * (r - h) * z
        tip = mid + sign * tip_length * z
        lat.add_beam(mid, tip, 0.5 * s, self.max_printable_radius, round_cap=False)
