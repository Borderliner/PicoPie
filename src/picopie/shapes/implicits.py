"""Implicit (signed-distance) primitives (port of ShapeKernel ``ImplicitUtility``).

Each class is a callable ``sdf(x, y, z) -> float`` (value < 0 inside), usable
directly with :meth:`Voxels.render_implicit_` / :meth:`Voxels.intersect_implicit_`.
Convenience :meth:`render` (within a bbox) and :meth:`intersect` (clip a volume)
wrap those. The distance functions match C# ``fSignedDistance`` (computed in
double precision, as upstream does).
"""

from __future__ import annotations

import math

from ..voxels import Voxels


class _Implicit:
    """Base: subclasses implement ``__call__(x, y, z)``."""

    def __call__(self, x: float, y: float, z: float) -> float:  # pragma: no cover
        raise NotImplementedError

    def render(self, bbox) -> Voxels:
        """Voxelise the region where ``sdf <= 0`` within ``bbox``."""
        return Voxels().render_implicit_(self, bbox)

    def intersect(self, voxels: Voxels) -> Voxels:
        """A copy of ``voxels`` clipped to the region where ``sdf <= 0``."""
        return voxels.copy().intersect_implicit_(self)


class ImplicitGyroid(_Implicit):
    """A gyroid TPMS shell. ``unit_size`` is the repeat length (mm)."""

    def __init__(self, unit_size: float, thickness_ratio: float):
        self.frequency = (2.0 * math.pi) / unit_size
        self.thickness_ratio = float(thickness_ratio)

    @staticmethod
    def thickness_ratio_for(wall_thickness: float, unit_size: float) -> float:
        """Thickness ratio approximating a physical wall thickness (mm)."""
        return wall_thickness * 10.0 / unit_size

    def __call__(self, x: float, y: float, z: float) -> float:
        f = self.frequency
        d = (math.sin(f * x) * math.cos(f * y)
             + math.sin(f * y) * math.cos(f * z)
             + math.sin(f * z) * math.cos(f * x))
        return abs(d) - 0.5 * self.thickness_ratio


class ImplicitSphere(_Implicit):
    """A solid sphere."""

    def __init__(self, center, radius: float):
        self.cx, self.cy, self.cz = (float(c) for c in center)
        self.radius = float(radius)

    def __call__(self, x: float, y: float, z: float) -> float:
        return math.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2
                         + (z - self.cz) ** 2) - self.radius


class ImplicitGenus(_Implicit):
    """A genus-2 implicit surface; ``gap`` controls the central hole."""

    def __init__(self, gap: float):
        self.gap = float(gap)

    def __call__(self, x: float, y: float, z: float) -> float:
        return (2 * y * (y * y - 3 * x * x) * (1 - z * z)
                + (x * x + y * y) ** 2
                - (9 * z * z - 1) * (1 - z * z) - self.gap)


class ImplicitSuperEllipsoid(_Implicit):
    """A super-ellipsoid (``epsilon1``/``epsilon2`` shape the squareness)."""

    def __init__(self, center, ax: float, ay: float, az: float,
                 epsilon1: float, epsilon2: float):
        self.cx, self.cy, self.cz = (float(c) for c in center)
        self.ax, self.ay, self.az = float(ax), float(ay), float(az)
        self.epsilon1 = float(epsilon1)
        self.epsilon2 = float(epsilon2)

    def __call__(self, x: float, y: float, z: float) -> float:
        dx = abs(x + self.cx) / self.ax
        dy = abs(y + self.cy) / self.ay
        dz = abs(z + self.cz) / self.az
        e1, e2 = self.epsilon1, self.epsilon2
        d = ((dx ** (2 / e2) + dy ** (2 / e2)) ** (e2 / e1)
             + dz ** (2 / e1))
        return d - 1.0
