"""Parametric sphere — a Pythonic port of ShapeKernel ``BaseSphere``.

The surface is sampled on a ``(theta, phi)`` grid in the shape's local frame,
exactly matching ShapeKernel's tessellation (so the rendered voxels match the
C# reference to float precision), then rendered to voxels via the mesh.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ._base import BaseShape, VertexTransform, quad_grid_to_mesh
from .frames import LocalFrame
from .modulations import SurfaceModulation


class Sphere(BaseShape):
    """A sphere defined by a :class:`LocalFrame` and a (possibly modulated) radius.

    Args:
        frame: the local frame the sphere is built in (default: origin frame).
        radius: a number, a ``callable(phi, theta)``, or a
            :class:`SurfaceModulation`.
        azimuthal_steps: theta-direction sampling resolution (C# default 360).
        polar_steps: phi-direction sampling resolution (C# default 180).
        transform: optional vectorised vertex transform.
    """

    def __init__(self, frame: LocalFrame | None = None, radius=10.0, *,
                 azimuthal_steps: int = 360, polar_steps: int = 180,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        self.frame = frame if frame is not None else LocalFrame()
        self.radius = radius
        self.azimuthal_steps = int(azimuthal_steps)
        self.polar_steps = int(polar_steps)

    @property
    def radius(self) -> SurfaceModulation:
        return self._radius

    @radius.setter
    def radius(self, value) -> None:
        self._radius = SurfaceModulation(value)

    def surface_point(self, phi_ratio: float, theta_ratio: float,
                      radius_ratio: float = 1.0) -> np.ndarray:
        """A single point on/inside the sphere (``radius_ratio=1`` -> surface).

        All ratios run ``0..1``. Mirrors C# ``vecGetSurfacePoint``.
        """
        theta = np.pi * theta_ratio
        phi = 2.0 * np.pi * phi_ratio
        r = radius_ratio * self._radius(phi, theta)
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        f = self.frame
        pt = f.position + x * f.local_x + y * f.local_y + z * f.local_z
        return self._apply_transform(np.asarray(pt, dtype=np.float64).reshape(1, 3))[0]

    def to_mesh(self) -> Mesh:
        a = self.azimuthal_steps
        p = self.polar_steps
        if a < 2 or p < 2:
            raise ValueError("azimuthal_steps and polar_steps must each be >= 2")

        # theta corners m=0..a-1 (ratio m/(a-1)); phi corners k=0..p (ratio (k-1)/(p-1)).
        theta = (np.pi * (np.arange(a, dtype=np.float64) / (a - 1)))[:, None]
        phi = (2.0 * np.pi * ((np.arange(p + 1, dtype=np.float64) - 1.0) / (p - 1)))[None, :]

        r = self._radius(phi, theta)                       # scalar or (a, p+1)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        x = r * cos_p * sin_t
        y = r * sin_p * sin_t
        z = r * cos_t * np.ones_like(phi)
        x, y, z = np.broadcast_arrays(x, y, z)             # all (a, p+1)

        f = self.frame
        grid = (f.position[None, None, :]
                + x[..., None] * f.local_x
                + y[..., None] * f.local_y
                + z[..., None] * f.local_z)                # (a, p+1, 3)

        if self._transform is not None:
            grid = self._apply_transform(grid.reshape(-1, 3)).reshape(a, p + 1, 3)

        return quad_grid_to_mesh(grid)
