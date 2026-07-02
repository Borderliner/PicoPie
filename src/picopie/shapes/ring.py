"""Parametric ring / torus (port of ShapeKernel ``BaseRing``).

A tube of (possibly modulated) radius swept around a circle of ``ring_radius``
in the frame's local XY plane. The sweep closes with a wrap-around seam.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ._base import BaseShape, SurfaceMeshBuilder, VertexTransform
from .frames import LocalFrame
from .modulations import SurfaceModulation


class Ring(BaseShape):
    """A torus: a tube of ``radius`` swept around a circle of ``ring_radius``."""

    def __init__(self, frame: LocalFrame | None = None, ring_radius: float = 50.0,
                 radius=5.0, *, radial_steps: int = 360, polar_steps: int = 360,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        self.frame = frame if frame is not None else LocalFrame()
        self.ring_radius = float(ring_radius)
        self.radius = radius
        self.radial_steps = max(5, radial_steps)
        self.polar_steps = max(5, polar_steps)

    @property
    def radius(self) -> SurfaceModulation:
        return self._radius

    @radius.setter
    def radius(self, value) -> None:
        self._radius = SurfaceModulation(value)

    def to_mesh(self) -> Mesh:
        radial, polar = self.radial_steps, self.polar_steps
        # alpha grid wraps: first cell joins alpha=1 (=2pi) back to alpha=0 (closed seam).
        a = np.concatenate([[1.0], np.arange(radial, dtype=np.float64) / (radial - 1)])
        p = np.arange(polar, dtype=np.float64) / (polar - 1)
        alpha = 2.0 * np.pi * a
        phi = 2.0 * np.pi * p

        f = self.frame
        pos, lx, ly, lz = f.position, f.local_x, f.local_y, f.local_z
        spine = (pos[None, :] + self.ring_radius * np.cos(alpha)[:, None] * lx[None, :]
                 + self.ring_radius * np.sin(alpha)[:, None] * ly[None, :])      # (nA, 3)
        radial_dir = spine - pos[None, :]
        norms = np.linalg.norm(radial_dir, axis=1, keepdims=True)
        local_x = np.divide(radial_dir, norms, out=np.zeros_like(radial_dir), where=norms > 0)
        local_y = lz                                                            # (3,)

        phi_row = phi[None, :]
        mod = np.broadcast_to(
            np.asarray(self._radius(phi_row, alpha[:, None]), dtype=np.float64),
            (len(a), len(p)))                                                   # (nA, nphi)
        fx = mod * np.cos(phi_row)
        fy = mod * np.sin(phi_row)
        grid = (spine[:, None, :] + fx[..., None] * local_x[:, None, :]
                + fy[..., None] * local_y[None, None, :])                       # (nA, nphi, 3)
        grid = self._xform_grid(grid)
        return SurfaceMeshBuilder().add(grid, flip=True).build()
