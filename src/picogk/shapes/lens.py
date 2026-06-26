"""Parametric lens (port of ShapeKernel ``BaseLens``).

A disc/annulus between an inner and outer radius, with separately modulated
lower and upper surfaces (the "lens" profile) in the frame's local Z.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ._base import BaseShape, SurfaceMeshBuilder, VertexTransform
from .frames import LocalFrame
from .modulations import SurfaceModulation


class Lens(BaseShape):
    """A lens between ``inner_radius`` and ``outer_radius`` with a modulated height.

    By default the lower surface is flat (z=0) and the upper is flat at
    ``height``; pass ``lower``/``upper`` modulations ``(phi, radius_ratio)`` for
    a curved lens.
    """

    def __init__(self, frame: LocalFrame | None = None, height: float = 5.0,
                 inner_radius: float = 0.0, outer_radius: float = 10.0, *,
                 lower=None, upper=None, radial_steps: int | None = None,
                 polar_steps: int = 360, height_steps: int = 5,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        self.frame = frame if frame is not None else LocalFrame()
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self._lower = SurfaceModulation(0.0 if lower is None else lower)
        self._upper = SurfaceModulation(float(height) if upper is None else upper)
        modulated = lower is not None or upper is not None
        rs = 500 if modulated else 5
        self.radial_steps = max(5, rs if radial_steps is None else radial_steps)
        self.polar_steps = max(5, polar_steps)
        self.height_steps = max(5, height_steps)

    def _surface(self, h, phi_r, rad_r) -> np.ndarray:
        h, phi_r, rad_r = np.broadcast_arrays(np.asarray(h, dtype=np.float64),
                                              np.asarray(phi_r, dtype=np.float64),
                                              np.asarray(rad_r, dtype=np.float64))
        phi = 2.0 * np.pi * phi_r
        radius = (self.outer_radius - self.inner_radius) * rad_r + self.inner_radius
        lower = np.asarray(self._lower(phi, rad_r), dtype=np.float64)
        upper = np.asarray(self._upper(phi, rad_r), dtype=np.float64)
        z = lower + h * (upper - lower)
        f = self.frame
        grid = (f.position[None, None, :]
                + (radius * np.cos(phi))[..., None] * f.local_x
                + (radius * np.sin(phi))[..., None] * f.local_y
                + np.broadcast_to(z, phi.shape)[..., None] * f.local_z)
        return self._xform_grid(grid)

    def to_mesh(self) -> Mesh:
        p = np.arange(self.polar_steps, dtype=np.float64) / (self.polar_steps - 1)
        rr = np.arange(self.radial_steps, dtype=np.float64) / (self.radial_steps - 1)
        hr = np.arange(self.height_steps, dtype=np.float64) / (self.height_steps - 1)
        b = SurfaceMeshBuilder()
        b.add(self._surface(1.0, p[:, None], rr[None, :]), flip=False)   # top
        b.add(self._surface(0.0, p[:, None], rr[None, :]), flip=True)    # bottom
        b.add(self._surface(hr[None, :], p[:, None], 0.0), flip=False)   # inner mantle
        b.add(self._surface(hr[None, :], p[:, None], 1.0), flip=True)    # outer mantle
        return b.build()
