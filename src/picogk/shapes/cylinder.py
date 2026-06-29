"""Parametric cylinder and cone (port of ShapeKernel ``BaseCylinder`` / ``BaseCone``).

A cylinder is a top cap + outer mantle + bottom cap over a :class:`Frames`
spine, with a (possibly modulated) radius. A cone is a cylinder with a radius
that varies linearly along the length.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ._base import BaseShape, SurfaceMeshBuilder, VertexTransform
from .frames import Frames, LocalFrame
from .modulations import SurfaceModulation


class Cylinder(BaseShape):
    """A cylinder over a length-spine with a radius ``modulation(phi, length_ratio)``."""

    def __init__(self, frame: LocalFrame | None = None, length: float = 20.0,
                 radius=10.0, *, frames: Frames | None = None,
                 polar_steps: int = 360, radial_steps: int = 5,
                 length_steps: int | None = None,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        spine = frames is not None
        if frames is not None:
            self.frames = frames
        else:
            self.frames = Frames.extrude(length, frame if frame is not None else LocalFrame())
        self.radius = radius
        self._spine = spine
        self.polar_steps = max(5, polar_steps)
        self.radial_steps = max(5, radial_steps)
        # length steps bump to 500 when the radius is modulated (or spine-based);
        # computed lazily so a radius set after construction still bumps.
        self._length_steps = length_steps

    @property
    def radius(self) -> SurfaceModulation:
        return self._radius

    @radius.setter
    def radius(self, value) -> None:
        self._radius = SurfaceModulation(value)

    @property
    def length_steps(self) -> int:
        if self._length_steps is not None:
            return max(5, self._length_steps)
        return 500 if (self._spine or self._radius.const_value is None) else 5

    @length_steps.setter
    def length_steps(self, value) -> None:
        self._length_steps = value

    def _cap(self, length_ratio: float) -> np.ndarray:
        p = np.arange(self.polar_steps, dtype=np.float64) / (self.polar_steps - 1)
        rr = np.arange(self.radial_steps, dtype=np.float64) / (self.radial_steps - 1)
        phi = 2.0 * np.pi * p
        spine, lx, ly, _ = self.frames.samples([length_ratio])
        spine, lx, ly = spine[0], lx[0], ly[0]
        mod = np.broadcast_to(np.asarray(self._radius(phi, length_ratio), dtype=np.float64),
                              (len(phi),))
        radius = rr[None, :] * mod[:, None]                      # (nphi, nrad)
        fx = radius * np.cos(phi)[:, None]
        fy = radius * np.sin(phi)[:, None]
        grid = spine + fx[..., None] * lx + fy[..., None] * ly
        return self._xform_grid(grid)

    def _mantle(self) -> np.ndarray:
        p = np.arange(self.polar_steps, dtype=np.float64) / (self.polar_steps - 1)
        lr = np.arange(self.length_steps, dtype=np.float64) / (self.length_steps - 1)
        phi = (2.0 * np.pi * p)[:, None]
        spine, lx, ly, _ = self.frames.samples(lr)              # (nl, 3)
        mod = np.broadcast_to(np.asarray(self._radius(phi, lr[None, :]), dtype=np.float64),
                              (len(p), len(lr)))                 # (nphi, nl)
        fx = mod * np.cos(phi)
        fy = mod * np.sin(phi)
        grid = (spine[None, :, :] + fx[..., None] * lx[None, :, :]
                + fy[..., None] * ly[None, :, :])
        return self._xform_grid(grid)

    def to_mesh(self) -> Mesh:
        b = SurfaceMeshBuilder()
        b.add(self._cap(1.0), flip=False)    # top
        b.add(self._cap(0.0), flip=True)     # bottom
        b.add(self._mantle(), flip=True)     # outer mantle
        return b.build()


class Cone(Cylinder):
    """A cone: a cylinder whose radius varies linearly from start to end."""

    def __init__(self, frame: LocalFrame | None, length: float,
                 start_radius: float, end_radius: float, *,
                 polar_steps: int = 360, radial_steps: int = 5,
                 transform: VertexTransform | None = None):
        s, e = float(start_radius), float(end_radius)

        def linear(phi, length_ratio):
            return s + np.clip(length_ratio, 0.0, 1.0) * (e - s)

        super().__init__(frame, length, radius=linear, polar_steps=polar_steps,
                         radial_steps=radial_steps, transform=transform)
