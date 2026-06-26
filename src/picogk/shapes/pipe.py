"""Parametric pipe and pipe segment (port of ``BasePipe`` / ``BasePipeSegment``).

A pipe is a hollow tube (inner + outer radius) over a :class:`Frames` spine —
top/bottom caps plus inner/outer mantles. A pipe *segment* is an angular slice
of a pipe (a ``mid +/- range`` sweep) with start/end caps.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ._base import BaseShape, SurfaceMeshBuilder, VertexTransform
from .frames import Frames, LocalFrame
from .modulations import LineModulation, SurfaceModulation


class Pipe(BaseShape):
    """A hollow tube of inner/outer radius over a length-spine."""

    def __init__(self, frame: LocalFrame | None = None, length: float = 20.0,
                 inner_radius=10.0, outer_radius=20.0, *, frames: Frames | None = None,
                 polar_steps: int = 360, radial_steps: int = 5,
                 length_steps: int | None = None,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        spine = frames is not None
        if frames is not None:
            self.frames = frames
        else:
            self.frames = Frames.extrude(length, frame if frame is not None else LocalFrame())
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        ic = self._inner.const_value is not None
        oc = self._outer.const_value is not None
        ls = 500 if (spine or not ic or not oc) else 5
        self.polar_steps = max(5, polar_steps)
        self.radial_steps = max(5, radial_steps)
        self.length_steps = max(5, ls if length_steps is None else length_steps)

    @property
    def inner_radius(self) -> SurfaceModulation:
        return self._inner

    @inner_radius.setter
    def inner_radius(self, value) -> None:
        self._inner = SurfaceModulation(value)

    @property
    def outer_radius(self) -> SurfaceModulation:
        return self._outer

    @outer_radius.setter
    def outer_radius(self, value) -> None:
        self._outer = SurfaceModulation(value)

    def _phi(self, phi_ratio, length_ratio):
        """Angle for a phi-ratio (full 2*pi sweep; segment overrides)."""
        return 2.0 * np.pi * np.asarray(phi_ratio, dtype=np.float64)

    def _surface(self, lr, phi_r, rad_r) -> np.ndarray:
        lr, phi_r, rad_r = np.broadcast_arrays(np.asarray(lr, dtype=np.float64),
                                               np.asarray(phi_r, dtype=np.float64),
                                               np.asarray(rad_r, dtype=np.float64))
        shape = lr.shape
        flat_lr = lr.reshape(-1)
        uniq, inv = np.unique(flat_lr, return_inverse=True)
        spine, lx, ly, _ = self.frames.samples(uniq)
        spine, lx, ly = spine[inv], lx[inv], ly[inv]
        phi = np.broadcast_to(np.asarray(self._phi(phi_r.reshape(-1), flat_lr),
                                         dtype=np.float64), flat_lr.shape)
        outer = np.asarray(self._outer(phi, flat_lr), dtype=np.float64)
        inner = np.asarray(self._inner(phi, flat_lr), dtype=np.float64)
        radius = rad_r.reshape(-1) * (outer - inner) + inner
        pts = (spine + (radius * np.cos(phi))[:, None] * lx
               + (radius * np.sin(phi))[:, None] * ly)
        return self._xform_grid(pts.reshape(*shape, 3))

    def _ratios(self):
        p = np.arange(self.polar_steps, dtype=np.float64) / (self.polar_steps - 1)
        rr = np.arange(self.radial_steps, dtype=np.float64) / (self.radial_steps - 1)
        lr = np.arange(self.length_steps, dtype=np.float64) / (self.length_steps - 1)
        return p, rr, lr

    def _surfaces(self):
        p, rr, lr = self._ratios()
        return [
            (self._surface(lr[-1], p[:, None], rr[None, :]), False),   # top cap
            (self._surface(lr[0], p[:, None], rr[None, :]), True),     # bottom cap
            (self._surface(lr[None, :], p[:, None], 0.0), False),      # inner mantle
            (self._surface(lr[None, :], p[:, None], 1.0), True),       # outer mantle
        ]

    def to_mesh(self) -> Mesh:
        b = SurfaceMeshBuilder()
        for grid, flip in self._surfaces():
            b.add(grid, flip)
        return b.build()


class PipeSegment(Pipe):
    """An angular slice of a pipe, swept ``mid +/- range/2`` around the spine.

    The sweep is given either as ``start``/``end`` angles (``method="start_end"``)
    or as ``mid``/``range`` (``method="mid_range"``) — each a number,
    ``callable(length_ratio)``, or :class:`LineModulation`.
    """

    def __init__(self, frame: LocalFrame | None = None, length: float = 20.0,
                 inner_radius=10.0, outer_radius=20.0, start=0.0, end=np.pi, *,
                 method: str = "start_end", frames: Frames | None = None,
                 polar_steps: int = 360, radial_steps: int = 5,
                 length_steps: int | None = None,
                 transform: VertexTransform | None = None):
        super().__init__(frame, length, inner_radius, outer_radius, frames=frames,
                         polar_steps=polar_steps, radial_steps=radial_steps,
                         length_steps=length_steps, transform=transform)
        a, b = LineModulation(start), LineModulation(end)
        if method == "start_end":
            self._mid = 0.5 * (a + b)
            self._range = b - a
        elif method == "mid_range":
            self._mid, self._range = a, b
        else:
            raise ValueError("method must be 'start_end' or 'mid_range'")

    def _phi(self, phi_ratio, length_ratio):
        phi_ratio = np.asarray(phi_ratio, dtype=np.float64)
        length_ratio = np.asarray(length_ratio, dtype=np.float64)
        return self._mid(length_ratio) + (phi_ratio - 0.5) * self._range(length_ratio)

    def _surfaces(self):
        _p, rr, lr = self._ratios()
        surfaces = super()._surfaces()
        surfaces.append((self._surface(lr[:, None], 0.0, rr[None, :]), False))   # start cap
        surfaces.append((self._surface(lr[:, None], 1.0, rr[None, :]), True))    # end cap
        return surfaces
