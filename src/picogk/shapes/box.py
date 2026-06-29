"""Parametric box (port of ShapeKernel ``BaseBox``).

A box defined in a :class:`Frames` spine with width (local X) and depth
(local Y) :class:`LineModulation`\\s along its length. Six faces are emitted
with outward-consistent winding.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ._base import BaseShape, SurfaceMeshBuilder, VertexTransform
from .frames import Frames, LocalFrame
from .modulations import LineModulation


class Box(BaseShape):
    """A box: width (X) x depth (Y) over a length-spine, with modulated sides.

    Args:
        frame: local frame the box is built in (ignored if ``frames`` given).
        length/width/depth: dimensions (numbers or, for width/depth,
            ``callable(length_ratio)`` / :class:`LineModulation`).
        frames: an explicit :class:`Frames` spine (replaces ``length``).
        *_steps: tessellation overrides (default 5, or 500 when modulated/spined).
    """

    def __init__(self, frame: LocalFrame | None = None, length: float = 20.0,
                 width=20.0, depth=20.0, *, frames: Frames | None = None,
                 width_steps: int | None = None, depth_steps: int | None = None,
                 length_steps: int | None = None,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        spine = frames is not None
        if frames is not None:
            self.frames = frames
        else:
            self.frames = Frames.extrude(length, frame if frame is not None else LocalFrame())
        self.width = width
        self.depth = depth
        self._spine = spine
        # step counts default to 5, but a modulated width/depth (or a spine) raises
        # that dimension's + the length tessellation to 500 (matching C#'s
        # SetWidth/SetDepth/spine ctor). Computed lazily so setting a modulation
        # AFTER construction still bumps; an explicit *_steps kwarg overrides.
        self._width_steps = width_steps
        self._depth_steps = depth_steps
        self._length_steps = length_steps

    @property
    def width(self) -> LineModulation:
        return self._width

    @width.setter
    def width(self, value) -> None:
        self._width = LineModulation(value)

    @property
    def depth(self) -> LineModulation:
        return self._depth

    @depth.setter
    def depth(self, value) -> None:
        self._depth = LineModulation(value)

    @property
    def width_steps(self) -> int:
        if self._width_steps is not None:
            return max(5, self._width_steps)
        return 5 if self._width.const_value is not None else 500

    @width_steps.setter
    def width_steps(self, value) -> None:
        self._width_steps = value

    @property
    def depth_steps(self) -> int:
        if self._depth_steps is not None:
            return max(5, self._depth_steps)
        return 5 if self._depth.const_value is not None else 500

    @depth_steps.setter
    def depth_steps(self, value) -> None:
        self._depth_steps = value

    @property
    def length_steps(self) -> int:
        if self._length_steps is not None:
            return max(5, self._length_steps)
        modulated = self._width.const_value is None or self._depth.const_value is None
        return 500 if (self._spine or modulated) else 5

    @length_steps.setter
    def length_steps(self, value) -> None:
        self._length_steps = value

    @classmethod
    def from_bbox(cls, bbox, *, transform: VertexTransform | None = None) -> Box:
        """A box matching a :class:`~picogk.types.BBox3` (origin at its min-Z centre)."""
        size = np.asarray(bbox.size, dtype=np.float64)
        centre = np.asarray(bbox.center, dtype=np.float64).copy()
        centre[2] = float(np.asarray(bbox.min, dtype=np.float64)[2])
        return cls(LocalFrame(centre), length=float(size[2]),
                   width=float(size[0]), depth=float(size[1]), transform=transform)

    # --- meshing ---
    def _offset_flat(self, wr, dr, lr, lz):
        """Per-point extra displacement (LogoBox overrides; default none)."""
        return 0.0

    def _surface(self, wr, dr, lr) -> np.ndarray:
        """Grid of surface points for broadcastable ratio arrays ``(wr, dr, lr)``."""
        wr, dr, lr = np.broadcast_arrays(np.asarray(wr, dtype=np.float64),
                                         np.asarray(dr, dtype=np.float64),
                                         np.asarray(lr, dtype=np.float64))
        shape = wr.shape
        flat_lr = lr.reshape(-1)
        uniq, inv = np.unique(flat_lr, return_inverse=True)
        spine, lx, ly, lz = self.frames.samples(uniq)
        wmod = np.array([float(self._width(r)) for r in uniq])
        dmod = np.array([float(self._depth(r)) for r in uniq])
        spine, lx, ly, lz = spine[inv], lx[inv], ly[inv], lz[inv]
        wmod, dmod = wmod[inv], dmod[inv]

        wrf, drf = wr.reshape(-1), dr.reshape(-1)
        pts = (spine + (0.5 * wrf * wmod)[:, None] * lx + (0.5 * drf * dmod)[:, None] * ly
               + self._offset_flat(wrf, drf, flat_lr, lz))
        grid = pts.reshape(*shape, 3)
        if self._transform is not None:
            grid = self._apply_transform(grid.reshape(-1, 3)).reshape(*shape, 3)
        return grid

    def to_mesh(self) -> Mesh:
        nw, nd, nl = self.width_steps, self.depth_steps, self.length_steps
        w = 2.0 * np.arange(nw, dtype=np.float64) / (nw - 1) - 1.0   # -1..1
        d = 2.0 * np.arange(nd, dtype=np.float64) / (nd - 1) - 1.0
        lr = np.arange(nl, dtype=np.float64) / (nl - 1)              # 0..1
        b = SurfaceMeshBuilder()
        b.add(self._surface(w[:, None], d[None, :], lr[-1]), flip=True)    # top
        b.add(self._surface(w[:, None], d[None, :], lr[0]), flip=False)    # bottom
        b.add(self._surface(w[0], d[None, :], lr[:, None]), flip=True)     # front
        b.add(self._surface(w[-1], d[None, :], lr[:, None]), flip=False)   # back
        b.add(self._surface(w[None, :], d[-1], lr[:, None]), flip=True)    # right
        b.add(self._surface(w[None, :], d[0], lr[:, None]), flip=False)    # left
        return b.build()
