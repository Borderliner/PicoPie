"""Surface of revolution (port of ShapeKernel ``BaseRevolve``).

A profile carried along a :class:`Frames` spine, with inward/outward radial
offsets (along the spine's local X), revolved around the axis frame's local Z.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from . import vectors
from ._base import BaseShape, SurfaceMeshBuilder, VertexTransform
from .frames import Frames, LocalFrame
from .modulations import LineModulation


class Revolve(BaseShape):
    """A solid of revolution about ``frame``'s local Z, profiled along ``frames``.

    Inward and outward radii are measured from the spine outward (inward is
    counted toward the axis). Always spine-based.
    """

    def __init__(self, frame: LocalFrame, frames: Frames,
                 inner_radius=3.0, outer_radius=3.0, *, radial_steps: int = 100,
                 polar_steps: int = 360, length_steps: int = 500,
                 transform: VertexTransform | None = None):
        super().__init__(transform)
        self.frame = frame
        self.frames = frames
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.radial_steps = max(5, radial_steps)
        self.polar_steps = max(5, polar_steps)
        self.length_steps = max(5, length_steps)

    @property
    def inner_radius(self) -> LineModulation:
        return self._inner

    @inner_radius.setter
    def inner_radius(self, value) -> None:
        self._inner = LineModulation(value)

    @property
    def outer_radius(self) -> LineModulation:
        return self._outer

    @outer_radius.setter
    def outer_radius(self, value) -> None:
        self._outer = LineModulation(value)

    @classmethod
    def frames_from_contour(cls, contour, frame: LocalFrame | None = None) -> Frames:
        """A cylindrical-aligned :class:`Frames` from a radius-vs-height contour."""
        frame = frame if frame is not None else LocalFrame()
        n = 500
        lr = np.arange(n, dtype=np.float64) / (n - 1)
        pts = [vectors.point_to_world(frame, [float(contour.modulation(r)), 0.0,
                                              float(r * contour.total_length)]) for r in lr]
        return Frames.aligned(pts, "cylindrical", spacing=0.5)

    def _rotate(self, pts: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Rotate each point about the axis frame's Z by its angle (quaternion, exact)."""
        axis = self.frame.local_z
        origin = self.frame.position
        half = 0.5 * angles
        s = np.sin(half)
        qx, qy, qz, qw = axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)
        rel = pts - origin
        x2, y2, z2 = qx + qx, qy + qy, qz + qz
        wx2, wy2, wz2 = qw * x2, qw * y2, qw * z2
        xx2, xy2, xz2 = qx * x2, qx * y2, qx * z2
        yy2, yz2, zz2 = qy * y2, qy * z2, qz * z2
        vx, vy, vz = rel[:, 0], rel[:, 1], rel[:, 2]
        out = np.stack([
            vx * (1 - yy2 - zz2) + vy * (xy2 - wz2) + vz * (xz2 + wy2),
            vx * (xy2 + wz2) + vy * (1 - xx2 - zz2) + vz * (yz2 - wx2),
            vx * (xz2 - wy2) + vy * (yz2 + wx2) + vz * (1 - xx2 - yy2),
        ], axis=-1)
        return out + origin

    def _surface(self, lr, phi_r, rad_r) -> np.ndarray:
        lr, phi_r, rad_r = np.broadcast_arrays(np.asarray(lr, dtype=np.float64),
                                               np.asarray(phi_r, dtype=np.float64),
                                               np.asarray(rad_r, dtype=np.float64))
        shape = lr.shape
        flat_lr = lr.reshape(-1)
        uniq, inv = np.unique(flat_lr, return_inverse=True)
        spine, lx, _, _ = self.frames.samples(uniq)
        spine, lx = spine[inv], lx[inv]
        outward = np.asarray(self._outer(flat_lr), dtype=np.float64)
        inward = -np.asarray(self._inner(flat_lr), dtype=np.float64)
        radius = rad_r.reshape(-1) * (outward - inward) + inward
        base = spine + radius[:, None] * lx
        phi = 2.0 * np.pi * phi_r.reshape(-1)
        pts = self._rotate(base, np.broadcast_to(phi, flat_lr.shape))
        return self._xform_grid(pts.reshape(*shape, 3))

    def to_mesh(self) -> Mesh:
        p = np.arange(self.polar_steps, dtype=np.float64) / (self.polar_steps - 1)
        rr = np.arange(self.radial_steps, dtype=np.float64) / (self.radial_steps - 1)
        lr = np.arange(self.length_steps, dtype=np.float64) / (self.length_steps - 1)
        b = SurfaceMeshBuilder()
        b.add(self._surface(lr[-1], p[:, None], rr[None, :]), flip=False)   # top
        b.add(self._surface(lr[None, :], p[:, None], 0.0), flip=False)      # inner mantle
        b.add(self._surface(lr[None, :], p[:, None], 1.0), flip=True)       # outer mantle
        b.add(self._surface(lr[0], p[:, None], rr[None, :]), flip=True)     # bottom
        return b.build()
