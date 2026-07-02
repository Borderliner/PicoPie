"""Local coordinate frames (port of ShapeKernel ``LocalFrame``).

A :class:`LocalFrame` is a position plus an orthonormal ``(x, y, z)`` basis,
with ``y = cross(z, x)`` (a right-handed system), exactly as in ShapeKernel.
Shapes are defined in their frame's local axes, so moving/orienting the frame
moves/orients the shape.

(The ``Frames`` spine — a field of frames along a path — is added in Phase 12c
together with the spline layer it is built on.)
"""

from __future__ import annotations

import numpy as np

from . import spline_ops, vectors


def _vec(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"frame vector must be finite, got {v!r}")
    return a


def _normalize(v: np.ndarray, name: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError(f"{name} has zero length")
    return v / n


class LocalFrame:
    """A position and a right-handed orthonormal basis."""

    __slots__ = ("_pos", "_x", "_y", "_z")

    def __init__(self, position=(0.0, 0.0, 0.0), local_z=None, local_x=None):
        pos = _vec(position)
        if local_z is None:
            z = np.array([0.0, 0.0, 1.0])
            x = np.array([1.0, 0.0, 0.0])
        elif local_x is None:
            z = _normalize(_vec(local_z), "local_z")
            x = vectors.orthogonal_dir(z)
        else:
            z = _normalize(_vec(local_z), "local_z")
            x = _normalize(_vec(local_x), "local_x")
        self._pos = pos
        self._x = x
        self._z = z
        self._y = np.cross(z, x)

    @property
    def position(self) -> np.ndarray:
        return self._pos

    @property
    def local_x(self) -> np.ndarray:
        return self._x

    @property
    def local_y(self) -> np.ndarray:
        return self._y

    @property
    def local_z(self) -> np.ndarray:
        return self._z

    @staticmethod
    def get_local_y(local_z, local_x) -> np.ndarray:
        """Right-handed Y from Z and X (``cross(z, x)``)."""
        return np.cross(_vec(local_z), _vec(local_x))

    def translated(self, offset) -> LocalFrame:
        """A copy moved by ``offset`` (axes unchanged)."""
        return LocalFrame(self._pos + _vec(offset), self._z, self._x)

    def rotated(self, angle: float, axis) -> LocalFrame:
        """A copy with all axes rotated by ``angle`` (rad) about ``axis`` (position fixed)."""
        new_x = vectors.rotate_around_axis(self._x, angle, axis)
        new_z = vectors.rotate_around_axis(self._z, angle, axis)
        return LocalFrame(self._pos, new_z, new_x)

    def inverted(self, mirror_z: bool, mirror_x: bool) -> LocalFrame:
        """A copy with Z and/or X negated (position fixed)."""
        new_z = -self._z if mirror_z else self._z
        new_x = -self._x if mirror_x else self._x
        return LocalFrame(self._pos, new_z, new_x)

    def __repr__(self) -> str:
        return (f"LocalFrame(position={self._pos.tolist()}, "
                f"local_z={self._z.tolist()}, local_x={self._x.tolist()})")


# --- frame fields along a spine (port of ShapeKernel ``Frames``) ---------------
_FRAME_TYPES = ("cylindrical", "spherical", "z", "min_rotation")


def align_with_target_x(local_z, target_x) -> np.ndarray:
    """The in-plane (orthogonal to ``local_z``) direction closest to ``target_x``.

    Brute-force search over 0..180 deg at 0.01 deg resolution, matching
    ShapeKernel (vectorised; same optimum to grid resolution).
    """
    local_z = np.asarray(local_z, dtype=np.float64).reshape(3)
    target_x = np.asarray(target_x, dtype=np.float64).reshape(3)
    init_x = vectors.orthogonal_dir(local_z)
    init_y = np.cross(init_x, local_z)
    phi = (2.0 * np.pi / 360.0) * np.arange(0.0, 180.0, 0.01)
    candidates = np.cos(phi)[:, None] * init_x + np.sin(phi)[:, None] * init_y
    best = candidates[int(np.argmax(np.abs(candidates @ target_x)))]
    return vectors.safe_normalized(vectors.flip_for_alignment(best, target_x))


def target_x_for(point, frame_type: str) -> np.ndarray:
    """The target local-X direction for a coordinate-system frame type."""
    point = np.asarray(point, dtype=np.float64).reshape(3)
    if frame_type == "cylindrical":
        return vectors.safe_normalized(np.array([point[0], point[1], 0.0]))
    if frame_type == "spherical":
        return vectors.safe_normalized(point)
    return np.array([0.0, 0.0, 1.0])     # "z"


def tangent_directions(points) -> np.ndarray:
    """Unit tangents at each point (endpoints duplicated for continuity)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inner = [vectors.safe_normalized(points[i] - points[i - 1])
             for i in range(1, len(points) - 1)]
    return np.array([inner[0], *inner, inner[-1]])


class Frames:
    """A field of local frames sampled along a spine (port of ``Frames``).

    Build with a classmethod; sample with ``frame_at(length_ratio)`` (or the
    per-component ``spine_at`` / ``local_x_at`` / ...). ``length_ratio`` runs
    ``0..1`` along the spine.
    """

    def __init__(self, points, local_x, local_y, local_z):
        self._points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._x = np.asarray(local_x, dtype=np.float64).reshape(-1, 3)
        self._y = np.asarray(local_y, dtype=np.float64).reshape(-1, 3)
        self._z = np.asarray(local_z, dtype=np.float64).reshape(-1, 3)

    # --- constructors ---
    @classmethod
    def extrude(cls, length: float, frame: LocalFrame, spacing: float = 1.0) -> Frames:
        """Extrude a constant frame along a straight line of ``length``."""
        start = frame.position
        end = start + length * frame.local_z
        pts = spline_ops.reparametrized_by_spacing([start, end], spacing)
        return cls._constant(pts, frame)

    @classmethod
    def along_spline(cls, points, frame: LocalFrame, spacing: float = 1.0) -> Frames:
        """Carry a constant frame's axes along an arbitrary spline."""
        pts = spline_ops.reparametrized_by_spacing(points, spacing)
        return cls._constant(pts, frame)

    @classmethod
    def _constant(cls, pts: np.ndarray, frame: LocalFrame) -> Frames:
        n = len(pts)
        return cls(pts, np.tile(frame.local_x, (n, 1)),
                   np.tile(frame.local_y, (n, 1)), np.tile(frame.local_z, (n, 1)))

    @classmethod
    def aligned_to_x(cls, points, target_x, spacing: float = 1.0) -> Frames:
        """Tangent local-Z along the spline, local-X aligned to a constant target."""
        target_x = vectors.safe_normalized(target_x)
        pts = spline_ops.reparametrized_by_spacing(points, spacing)
        lz = tangent_directions(pts)
        lx = np.array([align_with_target_x(lz[i], target_x) for i in range(len(pts))])
        ly = np.array([LocalFrame.get_local_y(lz[i], lx[i]) for i in range(len(pts))])
        n = len(pts)     # NURB post-processing (as in C#)
        return cls(spline_ops.nurb(pts, n), spline_ops.nurb(lx, n),
                   spline_ops.nurb(ly, n), spline_ops.nurb(lz, n))

    @classmethod
    def aligned(cls, points, frame_type: str = "z", spacing: float = 1.0) -> Frames:
        """Tangent local-Z, local-X from a coordinate system or min-rotation transport."""
        if frame_type not in _FRAME_TYPES:
            raise ValueError(f"frame_type must be one of {_FRAME_TYPES}")
        pts = spline_ops.reparametrized_by_spacing(points, spacing)
        lz = tangent_directions(pts)
        lx = np.empty((len(pts), 3))
        last_x = None
        for i in range(len(pts)):
            if frame_type == "min_rotation":
                if last_x is None:
                    last_x = align_with_target_x(lz[i], target_x_for(pts[i], "z"))
                last_x = align_with_target_x(lz[i], last_x)
                lx[i] = last_x
            else:
                lx[i] = align_with_target_x(lz[i], target_x_for(pts[i], frame_type))
        ly = np.array([LocalFrame.get_local_y(lz[i], lx[i]) for i in range(len(pts))])
        return cls(pts, lx, ly, lz)

    # --- sampling ---
    def _interp(self, arr: np.ndarray, length_ratio: float) -> np.ndarray:
        ratio = min(max(length_ratio, 0.0), 1.0)
        count = len(self._points)
        step = ratio * (count - 1)
        lower = int(min(step, count - 1))
        upper = int(min(step + 1, count - 1))
        ds = step - lower
        return arr[lower] + ds * (arr[upper] - arr[lower])

    def spine_at(self, length_ratio: float) -> np.ndarray:
        return self._interp(self._points, length_ratio)

    def local_x_at(self, length_ratio: float) -> np.ndarray:
        return self._interp(self._x, length_ratio)

    def local_y_at(self, length_ratio: float) -> np.ndarray:
        return self._interp(self._y, length_ratio)

    def local_z_at(self, length_ratio: float) -> np.ndarray:
        return self._interp(self._z, length_ratio)

    def frame_at(self, length_ratio: float) -> LocalFrame:
        """A :class:`LocalFrame` interpolated at ``length_ratio`` (axes re-normalised)."""
        return LocalFrame(self.spine_at(length_ratio),
                          self.local_z_at(length_ratio), self.local_x_at(length_ratio))

    def samples(self, ratios):
        """Sample ``(spine, local_x, local_y, local_z)`` at each ratio (each ``(N, 3)``)."""
        ratios = np.atleast_1d(np.asarray(ratios, dtype=np.float64))
        spine = np.array([self.spine_at(r) for r in ratios])
        lx = np.array([self.local_x_at(r) for r in ratios])
        ly = np.array([self.local_y_at(r) for r in ratios])
        lz = np.array([self.local_z_at(r) for r in ratios])
        return spine, lx, ly, lz

    @property
    def points(self) -> np.ndarray:
        return self._points

    def points_resampled(self, n_samples: int) -> np.ndarray:
        return spline_ops.reparametrized(self._points, n_samples)
