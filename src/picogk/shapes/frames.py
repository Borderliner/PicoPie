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

from . import vectors


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
