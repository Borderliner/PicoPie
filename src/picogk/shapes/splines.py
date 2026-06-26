"""B-spline curves and surfaces (port of ShapeKernel's Splines).

- :class:`ControlPointSpline` — a B-spline curve (open or closed ends).
- :class:`ControlPointSurface` — a B-spline surface patch (2D control grid).
- :class:`TangentialControlSpline` — a curve from endpoints + tangent directions.
- :class:`CylindricalControlSpline` — a curve built by cylindrical steps.

The Cox-de Boor basis (:func:`basis`) is replicated verbatim from C# so sampled
points match the reference. Curves expose ``point_at(ratio)`` and
``points(n)``; the surface exposes ``point_at(u, v)`` and ``grid(nu, nv)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from . import grids, vectors

if TYPE_CHECKING:
    from .frames import LocalFrame

_ERROR = 1e-7


@runtime_checkable
class ISpline(Protocol):
    """Anything that can be sampled into a point list."""

    def points(self, n_samples: int = 500) -> np.ndarray: ...


def basis(knot, t: float, i: int, degree: int, error: float = _ERROR) -> float:
    """Cox-de Boor B-spline basis function (recursive), matching ShapeKernel."""
    if degree == 0:
        if ((knot[i] <= t < knot[i + 1])
                or (abs(t - knot[i + 1]) < error and abs(t - knot[-1]) < error)):
            return 1.0
        return 0.0
    value = 0.0
    if abs(knot[i + degree] - knot[i]) > error:
        value += ((t - knot[i]) / (knot[i + degree] - knot[i])
                  * basis(knot, t, i, degree - 1, error))
    if abs(knot[i + degree + 1] - knot[i + 1]) > error:
        value += ((knot[i + degree + 1] - t) / (knot[i + degree + 1] - knot[i + 1])
                  * basis(knot, t, i + 1, degree - 1, error))
    return value


def _knot_vector(n_control: int, degree: int, clamp: bool) -> np.ndarray:
    n_knots = n_control + degree + 1
    valid_range = n_knots - degree - (degree + 1)
    d = 1.0 / valid_range
    k = -d * degree + d * np.arange(n_knots, dtype=np.float64)
    return np.clip(k, 0.0, 1.0) if clamp else k


class ControlPointSpline:
    """A B-spline curve through a control polygon (open or closed)."""

    def __init__(self, control_points, degree: int = 2, closed: bool = False):
        pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in control_points]
        self.degree = int(degree)
        self.closed = bool(closed)
        if self.closed:
            if float(np.linalg.norm(pts[0] - pts[-1])) < _ERROR:
                pts = pts[:-1]
            n_before = len(pts)
            for i in range(n_before - 1):       # wrap by appending n-1 points
                pts.append(pts[i])
            self.control_points = np.array(pts)
            self.knot = _knot_vector(len(pts), self.degree, clamp=False)
        else:
            self.control_points = np.array(pts)
            self.knot = _knot_vector(len(pts), self.degree, clamp=True)

    def point_at(self, length_ratio: float) -> np.ndarray:
        pt = np.zeros(3)
        for i in range(len(self.control_points)):
            pt = pt + basis(self.knot, length_ratio, i, self.degree) * self.control_points[i]
        return pt

    def points(self, n_samples: int = 500) -> np.ndarray:
        return np.array([self.point_at(i / (n_samples - 1)) for i in range(n_samples)])


class ControlPointSurface:
    """A B-spline surface patch over a 2D control grid ``(rows, cols, 3)``."""

    def __init__(self, control_grid, degree_u: int = 2, degree_v: int = 2,
                 closed_u: bool = False, closed_v: bool = False):
        grid = np.asarray(control_grid, dtype=np.float64)
        if grid.ndim != 3 or grid.shape[2] != 3:
            raise ValueError(f"control_grid must be (rows, cols, 3), got {grid.shape}")
        self.degree_u = int(degree_u)
        self.degree_v = int(degree_v)
        self.closed_u = bool(closed_u)
        self.closed_v = bool(closed_v)

        if self.closed_u and float(np.linalg.norm(grid[0, 0] - grid[-1, 0])) < _ERROR:
            grid = grids.remove_row_x(grid, grid.shape[0] - 1)
        if self.closed_v and float(np.linalg.norm(grid[0, 0] - grid[0, -1])) < _ERROR:
            grid = grids.remove_col_y(grid, grid.shape[1] - 1)

        if self.closed_u:
            before = grid.shape[0]
            for i in range(before - 1):
                grid = grids.add_row_x(grid, grid[i])
        if self.closed_v:
            before = grid.shape[1]
            for i in range(before - 1):
                grid = grids.add_col_y(grid, grid[:, i])

        self.control_grid = grid
        self.knot_u = _knot_vector(grid.shape[0], self.degree_u, clamp=not self.closed_u)
        self.knot_v = _knot_vector(grid.shape[1], self.degree_v, clamp=not self.closed_v)

    def point_at(self, u_ratio: float, v_ratio: float) -> np.ndarray:
        rows, cols, _ = self.control_grid.shape
        pt = np.zeros(3)
        for u in range(rows):
            bu = basis(self.knot_u, u_ratio, u, self.degree_u)
            if bu == 0.0:
                continue
            for v in range(cols):
                bv = basis(self.knot_v, v_ratio, v, self.degree_v)
                pt = pt + bu * bv * self.control_grid[u, v]
        return pt

    def grid(self, n_u: int = 500, n_v: int = 500) -> np.ndarray:
        out = np.empty((n_u, n_v, 3), dtype=np.float64)
        for i in range(n_u):
            ur = i / (n_u - 1)
            for j in range(n_v):
                out[i, j] = self.point_at(ur, j / (n_v - 1))
        return out

    def control_point(self, u: int, v: int) -> np.ndarray:
        return self.control_grid[u, v]

    def update_control_point(self, point, u: int, v: int) -> None:
        self.control_grid[u, v] = np.asarray(point, dtype=np.float64).reshape(3)


class TangentialControlSpline:
    """A B-spline curve from endpoints with start/end tangent directions."""

    def __init__(self, start, end, start_dir, end_dir,
                 start_strength: float = -1, end_strength: float = -1,
                 relative_start: bool = False, relative_end: bool = False):
        start = np.asarray(start, dtype=np.float64).reshape(3)
        end = np.asarray(end, dtype=np.float64).reshape(3)
        span = float(np.linalg.norm(start - end))
        if start_strength == -1:
            start_strength = 0.3 * span
        if end_strength == -1:
            end_strength = 0.3 * span
        if relative_start:
            start_strength *= span
        if relative_end:
            end_strength *= span
        cps = [
            start,
            start + start_strength * vectors.safe_normalized(start_dir),
            end - end_strength * vectors.safe_normalized(end_dir),
            end,
        ]
        self._spline = ControlPointSpline(cps)

    @classmethod
    def from_frames(cls, start_frame: LocalFrame, end_frame: LocalFrame,
                    start_strength: float = -1, end_strength: float = -1,
                    relative_start: bool = False, relative_end: bool = False
                    ) -> TangentialControlSpline:
        return cls(start_frame.position, end_frame.position,
                   start_frame.local_z, end_frame.local_z,
                   start_strength, end_strength, relative_start, relative_end)

    def point_at(self, length_ratio: float) -> np.ndarray:
        return self._spline.point_at(length_ratio)

    def points(self, n_samples: int = 500) -> np.ndarray:
        return self._spline.points(n_samples)


class CylindricalControlSpline:
    """A B-spline curve built by relative/absolute cylindrical steps."""

    def __init__(self, start):
        self._control_points = [np.asarray(start, dtype=np.float64).reshape(3)]

    def add_relative_step(self, direction: str, step_length: float) -> CylindricalControlSpline:
        last = self._control_points[-1]
        if direction == "z":
            new = last + step_length * np.array([0.0, 0.0, 1.0])
        elif direction == "radial":
            new = last + step_length * vectors.planar_dir(last)
        elif direction == "tangential":
            radial = vectors.planar_dir(last)
            tangential = np.cross(np.array([0.0, 0.0, 1.0]), radial)
            new = last + step_length * tangential
        else:
            raise ValueError("direction must be 'radial', 'tangential', or 'z'")
        self._control_points.append(new)
        return self

    def add_absolute_step(self, direction: str, new_value: float) -> CylindricalControlSpline:
        last = self._control_points[-1]
        if direction == "z":
            new = vectors.with_z(last, new_value)
        elif direction == "radial":
            new = vectors.with_radius(last, new_value)
        else:
            raise ValueError("absolute step direction must be 'radial' or 'z'")
        self._control_points.append(new)
        return self

    def points(self, n_samples: int = 500) -> np.ndarray:
        return ControlPointSpline(self._control_points).points(n_samples)
