"""Operations on point lists / splines (port of ShapeKernel ``SplineOperations``).

All functions take an ``(N, 3)`` array-like and return ``float64`` numpy arrays
(or scalars). Includes arc-length reparametrization, NURB smoothing, list
transforms, and nearest-point queries.
"""

from __future__ import annotations

import numpy as np

from . import vectors
from .splines import ControlPointSpline


def _pts(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def linear_interpolation(start, end, n_samples: int) -> np.ndarray:
    """``n_samples`` evenly interpolated points from ``start`` to ``end``."""
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    t = (np.arange(n_samples, dtype=np.float64) / (n_samples - 1))[:, None]
    return start + t * (end - start)


def lengths_at_indices(points) -> np.ndarray:
    """Cumulative arc length at each point (first entry 0)."""
    points = _pts(points)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def total_length(points) -> float:
    """Total polyline length."""
    points = _pts(points)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def average_spacing(points) -> float:
    points = _pts(points)
    return total_length(points) / (len(points) - 1)


def reparametrized(points, n_target: int) -> np.ndarray:
    """Resample to constant arc-length spacing (returns ``n_target + 1`` points).

    Start and end points are preserved (matches ShapeKernel's count convention).
    """
    points = _pts(points)
    lengths = lengths_at_indices(points)
    spine_length = lengths[-1]
    step = spine_length / n_target

    out = [points[0]]
    for j in range(1, n_target):
        target = step * j
        upper = 1
        while upper < len(points) and lengths[upper] < target:
            upper += 1
        upper = min(upper, len(points) - 1)
        lower = upper - 1
        denom = lengths[upper] - lengths[lower]
        ds = (target - lengths[lower]) / denom if denom != 0 else 0.0
        out.append(points[lower] + ds * (points[upper] - points[lower]))
    out.append(points[-1])
    return np.array(out)


def reparametrized_by_spacing(points, spacing: float) -> np.ndarray:
    """Resample to a target point spacing (at least 10 samples)."""
    n_target = int(max(10, total_length(points) / spacing))
    return reparametrized(points, n_target)


def nurb(control_points, n_samples: int) -> np.ndarray:
    """NURB-smooth a polyline (degree-2 open B-spline) to ``n_samples`` points."""
    return ControlPointSpline(_pts(control_points), 2, closed=False).points(n_samples)


def oversample(points, samples_per_step: int) -> np.ndarray:
    """Linearly upsample, inserting points per interval (endpoints preserved)."""
    points = _pts(points)
    out = []
    for i in range(1, len(points)):
        for j in range(samples_per_step):
            out.append(points[i - 1] + j / samples_per_step * (points[i] - points[i - 1]))
    out.append(points[-1])
    return np.array(out)


def subsample(points, sample_size: int) -> np.ndarray:
    """Keep every ``sample_size``-th point; the last point is always appended."""
    points = _pts(points)
    out = list(points[::sample_size])
    out.append(points[-1])
    return np.array(out)


def rotate_around_z(points, angle: float) -> np.ndarray:
    return np.array([vectors.rotate_around_z(p, angle) for p in _pts(points)])


def rotate_around_axis(points, angle: float, axis, origin=None) -> np.ndarray:
    return np.array([vectors.rotate_around_axis(p, angle, axis, origin) for p in _pts(points)])


def translate(points, shift) -> np.ndarray:
    shift = np.asarray(shift, dtype=np.float64).reshape(3)
    return _pts(points) + shift


def scale(points, factor: float) -> np.ndarray:
    return _pts(points) * float(factor)


def to_frame(frame, points) -> np.ndarray:
    """Place points (given in local coordinates) into the frame's world space."""
    return np.array([vectors.point_to_world(frame, p) for p in _pts(points)])


def express_in_frame(frame, points) -> np.ndarray:
    """Express world points in a frame's local coordinates."""
    return np.array([vectors.point_to_local(frame, p) for p in _pts(points)])


def average(points) -> np.ndarray:
    """Centroid of the points."""
    return _pts(points).mean(axis=0)


def closest_point(points, start) -> np.ndarray:
    """The point nearest to ``start``."""
    points = _pts(points)
    start = np.asarray(start, dtype=np.float64).reshape(3)
    return points[int(np.argmin(np.sum((points - start) ** 2, axis=1)))]


def distance_to_closest(points, start) -> float:
    points = _pts(points)
    start = np.asarray(start, dtype=np.float64).reshape(3)
    return float(np.sqrt(np.min(np.sum((points - start) ** 2, axis=1))))


def clustered(points, clustering_range: float) -> np.ndarray:
    """Greedily drop points within ``clustering_range`` of an accepted point."""
    points = _pts(points)
    d2 = clustering_range * clustering_range
    kept = [points[0]]
    for i in range(1, len(points)):
        near = closest_point(np.array(kept), points[i])
        if float(np.sum((near - points[i]) ** 2)) > d2:
            kept.append(points[i])
    return np.array(kept)


def snapped(points, voxels) -> np.ndarray:
    """Snap each point onto the surface of ``voxels`` (closest point)."""
    return np.array([np.asarray(voxels.closest_point(p), dtype=np.float64).reshape(3)
                     for p in _pts(points)])


def split(points, first_index_of_second):
    """Split into two lists at an index (no overlap)."""
    points = _pts(points)
    return points[:first_index_of_second], points[first_index_of_second:]


def combine(lists) -> np.ndarray:
    """Concatenate multiple point lists into one array."""
    return np.concatenate([_pts(p) for p in lists], axis=0)
