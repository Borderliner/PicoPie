"""Line decimation (port of ShapeKernel ``LineDecimation``).

Resamples a polyline by dropping points that lie within a perpendicular error
of a straight run. Start and end points are preserved. The error is a point's
perpendicular deviation (in mm) from the ideal line.
"""

from __future__ import annotations

import numpy as np

from . import vectors


def _max_error(line_points: np.ndarray) -> float:
    line_dir = line_points[-1] - line_points[0]
    max_err = 0.0
    for i in range(1, len(line_points) - 1):
        pointer = line_points[i] - line_points[0]
        angle = vectors.angle_between(pointer, line_dir)
        max_err = max(max_err, np.sin(angle) * float(np.linalg.norm(pointer)))
    return float(max_err)


def decimate(points, max_error: float) -> np.ndarray:
    """Return the decimated polyline within ``max_error`` of the original."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    def next_index(start: int) -> int:
        if start > len(pts) - 2:
            return len(pts) - 1
        end = start + 1
        for i in range(start + 2, len(pts)):
            if _max_error(pts[start:i + 1]) > max_error:
                break
            end = i
        return end

    indices = [0]
    start = end = 0
    while end < len(pts) - 1:
        end = next_index(start)
        start = end
        indices.append(start)
    return pts[indices]
