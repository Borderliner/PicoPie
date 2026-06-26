"""List resampling helpers (port of ShapeKernel ``ListOperations``).

Operate on 1-D float sequences; return numpy ``float64`` arrays.
"""

from __future__ import annotations

import numpy as np


def oversample(values, samples_per_step: int) -> np.ndarray:
    """Linearly upsample, inserting ``samples_per_step`` points per interval.

    The first and last values are preserved (the count changes).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = []
    for i in range(1, len(values)):
        for j in range(samples_per_step):
            out.append(values[i - 1] + j / samples_per_step * (values[i] - values[i - 1]))
    out.append(values[-1])
    return np.asarray(out, dtype=np.float64)


def subsample(values, sample_size: int) -> np.ndarray:
    """Take every ``sample_size``-th value; the last value is always appended."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    out = list(values[::sample_size])
    out.append(values[-1])
    return np.asarray(out, dtype=np.float64)


def index_of_max(values) -> int:
    """Index of the maximum value (-1 if empty)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return int(np.argmax(values)) if values.size else -1


def index_of_min(values) -> int:
    """Index of the minimum value (-1 if empty)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return int(np.argmin(values)) if values.size else -1
