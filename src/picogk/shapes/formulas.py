"""Scalar/point helper formulas (port of ShapeKernel ``Uf``).

Smooth (tanh) transitions, random sampling, and Fibonacci point distributions.

Randomness here uses Python's ``random`` (pass an ``rng`` for reproducibility);
unlike the deterministic geometry, random outputs are not parity-tested.
"""

from __future__ import annotations

import math
import random as _random

import numpy as np

_trans_bspline = None


def _get_trans_bspline():
    """Cached open BSpline driving the fixed-endpoint transition (lazy import)."""
    global _trans_bspline
    if _trans_bspline is None:
        from .splines import ControlPointSpline
        _trans_bspline = ControlPointSpline(
            [[0, 0, 0.0], [0, 0, 0.5], [1, 0, 0.5], [1, 0, 1.0]])
    return _trans_bspline


def trans_fixed(value1: float, value2: float, s: float) -> float:
    """Blend two values across ``s`` along a fixed-endpoint BSpline transition."""
    ratio = float(_get_trans_bspline().point_at(s)[0])
    return value1 + ratio * (value2 - value1)


def vec_trans_fixed(p1, p2, s: float) -> np.ndarray:
    """Vector form of :func:`trans_fixed` (per-component, endpoints fixed)."""
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2, dtype=np.float64).reshape(3)
    return np.array([trans_fixed(p1[k], p2[k], s) for k in range(3)])


def _normalized_tanh(s: float, transition_s: float, smooth: float) -> float:
    return 0.5 + 0.5 * math.tanh((s - transition_s) / smooth)


def trans_smooth(value1: float, value2: float, s: float,
                 transition_s: float, smooth: float) -> float:
    """Blend two values across ``s`` with a smooth tanh transition.

    ``smooth`` controls sharpness — smaller is sharper/shorter.
    """
    w = _normalized_tanh(s, transition_s, smooth)
    return value1 * (1.0 - w) + value2 * w


def vec_trans_smooth(p1, p2, s: float, transition_s: float, smooth: float) -> np.ndarray:
    """Vector form of :func:`trans_smooth`."""
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2, dtype=np.float64).reshape(3)
    w = _normalized_tanh(s, transition_s, smooth)
    return p1 * (1.0 - w) + p2 * w


def random_gaussian(mean: float, std_dev: float,
                    rng: _random.Random | None = None) -> float:
    """A Gaussian sample (Box-Muller, matching ShapeKernel's formula)."""
    r = rng or _random
    x1 = 1.0 - r.random()
    x2 = 1.0 - r.random()
    y1 = math.sqrt(-2.0 * math.log(x1)) * math.cos(2.0 * math.pi * x2)
    return y1 * std_dev + mean


def random_linear(min_value: float, max_value: float,
                  rng: _random.Random | None = None) -> float:
    """A uniform sample in ``[min_value, max_value)``."""
    r = rng or _random
    return min_value + (max_value - min_value) * r.random()


def random_bool(rng: _random.Random | None = None) -> bool:
    """A uniform random boolean."""
    r = rng or _random
    return r.random() > 0.5


def fibonacci_circle_points(outer_radius: float, n_samples: int) -> np.ndarray:
    """``(n, 3)`` points spread in a disc by the Fibonacci sequence (z=0)."""
    k = np.arange(n_samples, dtype=np.float64) + 0.5
    r = np.sqrt(k / n_samples) * outer_radius
    angle = math.pi * (1.0 + math.sqrt(5.0)) * k
    return np.column_stack([r * np.cos(angle), r * np.sin(angle), np.zeros(n_samples)])


def fibonacci_sphere_points(outer_radius: float, n_samples: int) -> np.ndarray:
    """``(n, 3)`` points spread on a sphere by the Fibonacci sequence."""
    k = np.arange(n_samples, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n_samples)
    angle = math.pi * (1.0 + math.sqrt(5.0)) * k
    return outer_radius * np.column_stack([
        np.cos(angle) * np.sin(phi),
        np.sin(angle) * np.sin(phi),
        np.cos(phi),
    ])
