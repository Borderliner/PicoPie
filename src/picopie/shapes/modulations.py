"""Modulations: dimensions that vary across a shape's parameters.

A *modulation* is the heart of ShapeKernel's parametricity — a dimension
(radius, width, ...) expressed as a constant, a function of parameter
ratio(s), or a composition. PicoPie keeps the C# names (``LineModulation`` for
1D, ``SurfaceModulation`` for 2D) but makes them callable and lets shape
parameters accept a plain number, a callable, or a modulation interchangeably.

Callables receive ratios/angles and may be passed numpy arrays during meshing,
so a function modulation should be written with numpy (e.g.
``lambda phi, lr: 10 + 2 * np.sin(phi)``).

Forms: constant, ``callable``, discrete points (``from_points`` /
``from_image``, linearly interpolated), and ``+ - *`` compositions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Number = int | float


class LineModulation:
    """A 1D modulation: a value as a function of a single ratio in ``[0, 1]``."""

    def __init__(self, value: Number | Callable | LineModulation):
        self.const_value: float | None = None
        self.knots: tuple[np.ndarray, np.ndarray] | None = None
        if isinstance(value, LineModulation):
            self._func: Callable = value._func
            self.const_value = value.const_value
        elif callable(value):
            self._func = value
        else:
            c = float(value)
            self.const_value = c
            self._func = lambda ratio: c

    @classmethod
    def from_points(cls, points, values: str = "z", axis: str = "x") -> LineModulation:
        """Build a modulation by linearly interpolating discrete ``(N, 3)`` points.

        ``axis`` selects the coordinate used as the ratio (the independent
        variable), ``values`` the coordinate used as the result — matching C#'s
        ``ECoord``. Points are sorted to strictly increasing ratio and the ends
        are held flat outside ``[0, 1]`` (as ShapeKernel does).
        """
        idx = {"x": 0, "y": 1, "z": 2}
        ai, vi = idx[axis], idx[values]
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        xs = [float(pts[0, ai])]
        ys = [float(pts[0, vi])]
        for i in range(1, len(pts)):
            x = float(pts[i, ai])
            if x > xs[-1]:
                xs.append(x)
                ys.append(float(pts[i, vi]))
        if xs[0] > 0.0:
            xs.insert(0, 0.0)
            ys.insert(0, ys[0])
        if xs[-1] < 1.0:
            xs.append(1.0)
            ys.append(ys[-1])
        xp = np.asarray(xs, dtype=np.float64)
        yp = np.asarray(ys, dtype=np.float64)

        def interp(ratio, xp=xp, yp=yp):
            return np.interp(np.clip(ratio, 0.0, 1.0), xp, yp)

        inst = cls(interp)
        inst.knots = (xp, yp)
        return inst

    def __call__(self, ratio):
        return self._func(ratio)

    def __mul__(self, factor: Number) -> LineModulation:
        f = float(factor)
        return LineModulation(lambda r: f * self._func(r))

    __rmul__ = __mul__

    def __add__(self, other) -> LineModulation:
        o = LineModulation(other)
        return LineModulation(lambda r: self._func(r) + o._func(r))

    def __sub__(self, other) -> LineModulation:
        o = LineModulation(other)
        return LineModulation(lambda r: self._func(r) - o._func(r))


class SurfaceModulation:
    """A 2D modulation: a value as a function of ``(phi, length_ratio)``.

    Accepts a constant, a ``callable(phi, length_ratio)``, another
    ``SurfaceModulation``, or a :class:`LineModulation` (broadcast across one
    axis — ``line="second"`` uses ``length_ratio`` as in C#'s default,
    ``line="first"`` uses ``phi``).
    """

    def __init__(self, value, *, line: str = "second"):
        self.const_value: float | None = None
        if isinstance(value, SurfaceModulation):
            self._func: Callable = value._func
            self.const_value = value.const_value
        elif isinstance(value, LineModulation):
            if line == "first":
                self._func = lambda phi, lr: value(phi)
            elif line == "second":
                self._func = lambda phi, lr: value(lr)
            else:
                raise ValueError("line must be 'first' or 'second'")
        elif callable(value):
            self._func = value
        else:
            c = float(value)
            self.const_value = c
            self._func = lambda phi, lr: c

    @classmethod
    def from_image(cls, image, mapping: Callable[[float], float]) -> SurfaceModulation:
        """Build a modulation from a 2D grayscale array (port of the Image form).

        ``image`` is indexed ``[x, y]`` (width, height); ``phi`` maps to x
        (reversed, as in C#) and ``length_ratio`` to y, then ``mapping`` turns
        the sampled gray value into a physical value. ``mapping`` should be
        numpy-aware so meshing can pass arrays.
        """
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2:
            raise ValueError("image must be a 2D grayscale array")
        nx = img.shape[0] - 1
        ny = img.shape[1] - 1

        def sample(phi, lr):
            x = np.clip(np.rint(nx - np.asarray(phi, dtype=np.float64) * nx).astype(int), 0, nx)
            y = np.clip(np.rint(np.asarray(lr, dtype=np.float64) * ny).astype(int), 0, ny)
            return mapping(img[x, y])

        return cls(sample)

    def __call__(self, phi, length_ratio):
        return self._func(phi, length_ratio)

    def __mul__(self, factor: Number) -> SurfaceModulation:
        f = float(factor)
        return SurfaceModulation(lambda phi, lr: f * self._func(phi, lr))

    __rmul__ = __mul__

    def __add__(self, other) -> SurfaceModulation:
        o = SurfaceModulation(other)
        return SurfaceModulation(lambda phi, lr: self._func(phi, lr) + o._func(phi, lr))

    def __sub__(self, other) -> SurfaceModulation:
        o = SurfaceModulation(other)
        return SurfaceModulation(lambda phi, lr: self._func(phi, lr) - o._func(phi, lr))


class Distribution:
    """A normalised :class:`LineModulation` paired with a physical length.

    Stores a data distribution consistently (port of ShapeKernel
    ``Distribution``).
    """

    def __init__(self, total_length: float, modulation):
        self.total_length = float(total_length)
        self.modulation = (modulation if isinstance(modulation, LineModulation)
                           else LineModulation(modulation))


class GenericContour(Distribution):
    """A :class:`Distribution` describing a rotationally symmetric contour."""

