"""Modulations: dimensions that vary across a shape's parameters.

A *modulation* is the heart of ShapeKernel's parametricity — a dimension
(radius, width, ...) expressed as a constant, a function of parameter
ratio(s), or a composition. PicoPie keeps the C# names (``LineModulation`` for
1D, ``SurfaceModulation`` for 2D) but makes them callable and lets shape
parameters accept a plain number, a callable, or a modulation interchangeably.

Callables receive ratios/angles and may be passed numpy arrays during meshing,
so a function modulation should be written with numpy (e.g.
``lambda phi, lr: 10 + 2 * np.sin(phi)``).

Note: image-driven modulation and the full operator set are completed in
Phase 12b; this module covers constant / callable / line-derived forms and the
``+ - *`` composition used by the base shapes.
"""

from __future__ import annotations

from collections.abc import Callable

Number = int | float


class LineModulation:
    """A 1D modulation: a value as a function of a single ratio in ``[0, 1]``."""

    def __init__(self, value: Number | Callable | LineModulation):
        if isinstance(value, LineModulation):
            self._func: Callable = value._func
        elif callable(value):
            self._func = value
        else:
            c = float(value)
            self._func = lambda ratio: c

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
        if isinstance(value, SurfaceModulation):
            self._func: Callable = value._func
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
            self._func = lambda phi, lr: c

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
