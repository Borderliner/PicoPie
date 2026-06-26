"""Colour utilities (port of ShapeKernel ``ColorPalette`` / ``ColorScale`` /
``RainboxSpectrum``).

Colours are RGB float tuples in ``0..1`` — ready for ``Viewer.set_group_material``
and ``show()``. Value→colour *scales* map a scalar to a colour for visualising a
field or per-triangle property (see :mod:`picogk.shapes.painter`).
"""

from __future__ import annotations

import random as _random

import numpy as np

from . import formulas, spline_ops


def _hex(code: str) -> tuple[float, float, float]:
    s = code.lstrip("#")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


class Palette:
    """LEAP 71's named colours (port of ``Cp``), as RGB ``0..1`` tuples."""

    BLACK = _hex("#000000")
    RACING_GREEN = _hex("#065c35")
    GREEN = _hex("#00b800")
    BILLIE = _hex("#02f70b")
    LEMONGRASS = _hex("#b8e031")
    YELLOW = _hex("#fcd808")
    WARNING = _hex("#fc6608")
    RED = _hex("#ff0000")
    RUBY = _hex("#b0002c")
    ORCHID = _hex("#c72483")
    PITAYA = _hex("#fa2a88")
    BUBBLEGUM = _hex("#ff66ce")
    LAVENDER = _hex("#c966ff")
    GRAY = _hex("#bdbdbd")
    ROCK = _hex("#6b7178")
    CRYSTAL = _hex("#0cc1f7")
    FROZEN = _hex("#6de2fc")
    BLUEBERRY = _hex("#4f0dbf")
    BLUE = _hex("#4287f5")
    TOOTHPASTE = _hex("#25e6c9")

    @staticmethod
    def random(index: int | None = None) -> tuple[float, float, float]:
        """A random RGB colour; reproducible per ``index`` if one is given."""
        rng = _random.Random(index) if index is not None else _random
        return (rng.random(), rng.random(), rng.random())


def rainbow_spectrum() -> list[tuple[float, float, float]]:
    """Blue→green→yellow→orange→red control colours (port of ``RainboxSpectrum``)."""
    return [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            (1.0, 130.0 / 255.0, 0.0), (1.0, 0.0, 0.0)]


class LinearColorScale:
    """Maps a value to a colour by linear RGB interpolation between two colours."""

    def __init__(self, min_color, max_color, min_value: float, max_value: float):
        self.min_color = np.asarray(min_color, dtype=np.float64)
        self.max_color = np.asarray(max_color, dtype=np.float64)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def _interp(self, c1: float, c2: float, value: float) -> float:
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        return c1 + ratio * (c2 - c1)

    def color(self, value: float) -> tuple[float, float, float]:
        v = min(max(value, self.min_value), self.max_value)
        c = [float(np.clip(self._interp(self.min_color[k], self.max_color[k], v), 0.0, 1.0))
             for k in range(3)]
        return (c[0], c[1], c[2])


class SmoothColorScale(LinearColorScale):
    """Like :class:`LinearColorScale` but with a smooth (BSpline) transition."""

    def _interp(self, c1: float, c2: float, value: float) -> float:
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        return formulas.trans_fixed(c1, c2, ratio)


class CustomColorScale(LinearColorScale):
    """A tanh-smooth colour transition with a custom position and sharpness."""

    def __init__(self, min_color, max_color, min_value: float, max_value: float,
                 transition: float, smoothness: float):
        super().__init__(min_color, max_color, min_value, max_value)
        self.transition = float(transition)
        self.smoothness = float(smoothness)

    def _interp(self, c1: float, c2: float, value: float) -> float:
        return formulas.trans_smooth(c1, c2, value, self.transition, self.smoothness)


class ColorScale3D:
    """Maps a value through a smooth multi-colour spectrum (port of ``ColorScale3D``)."""

    def __init__(self, spectrum, min_value: float, max_value: float):
        pts = np.asarray(spectrum, dtype=np.float64).reshape(-1, 3)
        smooth = spline_ops.nurb(pts, 500)
        self._rgb = spline_ops.reparametrized(smooth, 500)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def color(self, value: float) -> tuple[float, float, float]:
        v = min(max(value, self.min_value), self.max_value)
        ratio = (v - self.min_value) / (self.max_value - self.min_value)
        idx = int(ratio * (len(self._rgb) - 1))
        rgb = self._rgb[idx]
        return (float(np.clip(rgb[0], 0.0, 1.0)),
                float(np.clip(rgb[1], 0.0, 1.0)),
                float(np.clip(rgb[2], 0.0, 1.0)))
