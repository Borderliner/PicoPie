"""Logo box (port of ShapeKernel ``BaseLogoBox``).

A :class:`Box` whose top face is displaced by a grayscale image (an emboss),
mapped through a function from gray value to physical height.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .box import Box
from .frames import LocalFrame
from .modulations import SurfaceModulation


class LogoBox(Box):
    """A box with an image-embossed top surface.

    Args:
        frame: the local frame.
        length: box length (local Z).
        ref_width: physical width (local X); depth follows the image aspect.
        image: a 2D grayscale array indexed ``[x, y]`` (width x height).
        mapping: ``callable(gray) -> height`` (numpy-aware).
    """

    def __init__(self, frame: LocalFrame, length: float, ref_width: float,
                 image, mapping: Callable[[float], float], *, transform=None):
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2:
            raise ValueError("image must be a 2D grayscale array")
        img_w, img_h = img.shape
        depth = img_h / img_w * ref_width
        super().__init__(frame, length=length, width=ref_width, depth=depth,
                         width_steps=img_w, depth_steps=img_h, length_steps=5,
                         transform=transform)
        self._top = SurfaceModulation.from_image(img, mapping)

    def _offset_flat(self, wr, dr, lr, lz):
        out = np.zeros((len(wr), 3), dtype=np.float64)
        mask = np.abs(lr - 1.0) < 0.0003
        if np.any(mask):
            img_w_ratio = 1.0 - (0.5 + 0.5 * wr[mask])
            img_h_ratio = 1.0 - (0.5 + 0.5 * dr[mask])
            dz = np.asarray(self._top(img_w_ratio, img_h_ratio), dtype=np.float64)
            out[mask] = dz[:, None] * lz[mask]
        return out
