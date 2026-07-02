#!/usr/bin/env python3
"""Interactive viewer demo (Phase 10). Requires a display + OpenGL.

    python examples/viewer_demo.py             # open an interactive window
    python examples/viewer_demo.py out.png     # one-shot headless render to out.png

In the window: left-drag orbits, right/middle-drag (or Shift+left) pans, scroll
zooms; Esc/Q closes, F re-fits, S screenshots.
"""
import sys

import picopie
from picopie import Voxels


def main() -> int:
    picopie.init(voxel_size_mm=0.2)

    # A hollow shell with a bite taken out, so the wall is visible.
    part = Voxels.sphere(radius=12) - Voxels.sphere(center=(8, 0, 0), radius=7)
    part.shell_(1.5)

    if len(sys.argv) > 1:
        out = picopie.render_png(part, sys.argv[1], size=(1280, 960),
                                background=(0.16, 0.16, 0.2))
        print("wrote", out)
    else:
        print("opening viewer -- left-drag orbit, scroll zoom, Esc to close")
        picopie.show(part, background=(0.16, 0.16, 0.2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
