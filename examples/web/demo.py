#!/usr/bin/env python3
"""Quick test of the PicoPie web viewer — writes a self-contained HTML you can
open in any browser (no Jupyter needed; needs internet for the three.js CDN).

    python examples/web/demo.py              # writes web_demo.html and opens it
    python examples/web/demo.py out.html     # custom output path
    python examples/web/demo.py --no-open    # just write the file

Needs the web extra:  pip install -e ".[web]"   (or  pip install "picopie[web]")
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import picopie
from picopie.shapes import Box, ImplicitGyroid, ImplicitSphere, LocalFrame, Sphere
from picopie.web import export_html


def build_scene():
    box = Box(LocalFrame((-32, 0, 0)), 22, 22, 22).to_voxels()
    sphere = Sphere(LocalFrame((0, 0, 0)), radius=13).to_voxels()
    # a gyroid-filled sphere — exercises the implicit intersect fixed in 0.3.1
    gyroid = ImplicitSphere((32, 0, 0), 13).render(((19, -14, -14), (45, 14, 14)))
    gyroid.intersect_implicit_(ImplicitGyroid(4.0, 1.0))
    return [box, sphere, gyroid]


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--no-open"]
    auto_open = "--no-open" not in argv
    out = args[0] if args else "web_demo.html"

    picopie.init(voxel_size_mm=0.5)   # coarse-ish so the demo HTML stays light + fast
    export_html(build_scene(), out, title="PicoPie web viewer demo")

    uri = Path(out).resolve().as_uri()
    print(f"wrote {out}")
    print(f"open: {uri}")
    if auto_open:
        try:
            webbrowser.open(uri)
        except Exception:
            print("(could not auto-open a browser — open the URL above manually)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
