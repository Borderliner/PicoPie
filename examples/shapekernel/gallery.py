#!/usr/bin/env python3
"""ShapeKernel example gallery — a Python port of LEAP71_ShapeKernel/Examples.

Each ``build_*`` returns a *scene*: a list of ``(object, rgb)`` groups (objects
are :class:`Voxels` or :class:`Mesh`). The scenes are collected in ``SCENES``.

Usage::

    python examples/shapekernel/gallery.py                  # render every scene to PNG
    python examples/shapekernel/gallery.py out_dir          # ... into out_dir
    python examples/shapekernel/gallery.py --voxel-size 0.1 # finer/smoother (slower)
    python examples/shapekernel/gallery.py --show box       # open one scene interactively

Surface smoothness is set by the *voxel size*, not the parametric tessellation:
every scene is rasterised to a voxel field and re-meshed at that grid, so a
coarse grid gives faceted silhouettes and an "orange-peel" look. ``--voxel-size``
defaults to 0.2 mm for clean docs renders; drop it for stress tests.

Rendering needs a display + OpenGL; the geometry builds run headless (and are
exercised by tests/test_examples.py).
"""

from __future__ import annotations

import sys

import numpy as np

import picopie
from picopie.shapes import (
    Box,
    ControlPointSpline,
    Cylinder,
    Frames,
    ImplicitGenus,
    ImplicitGyroid,
    ImplicitSphere,
    ImplicitSuperEllipsoid,
    LatticeManifold,
    LatticePipe,
    Lens,
    LocalFrame,
    Pipe,
    PipeSegment,
    Ring,
    Sphere,
    functions,
    mesh_utils,
    painter,
)
from picopie.shapes import (
    Palette as P,
)
from picopie.shapes.colors import ColorScale3D, rainbow_spectrum


# --- shared modulations (from the upstream examples) ---------------------------
def _line1(lr):
    return 10.0 - 3.0 * np.cos(8.0 * lr)


def _line2(lr):
    return 8.0 - np.cos(40.0 * lr)


def _surf1(phi, lr):
    return 12.0 + 3.0 * np.cos(5.0 * phi)


def _surf3(phi, lr):
    return 8.0 + 5.0 * np.cos(5.0 * phi)


def _spline_points():
    return ControlPointSpline([[0, 0, 0], [0, 40, 0], [0, 50, 20], [0, 60, 60]]).points(500)


def _spine(x_offset=0.0):
    pts = np.asarray(_spline_points(), dtype=np.float64) + np.array([x_offset, 0, 0])
    return Frames.aligned_to_x(pts, (0, 1, 0))


# --- scenes --------------------------------------------------------------------
def build_box():
    return [
        (Box(LocalFrame((-50, 0, 0)), 20, 10, 15).to_voxels(), P.BLUE),
        (Box(LocalFrame((50, 0, 0)), 20, width=_line2, depth=_line1).to_voxels(), P.GREEN),
        (Box(frames=_spine(), width=_line2, depth=_line1).to_voxels(), P.YELLOW),
    ]


def build_sphere():
    return [
        (Sphere(LocalFrame((-100, 0, 0)), 40).to_voxels(), P.FROZEN),
        (Sphere(LocalFrame((0, 0, 0)),
                radius=lambda phi, theta: 40 - 10 * np.cos(6 * theta)).to_voxels(), P.PITAYA),
        (Sphere(LocalFrame((150, 0, 0)),
                radius=lambda phi, theta: 40 - 10 * np.cos(6 * theta) + 30 * np.cos(2 * phi)
                ).to_voxels(), P.WARNING),
    ]


def build_cylinder():
    return [
        (Cylinder(LocalFrame((-50, 0, 0)), 60, 40).to_voxels(), P.BLUE),
        (Cylinder(LocalFrame((50, 0, 0)), 60, radius=lambda phi, lr: _line1(lr)).to_voxels(),
         P.GREEN),
        (Cylinder(frames=_spine(), radius=_surf1).to_voxels(), P.YELLOW),
    ]


def build_ring():
    return [
        (Ring(LocalFrame((-50, -50, 0)), 30, 8).to_voxels(), P.FROZEN),
        (Ring(LocalFrame((-50, 50, 0)), 30,
              radius=lambda phi, alpha: 10 - 2 * np.cos(5 * phi)).to_voxels(), P.PITAYA),
        (Ring(LocalFrame((50, 50, 0)), 30,
              radius=lambda phi, alpha: 10 + 3 * np.cos(5 * alpha)).to_voxels(), P.WARNING),
        (Ring(LocalFrame((50, -50, 0)), 30,
              radius=lambda phi, alpha: 10 - 2 * np.cos(5 * (phi + alpha)) + 3 * np.cos(5 * alpha)
              ).to_voxels(), P.BLUEBERRY),
    ]


def build_lens():
    def surf(phi, rr):
        return 12.0 + 3.0 * np.cos(5.0 * phi)
    return [
        (Lens(LocalFrame((-50, -50, 0)), 10, 10, 40).to_voxels(), P.FROZEN),
        (Lens(LocalFrame((50, 50, 0)), 10, 10, 40,
              lower=lambda phi, rr: 5 - surf(phi, rr),
              upper=lambda phi, rr: 5 + surf(phi, rr)).to_voxels(), P.PITAYA),
        (Lens(LocalFrame((-50, 50, 0)), 10, 10, 40,
              lower=lambda phi, rr: 5 - surf(phi, rr),
              upper=lambda phi, rr: 5 + np.cos(6 * (phi + 0.3 * np.pi * rr)) + 3 * np.cos(20 * rr)
              ).to_voxels(), P.WARNING),
    ]


def build_pipe():
    def trafo(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        return np.column_stack([y + 0.2 * z - 50, 0.5 * z + 50, 0.5 * x])
    return [
        (Pipe(LocalFrame((-50, 0, 0)), 60, 10, 20).to_voxels(), P.BLUE),
        (Pipe(LocalFrame((0, 0, 0)), 60, 10, 20, transform=trafo).to_voxels(), P.GREEN),
        (Pipe(LocalFrame((50, -50, 0)), 60, inner_radius=6,
              outer_radius=lambda phi, lr: _line1(lr)).to_voxels(), P.LEMONGRASS),
        (Pipe(frames=_spine(), inner_radius=_surf3, outer_radius=_surf1).to_voxels(), P.ORCHID),
    ]


def build_pipe_segment():
    return [
        (PipeSegment(LocalFrame((-50, 0, 0)), 60, 20, 40,
                     start=np.pi, end=0.5 * np.pi, method="mid_range").to_voxels(), P.BLUE),
        (PipeSegment(LocalFrame((50, 0, 0)), 60, inner_radius=_surf3, outer_radius=_surf1,
                     start=lambda lr: -np.pi * lr, end=1.75 * np.pi,
                     method="mid_range").to_voxels(), P.RUBY),
        (PipeSegment(frames=_spine(), inner_radius=_surf3, outer_radius=_surf1,
                     start=lambda lr: 4 * np.pi * lr, end=1.5 * np.pi,
                     method="mid_range").to_voxels(), P.RACING_GREEN),
    ]


def build_basic_lattices():
    lat = functions.lat_from_point((1, 5, -10), 5)
    lat.add_beam((5, 3, 0), (-3, 0, 7), 1, 3, True)
    return [(lat.to_voxels(), P.BLUEBERRY)]


def build_lattice_pipe():
    return [
        (LatticePipe(LocalFrame((-50, 0, 0)), 60, 10).to_voxels(), P.YELLOW),
        (LatticePipe(LocalFrame((50, -50, 0)), 60, radius=_line1).to_voxels(), P.FROZEN),
        (LatticePipe(frames=_spine(), radius=_line1).to_voxels(), P.RACING_GREEN),
    ]


def build_lattice_manifold():
    return [
        (LatticeManifold(LocalFrame((-50, 0, 0), local_z=(0, 1, 0)), 50, 5, 45).to_voxels(),
         P.YELLOW),
        (LatticeManifold(LocalFrame((0, 0, 0), local_z=(0, 1, 0)), 50, 10, 30,
                         extend_both_sides=True).to_voxels(), P.CRYSTAL),
        (LatticeManifold(LocalFrame((50, 0, 0), local_z=(0, 1, 0)), 50, 5, 60,
                         extend_both_sides=True).to_voxels(), P.GREEN),
    ]


def build_gyroid_sphere():
    vox = ImplicitSphere((0, 0, 0), 10).render(((-12, -12, -12), (12, 12, 12)))
    gyroid = vox.copy().intersect_implicit_(ImplicitGyroid(3, 1))
    return [(gyroid, P.BILLIE)]


def build_gyroid_genus():
    from picopie import Voxels
    s = 8.0   # the genus lives in ~unit space -> scale it up so the surface resolves cleanly
    genus = ImplicitGenus(0.0)
    vox = Voxels().render_implicit_(lambda x, y, z: genus(x / s, y / s, z / s),
                                    ((-3 * s, -3 * s, -1.6 * s), (3 * s, 3 * s, 1.6 * s)))
    gyroid = vox.copy().intersect_implicit_(ImplicitGyroid(6, 0.6))
    return [(gyroid, P.LAVENDER)]


def build_superellipsoid():
    # scaled up from the unit examples so the surface resolves cleanly (no dimpling)
    a = 16.0
    specs = [((-45, 0, 0), 3.0, 0.25, P.RUBY),
             ((0, 0, 0), 1.5, 1.5, P.BLUE),
             ((45, 0, 0), 0.25, 0.25, P.BUBBLEGUM)]
    scene = []
    for centre, e1, e2, rgb in specs:
        vox = ImplicitSuperEllipsoid((0, 0, 0), a, a, a, e1, e2).render(((-a, -a, -a), (a, a, a)))
        placed = mesh_utils.vox_apply_transformation(
            vox, lambda p, c=centre: p + np.array(c, dtype=float))
        scene.append((placed, rgb))
    return scene


def build_mesh_painter():
    mesh = Sphere(radius=40).to_mesh()
    scale = ColorScale3D(rainbow_spectrum(), 0.0, 90.0)
    return painter.split_by_overhang_angle(mesh, scale, n_classes=50)


def build_mesh_trafo():
    def rot45(p):
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        return np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]])
    box = Box(LocalFrame((0, 0, 0)), 40, 30, 20).to_voxels()
    rotated = mesh_utils.vox_apply_transformation(box, rot45)   # mesh -> rotate -> voxelise
    return [(box, P.GRAY), (rotated, P.ORCHID)]


def build_over_offset():
    box = (Box(LocalFrame((-15, 0, 0)), 30, 30, 30).to_voxels()
           + Box(LocalFrame((15, 0, 0)), 30, 30, 30).to_voxels())
    return [(box.copy().double_offset_(4.0, -4.0), P.BUBBLEGUM)]


SCENES = {
    "box": build_box, "sphere": build_sphere, "cylinder": build_cylinder,
    "ring": build_ring, "lens": build_lens, "pipe": build_pipe,
    "pipe_segment": build_pipe_segment, "basic_lattices": build_basic_lattices,
    "lattice_pipe": build_lattice_pipe, "lattice_manifold": build_lattice_manifold,
    "gyroid_sphere": build_gyroid_sphere, "gyroid_genus": build_gyroid_genus,
    "superellipsoid": build_superellipsoid, "mesh_painter": build_mesh_painter,
    "mesh_trafo": build_mesh_trafo, "over_offset": build_over_offset,
}


def render_scene(scene, path, size=(1280, 960), background=(0.16, 0.16, 0.2)):
    """Render a ``[(object, rgb), ...]`` scene to a PNG (needs a display)."""
    from picopie import Viewer
    with Viewer(size=size) as v:
        for i, (obj, rgb) in enumerate(scene):
            v.add(obj, group=i)
            v.set_group_material(i, rgb)
        v.set_background(background)
        v.screenshot(path)
    return path


def main(argv) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default="examples/_out/gallery",
                        help="output directory for the PNGs (default: examples/_out/gallery)")
    parser.add_argument("--voxel-size", type=float, default=0.2, metavar="MM",
                        help="kernel voxel size in mm; smaller = smoother but slower "
                             "(default: 0.2)")
    parser.add_argument("--show", metavar="SCENE", choices=sorted(SCENES),
                        help="open one scene interactively instead of rendering PNGs")
    args = parser.parse_args(argv)

    picopie.init(voxel_size_mm=args.voxel_size)

    if args.show:
        from picopie import Viewer
        with Viewer() as v:
            for i, (obj, rgb) in enumerate(SCENES[args.show]()):
                v.add(obj, group=i)
                v.set_group_material(i, rgb)
            v.run()
        return 0

    import os
    os.makedirs(args.out, exist_ok=True)
    for name, build in SCENES.items():
        path = os.path.join(args.out, f"{name}.png")
        render_scene(build(), path)
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
