#!/usr/bin/env python3
"""End-to-end PicoPie demo: primitives, booleans, offsets, a lattice, and STL.

Run from the repo root (so the dev runtime under native/ is found)::

    PYTHONPATH=src python examples/hello_picogk.py
"""
import math
from pathlib import Path

import picogk
from picogk import Lattice, Voxels

OUT = Path(__file__).parent / "_out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    picogk.init(voxel_size_mm=0.2)
    print(f"PicoGK {picogk.version()}  ({picogk.name()})")

    # 1) a sphere with a smaller sphere subtracted, then hollowed to a shell
    body = Voxels.sphere(radius=10)
    hole = Voxels.sphere(center=(6, 0, 0), radius=6)
    part = body - hole
    print(f"   sphere-minus-sphere volume: {part.volume_mm3():.1f} mm^3")
    part.shell_(1.0)
    mesh = part.to_mesh()
    mesh.save_stl(str(OUT / "shelled_part.stl"))
    print(f"   -> shelled_part.stl  ({mesh.triangle_count()} triangles, "
          f"bbox size {mesh.bounding_box().size.round(2).tolist()} mm)")

    # 2) a beam-and-node lattice voxelized into a solid
    lat = Lattice()
    corners = [(x, y, z) for x in (-8, 8) for y in (-8, 8) for z in (-8, 8)]
    for c in corners:
        lat.add_sphere(c, 1.5)
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            # connect only edges of the cube (differ in exactly one axis)
            if sum(1 for p, q in zip(a, b) if p != q) == 1:
                lat.add_beam(a, b, 1.0)
    cage = lat.to_voxels()
    cage_mesh = cage.to_mesh()
    cage_mesh.save_stl(str(OUT / "lattice_cage.stl"))
    print(f"   -> lattice_cage.stl  (volume {cage.volume_mm3():.1f} mm^3, "
          f"{cage_mesh.triangle_count()} triangles)")

    # 3) a TPMS gyroid clipped to a sphere, built from a single implicit SDF.
    #    Composing the clip INSIDE the SDF (max of the two fields) keeps the
    #    result a valid level set -- the robust alternative to a boolean CSG.
    #    NOTE: the SDF runs once per voxel from native code, so this pure-Python
    #    callback is the slow path; fine for a demo.
    print("   building gyroid (implicit, per-voxel Python callback -- slow)...")
    k = 2 * math.pi / 6.0  # ~6 mm period
    r, wall = 10.0, 0.4

    def gyroid_in_sphere(x, y, z):
        g = (math.sin(k * x) * math.cos(k * y)
             + math.sin(k * y) * math.cos(k * z)
             + math.sin(k * z) * math.cos(k * x))
        sphere = math.sqrt(x * x + y * y + z * z) - r   # <=0 inside
        return max(abs(g) - wall, sphere)               # thin walls AND inside

    pad = 2.0
    tpms = Voxels()
    tpms.render_implicit_(gyroid_in_sphere,
                          ((-r - pad, -r - pad, -r - pad), (r + pad, r + pad, r + pad)))
    tpms_mesh = tpms.to_mesh()
    tpms_mesh.save_stl(str(OUT / "gyroid_sphere.stl"))
    print(f"   -> gyroid_sphere.stl  (volume {tpms.volume_mm3():.1f} mm^3, "
          f"{tpms_mesh.triangle_count()} triangles)")

    print(f"\nnative memory in use: {picogk.total_memory_bytes()/1e6:.1f} MB")
    print("done.")


if __name__ == "__main__":
    main()
