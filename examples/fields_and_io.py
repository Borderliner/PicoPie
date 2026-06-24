#!/usr/bin/env python3
"""Phase 1 demo: fields, metadata, VDB persistence, and mesh import.

    PYTHONPATH=src python examples/fields_and_io.py
"""
from pathlib import Path

import picogk
from picogk import Mesh, Metadata, ScalarField, Voxels, load_vdb, save_vdb

OUT = Path(__file__).parent / "_out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    picogk.init(voxel_size_mm=0.3)
    print(f"PicoGK {picogk.version()}")

    # --- build a part, attach a scalar field + metadata ---
    part = Voxels.sphere(radius=10) - Voxels.sphere(center=(6, 0, 0), radius=6)

    heat = ScalarField.from_voxels(part)
    heat.set((0, 0, 0), 100.0)             # a hot spot at the origin

    md = Metadata.from_voxels(part)
    md["material"] = "Ti-6Al-4V"
    md["wall_mm"] = 1.0
    md["build_dir"] = (0, 0, 1)
    print(f"   metadata: {md.to_dict()}")

    # --- persist everything to a single .vdb, then reload ---
    vdb_path = OUT / "part_with_fields.vdb"
    save_vdb(str(vdb_path), body=part, heat=heat)
    print(f"   -> {vdb_path.name}")

    objs = load_vdb(str(vdb_path))
    print(f"   reloaded: { {k: type(v).__name__ for k, v in objs.items()} }")
    print(f"   volume orig/loaded: {part.volume_mm3():.1f} / "
          f"{objs['body'].volume_mm3():.1f} mm^3")

    # --- export a mesh, re-import it, and re-voxelize (round trip) ---
    stl = OUT / "part.stl"
    objs["body"].to_mesh().save_stl(str(stl))
    reimported = Mesh.load_stl(str(stl))
    print(f"   STL round trip: {reimported.vertex_count()} verts, "
          f"{reimported.triangle_count()} tris")

    revox = Voxels.from_mesh(reimported)   # voxelize the imported mesh
    revox.offset_(0.5)                      # thicken it 0.5 mm
    revox.to_mesh().save_stl(str(OUT / "part_reimported_offset.stl"))
    print(f"   re-voxelized + offset volume: {revox.volume_mm3():.1f} mm^3")
    print("done.")


if __name__ == "__main__":
    main()
