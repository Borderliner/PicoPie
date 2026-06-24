#!/usr/bin/env python3
"""Phase 3 demo: headless visualization (slice PNGs + 3D mesh preview).

Needs the viz extras:  pip install 'picopie[viz]'

    PYTHONPATH=src python examples/visualize.py
"""
from pathlib import Path

import picogk
from picogk import Voxels, mesh_preview, save_slice_png, save_slice_sheet

OUT = Path(__file__).parent / "_out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    picogk.init(voxel_size_mm=0.3)

    # a hollow shelled part with a through-hole
    part = Voxels.sphere(radius=12) - Voxels.sphere(center=(7, 0, 0), radius=7)
    part.shell_(1.5)
    _, size = part.voxel_dimensions()
    print(f"part voxel size: {size.tolist()}")

    # 1) a single mid Z slice as a signed-distance heatmap
    p1 = save_slice_png(part, str(OUT / "slice_sdf.png"), axis="z", mode="sdf")
    print(f"   -> {Path(p1).name}  (signed-distance heatmap)")

    # 2) a montage of inside/outside masks through the part
    p2 = save_slice_sheet(part, str(OUT / "slice_sheet.png"),
                          axis="z", count=16, cols=4, mode="mask")
    print(f"   -> {Path(p2).name}  (16-slice mask montage)")

    # 3) a 3D preview of the meshed surface
    p3 = OUT / "mesh_preview.png"
    mesh_preview(part.to_mesh(), str(p3), elev=22, azim=-55)
    print(f"   -> {p3.name}  (3D mesh render)")
    print("done.")


if __name__ == "__main__":
    main()
