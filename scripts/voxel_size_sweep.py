#!/usr/bin/env python3
"""Empirical voxel-size sweep: exercise every native op across fine->coarse voxel
sizes and report which (op, voxel_size) combinations fail, abort, go empty, or
produce a wildly inconsistent measurement.

This is the method that found the IntersectImplicit narrow-band bug: it only broke
below ~0.33 mm, but the test suite runs only at 0.5 mm. Run with the *fixed*
runtime to confirm green; run patterns here mirror real usage.

    python scripts/voxel_size_sweep.py
"""
from __future__ import annotations

import math
import traceback

import picogk
from picogk import Voxels
from picogk.shapes import (
    Box,
    Cylinder,
    ImplicitGyroid,
    ImplicitSphere,
    LocalFrame,
    Sphere,
    functions,
)

VOXEL_SIZES = [0.1, 0.2, 0.5, 1.0, 2.0]
SPHERE_VOL = 4.0 / 3.0 * math.pi * 10**3   # r=10 -> ~4188.79 mm^3

results: dict[str, dict[float, str]] = {}


def run(op: str, vs: float, thunk):
    """Run thunk; classify OK / EMPTY / RAISED / value. Returns the result or None."""
    try:
        val = thunk()
        results.setdefault(op, {})[vs] = "OK" if val is None else str(val)
        return val
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:80]
        results.setdefault(op, {})[vs] = f"RAISE {msg}"
        return None


def vol(v: Voxels) -> float:
    return v.calculate_properties()[0]


def nonempty(v: Voxels) -> str:
    vv = vol(v)
    return f"vol={vv:.0f}" if vv > 0 else "EMPTY"


def sphere(r=10.0, c=(0, 0, 0)) -> Voxels:
    return Voxels.sphere(center=c, radius=r)


def sweep_one(vs: float):
    picogk.shutdown()
    picogk.init(voxel_size_mm=vs)

    # --- primitives + measurement ---
    run("sphere+volume", vs, lambda: nonempty(sphere()))
    run("sphere->mesh", vs, lambda: f"tris={sphere().to_mesh().triangle_count()}")
    run("mesh->voxels", vs, lambda: nonempty(Voxels.from_mesh(sphere().to_mesh())))

    # --- booleans ---
    run("bool_add", vs, lambda: nonempty(sphere(c=(-5, 0, 0)) + sphere(c=(5, 0, 0))))
    run("bool_sub", vs, lambda: nonempty(sphere() - sphere(c=(8, 0, 0))))

    run("bool_intersect", vs, lambda: nonempty(sphere() & sphere(c=(8, 0, 0))))

    # --- offsets (mm-valued -> the same family as the bug) ---
    def _offset(d):
        v = sphere()
        v.offset_(d)
        return nonempty(v)
    run("offset_+1mm", vs, lambda: _offset(1.0))
    run("offset_-1mm", vs, lambda: _offset(-1.0))
    # sub-voxel offset: smaller than one voxel at coarse sizes
    run("offset_+0.2mm", vs, lambda: _offset(0.2))

    def _double():
        v = sphere()
        v.double_offset_(2.0, -2.0)
        return nonempty(v)
    run("double_offset_", vs, _double)

    def _triple():
        v = sphere()
        v.triple_offset_(1.0)
        return nonempty(v)
    run("triple_offset_", vs, _triple)

    def _shell():
        return nonempty(Voxels.mesh_shell(sphere().to_mesh(), 1.0))
    run("mesh_shell_1mm", vs, _shell)
    run("mesh_shell_0.2mm", vs, lambda: nonempty(Voxels.mesh_shell(sphere().to_mesh(), 0.2)))

    def _project():
        v = sphere()
        v.project_z_slice_(-3.0, 3.0)
        return nonempty(v)
    run("project_z_slice_", vs, _project)

    # --- implicit render + intersect (the bug's family) ---
    run("render_implicit", vs, lambda: nonempty(
        Voxels().render_implicit_(
            lambda x, y, z: (x * x + y * y + z * z) ** 0.5 - 10.0,
            ((-12, -12, -12), (12, 12, 12)))))

    def _intersect_impl():
        v = sphere()
        v.intersect_implicit_(lambda x, y, z: x)   # keep x<=0 hemisphere
        return nonempty(v)
    run("intersect_implicit_plane", vs, _intersect_impl)

    def _gyroid():
        v = ImplicitSphere((0, 0, 0), 10).render(((-12, -12, -12), (12, 12, 12)))
        v.intersect_implicit_(ImplicitGyroid(3, 1))
        return nonempty(v)
    run("implicit_sphere&gyroid", vs, _gyroid)

    # --- lattices ---
    def _lat():
        lat = functions.lat_from_point((0, 0, 0), 5)
        lat.add_beam((-8, 0, 0), (8, 0, 0), 2, 3, True)
        return nonempty(lat.to_voxels())
    run("lattice_beam", vs, _lat)

    # --- shapes layer ---
    run("Sphere.to_voxels", vs, lambda: nonempty(Sphere(radius=10).to_voxels()))
    run("Box.to_voxels", vs, lambda: nonempty(Box(LocalFrame((0, 0, 0)), 20, 15, 10).to_voxels()))
    run("Cylinder.to_voxels", vs, lambda: nonempty(
        Cylinder(LocalFrame((0, 0, 0)), 20, 8).to_voxels()))

    # --- fields ---
    def _scalar():
        from picogk import ScalarField
        sf = ScalarField.from_voxels(sphere())
        return f"ok={sf is not None}"
    run("ScalarField.from_voxels", vs, _scalar)


def main() -> int:
    for vs in VOXEL_SIZES:
        try:
            sweep_one(vs)
        except Exception:
            print(f"!! sweep crashed at voxel_size={vs}")
            traceback.print_exc()

    # --- print matrix ---
    hdr = "op".ljust(26) + "".join(f"{vs:>11}" for vs in VOXEL_SIZES)
    print(hdr)
    print("-" * len(hdr))
    flagged = []
    for op in results:
        row = op.ljust(26)
        for vs in VOXEL_SIZES:
            cell = results[op].get(vs, "-")
            short = cell if cell.startswith(("OK", "vol", "tris", "ok", "EMPTY", "-")) else "RAISE"
            row += f"{short:>11}"
            if cell.startswith("RAISE") or cell == "EMPTY":
                flagged.append((op, vs, cell))
        print(row)

    print("\n=== FLAGGED (raised / empty) ===")
    if not flagged:
        print("none — every op succeeded non-empty at every voxel size")
    for op, vs, cell in flagged:
        print(f"  [{vs} mm] {op}: {cell}")

    # --- volume consistency for the sphere (should be ~SPHERE_VOL, looser at coarse) ---
    print("\n=== sphere volume vs voxel size (expect ~4189 mm^3) ===")
    for vs in VOXEL_SIZES:
        cell = results.get("sphere+volume", {}).get(vs, "-")
        print(f"  {vs:>5} mm: {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
