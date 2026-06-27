#!/usr/bin/env python3
"""Deep, subprocess-isolated fuzz campaign for the never-abort guard (Phase 11b).

Each worker runs a long sequence of randomized API calls (including the
abort-prone stateful chains -- repeated implicit intersects, booleans on
degenerate grids, NaN/inf everywhere) in its own process, each at a different
voxel size drawn from a fine->coarse sweep (so resolution-dependent native bugs,
like the narrow-band truncations, are exercised too). With the Phase-11a
guard, native errors are caught as exceptions and the worker finishes with
``WORKER_OK``; if any input slips past the guard the worker is killed by the
native abort (SIGABRT / 0xC0000409) and the driver reports the exact seed +
op index + the last operation it logged.

    python scripts/fuzz_abort.py                 # default 8 workers x 600 ops
    python scripts/fuzz_abort.py 16 1000         # 16 workers x 1000 ops

Exit code is nonzero if any worker aborted.
"""
from __future__ import annotations

import contextlib
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Per-worker voxel size (fine->coarse). Resolution-dependent native bugs only
# fire at fine sizes; a fixed-0.5 mm campaign never reached them.
VOXEL_SWEEP = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def _weird(rng: random.Random) -> float:
    return rng.choice([
        float("nan"), float("inf"), -float("inf"), 1e30, -1e30, 0.0,
        rng.uniform(-50, 50), rng.uniform(-1e6, 1e6), rng.uniform(-1e-6, 1e-6),
    ])


def _pt(rng: random.Random) -> tuple[float, float, float]:
    return (_weird(rng), _weird(rng), _weird(rng))


def _size(rng: random.Random) -> float:
    # grid-sizing magnitude: bounded (NaN/0/negative still in play, no huge/inf)
    return rng.choice([float("nan"), 0.0, rng.uniform(-12, 12), rng.uniform(-3, 3)])


def _spt(rng: random.Random) -> tuple[float, float, float]:
    return (_size(rng), _size(rng), _size(rng))


def worker(seed: int, n: int) -> int:
    import numpy as np

    import picogk
    from picogk import Mesh, Metadata, ScalarField, VdbFile, Voxels

    rng = random.Random(seed)
    vs = VOXEL_SWEEP[seed % len(VOXEL_SWEEP)]
    picogk.init(voxel_size_mm=vs)

    # Keep voxel counts bounded at fine resolutions (radius ~ proportional to the
    # voxel size); NaN/inf/0 radii still flow through to exercise the guards.
    base_r = max(1.0, min(5.0, 10.0 * vs))
    rmax = 2.0 * base_r
    dmax = max(0.5, min(12.0, 20.0 * vs))      # offset/extent magnitude
    bbox_h = max(0.4, min(8.0, 8.0 * vs))      # render bbox half-extent
    # The finite magnitudes above are capped relative to the voxel size so an op's
    # cost (voxel-offset distance, per-voxel SDF callbacks) stays bounded at fine
    # resolutions; NaN/inf/0 still pass through to keep exercising the guards.

    def _bounded(hi: float) -> float:
        v = _size(rng)
        return max(-hi, min(hi, v)) if math.isfinite(v) else v

    def _radius() -> float:
        return _bounded(rmax)

    def _dist() -> float:
        return _bounded(dmax)

    # Stateless ops: each builds FRESH operands so a grid can't grow unboundedly
    # across the sequence (that would be slow, not an abort). Abort-prone *chains*
    # are exercised within a single op.
    def base() -> Voxels:
        return Voxels.sphere(radius=base_r)

    def op_new(_):
        Voxels.sphere(center=_spt(rng), radius=_radius())

    def op_capsule(_):
        Voxels.capsule(_spt(rng), _spt(rng), _radius(), _radius())

    def op_offset(_):
        base().offset_(_dist())

    def op_double_offset(_):
        base().double_offset_(_dist(), _dist())

    def op_shell(_):
        base().shell_(_dist())

    def op_bool(_):
        getattr(base(), rng.choice(["bool_add_", "bool_subtract_", "bool_intersect_"]))(base())

    def _bcorner() -> float:
        return rng.choice([float("nan"), rng.uniform(-bbox_h, bbox_h)])

    def op_render(_):
        r = _weird(rng)
        Voxels().render_implicit_(
            lambda x, y, z: r,
            ((_bcorner(), _bcorner(), _bcorner()), (_bcorner(), _bcorner(), _bcorner())))

    def op_intersect_chain(_):
        v = base()
        v.intersect_implicit_(lambda x, y, z: _weird(rng))
        v.intersect_implicit_(lambda x, y, z: _weird(rng))   # the abort-prone repeat

    def op_mesh(_):
        verts = np.array([_spt(rng) for _ in range(rng.randint(0, 8))] or [(0, 0, 0)], np.float32)
        ntri = rng.randint(0, 8)
        tris = np.array([[rng.randint(-3, 12) for _ in range(3)] for _ in range(ntri)]
                        or [(0, 0, 0)], np.int32)
        Voxels.from_mesh(Mesh.from_arrays(verts, tris))

    def op_field(_):
        f = ScalarField.from_voxels(base())
        arr = np.array([_pt(rng) for _ in range(rng.randint(0, 32))] or [(0, 0, 0)], np.float32)
        f.set_many(arr, np.full(len(arr), _weird(rng), np.float32))
        f.get_many(arr)

    def _qry() -> tuple[float, float, float]:
        # near the (small) grid: a far point makes closest_point/ray_cast march
        # ~distance/voxel_size steps, which explodes at fine resolutions.
        return (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))

    def op_query(_):
        v = base()
        v.is_inside(_qry())
        v.closest_point(_qry())
        v.ray_cast(_qry(), (1.0, 0.0, 0.0))      # fixed dir: a degenerate one would hang, not abort

    def op_slice(_):
        base().slice_z(rng.randint(-1000, 1000))

    def op_meta(_):
        md = Metadata.from_voxels(base())
        md.set_float("k", _weird(rng))
        md.get("k")

    def op_vdb(_):
        f = VdbFile()
        f.add_voxels("v", base())
        f.get(rng.randint(-50, 50))

    def op_triple_offset(_):
        base().triple_offset_(_dist())

    def op_project_z(_):
        base().project_z_slice_(_dist(), _dist())

    def op_mesh_shell(_):
        Voxels.mesh_shell(base().to_mesh(), _dist())

    ops = [op_new, op_capsule, op_offset, op_double_offset, op_shell, op_bool, op_render,
           op_intersect_chain, op_mesh, op_field, op_query, op_slice, op_meta, op_vdb,
           op_triple_offset, op_project_z, op_mesh_shell]

    for i in range(n):
        op = rng.choice(ops)
        sys.stderr.write(f"[{seed}:{i}|vs={vs}] {op.__name__}\n")
        sys.stderr.flush()
        with contextlib.suppress(Exception):
            op(None)
    print(f"WORKER_OK {n} voxel={vs}")
    return 0


def _run_worker(seed: int, ops: int):
    return seed, subprocess.run([sys.executable, __file__, "--worker", str(seed), str(ops)],
                                capture_output=True, text=True)


def driver(workers: int, ops: int) -> int:
    print(f"fuzz campaign: {workers} workers x {ops} ops = {workers * ops} operations "
          f"(parallel x{min(workers, os.cpu_count() or 1)})\n")
    aborts = 0
    total = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = sorted(pool.map(lambda s: _run_worker(s, ops), range(workers)))
    for seed, r in results:
        if r.returncode == 0 and "WORKER_OK" in r.stdout:
            total += ops
            print(f"  worker {seed:2}: OK ({ops} ops)")
        else:
            aborts += 1
            last = (r.stderr.strip().splitlines() or ["<no log>"])[-1]
            print(f"  worker {seed:2}: ABORTED rc={r.returncode}  last op: {last}")
    print(f"\n{total} operations survived, {aborts} worker(s) aborted.")
    return 1 if aborts else 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--worker":
        return worker(int(argv[2]), int(argv[3]))
    workers = int(argv[1]) if len(argv) > 1 else 8
    ops = int(argv[2]) if len(argv) > 2 else 600
    return driver(workers, ops)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
