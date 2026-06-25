#!/usr/bin/env python3
"""Deep, subprocess-isolated fuzz campaign for the never-abort guard (Phase 11b).

Each worker runs a long sequence of randomized API calls (including the
abort-prone stateful chains -- repeated implicit intersects, booleans on
degenerate grids, NaN/inf everywhere) in its own process. With the Phase-11a
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
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


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


def _bpt(rng: random.Random) -> tuple[float, float, float]:
    # render bbox corner kept tiny (the SDF callback runs per voxel, in Python)
    def f() -> float:
        return rng.choice([float("nan"), rng.uniform(-5, 5)])
    return (f(), f(), f())


def _qpt(rng: random.Random) -> tuple[float, float, float]:
    # query point: bounded. A far *finite* point makes closest_point/ray_cast
    # search/march unboundedly (a hang, not an abort -- out of scope here).
    return (rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50))


def worker(seed: int, n: int) -> int:
    import numpy as np

    import picogk
    from picogk import Mesh, Metadata, ScalarField, VdbFile, Voxels

    rng = random.Random(seed)
    picogk.init(voxel_size_mm=0.5)

    # Stateless ops: each builds FRESH operands so a grid can't grow unboundedly
    # across the sequence (that would be slow, not an abort). Abort-prone *chains*
    # are exercised within a single op.
    def base() -> Voxels:
        return Voxels.sphere(radius=5.0)

    def op_new(_):
        Voxels.sphere(center=_spt(rng), radius=_size(rng))

    def op_capsule(_):
        Voxels.capsule(_spt(rng), _spt(rng), _size(rng), _size(rng))

    def op_offset(_):
        base().offset_(_size(rng))

    def op_double_offset(_):
        base().double_offset_(_size(rng), _size(rng))

    def op_shell(_):
        base().shell_(_size(rng))

    def op_bool(_):
        getattr(base(), rng.choice(["bool_add_", "bool_subtract_", "bool_intersect_"]))(base())

    def op_render(_):
        r = _weird(rng)
        Voxels().render_implicit_(lambda x, y, z: r, (_bpt(rng), _bpt(rng)))

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

    def op_query(_):
        v = base()
        v.is_inside(_qpt(rng))
        v.closest_point(_qpt(rng))
        v.ray_cast(_qpt(rng), (1.0, 0.0, 0.0))   # fixed dir: a degenerate one would hang, not abort

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

    ops = [op_new, op_capsule, op_offset, op_double_offset, op_shell, op_bool, op_render,
           op_intersect_chain, op_mesh, op_field, op_query, op_slice, op_meta, op_vdb]

    for i in range(n):
        op = rng.choice(ops)
        sys.stderr.write(f"[{seed}:{i}] {op.__name__}\n")
        sys.stderr.flush()
        with contextlib.suppress(Exception):
            op(None)
    print(f"WORKER_OK {n}")
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
