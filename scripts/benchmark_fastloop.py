#!/usr/bin/env python3
"""Benchmark the compiled fast path vs the pure-Python transfer loops.

    PYTHONPATH=src python scripts/benchmark_fastloop.py
"""
import time

import numpy as np

import picogk
from picogk import Mesh, ScalarField, Voxels, _fast


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, (time.perf_counter() - t) * 1000.0


def main() -> None:
    print(f"compiled _fastloop available: {_fast.available()}\n")
    picogk.init(voxel_size_mm=0.15)
    mesh = Voxels.sphere(radius=15).to_mesh()
    nv, nt = mesh.vertex_count(), mesh.triangle_count()
    print(f"benchmark mesh: {nv:,} vertices  {nt:,} triangles\n")

    print(f"{'operation':<26}{'pure (ms)':>12}{'fast (ms)':>12}{'speedup':>10}")
    print("-" * 60)

    _, sp = timed(lambda: mesh._vertices_py(nv))
    verts, fp = timed(lambda: mesh.vertices)
    print(f"{'read vertices':<26}{sp:>12.1f}{fp:>12.1f}{sp / fp:>9.1f}x")

    _, sp = timed(lambda: mesh._triangles_py(nt))
    tris, fp = timed(lambda: mesh.triangles)
    print(f"{'read triangles':<26}{sp:>12.1f}{fp:>12.1f}{sp / fp:>9.1f}x")

    # build: temporarily disable fast path to time the pure loop
    saved = _fast.lib
    try:
        _fast.lib = None
        _, sp = timed(lambda: Mesh.from_arrays(verts, tris))
    finally:
        _fast.lib = saved
    _, fp = timed(lambda: Mesh.from_arrays(verts, tris))
    print(f"{'build from_arrays':<26}{sp:>12.1f}{fp:>12.1f}{sp / fp:>9.1f}x")

    # field bulk set/get on distinct voxels
    g = np.mgrid[0:40, 0:40, 0:10].reshape(3, -1).T.astype(np.float32) * 0.5
    vals = g.sum(axis=1).astype(np.float32)
    sf = ScalarField()
    saved = _fast.lib
    try:
        _fast.lib = None
        _, sp = timed(lambda: ScalarField().set_many(g, vals))
    finally:
        _fast.lib = saved
    _, fp = timed(lambda: sf.set_many(g, vals))
    print(f"{'scalar set_many (%d)' % len(g):<26}{sp:>12.1f}{fp:>12.1f}{sp / fp:>9.1f}x")


if __name__ == "__main__":
    main()
