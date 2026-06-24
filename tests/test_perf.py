"""Phase 9: performance-regression guard for the compiled ``_fastloop`` path.

Two kinds of check:

* ``test_fastloop_available`` -- a binary, never-flaky assert that the compiled
  extension is actually loaded. This catches the most important regression: a
  build that silently ships no fast loop and falls back to pure Python.
* the ``perf``-marked tests -- compare the fast path against the pure-Python
  fallback *in the same process* and assert a *relative* speedup. Ratios are
  stable across machines (both halves run on the same CPU under the same load),
  unlike absolute wall-clock thresholds. The real speedup is ~13-21x; we assert
  only >= 2x so the check never flakes. They are excluded from CI (which can be
  noisy) via ``-m "not perf"`` but run by default locally.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from picogk import ScalarField, Voxels, _fast

# generous floor vs the observed ~13-21x -- catches a real regression, never flakes
_MIN_SPEEDUP = 2.0


def test_fastloop_available():
    # If this fails, the wheel/sdist built without the compiled extension and
    # every bulk op silently uses the slow Python path.
    assert _fast.available(), "_fastloop extension not built -- bulk ops fall back to Python"


def _best(fn, repeats: int) -> float:
    fn()                                         # warmup
    return min(_one(fn) for _ in range(repeats))


def _one(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _fast_vs_fallback(fn, fast_repeats: int, slow_repeats: int) -> float:
    """Return fallback_time / fast_time for the same operation."""
    assert _fast.lib is not None
    fast = _best(fn, fast_repeats)
    saved = _fast.lib
    _fast.lib = None                             # force the pure-Python branch
    try:
        slow = _best(fn, slow_repeats)
    finally:
        _fast.lib = saved
    return slow / fast if fast > 0 else float("inf")


@pytest.mark.perf
def test_mesh_read_fast_beats_fallback():
    mesh = Voxels.sphere(radius=20.0).to_mesh()
    assert mesh.vertex_count() > 5000          # ensure a workload worth measuring
    ratio = _fast_vs_fallback(lambda: mesh.vertices, fast_repeats=5, slow_repeats=3)
    assert ratio >= _MIN_SPEEDUP, f"mesh.vertices fast path only {ratio:.1f}x fallback"


@pytest.mark.perf
def test_scalar_field_bulk_fast_beats_fallback():
    part = Voxels.sphere(radius=20.0)
    field = ScalarField.from_voxels(part)
    rng = np.random.default_rng(0)
    pos = (rng.random((50_000, 3), dtype=np.float32) - 0.5) * 30.0
    vals = rng.random(50_000, dtype=np.float32)

    set_ratio = _fast_vs_fallback(lambda: field.set_many(pos, vals),
                                  fast_repeats=3, slow_repeats=1)
    assert set_ratio >= _MIN_SPEEDUP, f"set_many fast path only {set_ratio:.1f}x fallback"

    get_ratio = _fast_vs_fallback(lambda: field.get_many(pos),
                                  fast_repeats=3, slow_repeats=1)
    assert get_ratio >= _MIN_SPEEDUP, f"get_many fast path only {get_ratio:.1f}x fallback"
