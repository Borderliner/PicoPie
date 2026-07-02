#!/usr/bin/env python3
"""Perf trend tracking (Phase 11e): measure the fast-path speedups, append them
to a ledger, and flag regressions vs the last like-for-like baseline.

Phase 9 asserts the compiled fast path *exists* and clears a coarse 2x floor;
this tracks the *actual* numbers across commits so gradual drift -- e.g. an op
sliding 20x -> 9x while still above the floor -- becomes visible. We compare
SPEEDUP RATIOS (fast vs the pure-Python fallback in the same run), which are
machine-stable unlike raw wall-clock, and only against prior entries from the
same os/arch/python. Advisory, not a per-wheel CI gate (those runners are too
noisy): run it on a consistent machine before releases, or in a dedicated job.

  python scripts/perf_trend.py           # measure, compare, append to the ledger, report
  python scripts/perf_trend.py --check   # measure + compare + report, do NOT append

Exit status is nonzero if any op regressed past the band (so --check can gate).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import picopie
from picopie import ScalarField, Voxels, _fast

LEDGER = Path(__file__).resolve().parents[1] / "benchmarks" / "history.jsonl"
REGRESSION_BAND = 0.25   # flag if speedup drops > 25% vs the last like-for-like baseline


def _best(fn, repeats: int) -> float:
    fn()                                          # warmup
    return min(_one(fn) for _ in range(repeats))


def _one(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def _fast_vs_fallback(fn, fast_repeats: int, slow_repeats: int) -> tuple[float, float, float]:
    fast = _best(fn, fast_repeats)
    saved = _fast.lib
    _fast.lib = None                              # force the pure-Python path
    try:
        slow = _best(fn, slow_repeats)
    finally:
        _fast.lib = saved
    return fast, slow, (slow / fast if fast > 0 else float("inf"))


def measure() -> dict:
    picopie.init(voxel_size_mm=0.5)
    mesh = Voxels.sphere(radius=20.0).to_mesh()
    field = ScalarField.from_voxels(Voxels.sphere(radius=20.0))
    rng = np.random.default_rng(0)
    pos = (rng.random((50_000, 3), dtype=np.float32) - 0.5) * 30.0
    vals = rng.random(50_000, dtype=np.float32)
    ops = {
        "mesh_vertices": (lambda: mesh.vertices, 5, 3),
        "mesh_triangles": (lambda: mesh.triangles, 5, 3),
        "field_set_many": (lambda: field.set_many(pos, vals), 3, 1),
        "field_get_many": (lambda: field.get_many(pos), 3, 1),
    }
    out = {}
    for name, (fn, fr, sr) in ops.items():
        fast, slow, sp = _fast_vs_fallback(fn, fr, sr)
        out[name] = {"fast_ms": round(fast * 1e3, 3),
                     "fallback_ms": round(slow * 1e3, 3),
                     "speedup": round(sp, 2)}
    picopie.shutdown()
    return out


def _env() -> dict:
    return {"os": platform.system(), "arch": platform.machine(),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "cpu": os.cpu_count()}


def _commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=False)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(ln) for ln in LEDGER.read_text().splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="fast-path perf trend tracker")
    ap.add_argument("--check", action="store_true", help="don't append to the ledger")
    args = ap.parse_args()

    env = _env()
    results = measure()
    baseline = next((e for e in reversed(_load()) if e.get("env") == env), None)

    print(f"perf trend ({env['os']}/{env['arch']} py{env['python']}, {env['cpu']} cpu):\n")
    print(f"  {'op':<16}{'speedup':>9}{'baseline':>10}{'delta':>8}   fast/fallback ms")
    regressed = []
    for op, r in results.items():
        sp = r["speedup"]
        bsp = (baseline or {}).get("results", {}).get(op, {}).get("speedup")
        if bsp is None:
            delta, bstr = "(new)", "-"
        else:
            delta = f"{(sp - bsp) / bsp * 100:+.0f}%"
            bstr = f"{bsp:.1f}x"
            if sp < bsp * (1 - REGRESSION_BAND):
                regressed.append((op, bsp, sp))
        print(f"  {op:<16}{sp:>8.1f}x{bstr:>10}{delta:>8}   {r['fast_ms']}/{r['fallback_ms']}")

    if baseline:
        print(f"\n  baseline: {baseline.get('commit', '?')} @ {baseline.get('date', '?')}")
    if regressed:
        print(f"\n  REGRESSION: {len(regressed)} op(s) dropped >{int(REGRESSION_BAND * 100)}%:")
        for op, b, s in regressed:
            print(f"    {op}: {b:.1f}x -> {s:.1f}x")
    else:
        print("\n  OK: no regression vs baseline.")

    if not args.check:
        entry = {"commit": _commit(),
                 "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "env": env, "results": results}
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\n  appended to {LEDGER}")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
