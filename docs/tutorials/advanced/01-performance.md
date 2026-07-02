# Advanced 1 — Performance

PicoPie is a thin layer over a fast C++ kernel, but how you call it matters a lot.
Here's how to keep things quick.

## 1. Voxel size dominates everything

The voxel size you pass to `init()` sets the resolution. Cost scales roughly with
the **surface area / voxel²** (level sets store a narrow band around the surface),
and finer voxels mean more memory and time for every operation.

```python
picopie.init(voxel_size_mm=0.5)   # coarse: fast iteration, prototyping
picopie.init(voxel_size_mm=0.1)   # fine:   final detail, much heavier
```

Model and iterate coarse; only drop the voxel size for the final export.

## 2. Use the bulk array APIs (the compiled fast path)

Mesh vertex/triangle transfer and field get/set go through a compiled `_fastloop`
extension that copies in one `nogil` pass — **~13–28× faster** than a Python loop.
It's automatic for:

```python
mesh.vertices            # (N,3) in one fast copy
mesh.triangles           # (M,3) in one fast copy
field.set_many(pts, vals)
field.get_many(pts)      # -> (values, found)
```

So **never** write per-point loops when an array call exists:

```python
# SLOW — a native round-trip per point:
for p, v in zip(pts, vals):
    field.set(p, v)

# FAST — one compiled pass:
field.set_many(pts, vals)
```

Check the fast path is active (it falls back to pure Python if the extension
wasn't built):

```python
from picopie import _fast
assert _fast.available()        # True in the published wheels
```

## 3. Prefer primitives + booleans over per-voxel Python SDFs

`render_implicit_(sdf, bbox)` calls your Python `sdf` **once per voxel** — fine for
math surfaces (gyroids), but for bulk solids a primitive + boolean is far cheaper
because the loop stays in C:

```python
# Cheap (native): a ball with a bore
part = Voxels.sphere(radius=12) - Voxels.capsule((-13,0,0),(13,0,0), radius=4)

# Expensive if used for the same thing: Python callback over the whole box
```

When you *do* render implicitly, keep the `bbox` snug — every extra voxel in the box
is another Python call.

### Compiled callbacks (numba / ctypes): ~30× faster

Some SDFs *can't* be expressed as primitives + booleans — a gyroid, or an implicit
driven by a learned model (a CPPN / PINN / neuroevolved field). For those, the
per-voxel Python call is the dominant cost. `render_implicit_` and
`intersect_implicit_` therefore also accept a **compiled** callback and hand it
straight to the native loop, with no interpreter in the inner loop:

```python
import numba
from numba import types

# ABI must be float(const PKVector3*): a pointer to three contiguous float32.
sig = types.float32(types.CPointer(types.float32))

@numba.cfunc(sig, nopython=True)
def gyroid(p):
    x, y, z = p[0], p[1], p[2]
    import math
    return (math.sin(x)*math.cos(y) + math.sin(y)*math.cos(z)
            + math.sin(z)*math.cos(x))

part.render_implicit_(gyroid, bbox)   # detected as compiled, runs natively
```

On a sphere at 0.2 mm voxels this is **~30× faster** than the equivalent Python
callback (measured 724 ms → 23 ms), with identical geometry. A `numba.cfunc`, a
hand-written C shared library loaded with `ctypes.CDLL`, or a cffi callback all
work — numba is **not** a dependency of PicoPie; a compiled function pointer is
just detected by duck typing.

!!! warning "The compiled path bypasses the safety guard"
    A plain Python `sdf` is wrapped so that an exception or a NaN/inf return can
    never reach the native loop (it would silently inject a zero-distance
    surface). That guard is exactly what makes the Python path both safe **and**
    slow, so the compiled path **skips it**: your compiled function owns its own
    correctness and must return finite values. This matches upstream C# PicoGK
    behaviour — a bad return corrupts the level set rather than raising. Keep the
    body `nopython` / `nogil` and touch no Python objects (that is also what lets
    it run without the GIL).

## 4. `volume_mm3()` vs `calculate_properties()`

```python
v = part.volume_mm3()                     # fast: a direct native measurement
vol, bbox = part.calculate_properties()   # accurate: meshes + re-voxelizes (matches C#)
```

`volume_mm3()` is instant but can be skewed by zero-distance surface voxels left by
boolean ops; `calculate_properties()` is the accurate figure (a mesh round-trip), at
the cost of meshing. Use the fast one in tight loops, the accurate one for reports.

## 5. Track regressions over time

PicoPie ships a trend tracker so you can catch gradual slow-downs:

```bash
python scripts/perf_trend.py          # measures fast-vs-fallback speedups,
                                      # appends to benchmarks/history.jsonl,
                                      # flags any op that dropped >25% vs the baseline
```

It compares *speedup ratios* (machine-stable) rather than raw wall-clock, so it
won't flake across machines. Run it on a consistent box before a release.

Next: [reliability & the never-abort contract →](02-reliability.md)
