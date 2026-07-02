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
