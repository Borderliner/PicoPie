# Guide

## Voxels — the core object

A `Voxels` is a signed-distance / level-set volume (values ≤ 0 are *inside*).
Operations come in two flavors: **functional** operators that return a new volume,
and **in-place** methods (trailing `_`) that mutate and are faster (no copy).

```python
from picogk import Voxels

a = Voxels.sphere(radius=10)
b = Voxels.sphere(center=(6, 0, 0), radius=6)

union   = a + b      # or a | b
diff    = a - b
common  = a & b
grown   = a.offset(2.0)      # functional: returns new
a.offset_(2.0)               # in-place
```

Primitives: `Voxels.sphere`, `Voxels.capsule`, `Voxels.from_mesh`,
`Voxels.mesh_shell`, `Voxels.from_lattice`.

Hollowing — `shell_(thickness)` keeps a wall of `thickness` mm inside the surface
(`solid − erode(solid, thickness)`):

```python
part = Voxels.sphere(radius=12)
part.shell_(1.5)             # 1.5 mm wall
```

Queries: `volume_mm3()`, `is_empty()`, `is_inside(p)`, `surface_normal(p)`,
`closest_point(p)`, `ray_cast(o, d)`, `voxel_dimensions()`, `bounding_box()`.

### Implicit modeling

Render a signed-distance function over a box. The callback runs once per voxel in
native code, so a pure-Python SDF is the slow path — compose fields inside one
callback rather than doing boolean CSG on the result.

```python
import math
k = 2 * math.pi / 6.0
def gyroid_in_sphere(x, y, z):
    g = (math.sin(k*x)*math.cos(k*y) + math.sin(k*y)*math.cos(k*z)
         + math.sin(k*z)*math.cos(k*x))
    sphere = math.sqrt(x*x + y*y + z*z) - 10      # <= 0 inside
    return max(abs(g) - 0.4, sphere)              # thin walls AND inside

v = Voxels()
v.render_implicit_(gyroid_in_sphere, ((-12, -12, -12), (12, 12, 12)))
```

## Mesh

```python
mesh = part.to_mesh()                 # marching-cubes surface
mesh.vertices                         # (N, 3) float32  (compiled fast path)
mesh.triangles                        # (M, 3) int32
mesh.bounding_box()
mesh.save_stl("part.stl"); mesh.save_obj("part.obj")

m = picogk.Mesh.load_stl("in.stl")    # binary or ASCII
vox = Voxels.from_mesh(m)             # voxelize an imported mesh
```

## Fields & metadata

```python
from picogk import ScalarField, VectorField, Metadata

heat = ScalarField.from_voxels(part)
heat.set((0, 0, 0), 100.0)
heat.set_many(positions, values)      # bulk (compiled)
vals, found = heat.get_many(positions)

md = Metadata.from_voxels(part)
md["material"] = "Ti-6Al-4V"; md["wall_mm"] = 1.0
```

## File I/O (OpenVDB)

```python
from picogk import save_vdb, load_vdb
save_vdb("part.vdb", body=part, heat=heat)        # voxels + fields, one file
objs = load_vdb("part.vdb")                        # -> {"body": Voxels, "heat": ScalarField}
```

## Visualization (headless)

```python
from picogk import save_slice_png, save_slice_sheet, mesh_preview
save_slice_png(part, "z.png", axis="z", mode="sdf")   # mask / sdf / gray
save_slice_sheet(part, "sheet.png", count=16)
mesh_preview(part.to_mesh(), "preview.png")           # close=True when saving
```

## Performance notes

- Mesh and bulk field transfer use a compiled `_fastloop` extension (~17–28×);
  if it isn't built, PicoPie falls back to pure-Python loops automatically.
- Implicit SDF callbacks and per-element field set/get are inherently per-point —
  prefer native primitives + booleans, or the bulk `*_many` field APIs.
- `scripts/perf_trend.py` records the fast-path speedups to `benchmarks/history.jsonl`
  and flags regressions vs the last like-for-like baseline — run it (on a consistent
  machine) before releases to catch gradual drift the 2× CI floor wouldn't.

## Lifetime

Native objects free themselves when garbage-collected, but you can be explicit:

```python
with Voxels.sphere(radius=5) as v:
    ...                               # v.close() on exit
```

## Safety & limits (process-termination contract)

PicoPie binds a native runtime (OpenVDB-based) over a flat C ABI. A C++/OpenVDB
exception escaping that ABI would normally `std::terminate` the whole process,
uncatchably (upstream C# has this property). **PicoPie's bundled runtime is built
with a never-abort guard**: every C entry point is wrapped so an exception is
captured and surfaced to Python as a `PicoGKError` instead of aborting — so *any*
native error, even an unanticipated one, is catchable. (Building against an
*unpatched* runtime loses this; the binding still works but can abort.)

On top of that, PicoPie turns the common mistakes into specific, catchable
exceptions *before* they reach native code:

- **Out-of-range / closed handles** → `IndexError` / `InvalidHandleError`
  (VDB field indices, slice indices, ops on a closed object or after `shutdown()`).
- **Wrong-type operands** → `TypeError`. Handles are bare `uint64` across the
  ABI, so any method that takes another native object (`Voxels.from_mesh`,
  `VdbFile.add_voxels`, `ScalarField.from_voxels`, …) checks the type before the
  handle reaches C, and `VdbFile.get_scalar_field()` on a level-set field is
  rejected rather than returning silently wrong data.
- **Reserved metadata** → `ValueError`: writing/removing a `PicoGK.`-prefixed
  name (the runtime's internal fields) is refused so you can't corrupt grid state.
- **Non-finite geometry inputs** → `ValueError`: NaN/inf coordinates, radii, or
  offset distances are rejected up front. Unlike a C++ exception, these reach
  native code as a hard `SIGSEGV` or an unbounded loop that the guard *can't*
  catch (found by fuzzing -- see scripts/fuzz_abort.py). Note this covers
  non-finite values, not huge-but-finite ones: asking for a kilometre sphere at
  micron resolution is a resource limit, your responsibility.
- **Bad file paths** → `FileNotFoundError` / `PicoGKError` (load/save check first).
- **Failing SDF callbacks** → your exception is re-raised; non-finite returns are
  treated as "outside" rather than corrupting the grid.
- **Runtime version mismatch** → `PicoGKError` at `init()` (see Phase 6 version gate).

**Note on `intersect_implicit_`:** calling it **more than once** on the same volume
(or a copy) leaves the grid in a non-level-set state. PicoPie detects the repeat and
raises `PicoGKError` early with an actionable message; even if you bypass that, the
runtime guard now turns the underlying OpenVDB error into a `PicoGKError` rather than
a crash. Still, **prefer composing the clip into one `render_implicit_` callback**
(`max(feature_sdf, clip_sdf)`) — it's correct by construction. Empirically every
other operation — booleans, offsets, meshing, `calculate_properties`, field/slice
access on empty or degenerate geometry — is already crash-safe.
