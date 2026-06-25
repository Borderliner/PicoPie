# Advanced 2 — Reliability: errors, lifetimes, and the never-abort contract

PicoPie binds a C++ runtime over a flat C ABI. A naive binding of such a runtime
can hard-crash your Python process; PicoPie is hardened so it doesn't. This tutorial
explains the guarantees and how to work with them.

## Errors are catchable, not fatal

The runtime is built with a **never-abort guard**: every C entry point is wrapped so
a C++/OpenVDB exception becomes a catchable `PicoGKError` instead of terminating the
process.

```python
import picogk
from picogk import Voxels
from picogk._errors import PicoGKError

picogk.init(0.5)
try:
    risky_operation()
except PicoGKError as e:
    print("native error:", e)        # caught — your program keeps running
```

The exception hierarchy:

- `PicoGKError` — base / native errors and version mismatches.
- `InvalidHandleError` — using an object after it's closed (or after `shutdown()`).
- `NotInitializedError` — calling the API before `picogk.init()`.

## Non-finite inputs are rejected

`NaN`/`inf` coordinates, radii, or offset distances would make native code
segfault or hang (which a `try/catch` *can't* catch) — so PicoPie rejects them at
the boundary with a `ValueError`:

```python
Voxels.sphere(radius=float("nan"))     # ValueError: radius must be finite, got nan
Voxels.sphere(radius=10).offset_(float("inf"))   # ValueError
```

This was found by fuzzing the whole API with thousands of adversarial inputs — none
abort the process. (Huge-but-*finite* values are a different matter: asking for a
kilometre sphere at micron resolution is a resource limit, your responsibility.)

## Type safety at the boundary

Handles are opaque integers in C, so passing the wrong object would corrupt state.
PicoPie type-checks every call that takes another object:

```python
mesh = Voxels.sphere(radius=5).to_mesh()
Voxels.from_mesh(Voxels.sphere(radius=5))   # TypeError: voxels where a Mesh is expected
```

## Object lifetimes

Native objects free themselves when garbage-collected, but you can be explicit. For
most objects this is optional; for the **`Viewer` it's required** (see the
[viewer tutorial](03-viewer.md)).

```python
with Voxels.sphere(radius=5) as v:
    ...                              # v.close() on exit

# or manually
v = Voxels.sphere(radius=5)
v.close()
v.volume_mm3()                       # InvalidHandleError — using a closed object raises (no crash)
```

`picogk.shutdown()` invalidates all live objects first, so any later use raises
cleanly rather than touching freed native memory.

## A note on `intersect_implicit_`

The one operation with a sharp edge: calling `intersect_implicit_` twice on the same
volume (or a copy) leaves a non-level-set grid. PicoPie detects the repeat and raises
`PicoGKError`; and even if you bypass that, the never-abort guard turns the underlying
OpenVDB error into an exception rather than a crash. Still — **prefer composing the
clip into one `render_implicit_`** (`max(feature, clip)`); it's correct by construction.

## Provenance

Every build input is pinned (the runtime, OpenVDB, Boost, TBB, Blosc, …). Each
release ships a CycloneDX SBOM (`sbom.cdx.json`) inventorying the bundled components
with versions and licenses, plus hash-pinned Python build dependencies — so you can
audit exactly what's inside a wheel.

Next: [the interactive viewer →](03-viewer.md)
