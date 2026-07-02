# Changelog

All notable changes to PicoPie. Versions follow [SemVer](https://semver.org).

## 0.7.0 — 2026-07-02

### Changed — `active_values()` is now compiled (~30× faster)

`ScalarField.active_values()` and `VectorField.active_values()` previously walked
the sparse active-voxel set with a per-voxel Python callback (the same overhead
class as the implicit-SDF path fixed in 0.6.0). They now use the compiled
`_fastloop` extension: a native traversal that writes straight into NumPy arrays
(two `nogil` passes — count, then fill an exactly-sized buffer).

- **~30× faster** — 335k active voxels went from ~320 ms to ~10 ms — with
  **bit-identical** results (same coords, values, dtype, shape, and ordering).
- **Same public API, transparent.** No new method; it falls back to the pure-Python
  callback when the extension isn't built, exactly like `get_many`/`set_many`.
- Concurrent calls from multiple threads are serialised with a lock (the native
  callback ABI carries no user-data pointer, so the collector uses module state);
  the pure-Python fallback needs no lock.

## 0.6.0 — 2026-07-02

### Added — compiled callbacks for `render_implicit_` / `intersect_implicit_`

`render_implicit_(sdf, bbox)` and `intersect_implicit_(sdf)` now accept a
**compiled** SDF as well as a plain Python callable. Pass a `numba.cfunc` or any
ctypes function pointer with the ABI `float(const PKVector3*)` (a pointer to three
contiguous `float32`) and it is handed straight to the native loop — no per-voxel
Python round-trip. Measured **~30×** faster than the Python-callback path on a
sphere at 0.2 mm voxels, with bit-identical geometry.

```python
sig = types.float32(types.CPointer(types.float32))  # matches PKPFnfSdf

@numba.cfunc(sig)
def my_sdf(p):
    return (p[0]**2 + p[1]**2 + p[2]**2) ** 0.5 - 8.0

part.render_implicit_(my_sdf, bbox)   # detected and run natively
```

- Plain `sdf(x, y, z)` callables are unchanged and keep the never-abort finite
  guard. Detection is duck-typed, so **numba stays an optional, undeclared
  dependency** — the feature works equally with numba, a hand-written C shared
  library, or cffi.
- **Caveat:** the compiled path deliberately bypasses the finite/exception guard
  (that guard is what makes the Python path safe *and* slow). A compiled callback
  owns its own correctness — returning NaN/inf injects a zero-distance surface,
  the same behaviour as upstream C# PicoGK. See the "Compiled callbacks" section
  in `docs/tutorials/advanced/01-performance.md`.
- Requested in [#1](https://github.com/Borderliner/PicoPie/issues/1).

## 0.5.0 — 2026-07-02

### Changed — BREAKING: the import namespace is now `picopie`, not `picogk`

At the request of LEAP 71 (PicoGK's maintainers), the package now consistently
uses the **PicoPie** name so there's no confusion with the official PicoGK
library: this is an independent, community-maintained binding — **not** an
official LEAP 71 project.

- **Migration:** replace `import picogk` with `import picopie` (and `from picogk…`
  with `from picopie…`). The parametric layer is now `picopie.shapes`; the web
  viewer is `picopie.web`.
- The exception type `PicoGKError` is renamed to **`PicoPieError`**.
- The **PyPI package name is unchanged** (`pip install picopie`), and the bundled
  native **PicoGK runtime is unchanged** (it remains `picogk.so`, as it is the
  actual PicoGK runtime binary).
- README and NOTICE now state clearly that PicoPie is unofficial and not
  affiliated with or endorsed by LEAP 71.

## 0.4.2 — 2026-06-29

### Fixed
- **Shape tessellation now follows post-construction modulation.** Setting a
  modulated dimension *after* construction (`box.width = fn`,
  `cylinder.radius = fn`, `pipe.inner_radius` / `outer_radius = fn`, and `Cone`)
  now raises the tessellation step count to match the constructor path; it
  previously stayed silently coarse. Step counts are computed lazily from the
  current modulation state; an explicit `*_steps` argument still overrides.

### Testing
- Added direct coverage for previously-untested public wrappers
  (`Voxels.voxel_size_mm` / `bounding_box`, `Lattice` / `VdbFile.is_valid`, field
  `memory_bytes`, `Metadata.to_dict`, the `VdbFile.save` failure path, and the
  `picopie.__all__` re-exports) and shape branches (spherical frame alignment, the
  cylindrical control-spline tangential step, closed control surfaces,
  `LatticeManifold(extend_both_sides=...)`, the down-facing painter split, mesh
  STL export). 436 CI tests.

## 0.4.1 — 2026-06-29

A polish pass on the 0.4.0 web viewer (from a final gap audit) plus consistency
hardening. The core was verified to have no crash/hang gaps.

### Fixed (web viewer)
- **Matrix convention now matches the desktop `Viewer`** — `set_group_matrix` /
  `set_object_matrix` use the same row-major System.Numerics convention;
  previously a non-symmetric transform rendered transposed in the browser.
- **The camera no longer re-fits on every update** — recolouring, toggling
  visibility, or transforming a group preserves the current orbit (it auto-frames
  only on first load and on `reset_view`).
- `export_html` now carries per-object transforms; the identity matrix is no
  longer a shared-mutable default.

### Changed (web viewer)
- **geometry/style split** — heavy vertex/index buffers (the `geometry` trait)
  are separate from light per-object style (color, material, visibility,
  transform). A style tweak does a cheap in-place update instead of
  re-transmitting and rebuilding all geometry.

### Hardening
- `Voxels.mesh_shell`, `ScalarField`/`VectorField` `set_many`/`get_many`,
  `ScalarField.slice`, and `Mesh.add_triangle` now reject non-finite /
  out-of-range input with a clear error (consistent with the rest of the API)
  instead of producing silent garbage.

### Testing
- Assert the `require_finite` / `to_vec3` / bounds guards actually raise (the fuzz
  suite only proved "no abort"); add library-lifecycle tests
  (`NotInitializedError`, `init(<=0)`, `session()`); bring the web serialization
  and viewer camera-state math into per-wheel CI. 420 CI tests.

## 0.4.0 — 2026-06-29

### Added
- **`picopie.web`** — a browser-based 3D viewer (three.js inside an
  [anywidget](https://anywidget.dev)), via the optional `[web]` extra. Geometry is
  computed in Python and streamed to the browser as binary buffers, so it renders
  where the desktop GLFW `picopie.Viewer` can't: JupyterLab, VS Code notebooks,
  Google Colab, and remote/SSH sessions (compute-in-Python, render-in-browser).
  - `WebViewer` mirrors the desktop `Viewer` — `add` (Voxels/Mesh/PolyLine),
    `remove`, `set_group_material`, `set_group_visible`, `set_background`,
    `set_group_matrix` / `set_object_matrix`, `reset_view`, `screenshot` — plus a
    `show(*objects)` one-liner.
  - PBR materials + lighting, orbit/pan/zoom, an auto-framed camera, a wireframe
    toggle, and keyboard shortcuts (F fit · W wireframe · S save PNG).
  - `export_html(objects, path)` writes a self-contained, double-click-to-open
    HTML file (three.js renderer inlined, geometry embedded) — no Jupyter needed.
  - Runnable demo: `python examples/web/demo.py`.

## 0.3.2 — 2026-06-27

### Fixed
- **`project_z_slice_` at fine voxel sizes.** Found by auditing for more of the
  0.3.1 bug class (a *millimetre* value used as an integer *voxel count*).
  PicoGK's `ProjectZSliceDn`/`Up` sealed the end cap by iterating
  `(int)(0.5 + background_mm)` slices — but the background is in mm
  (`narrowBand × voxel_size`), not a voxel count. Below ~0.167 mm it truncated to
  0, so the cap was never sealed and the projected solid came out **non-watertight**
  (silently — no error; re-voxelising it collapsed the volume, e.g. 818 → 22 mm³
  at 0.1 mm). Above ~1.5 mm it over-iterated. The bundled runtime now uses the
  narrow band directly, so the result is consistent across voxel sizes.
- `Voxels.triple_offset_` and `project_z_slice_` now reject non-finite (NaN/inf)
  arguments with a `ValueError`, matching `offset_` / `double_offset_`.

### Testing
- New voxel-size **sweep tripwire** (`tests/test_voxel_size_sweep.py`): exercises
  the resolution-sensitive native ops across 0.1–1.0 mm and asserts results stay
  valid and consistent with the 0.5 mm baseline. The whole suite previously ran
  only at 0.5 mm — the one resolution where these narrow-band truncations can't
  fire — which is why the class shipped twice.
- `scripts/fuzz_abort.py` now sweeps voxel size across workers (0.05–2.0 mm) and
  fuzzes the previously-uncovered `triple_offset_`, `project_z_slice_`, and
  `mesh_shell` ops.

## 0.3.1 — 2026-06-27

### Fixed
- **Implicit intersect at fine voxel sizes.** `Voxels.intersect_implicit_` (and
  the `ImplicitGyroid`/`ImplicitSphere`/etc. intersect helpers) aborted with
  *"expected grid A outside value > 0, got 0"* whenever the voxel size was below
  ~0.33 mm. Upstream PicoGK's `IntersectImplicit` built its temporary grid from
  the background *in millimetres* (a float) passed into an `int nNarrowBand`
  argument, which truncated to `0` at fine resolutions → a grid with background
  `0` → OpenVDB's `csgIntersection` rejected it. The bundled runtime is now
  patched at build time to pass the source's narrow band instead, so implicit
  intersects work at any voxel size.

### Changed
- The example gallery (`examples/shapekernel/gallery.py`) now renders at a 0.2 mm
  voxel size by default (was 0.5 mm) with a new `--voxel-size` flag, and the
  small implicit shapes are scaled up — the docs gallery images are noticeably
  smoother (no faceted silhouettes / "orange-peel"). Surface smoothness is set
  by the voxel size, not the parametric tessellation.

## 0.3.0 — 2026-06-26

### Added
- **`picopie.shapes`** — a full, Pythonic port of LEAP 71's
  [ShapeKernel](https://github.com/leap71/LEAP71_ShapeKernel) (pinned at tag
  `ShapeKernel-v2.1.0`), **parity-tested against the reference C#** (geometry to
  float precision):
  - Local frames (`LocalFrame`, `Frames`), modulations (`LineModulation`,
    `SurfaceModulation`), vector/spline/list/grid/formula utilities, and splines
    (`ControlPointSpline`/`Surface`, `Tangential`/`CylindricalControlSpline`).
  - All 10 base shapes: `Sphere`, `Box`, `Cylinder`, `Cone`, `Ring`, `Lens`,
    `Pipe`, `PipeSegment`, `Revolve`, `LogoBox` — built from frames + modulations,
    optionally following a spine, with `to_voxels()` / `to_mesh()`.
  - Lattice shapes (`LatticePipe`, `LatticeManifold`) and lattice builders;
    implicit SDF primitives (`ImplicitGyroid`, `ImplicitSphere`, `ImplicitGenus`,
    `ImplicitSuperEllipsoid`) with `render` / `intersect`.
  - Measurement (`volume`, `surface_area`, `centre_of_gravity`,
    `moment_of_inertia`), mesh utilities, CSV/STL/VDB export, and a colour
    palette + value→colour scales + `MeshPainter`.
- Runnable example gallery (`examples/shapekernel/gallery.py`, a port of the
  upstream Examples) + rendered docs gallery, and a "Shapes" tutorial track.

### Changed / Fixed
- `Mesh.from_arrays` and `Lattice.add_*` now reject non-finite (NaN/inf) input
  with a clear `ValueError` on both the fast and fallback paths — so every shape
  validates its dimensions up front instead of producing silent-garbage voxels.
- B-splines with too few control points raise a clear `ValueError` instead of a
  `ZeroDivisionError`.

## 0.2.1 — earlier

- Documentation: leveled tutorials (Novice/Intermediate/Advanced) + QuickLearn.

## 0.2.0 — earlier

- Interactive viewer (`show`, `render_png`, `Viewer`); PyPI trusted publishing.

## 0.1.x — earlier

- Initial release: `ctypes` binding of the PicoGK native runtime — `Voxels`,
  `Mesh`, `Lattice`, fields, VDB I/O, headless viz, `_fastloop`; cross-platform
  wheels; C# parity tests; never-abort hardening.
