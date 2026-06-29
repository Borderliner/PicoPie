# Changelog

All notable changes to PicoPie. Versions follow [SemVer](https://semver.org).

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
- **`picogk.web`** — a browser-based 3D viewer (three.js inside an
  [anywidget](https://anywidget.dev)), via the optional `[web]` extra. Geometry is
  computed in Python and streamed to the browser as binary buffers, so it renders
  where the desktop GLFW `picogk.Viewer` can't: JupyterLab, VS Code notebooks,
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
- **`picogk.shapes`** — a full, Pythonic port of LEAP 71's
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
