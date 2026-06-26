# Changelog

All notable changes to PicoPie. Versions follow [SemVer](https://semver.org).

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
