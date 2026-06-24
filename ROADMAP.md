# PicoPie Roadmap

Ordered, phased plan. See [`PLAN.md`](PLAN.md) for the architecture rationale.

Strategy (decided): bind the native PicoGK runtime via **ctypes**, headless-first,
shipped as cross-platform wheels. Pure-Python idiomatic layer on top.

---

## ✅ Phase 0 — Foundation (done)

- [x] Build the native runtime on Linux (OpenVDB 13, PicoGK 26.2 → `picogk.so`)
- [x] Header-driven codegen: ctypes signatures for all **173** C functions
- [x] Runtime loader (env override + dev/bundled search), packed struct/callback mirrors
- [x] Core Pythonic layer: `Voxels`, `Mesh`, `Lattice`, `library` session
- [x] Handle-lifetime management (context managers, leak-free)
- [x] Smoke test + end-to-end example + 12 passing tests

## ✅ Phase 1 — Complete the headless data model (done)

Goal: every native capability reachable from idiomatic Python, plus persistence.

- [x] `ScalarField` + `VectorField` wrappers (get/set/remove, slice, traverse→numpy, build-from-voxels)
- [x] VDB file I/O — `VdbFile` save/load; add/get voxels + scalar + vector fields; field names & types
- [x] `Metadata` wrapper (string/float/vector get/set; names; types)
- [x] `PolyLine` wrapper (vertices, color, bbox)
- [x] Mesh import: STL/OBJ → `Mesh` → voxelize (`Mesh.load_stl/load_obj/from_arrays`)
- [x] Abort-hardening: validity guards before CSG to reduce uncaught-native-exception aborts
- [x] Tests for all of the above; example covering save/load + mesh import

## ✅ Phase 2 — Performance: compiled bulk transfer (done)

- [x] Cython `_fastloop` extension: bulk mesh vertex/triangle read+build and field
      get/set in `nogil` loops over native function pointers → one NumPy copy
- [x] Pure-Python fallback when the extension isn't compiled (auto-detected)
- [x] Benchmark suite + parity tests vs the pure-Python loop
- *Measured ~17-18x faster mesh read, ~28x faster mesh build, ~13x field bulk set.*

## Phase 3 — Headless visualization

- [ ] Slice extraction (`Voxels_GetZSlice`…) → numpy → PNG (Pillow)
- [ ] matplotlib 3D mesh preview for notebooks
- *Independent of P2.*

## Phase 4 — Packaging & cross-platform wheels

- [ ] Scripted, reproducible native build
- [ ] `cibuildwheel` matrix: Linux (manylinux, build runtime in-container), macOS, Windows
- [ ] Bundle runtime into wheels (`auditwheel`/`delocate`/`delvewheel`) under `picogk/_lib/`
- [ ] GitHub Actions CI: build + test on all three OSes
- *Heaviest infra; loader + `_lib/` hooks already in place.*

## Phase 5 — Parity validation & docs

- [ ] Golden tests vs C# PicoGK (volumes, bbox, counts, VDB round-trip)
- [ ] API docs (mkdocs), gallery, richer examples
- *Deliverable: documented, parity-validated v0.2.*

## Phase 6 — Interactive viewer (optional, last)

- [ ] Wrap the 33 viewer fns; GUI-thread/GIL/main-thread-GL model (viewer on main, model on worker)
- *Hardest integration; do only if wanted.*

## Phase 7 — Higher-level kernel (future)

- [ ] Port LEAP71 ShapeKernel (pure C# atop PicoGK) to Python on our core
