# PicoPie Roadmap

**Status: complete — released on PyPI as [`picopie`](https://pypi.org/project/picopie).**
Every phase below is done; the binding is feature-complete vs C# PicoGK (including
the interactive viewer) and hardened beyond it. See [`PLAN.md`](PLAN.md) for the
architecture rationale.

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

## ✅ Phase 3 — Headless visualization (done)

- [x] Voxel slice extraction (`slice_x/y/z` + interpolated z) → NumPy (one native call)
- [x] Slice → PNG via Pillow (`save_slice_png`, `save_slice_sheet` montage; mask/sdf/gray)
- [x] matplotlib 3D mesh preview (`mesh_preview`, Agg/headless)
- [x] Optional `[viz]` extra; lazy imports so core stays dependency-free
- *Surfaced and fixed a real `shell_` bug (it wasn't hollowing).*

## ✅ Phase 4 — Packaging & cross-platform wheels (done)

- [x] Scripted, reproducible native build (`scripts/build_runtime.sh` + `stage_runtime.py`)
- [x] Bundle runtime into wheels under `picogk/_lib/` — **verified locally**: wheel installs
      in a clean venv (no source tree, no env var) and runs off the bundled runtime
- [x] `auditwheel repair` vendors the runtime's deps (TBB/Blosc/Boost/…) and patches RPATH;
      confirmed all resolve to `picopie.libs/` (manylinux self-contained, even dlopen'd)
- [x] **Docker-verified**: the repaired `manylinux_2_39` wheel runs a full pipeline (booleans,
      meshing via fast path, VDB round-trip) in a bare `ubuntu:24.04` container with **no system
      TBB/Blosc/Boost** — proven to run purely on the vendored libs
- [x] CI asserts `_fastloop` builds from the sdist; `ci_check.py` asserts bundled runtime in wheel test
- [x] `cibuildwheel` config + GitHub Actions matrix (Linux/macOS/Windows) authored
- [x] **Matrix green in real CI** — Linux (manylinux_2_34), macOS (Apple Silicon),
      Windows (x64), cp310–cp313; each wheel runs the full test suite. Required
      per-platform fixes: Boost-1.86-from-source on AlmaLinux 9, deployment-target
      14.0 on macOS, vcpkg + an upstream `strncpy_s`/`_WINDOWS` fix on Windows.
- *Cross-platform wheels build, test, and publish from CI.*

## ✅ Phase 5 — Parity validation & docs (done)

- [x] Golden tests vs **C# PicoGK** running on the same native runtime (16 checks): sphere/
      capsule volumes, boolean union/subtract/intersect, ±offset & double-offset, mesh→voxels,
      voxel dims, mesh counts & bbox, `is_inside` queries, and a **VDB round-trip** (a `.vdb`
      written by C# loads in PicoPie with matching geometry) — all match to float precision
      (`parity/`, `tests/test_parity_csharp.py`, `tests/golden/`)
- [x] Surfaced a real semantic gap and added `Voxels.calculate_properties()` (accurate
      volume+bbox via mesh round-trip, matching C# `CalculateProperties`)
- [x] API docs (mkdocs-material + mkdocstrings): home, quickstart, guide, API reference
      (`mkdocs.yml`, `docs/`) — **`mkdocs build --strict` passes** (5 pages, API auto-generated
      from the source docstrings via griffe). Build: `pip install -e ".[docs]" && mkdocs build`
- [x] Examples: `hello_picogk`, `fields_and_io`, `visualize`
- *Parity-validated against the reference implementation; docs ready to publish.*

## ✅ Phase R — Reliability hardening (done)

- [x] **Never-abort runtime**: a build-time patch wraps all 173 C ABI functions so a
      C++/OpenVDB exception becomes a catchable `PicoGKError` instead of `SIGABRT`;
      a ctypes `errcheck` raises it on the Python side (the Cython fast loop bypasses it)
- [x] **Fuzzing** (Hypothesis + a subprocess campaign, 12k+ ops) — found a NaN-capsule
      segfault and an inf-offset hang; fixed by rejecting non-finite geometry inputs
- [x] **Type safety** at every native boundary (`require_type`), reserved-metadata guard,
      runtime version gate, ABI struct-size self-check
- [x] **Reproducible/auditable supply chain**: pinned PicoGKRuntime / vcpkg / c-blosc /
      Boost / homebrew-core; CycloneDX SBOM; hash-pinned Python build deps
- [x] Perf-regression guard + trend ledger (`scripts/perf_trend.py`)

## ✅ Phase 10 — Interactive viewer (done)

- [x] Bound the native GLFW/OpenGL viewer: `Viewer` (window, PBR materials, IBL
      lighting), orbit camera (mouse/scroll/keys), `picogk.show()` one-liner,
      offscreen `render_png()`, `screenshot()`
- [x] Main-thread affinity guard, use-after-close protection, leak-free lifetime
- *Closes the last C# feature gap; headless usage still preferred for batch/CI.*

## Phase 7 — Higher-level kernel (future, not started)

- [ ] Port LEAP71 ShapeKernel (pure C# atop PicoGK) to Python on our core
