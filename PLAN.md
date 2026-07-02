# PicoPie — A Python binding for PicoGK

> Goal: bring PicoGK's voxel/level-set geometry kernel to Python with an idiomatic,
> NumPy-friendly API — by binding the **same native runtime** the C# library uses,
> not by reimplementing the kernel.

**Status: shipped.** Released on PyPI as `picopie`; all phases in [`ROADMAP.md`](ROADMAP.md)
are complete (cross-platform wheels, parity-validated, hardened, interactive viewer).
This document records the architecture decisions that got us there.

## 1. Strategy (decided)

| Decision | Choice | Consequence |
|---|---|---|
| Binding mechanism | **ctypes** over the native C ABI + a Pythonic layer on top | No compiler needed by end users; no .NET; full control; mirrors `Internals/Interop.cs` + `Base/*.cs` |
| Platforms | **Linux + macOS + Windows**, shipped as **bundled wheels** | Native runtime must be built/obtained per-platform and packaged via `cibuildwheel` |
| Viewer | **Headless first** | Skip the 33 viewer fns + GUI-thread/GIL/main-thread-GL complexity in v1 |

### Why a binding, not a rewrite
PicoGK is itself a thin wrapper. The real engine is **PicoGKRuntime** (C++ on **OpenVDB**
sparse level sets + **GLFW/OpenGL** viewer). The C# side is just: `[DllImport]` declarations,
handle-wrapping classes, and pure-C# conveniences (IO, shapes, numerics). The native boundary
is a **flat C ABI** — `API/PicoGK.h`, **177 `extern "C"` functions**, every object an opaque
`uint64_t` handle, trivial packed structs. That maps almost 1:1 onto ctypes. Reimplementing the
kernel would mean reimplementing OpenVDB — huge and slow. We do what C# does: bind the runtime.

## 2. Native C ABI — the surface we bind

- Objects are opaque handles: `PKINSTANCE/PKVOXELS/PKMESH/PKLATTICE/PKSCALARFIELD/PKVECTORFIELD/PKPOLYLINE/PKVDBFILE/PKMETADATA = uint64_t`.
- First arg of (almost) every function is the **library instance** handle → we hold it as a global singleton, so Python users never pass it (matches C# static `Library`).
- Structs are `#pragma pack(1)`: `PKVector3{f,f,f}`, `PKVector2`, `PKVector4`, `PKTriangle{i32,i32,i32}`, `PKCoord{i32×3}`, `PKBBox3{vec3,vec3}`, `PKMatrix4x4{vec4×4}`, `PKColorFloat{f×4}`.
- Library name string: `picogk.26.2` (resolves to `picogk.26.2.dll` / `libpicogk.26.2.dylib` / `libpicogk.26.2.so`).

### Function count by category (177 total)
`Voxels` 33 · `Viewer` 33 *(skip v1)* · `Library` 24 · `VdbFile` 15 · `Metadata` 15 ·
`ScalarField` 13 · `Mesh` 13 · `VectorField` 11 · `PolyLine` 9 · `Lattice` 6.
→ **~110 functions for the headless v1.**

### Two facts that shape the design
1. **No bulk array transfer.** Mesh verts/tris and field values are accessed one element at a
   time (`Mesh_GetVertex`, `ScalarField_SetValue`, …). Millions of ctypes calls per mesh is a
   real perf cliff → plan a compiled fast-loop (see §6). Not v1-blocking.
2. **Implicit modeling is callback-driven** (`Voxels_RenderImplicit(PKPFnfSdf)` called per point).
   A Python SDF lambda per-voxel is very slow. Fast path = native primitives + booleans
   (sphere/capsule/lattice/mesh-render/offset/bool — all present). Python-callback SDFs become an
   opt-in "flexible but slow" mode.

## 3. Package architecture

```
picopie/                      # pip name: picopie  ·  import name: picopie
├── pyproject.toml            # build via scikit-build-core or setuptools + cibuildwheel
├── src/picopie/
│   ├── __init__.py           # public API: init(), Voxels, Mesh, Lattice, ScalarField, ...
│   ├── _native/
│   │   ├── loader.py         # locate + dlopen runtime (bundled → env var → system)
│   │   ├── ctypes_types.py   # ctypes Structures (_pack_=1) + numpy dtype mirrors
│   │   ├── prototypes.py     # GENERATED: argtypes/restype for every C fn (from PicoGK.h)
│   │   └── _gen_prototypes.py# header parser/codegen (keeps us in sync per runtime version)
│   ├── library.py            # Library singleton: init(voxel_size), instance handle, log cb
│   ├── voxels.py             # Voxels (primitives, booleans, offsets, volume, slices, render*)
│   ├── mesh.py               # Mesh (+ from_voxels, numpy verts/tris)
│   ├── lattice.py            # Lattice (beams/spheres)
│   ├── fields.py             # ScalarField, VectorField
│   ├── polyline.py           # PolyLine
│   ├── types.py              # Vector3/BBox3/Matrix4x4 helpers ↔ numpy/tuples
│   ├── metadata.py           # field/voxel metadata
│   ├── io/                   # stl/obj export, vdb save/load, png slice export
│   └── _fastloop/            # OPTIONAL Cython/C ext: bulk mesh/field transfer (Phase 4)
├── tests/                    # pytest, incl. golden comparisons vs C# PicoGK
├── examples/                 # gyroid lattice, boolean part, mesh→voxelize→offset→export
└── native/                   # build scripts/CI for PicoGKRuntime per platform
```

### Core design rules
- **Handle lifetime** → each wrapper holds `(instance, handle)`; implement `close()`,
  `__enter__/__exit__`, and `__del__` calling the matching `*_Destroy`. Guard double-free; tear
  down objects before the library instance. Mirrors C# `IDisposable`.
- **Library singleton** → `picopie.init(voxel_size_mm=0.1)` creates the instance via
  `Library_hCreateInstance`; everything else pulls it implicitly (like C# static `Library`).
- **Log callback** → register a `PKFInfo` CFUNCTYPE routed to Python `logging`; keep a hard ref
  so it isn't GC'd. Fatal-error flag → raise `PicoPieError`.
- **NumPy first** → vectors/bboxes accept/return numpy or tuples; fields and mesh buffers expose
  numpy views; conversions centralized in `types.py`.
- **Prototype codegen** → parse `PicoGK.h` (regular `PICOGK_API <ret> <Name>(<args>);` grammar)
  to generate `prototypes.py`. Re-run per runtime version; no hand-maintained 177-fn table.

## 4. C# → Python mapping

| C# (`Base/*.cs`) | Python | Notes |
|---|---|---|
| `Library` (static, `Go`) | `picopie.init()` + module globals | hides instance handle |
| `Voxels` | `picopie.Voxels` | primitives, `+ - &` → `__add__/__sub__/__and__` (BoolAdd/Subtract/Intersect), offsets, `volume()`, slices, `render_mesh/lattice/implicit` |
| `Mesh` | `picopie.Mesh` | `from_voxels`, `.vertices`/`.triangles` numpy |
| `Lattice` | `picopie.Lattice` | `add_beam`, `add_sphere` |
| `ScalarField`/`VectorField` | `picopie.ScalarField`/`VectorField` | get/set, traverse, from-voxels |
| `PolyLine` | `picopie.PolyLine` | |
| `BBox3`/`Matrix4x4`/`Vector3` | `picopie.types` | numpy-backed |
| IO (`MeshIo`,`OpenVdbFile`,`VoxelsIo`,`ImageIo`) | `picopie.io` | STL/OBJ/VDB/PNG |
| `Viewer` | — | deferred (post-v1) |
| `Shapes/*`, ShapeKernel | — | follow-on pure-Python port (it's pure C# atop PicoGK) |

## 5. Milestones

- **M0 — Runtime on Linux (gating).** Build PicoGKRuntime from source (vcpkg/CMake: OpenVDB, TBB,
  Boost, Blosc, zlib, GLFW). Success: `ctypes.CDLL` loads it and `Library_GetVersion` returns a
  string. *Biggest practical risk; do first.*
- **M1 — ctypes core.** Header→`prototypes.py` codegen, ctypes structs, loader, `init()`, log
  callback, handle/dispose. Smoke test: create instance → `Voxels.sphere()` → `volume()`.
- **M2 — Geometry layer.** Voxels (primitives/booleans/offsets/volume/slices/render), Mesh,
  Lattice, ScalarField, VectorField, PolyLine, BBox, metadata, NumPy interop.
- **M3 — IO + examples.** STL/OBJ export, VDB save/load, PNG slice export; 3 headless examples.
- **M4 — Performance.** Cython/C `_fastloop` for bulk mesh/field transfer (loops in compiled code
  over the runtime's per-element fns — no runtime rebuild needed); benchmarks.
- **M5 — Packaging.** `pyproject`, `cibuildwheel` matrix, bundle runtime via
  `auditwheel`/`delocate`/`delvewheel`; reuse LEAP71 prebuilt mac/Windows runtimes to unblock CI.
- **M6 — Tests vs C# golden + docs.** Compare volumes/bbox/counts/VDB round-trips against C#
  reference scripts; API docs; quickstart.
- **Later** — native viewer (thread/GIL/main-thread-GL), ShapeKernel port.

## 6. Performance plan
v1: correctness-first, per-element loops in Python with numpy-shaped properties; document the
hotspot. Phase 4: small Cython extension that re-declares the runtime's per-element fns and loops
in C, filling contiguous buffers → single numpy copy. Avoids rebuilding the runtime and keeps the
ctypes layer pure-Python for everything else.

## 7. Testing
- Golden comparisons vs C# PicoGK on deterministic ops (sphere volume, boolean volumes, offset
  results, mesh vert/tri counts, voxel dims, VDB round-trip).
- pytest unit tests per wrapper; memory/handle-leak checks via `Library_n*Allocated` counters.
- CI runs the headless suite on all three platforms.

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Building OpenVDB-based runtime on Linux is heavy | Use vcpkg manifest in a manylinux container; cache; reuse LEAP71 prebuilt mac/Win runtimes |
| Per-element transfer too slow | Cython `_fastloop` (Phase 4); keep ops native where possible |
| Python-callback SDFs slow | Steer users to native primitives+booleans; mark SDF mode "flexible/slow" |
| Runtime version drift (`picogk.26.x`) | Generate prototypes from header; pin/version-check at load |
| Handle lifetime / double-free | Centralized dispose + allocation-counter leak tests |
| Cross-platform `.so/.dylib/.dll` name resolution | Normalize in `loader.py`; `PICOGK_RUNTIME` env override |

## 9. Immediate next steps
1. Scaffold the package (`pyproject.toml`, `src/picopie/`, `tests/`, `examples/`).
2. Vendor `PicoGK.h` + `PicoGKApiTypes.h`; write `_gen_prototypes.py` and emit `prototypes.py`.
3. Stand up the Linux runtime build (M0) — the gating prerequisite.
4. Implement `loader.py` + `library.py` + a smoke test (sphere → volume).
