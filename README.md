# PicoPie

A **Pythonic binding for [PicoGK](https://github.com/leap71/PicoGK)** — LEAP 71's
compact computational-geometry kernel for engineering (voxel / level-set modeling
on OpenVDB).

PicoPie binds the *same native runtime* the official C# library uses (so you get
identical geometry and performance), then wraps it in an idiomatic, NumPy-friendly
Python API. No .NET required.

```python
import picogk
from picogk import Voxels

picogk.init(voxel_size_mm=0.2)

body = Voxels.sphere(radius=10)
hole = Voxels.sphere(center=(6, 0, 0), radius=6)
part = (body - hole)        # boolean subtract → new Voxels
part.shell_(1.0)            # hollow to a 1 mm wall (in place)

print(part.volume_mm3())
part.to_mesh().save_stl("part.stl")
```

## Status

Early but functional. Headless modeling works end-to-end on Linux:

- ✅ Native runtime builds & loads (OpenVDB 13, PicoGK 26.2)
- ✅ Auto-generated ctypes bindings for **all 173** C-API functions
- ✅ `Voxels` (primitives, booleans `+ - &`, offsets/shell, implicit SDF, queries)
- ✅ `Mesh` (from voxels, NumPy vertices/triangles, STL/OBJ import **and** export)
- ✅ `Lattice` (beams + spheres)
- ✅ `ScalarField` / `VectorField` (get/set, slices, traverse→NumPy)
- ✅ `Metadata` (string/float/vector) and `PolyLine`
- ✅ OpenVDB file I/O — `save_vdb` / `load_vdb` round-trips voxels + fields
- ✅ Handle lifetime management (context managers, leak-free)
- ✅ Compiled `_fastloop` bulk transfer (~17–28× faster mesh/field I/O) with
  automatic pure-Python fallback
- ✅ Headless visualization (`pip install picopie[viz]`): voxel slices → PNG
  (`save_slice_png`, `save_slice_sheet`) and 3D `mesh_preview`
- ✅ **Parity-validated against C# PicoGK** (same native runtime): volumes, voxel
  dims, mesh counts/bbox match to float precision (`tests/test_parity_csharp.py`)
- ✅ Docs site (`pip install -e ".[docs]" && mkdocs build`) — see [`docs/`](docs/)
- 🔜 Cross-platform wheels in CI, the GLFW viewer. See [`ROADMAP.md`](ROADMAP.md).

See [`PLAN.md`](PLAN.md) for the full roadmap.

## Install (development)

The native runtime must be built once (see below), then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python examples/hello_picogk.py
pytest
```

### Building the native runtime (Linux)

```bash
# system deps (Debian/Ubuntu/Mint):
sudo apt-get install -y --no-install-recommends \
  libboost-all-dev libblosc-dev libtbb-dev extra-cmake-modules \
  xorg-dev mesa-common-dev libgl1-mesa-dev ninja-build cmake g++ git

scripts/build_runtime.sh        # clones + builds PicoGKRuntime, stages it into the package
```

PicoPie locates the runtime automatically: bundled in the wheel (`picogk/_lib/`),
else under `native/`, else via the `$PICOGK_RUNTIME` environment variable (full
path to `picogk.so`/`.dylib`/`.dll`).

### Wheels (self-contained)

`scripts/build_runtime.sh` stages the native runtime into `picogk/_lib/` so a
built wheel bundles it. The wheel also bundles the compiled `_fastloop`
extension, and `auditwheel`/`delocate`/`delvewheel` vendor the runtime's C++
deps (TBB, Blosc, Boost, …) — so `pip install` needs no system libraries:

```bash
scripts/build_runtime.sh && python -m build --wheel
auditwheel repair dist/*.whl -w dist/repaired   # Linux -> manylinux, vendoring deps
```

Cross-platform wheels (Linux/macOS/Windows) are built in CI via
[`cibuildwheel`](pyproject.toml) and `.github/workflows/wheels.yml`.

## Performance & safety notes

- Mesh vertices/triangles and bulk field get/set use a compiled `_fastloop`
  extension (built from `_fastloop.pyx`); if it isn't compiled, PicoPie falls
  back to slower pure-Python loops automatically.
- Implicit SDF callbacks run once **per voxel** from native code — a pure-Python
  SDF is the slow path. Prefer primitives + booleans, or compose fields inside a
  single `render_implicit_` call.
- Some native functions let OpenVDB exceptions escape (e.g. a CSG boolean on a
  non-level-set grid), which **aborts the process**. Feed valid level sets;
  prefer composed-SDF clipping over `intersect_implicit_` + boolean.

## License

PicoPie is Apache-2.0, matching upstream PicoGK / PicoGKRuntime.
