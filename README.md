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
- 🔜 Compiled bulk-transfer fast path, slice/PNG export, cross-platform wheels,
  the GLFW viewer. See [`ROADMAP.md`](ROADMAP.md).

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
  xorg-dev mesa-common-dev libgl1-mesa-dev

git clone --recurse-submodules https://github.com/leap71/PicoGKRuntime native/PicoGKRuntime
cmake -S native/PicoGKRuntime -B native/PicoGKRuntime/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_CORE_SHARED=OFF -DOPENVDB_CORE_STATIC=ON \
  -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF
ninja -C native/PicoGKRuntime/build
```

PicoPie locates the runtime automatically under `native/`, or via the
`$PICOGK_RUNTIME` environment variable (full path to `picogk.so`/`.dylib`/`.dll`).

## Performance & safety notes

- Mesh vertices/triangles transfer one element at a time today (Python loop). A
  compiled fast path is planned; large meshes are slower than C# for now.
- Implicit SDF callbacks run once **per voxel** from native code — a pure-Python
  SDF is the slow path. Prefer primitives + booleans, or compose fields inside a
  single `render_implicit_` call.
- Some native functions let OpenVDB exceptions escape (e.g. a CSG boolean on a
  non-level-set grid), which **aborts the process**. Feed valid level sets;
  prefer composed-SDF clipping over `intersect_implicit_` + boolean.

## License

PicoPie is Apache-2.0, matching upstream PicoGK / PicoGKRuntime.
