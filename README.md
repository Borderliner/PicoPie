# PicoPie

A **Pythonic binding for [PicoGK](https://github.com/leap71/PicoGK)** — LEAP 71's
compact computational-geometry kernel for engineering (voxel / level-set modeling
on OpenVDB).

PicoPie binds the *same native runtime* the official C# library uses — so you get
identical geometry and performance — then wraps it in an idiomatic, NumPy-friendly
Python API. No .NET, no compiler, no system libraries: just `pip install picopie`.

```python
import picogk
from picogk import Voxels

picogk.init(voxel_size_mm=0.2)

body = Voxels.sphere(radius=10)
hole = Voxels.sphere(center=(6, 0, 0), radius=6)
part = body - hole          # boolean subtract → new Voxels
part.shell_(1.0)            # hollow to a 1 mm wall (in place)

vol, bbox = part.calculate_properties()
print(f"{vol:.1f} mm³")
part.to_mesh().save_stl("part.stl")
```

## Install

```bash
pip install picopie            # core
pip install "picopie[viz]"     # + headless slice/mesh PNG helpers (Pillow, matplotlib)
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add picopie
```

Wheels are self-contained — the native runtime (OpenVDB 13, PicoGK 26.2) and all
its C++ deps are bundled. Prebuilt for CPython 3.10–3.13 on **Linux (manylinux
x86-64), macOS (Apple Silicon), and Windows (x64)**.

## Documentation

- **Tutorials** — a leveled path:
  [Novice](docs/tutorials/novice/01-setup.md) (uv setup + install → shapes → booleans),
  [Intermediate](docs/tutorials/intermediate/01-implicit-modeling.md) (implicit modeling,
  meshes/files, fields), [Advanced](docs/tutorials/advanced/01-performance.md)
  (performance, reliability, viewer)
- **[QuickLearn](docs/tutorials/QuickLearn.md)** — the whole API in one annotated file
  (learn-x-in-y-minutes style)
- **[API reference](docs/api.md)** · docs build with `mkdocs build` (`pip install -e ".[docs]"`)

## Features

- **`Voxels`** — primitives (`sphere`, `capsule`), booleans (`+ - &`), offsets,
  `shell_` (hollow), implicit SDF modeling, and queries (`is_inside`,
  `closest_point`, `surface_normal`, `ray_cast`, `calculate_properties`).
- **`Mesh`** — from voxels (marching cubes), NumPy `vertices`/`triangles`, STL/OBJ
  import **and** export.
- **`Lattice`** (beams + spheres), **`ScalarField`** / **`VectorField`** (bulk
  `set_many`/`get_many`), **`Metadata`**, **`PolyLine`**.
- **File I/O** — `save_vdb` / `load_vdb` round-trips voxels + fields (OpenVDB).
- **Headless viz** (`[viz]`) — `save_slice_png`, `save_slice_sheet`, `mesh_preview`.
- **Interactive viewer** — `picogk.show(part)` (GLFW/OpenGL, orbit camera) and
  `render_png(part, "out.png")` for offscreen renders.
- **Fast path** — a compiled `_fastloop` extension (~13–28× faster bulk mesh/field
  transfer) with automatic pure-Python fallback.

## Reliability

PicoPie is validated against the reference and hardened well beyond a naive binding:

- **Parity-tested vs C# PicoGK** on the same runtime — volumes, dims, mesh
  counts/bbox, and geometric queries match to float precision.
- **Never aborts** — the native runtime is patched so a C++/OpenVDB error surfaces
  as a catchable `PicoGKError` instead of killing the process; non-finite inputs are
  rejected up front. Verified against thousands of fuzzed inputs.
- **Reproducible & auditable** — every upstream dependency is version-pinned; builds
  ship an SBOM (`sbom.cdx.json`) and hash-pinned build deps.
- Green CI across Linux/macOS/Windows; mypy + ruff clean; ~190 tests.

## Development

```bash
git clone https://github.com/Borderliner/PicoPie && cd PicoPie
python -m venv .venv && source .venv/bin/activate
scripts/build_runtime.sh         # build + stage the native runtime (see below)
pip install -e ".[dev]"
pytest
```

Building the native runtime (Linux) needs system deps:

```bash
sudo apt-get install -y --no-install-recommends \
  libboost-all-dev libblosc-dev libtbb-dev extra-cmake-modules \
  xorg-dev mesa-common-dev libgl1-mesa-dev ninja-build cmake g++ git
scripts/build_runtime.sh
```

PicoPie locates the runtime automatically: bundled in the wheel (`picogk/_lib/`),
else under `native/`, else via `$PICOGK_RUNTIME` (full path to the shared library).
Cross-platform wheels are built by `cibuildwheel` in `.github/workflows/wheels.yml`.

See [`ROADMAP.md`](ROADMAP.md) for the (now-complete) phase history and
[`PLAN.md`](PLAN.md) for the architecture rationale.

## License

Apache-2.0, matching upstream PicoGK / PicoGKRuntime. Ported by
Mohammadreza Hajianpour and Vish Vadlamani.
