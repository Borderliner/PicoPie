# PicoPie

A **Pythonic binding for [PicoGK](https://github.com/leap71/PicoGK)** — LEAP 71's
compact computational-geometry kernel for engineering, built on voxel / level-set
modeling (OpenVDB).

PicoPie binds the **same native runtime** the official C# library uses, so you get
identical geometry and performance, wrapped in an idiomatic, NumPy-friendly API.
No .NET required.

```python
import picogk
from picogk import Voxels

picogk.init(voxel_size_mm=0.2)

body = Voxels.sphere(radius=10)
hole = Voxels.sphere(center=(6, 0, 0), radius=6)
part = (body - hole)        # boolean subtract -> new Voxels
part.shell_(1.0)            # hollow to a 1 mm wall (in place)

print(part.volume_mm3())
part.to_mesh().save_stl("part.stl")
```

## Why a binding, not a rewrite

PicoGK's C# layer is itself a thin wrapper over a native C++ runtime
(`PicoGKRuntime`, on OpenVDB + GLFW). PicoPie wraps that *same* runtime through its
flat C ABI via `ctypes`, then adds a Pythonic layer on top — so the heavy geometry
is the identical, battle-tested kernel.

## Feature highlights

- `Voxels`: primitives, booleans (`+ - &`), offsets, hollow `shell_`, implicit SDF,
  queries, meshing
- `Mesh`: NumPy vertices/triangles, STL/OBJ import & export
- `Lattice`: beams + spheres
- `ScalarField` / `VectorField`, `Metadata`, `PolyLine`
- OpenVDB file I/O (`save_vdb` / `load_vdb`)
- Compiled bulk transfer (`_fastloop`, ~17–28× faster mesh/field I/O) with a
  pure-Python fallback
- Headless visualization (`save_slice_png`, `save_slice_sheet`, `mesh_preview`)

See the [Quickstart](quickstart.md) to install and build your first model, the
[Guide](guide.md) for the full API in context, and the
[API reference](api.md) for details.

## Verified against C# PicoGK

PicoPie's core operations are checked for **parity** against the reference C#
wrapper running on the same native runtime: sphere/capsule volumes, boolean and
offset volumes, voxel dimensions, and mesh vertex/triangle counts and bounding
boxes match exactly. (See `tests/test_parity_csharp.py`.)
