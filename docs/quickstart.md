# Quickstart

## Install

PicoPie needs the native PicoGK runtime. Released wheels bundle it; for a source
checkout you build it once.

=== "From a wheel (bundled runtime)"

    ```bash
    pip install picopie            # bundles the native runtime + fast extension
    pip install "picopie[viz]"     # + Pillow/matplotlib for visualization
    ```

=== "From source (build the runtime)"

    ```bash
    # system deps (Debian/Ubuntu/Mint)
    sudo apt-get install -y --no-install-recommends \
      libboost-all-dev libblosc-dev libtbb-dev extra-cmake-modules \
      xorg-dev mesa-common-dev libgl1-mesa-dev ninja-build cmake g++ git

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,viz]"
    scripts/build_runtime.sh       # clone + build PicoGKRuntime, stage into the package
    pytest
    ```

PicoPie finds the runtime automatically: bundled in the wheel (`picogk/_lib/`),
else under `native/`, else via `$PICOGK_RUNTIME` (full path to the shared library).

## Your first model

```python
import picogk
from picogk import Voxels

picogk.init(voxel_size_mm=0.2)        # voxel size = resolution; set once

# build with primitives + booleans
shaft = Voxels.capsule((-20, 0, 0), (20, 0, 0), radius=4)
ball = Voxels.sphere(center=(20, 0, 0), radius=8)
part = shaft + ball                   # union -> new Voxels
part.shell_(1.0)                      # hollow to a 1 mm wall

print("volume:", round(part.volume_mm3(), 1), "mm^3")

# mesh + export
mesh = part.to_mesh()
print("triangles:", mesh.triangle_count())
mesh.save_stl("part.stl")
```

!!! note "The session"
    `picogk.init(voxel_size_mm=...)` creates the one library instance everything
    uses; call `picogk.shutdown()` when done, or use the context manager:

    ```python
    with picogk.session(voxel_size_mm=0.2):
        ...
    ```

## Visualize headlessly

```python
from picogk import save_slice_png, save_slice_sheet, mesh_preview

save_slice_png(part, "slice.png", axis="z", mode="sdf")     # signed-distance heatmap
save_slice_sheet(part, "sheet.png", axis="z", count=16)     # montage of slices
mesh_preview(part.to_mesh(), "preview.png")                 # 3D render (matplotlib)
```
