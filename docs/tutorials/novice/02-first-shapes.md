# Novice 2 — First shapes and how to "see" them

PicoPie models with **voxels** — a 3D grid where each cell stores the signed
distance to the surface (negative inside, positive outside). You rarely touch that
directly; you build with primitives and combine them.

## Primitives

There are two built-in solids: spheres and capsules (a cylinder with rounded ends).

```python
import picopie
from picopie import Voxels

picopie.init(voxel_size_mm=0.2)

ball   = Voxels.sphere(center=(0, 0, 0), radius=10)
rod    = Voxels.capsule(start=(-15, 0, 0), end=(15, 0, 0), radius=3)
cone   = Voxels.capsule(start=(0, 0, -10), end=(0, 0, 10),
                        radius=6, radius2=1)   # tapered: different end radii
```

Need a box, a torus, a gyroid? Those come from **implicit modeling** (an
[Intermediate tutorial](../intermediate/01-implicit-modeling.md)) or from combining
primitives — covered next.

## Seeing your geometry

Modeling is headless by default, so you "look" at a shape by exporting it. Three easy ways:

### A. Save a mesh (open it in any 3D viewer / slicer)

```python
mesh = ball.to_mesh()          # marching-cubes surface
mesh.save_stl("ball.stl")      # or mesh.save_obj("ball.obj")
print(mesh.vertex_count(), "vertices,", mesh.triangle_count(), "triangles")
```

### B. Render a PNG (needs the `[viz]` extra)

```python
from picopie import save_slice_png, mesh_preview

save_slice_png(ball, "slice.png", axis="z", mode="sdf")   # a cross-section image
mesh_preview(ball.to_mesh(), "preview.png")               # a shaded 3D preview
```

- `save_slice_png(..., mode=...)` — `"mask"` (inside/outside), `"sdf"` (distance
  heatmap), or `"gray"`.
- `mesh_preview` renders the mesh with matplotlib (works headless / on a server).

### C. Open an interactive window (needs a display)

```python
picopie.show(ball)              # orbit with the mouse; press Esc to close
```

On a machine with no display (CI, a server) this raises `PicoPieError` — use A or B
there. (More in the [viewer tutorial](../advanced/03-viewer.md).)

## A tiny complete example

```python
import picopie
from picopie import Voxels

with picopie.session(voxel_size_mm=0.2):        # auto-init + shutdown
    rod = Voxels.capsule((-15, 0, 0), (15, 0, 0), radius=3)
    rod.to_mesh().save_stl("rod.stl")
    print("wrote rod.stl —", rod.calculate_properties()[0], "mm³")
```

Using `with picopie.session(...)` is the tidy way to scope a session: it initializes
on entry and shuts down on exit.

Next: [combining shapes with booleans, and exporting →](03-booleans-and-export.md)
