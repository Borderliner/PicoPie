# Shapes 1 — Parametric shapes

`picopie.shapes` is a Pythonic port of LEAP 71's
[ShapeKernel](https://github.com/leap71/LEAP71_ShapeKernel) — a higher-level,
*parametric* modeling layer built on the PicoPie core. Instead of composing raw
booleans and SDFs, you instantiate engineering primitives (boxes, spheres,
cylinders, pipes, lenses, rings, ...) defined by a **local frame** and
**modulations**, then rasterise them to voxels.

It's pure Python on top of `Voxels.from_mesh` / `Mesh.from_arrays` — no extra
native code — and it's **parity-tested against the reference C# ShapeKernel** to
float precision.

```python
import picopie
from picopie.shapes import Sphere, Box, Cylinder, LocalFrame

picopie.init(0.5)

part = Sphere(LocalFrame(position=(0, 0, 0)), radius=10).to_voxels()
vol, _ = part.calculate_properties()
print(f"{vol:.0f} mm³")
part.to_mesh().save_stl("ball.stl")
```

Every shape exposes `.to_voxels()` (and `.to_mesh()`) — the Pythonic equivalents
of C#'s `voxConstruct()` / `mshConstruct()`.

## Local frames

A `LocalFrame` is a position plus a right-handed orthonormal basis
(`local_x`, `local_y`, `local_z` with `y = cross(z, x)`). Shapes are built in
their frame's axes, so moving or orienting the frame moves/orients the shape.

```python
from picopie.shapes import LocalFrame

f = LocalFrame(position=(10, 0, 0))               # origin frame, default axes
tilted = LocalFrame((0, 0, 0), local_z=(0, 1, 1)) # choose the Z axis; X auto-picked
tilted2 = LocalFrame((0, 0, 0), local_z=(0, 1, 0), local_x=(1, 0, 0))  # both axes
moved = f.translated((0, 0, 5))                   # copy, shifted
spun = f.rotated(angle=0.5, axis=(0, 0, 1))       # copy, rotated about Z
```

## The base shapes

| Shape | Constructor (essentials) |
|-------|--------------------------|
| `Sphere` | `Sphere(frame, radius=10)` |
| `Box` | `Box(frame, length, width, depth)` |
| `Cylinder` | `Cylinder(frame, length, radius)` |
| `Cone` | `Cone(frame, length, start_radius, end_radius)` |
| `Ring` | `Ring(frame, ring_radius, radius)` (a torus) |
| `Lens` | `Lens(frame, height, inner_radius, outer_radius)` |
| `Pipe` | `Pipe(frame, length, inner_radius, outer_radius)` (a hollow tube) |
| `PipeSegment` | `PipeSegment(frame, length, inner_radius, outer_radius, start, end)` (an angular slice) |
| `Revolve` | `Revolve(axis_frame, spine, inner_radius, outer_radius)` (surface of revolution) |
| `LogoBox` | `LogoBox(frame, length, ref_width, image, mapping)` (image-embossed top) |

```python
from picopie.shapes import Box, Cylinder, Cone, Ring, Lens

picopie.init(0.5)
box  = Box(LocalFrame((-30, 0, 0)), length=20, width=10, depth=8).to_voxels()
cyl  = Cylinder(LocalFrame((0, 0, 0)), length=30, radius=8).to_voxels()
cone = Cone(LocalFrame((30, 0, 0)), length=30, start_radius=8, end_radius=2).to_voxels()
ring = Ring(LocalFrame((0, 40, 0)), ring_radius=20, radius=5).to_voxels()
lens = Lens(LocalFrame((0, -40, 0)), height=6, inner_radius=0, outer_radius=15).to_voxels()
```

## Modulations — the parametric part

Dimensions aren't limited to constants. A radius / width / height accepts a
**number**, a **callable**, or a `Modulation`, so a dimension can vary smoothly
across the shape — a tapering nozzle, a fluted cylinder, a wavy sphere.

- `SurfaceModulation` (2D) — `f(phi, length_ratio)` — for radii (sphere, cylinder,
  pipe, ring, lens).
- `LineModulation` (1D) — `f(length_ratio)` — for box width/depth along the length.

```python
import numpy as np
from picopie.shapes import Sphere, Cylinder

# a sphere whose radius ripples with the azimuthal angle
wavy = Sphere(radius=lambda phi, theta: 20 + 3 * np.cos(6 * phi)).to_voxels()

# a cylinder that bulges along its length
fluted = Cylinder(LocalFrame(), length=40,
                  radius=lambda phi, lr: 10 - 3 * np.cos(8 * lr)).to_voxels()
```

Callables must be numpy-aware (they're evaluated on arrays during meshing).
Modulations compose with `+ - *`:

```python
from picopie.shapes import SurfaceModulation
base = SurfaceModulation(10.0)
ripple = SurfaceModulation(lambda phi, lr: np.cos(5 * phi))
radius = base + 2.0 * ripple        # a SurfaceModulation; pass it as radius=
```

A modulated dimension automatically refines the tessellation (matching C#), so
the curve is smooth.

## Transforming a shape

Pass `transform=` — a **vectorised** point map taking and returning an `(N, 3)`
array — to deform every vertex during construction:

```python
def shear(p):                       # p is (N, 3)
    out = p.copy()
    out[:, 0] += 0.3 * p[:, 2]      # shear X by Z
    return out

sheared = Box(LocalFrame(), 20, 20, 40, transform=shear).to_voxels()
```

## Coming from C# ShapeKernel?

The API is Pythonic (snake_case; numbers/lambdas instead of `Modulation`
objects), but the concepts map 1:1:

| C# ShapeKernel | PicoPie |
|----------------|---------|
| `voxConstruct()` | `.to_voxels()` |
| `mshConstruct()` | `.to_mesh()` |
| `BaseSphere`, `BaseBox`, … | `Sphere`, `Box`, … (drop `Base`) |
| `vecGetSurfacePoint(r1, r2, r3)` | `.surface_point(r1, r2, r3)` |
| `SetRadius(SurfaceModulation(...))` | `radius=` (number \| callable \| `SurfaceModulation`) |
| `SetTransformation(fn)` | `transform=` (vectorised, takes/returns `(N, 3)`) |
| `oFrame.vecGetLocalX()` | `frame.local_x` |

Next: [Frames, splines & spined shapes](02-frames-and-spines.md).
