# Intermediate 1 — Implicit modeling with signed-distance functions

Primitives + booleans get you far, but the real power of a voxel kernel is
**implicit modeling**: you define a shape by a function `f(x, y, z)` where `f ≤ 0`
is *inside* the solid. PicoPie evaluates it over a box and fills the voxels.

## The basics

`Voxels.render_implicit_(sdf, bbox)` renders an SDF in place:

```python
import math
import picopie
from picopie import Voxels

picopie.init(voxel_size_mm=0.3)

def sphere_sdf(x, y, z):
    return math.sqrt(x*x + y*y + z*z) - 10        # <= 0 inside radius 10

v = Voxels()                                       # empty volume
v.render_implicit_(sphere_sdf, ((-12, -12, -12), (12, 12, 12)))
```

- `sdf(x, y, z) -> float` — return the **signed distance** (negative inside). It
  doesn't have to be an *exact* distance; the sign and rough magnitude are what
  matter.
- `bbox` is `((xmin, ymin, zmin), (xmax, ymax, zmax))` — only voxels in this box are
  evaluated, so keep it snug around your shape.

> **Performance note:** the callback runs **once per voxel**, from native code, so
> a pure-Python SDF over a big/fine box is the slow path. Keep the bbox tight and
> the voxel size as coarse as your detail allows. For bulk shapes, primitives +
> booleans are far faster.

## Composing shapes inside one callback

The clean way to combine implicit shapes is set operations on distances, *inside a
single render* — `max` = intersection, `min` = union, negation = complement:

```python
def shape(x, y, z):
    ball  = math.sqrt(x*x + y*y + z*z) - 12        # solid ball, r=12
    bore  = 4 - math.sqrt(y*y + z*z)               # solid cylinder along x, r=4
    return max(ball, -bore)                         # ball with the cylinder removed

v = Voxels()
v.render_implicit_(shape, ((-13, -13, -13), (13, 13, 13)))
```

This is preferred over doing a boolean on a separately-rendered cylinder: it's one
pass and always produces a clean level set.

## A gyroid infill — the classic TPMS

Triply-periodic minimal surfaces (gyroid, schwarz-p, …) are a signature use of
implicit modeling — great for lightweight infill:

```python
def gyroid_in_sphere(x, y, z):
    k = 2 * math.pi / 6.0                           # cell size 6 mm
    g = (math.sin(k*x)*math.cos(k*y)
         + math.sin(k*y)*math.cos(k*z)
         + math.sin(k*z)*math.cos(k*x))
    walls  = abs(g) - 0.4                            # thin gyroid walls
    sphere = math.sqrt(x*x + y*y + z*z) - 12         # clip to a sphere
    return max(walls, sphere)                        # walls AND inside the sphere

infill = Voxels()
infill.render_implicit_(gyroid_in_sphere, ((-13, -13, -13), (13, 13, 13)))
infill.to_mesh().save_stl("gyroid_ball.stl")
```

## Mixing implicit and primitives

Render an implicit shape, then combine it with primitives via ordinary booleans:

```python
shell  = Voxels.sphere(radius=12).shell_(1.0)       # a 1 mm spherical shell
filled = shell + infill                             # shell + gyroid lattice inside
```

## A caution on `intersect_implicit_`

There is a `Voxels.intersect_implicit_(sdf)` that clips an *existing* volume by
`sdf ≤ 0`. It works, but the result may not be a clean level set, and chaining two
of them is unsupported (PicoPie raises rather than risk it). **Prefer composing the
clip into one `render_implicit_`** — `max(feature_sdf, clip_sdf)` — which has none
of these caveats.

Next: [meshes, lattices, and files →](02-meshes-and-files.md)
