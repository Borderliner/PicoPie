# Novice 3 — Booleans, offsets, and hollowing

Real parts come from *combining* and *modifying* solids. PicoPie gives you CSG
booleans, surface offsets, and shelling.

## Booleans

Use the operators — they read like set algebra and return a **new** `Voxels`:

```python
import picopie
from picopie import Voxels

picopie.init(voxel_size_mm=0.2)

a = Voxels.sphere(radius=10)
b = Voxels.sphere(center=(8, 0, 0), radius=8)

union     = a + b      # or a | b   (everything in either)
cut       = a - b      #            (a with b removed)
overlap   = a & b      #            (only where both)
```

Each operator makes a copy, so `a` and `b` are unchanged. If you want to modify a
volume **in place** (faster — no copy), use the trailing-underscore methods:

```python
part = Voxels.sphere(radius=10)
hole = Voxels.sphere(center=(8, 0, 0), radius=8)
part.bool_subtract_(hole)          # mutates `part`; same as `part -= hole`
```

> **Convention:** a trailing `_` means "in place" (mutates `self`, returns `self`).
> No underscore means "functional" (returns a new object). This pattern runs
> through the whole API (`offset_` vs `offset`, etc.).

## Offsets — grow and shrink

Offsetting moves the surface outward (positive) or inward (negative):

```python
grown  = part.offset(2.0)      # 2 mm larger all around (new Voxels)
part.offset_(-1.0)             # 1 mm smaller, in place

# A double offset (grow then shrink, or vice-versa) rounds edges / fills gaps —
# it is NOT a hollow:
rounded = Voxels.sphere(radius=10).double_offset_(2.0, -2.0)
```

## Hollowing — `shell_`

To make a wall of a given thickness (hollow the inside):

```python
part = Voxels.sphere(radius=12)
part.shell_(1.5)               # 1.5 mm wall, hollow inside (in place)
```

`shell_` keeps a wall of `thickness_mm` just inside the current surface
(implemented as `solid − erode(solid, thickness)`).

## Putting it together — a vented hollow ball

```python
import picopie
from picopie import Voxels

with picopie.session(voxel_size_mm=0.2):
    part = Voxels.sphere(radius=12)
    part -= Voxels.sphere(center=(9, 0, 0), radius=6)   # cut an opening
    part.shell_(1.5)                                    # hollow it

    vol, bbox = part.calculate_properties()
    print(f"{vol:.0f} mm³, bbox {bbox.size.round(1)} mm")
    part.to_mesh().save_stl("vented_ball.stl")
```

## Querying geometry

Beyond `calculate_properties()`, you can probe a volume:

```python
part.is_inside((0, 0, 0))           # True/False
part.closest_point((20, 0, 0))      # nearest point on the surface → np.array
part.surface_normal((12, 0, 0))     # outward normal at a surface point
part.volume_mm3()                   # fast raw volume (calculate_properties is the accurate one)
```

You now know enough to model real parts. The **Intermediate** tutorials add
implicit (math-defined) modeling, mesh import, fields, and file I/O.

→ [Intermediate: implicit modeling](../intermediate/01-implicit-modeling.md)
