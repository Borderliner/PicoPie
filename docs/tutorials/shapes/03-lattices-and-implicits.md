# Shapes 3 — Lattices, implicits, measurement & colour

The last layer of the ShapeKernel port: lattice-based shapes, implicit (SDF)
primitives, measurement, and the colour/painter visualisation helpers.

## Lattice shapes

`LatticePipe` builds a tube from lattice beams along a spine; `LatticeManifold`
adds self-supporting tear-drop tips for printable overhangs.

```python
import picogk
from picogk.shapes import LatticePipe, LatticeManifold, LocalFrame

picogk.init(0.5)
pipe = LatticePipe(LocalFrame((0, 0, 0)), length=60, radius=8).to_voxels()
manifold = LatticeManifold(LocalFrame((0, 0, 0), local_z=(0, 1, 0)),
                           length=50, radius=6, max_overhang_angle=45).to_voxels()
```

Build raw lattices from points / splines / grids with `picogk.shapes.functions`,
or quick axis-aligned solids with the `*_between_z` helpers:

```python
from picogk.shapes import functions as fn

lat = fn.lat_from_line([[0, 0, 0], [20, 0, 0], [20, 20, 0]], beam=2)
cyl = fn.cylinder_between_z(0, 30, radius=6)        # also box/cone/pipe_between_z
```

## Implicit (SDF) primitives

Each implicit is a callable signed-distance function — render it into a bounding
box, or intersect it with an existing volume (e.g. to fill a part with a gyroid
lattice):

```python
from picogk import Voxels
from picogk.shapes import ImplicitSphere, ImplicitGyroid, ImplicitSuperEllipsoid

# render an SDF straight into voxels
ball = ImplicitSphere(center=(0, 0, 0), radius=10).render(((-12,) * 3, (12,) * 3))

# fill a solid with a gyroid TPMS shell
solid = Voxels.sphere(radius=10)
gyroid_ball = ImplicitGyroid(unit_size=3, thickness_ratio=1).intersect(solid)

blob = ImplicitSuperEllipsoid((0, 0, 0), 8, 8, 8, 1.5, 0.5).render(((-8,) * 3, (8,) * 3))
```

(`ImplicitGenus` is also available.) Implicit rendering invokes the SDF once per
voxel from Python, so keep the bounding box modest.

## Measurement

`picogk.shapes.measure` reports volume, surface area, centre of gravity, and the
inertia tensor (integrated over the meshed solid):

```python
from picogk.shapes import measure as me

vox = ball
print(me.volume(vox), "mm³")
print(me.surface_area(vox), "mm²")
cg = me.centre_of_gravity(vox)
inertia = me.moment_of_inertia(vox, LocalFrame(tuple(cg)), density=7800)  # kg/m³
```

`picogk.shapes.mesh_utils` adds mesh-level helpers: `mesh_from_grid`,
`apply_transformation`, `translate_mesh_onto_frame`, `append`.

## Colour & the mesh painter

`picogk.shapes.colors` provides a named palette, spectra, and value→colour
scales; `painter` splits a mesh into coloured sub-meshes by a property:

```python
from picogk.shapes import Palette, painter
from picogk.shapes.colors import ColorScale3D, rainbow_spectrum

mesh = ball.to_mesh()
scale = ColorScale3D(rainbow_spectrum(), 0.0, 90.0)
groups = painter.split_by_overhang_angle(mesh, scale)   # [(sub_mesh, rgb), ...]
painter.preview(groups)                                 # show in the viewer (needs a display)
```

Colours are plain RGB `0..1` tuples — drop them straight into
`Viewer.set_group_material` or `picogk.show()`. See the
[gallery](../../gallery.md) for rendered examples, and
[QuickLearn](../QuickLearn.md) for the whole API on one page.
