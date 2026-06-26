# Shapes 2 — Frames, splines & spined shapes

The frame-based shapes from [part 1](01-parametric-shapes.md) get their real
power from **spines**: instead of a straight axis, a shape can follow an
arbitrary curve, carrying a field of local frames along it.

## Splines

Build smooth curves from control points:

```python
import picogk
from picogk.shapes import ControlPointSpline, TangentialControlSpline

picogk.init(0.5)

# a B-spline through control points (open by default; closed=True for a loop)
curve = ControlPointSpline([[0, 0, 0], [0, 40, 0], [0, 50, 20], [0, 60, 60]])
pts = curve.points(500)            # (500, 3) sampled points
mid = curve.point_at(0.5)          # a single point at length-ratio 0.5

# a curve from endpoints + tangent directions (a smooth blend)
blend = TangentialControlSpline([0, 0, 0], [50, 0, 0], [0, 1, 0], [0, -1, 0])
```

Also available: `ControlPointSurface` (a 2D B-spline patch) and
`CylindricalControlSpline` (built from radial/tangential/axial steps).

## Frames — a spine of local frames

A `Frames` object samples a field of local frames along a path. The classmethods
choose how the local axes are oriented:

```python
from picogk.shapes import Frames, LocalFrame

spine_pts = ControlPointSpline([[0, 0, 0], [0, 40, 0], [0, 50, 30]]).points(500)

# carry a constant frame's axes along the curve
f1 = Frames.along_spline(spine_pts, LocalFrame())

# tangent-following Z, with X aligned to a target / a coordinate system
f2 = Frames.aligned_to_x(spine_pts, target_x=(0, 1, 0))
f3 = Frames.aligned(spine_pts, "cylindrical")   # "z" | "cylindrical" | "spherical" | "min_rotation"

# a straight extrusion (what the non-spine constructors use internally)
f4 = Frames.extrude(length=40, frame=LocalFrame())

frame_at_mid = f3.frame_at(0.5)     # a LocalFrame anywhere along the spine
```

`"min_rotation"` transports the frame with minimal twist — ideal for pipes that
shouldn't rotate as they bend.

## Spined shapes

Pass `frames=` instead of a frame + length, and the shape follows the spine.
`Cylinder`, `Box`, `Pipe`, `PipeSegment`, and `LatticePipe` all support it:

```python
from picogk.shapes import Cylinder, Pipe

spine = Frames.aligned_to_x(spine_pts, target_x=(0, 1, 0))

tube = Pipe(frames=spine, inner_radius=4, outer_radius=8).to_voxels()
rod  = Cylinder(frames=spine, radius=lambda phi, lr: 8 + 2 * lr).to_voxels()
```

Because the radius is still a modulation, you can taper or flute a shape *as it
travels along the spine*.

## Surfaces of revolution

`Revolve` sweeps a profile (carried along a spine, offset inward/outward) around
an axis frame's local Z:

```python
from picogk.shapes import Revolve

axis = LocalFrame((0, 0, 0))
profile = Frames.extrude(40, LocalFrame((0, 0, 0)))
vase = Revolve(axis, profile, inner_radius=0,
               outer_radius=lambda lr: 10 + 4 * (lr - 0.5) ** 2).to_voxels()
```

`Revolve.frames_from_contour(contour)` builds a suitable spine from a
radius-vs-height contour.

## Point-list utilities

`picogk.shapes.spline_ops` operates on point lists — arc-length resampling, NURB
smoothing, transforms, nearest-point queries:

```python
from picogk.shapes import spline_ops as so

even = so.reparametrized(spine_pts, 200)     # constant arc-length spacing
smooth = so.nurb(spine_pts, 200)             # NURB-smoothed
length = so.total_length(spine_pts)
```

Next: [Lattices, implicits & visualization](03-lattices-and-implicits.md).
