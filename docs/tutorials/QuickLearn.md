# Learn PicoPie in Y minutes

The whole API as one annotated script (learn-x-in-y-minutes style). Every line is
real, runnable code. Headless by default; the viewer bits need a display.

```python
# ============================================================================
# INSTALL    pip install "picopie[viz]"      (or: uv add "picopie[viz]")
#            Self-contained wheels: native runtime + deps bundled. No .NET/compiler.
# ============================================================================

import math
import numpy as np
import picopie
from picopie import (Voxels, Mesh, Lattice, ScalarField, VectorField,
                    Metadata, save_vdb, load_vdb)

# ---- SESSION ---------------------------------------------------------------
# One session = one native instance with a fixed voxel size (the resolution, mm).
picopie.init(voxel_size_mm=0.3)        # call once before modeling
picopie.version()                      # "26.2.0"
picopie.voxel_size()                   # 0.3
# ... or scope it:  with picopie.session(0.3): ...     # auto init + shutdown

# ---- VOXELS: the core object (a signed-distance / level-set volume) ---------
# value <= 0 is INSIDE. Two primitives: sphere and capsule.
ball = Voxels.sphere(center=(0, 0, 0), radius=10)
rod  = Voxels.capsule((-15, 0, 0), (15, 0, 0), radius=3)          # rounded cylinder
cone = Voxels.capsule((0, 0, -8), (0, 0, 8), radius=6, radius2=1) # tapered

# ---- BOOLEANS --------------------------------------------------------------
# Operators return a NEW volume (a, b unchanged):
union = ball + rod        # or ball | rod
cut   = ball - rod
both  = ball & rod
# Trailing-underscore methods mutate IN PLACE (faster, no copy):
part = Voxels.sphere(radius=10)
part.bool_subtract_(Voxels.sphere(center=(8, 0, 0), radius=8))    # also: part -= other
# Convention everywhere:  name_()  = in place;  name()  = returns new.

# ---- OFFSET / SHELL --------------------------------------------------------
grown = part.offset(2.0)              # grow 2 mm (new);  part.offset_(-1) shrinks in place
part.shell_(1.5)                      # hollow to a 1.5 mm wall (in place)
part.double_offset_(2.0, -2.0)        # grow-then-shrink = rounding/fillet (NOT a hollow)

# ---- IMPLICIT MODELING: define a shape by f(x,y,z) <= 0 --------------------
def gyroid(x, y, z):
    k = 2 * math.pi / 6.0             # 6 mm cell
    g = (math.sin(k*x)*math.cos(k*y) + math.sin(k*y)*math.cos(k*z)
         + math.sin(k*z)*math.cos(k*x))
    return max(abs(g) - 0.4,                                  # thin walls ...
               math.sqrt(x*x + y*y + z*z) - 12)              # ... clipped to a sphere
v = Voxels()
v.render_implicit_(gyroid, ((-13, -13, -13), (13, 13, 13)))  # sdf runs once PER VOXEL
# Compose with max()=intersect, min()=union, -f=complement, INSIDE one callback.
# (Keep the bbox tight: a pure-Python sdf over a big/fine box is the slow path.)

# ---- QUERIES ---------------------------------------------------------------
part.volume_mm3()                     # fast raw volume
vol, bbox = part.calculate_properties()   # accurate volume + surface bbox (matches C#)
part.is_inside((0, 0, 0))             # bool
part.closest_point((20, 0, 0))       # nearest surface point -> np.array (or None)
part.surface_normal((10, 0, 0))      # outward normal at a surface point
part.ray_cast((20, 0, 0), (-1, 0, 0))   # first surface hit along a ray (or None)
part.bounding_box()                   # voxel-grid bbox (looser than the surface bbox)

# ---- MESH <-> VOXELS -------------------------------------------------------
mesh = part.to_mesh()                 # marching cubes
mesh.vertices                         # (N,3) float32 NumPy  | mesh.triangles (M,3) int32
mesh.save_stl("part.stl")             # or .save_obj(...)
back = Voxels.from_mesh(mesh)         # voxelize a closed mesh
imported = Mesh.load_stl("in.stl")    # binary or ASCII; also Mesh.load_obj
tet = Mesh.from_arrays(np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.int32))

# ---- LATTICE (beams + nodes) -> voxels -------------------------------------
lat = Lattice()
lat.add_sphere((-10, 0, 0), 2.0); lat.add_sphere((10, 0, 0), 2.0)
lat.add_beam((-10, 0, 0), (10, 0, 0), 1.0, 1.0)     # start, end, r_start, r_end
struts = lat.to_voxels()

# ---- FIELDS (data over space) ----------------------------------------------
f = ScalarField.from_voxels(part)     # one float per active voxel
f.set((0, 0, 0), 100.0); f.get((0, 0, 0))          # single point (get -> None if inactive)
pts  = np.array([(0, 0, 0), (5, 0, 0)], np.float32)
f.set_many(pts, np.array([1.0, 2.0], np.float32))  # BULK = fast; prefer over loops
vals, found = f.get_many(pts)                       # (N,) float32, (N,) bool
vf = VectorField.from_voxels(part); vf.set((0, 0, 0), (1, 0, 0))   # 3-vector per voxel

# ---- METADATA (typed annotations: str / float / 3-vector) ------------------
md = Metadata.from_voxels(part)
md["material"] = "Ti-6Al-4V"; md["wall_mm"] = 1.25; md["dir"] = (0, 0, 1)
md["material"]; "wall_mm" in md; md.to_dict()
# Names starting with "PicoGK." are reserved -> writing one raises ValueError.

# ---- FILE I/O (OpenVDB; voxels + fields, one file) -------------------------
save_vdb("model.vdb", body=part, heat=f)            # name=object
objs = load_vdb("model.vdb")                         # {"body": Voxels, "heat": ScalarField}

# ---- HEADLESS VISUALIZATION  (needs picopie[viz]) --------------------------
from picopie import save_slice_png, save_slice_sheet, mesh_preview
save_slice_png(part, "z.png", axis="z", mode="sdf")   # mask | sdf | gray
save_slice_sheet(part, "sheet.png", count=16)          # montage of slices
mesh_preview(part.to_mesh(), "preview.png")            # matplotlib 3D preview

# ---- INTERACTIVE VIEWER  (needs a display; main thread only) ---------------
picopie.show(part)                     # window: left-drag orbit, scroll zoom, Esc close
picopie.render_png(part, "render.png", size=(1920, 1080))   # offscreen one-shot
from picopie import Viewer
with Viewer() as viewer:              # 'with' guarantees cleanup (close() is required)
    viewer.add(part).set_group_material(0, (0.35, 0.6, 0.95))
    viewer.screenshot("shot.png")
    # viewer.run()                    # blocks until the window closes

# ---- PARAMETRIC SHAPES  (picopie.shapes: a Pythonic ShapeKernel port) -------
from picopie.shapes import (Sphere, Box, Cylinder, Cone, Ring, Lens, Pipe,
                           PipeSegment, Revolve, LocalFrame, Frames)
# Build engineering primitives from a local frame; .to_voxels() / .to_mesh().
f = LocalFrame(position=(0, 0, 0))            # position + orthonormal x/y/z axes
ball   = Sphere(f, radius=10).to_voxels()
box    = Box(f, length=20, width=10, depth=8).to_voxels()
cyl    = Cylinder(f, length=30, radius=8).to_voxels()
cone   = Cone(f, 30, start_radius=8, end_radius=2).to_voxels()
ring   = Ring(f, ring_radius=20, radius=5).to_voxels()          # torus
tube   = Pipe(f, length=30, inner_radius=4, outer_radius=8).to_voxels()  # hollow

# Dimensions accept a number, a numpy lambda, or a Modulation (the parametric part):
wavy   = Sphere(radius=lambda phi, theta: 20 + 3 * np.cos(6 * phi)).to_voxels()
fluted = Cylinder(f, 40, radius=lambda phi, lr: 10 - 3 * np.cos(8 * lr)).to_voxels()

# SPINED: follow a curve. Frames carry local frames along a spline.
from picopie.shapes import ControlPointSpline
spine  = Frames.aligned_to_x(ControlPointSpline([[0,0,0],[0,40,0],[0,50,30]]).points(500),
                             target_x=(0, 1, 0))
bent   = Pipe(frames=spine, inner_radius=3, outer_radius=6).to_voxels()

# LATTICES + IMPLICITS (SDFs you can render or intersect):
from picopie.shapes import LatticePipe, ImplicitGyroid
lat    = LatticePipe(f, length=60, radius=8).to_voxels()        # tube of beams
gyroid = ImplicitGyroid(unit_size=3, thickness_ratio=1).intersect(ball)  # TPMS-filled

# MEASURE + COLOUR:
from picopie.shapes import measure as me, Palette
print(me.volume(ball), me.surface_area(ball), me.centre_of_gravity(ball))
picopie.show(box) if False else None           # colours are RGB 0..1 (Palette.BLUE, ...)

# ---- RELIABILITY -----------------------------------------------------------
# Native errors are CATCHABLE, never fatal:
from picopie._errors import PicoPieError, InvalidHandleError
try:
    part.intersect_implicit_(lambda x, y, z: x)   # sharp edge: prefer composing in render_implicit_
except PicoPieError as e:
    print(e)
# NaN/inf geometry inputs raise ValueError up front (would otherwise segfault/hang).
# Wrong-typed args raise TypeError. Using a closed object raises InvalidHandleError.

# ---- LIFETIME --------------------------------------------------------------
part.close()                          # optional for most objects (GC frees them);
                                      # REQUIRED for Viewer (use 'with' or run()).
picopie.shutdown()                     # end the session (also runs atexit)
```

For deeper, narrative coverage see the tutorials:
[Novice](novice/01-setup.md) · [Intermediate](intermediate/01-implicit-modeling.md) ·
[Advanced](advanced/01-performance.md) · [Shapes](shapes/01-parametric-shapes.md),
and the [gallery](../gallery.md).
