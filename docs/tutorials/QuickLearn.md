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
import picogk
from picogk import (Voxels, Mesh, Lattice, ScalarField, VectorField,
                    Metadata, save_vdb, load_vdb)

# ---- SESSION ---------------------------------------------------------------
# One session = one native instance with a fixed voxel size (the resolution, mm).
picogk.init(voxel_size_mm=0.3)        # call once before modeling
picogk.version()                      # "26.2.0"
picogk.voxel_size()                   # 0.3
# ... or scope it:  with picogk.session(0.3): ...     # auto init + shutdown

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
from picogk import save_slice_png, save_slice_sheet, mesh_preview
save_slice_png(part, "z.png", axis="z", mode="sdf")   # mask | sdf | gray
save_slice_sheet(part, "sheet.png", count=16)          # montage of slices
mesh_preview(part.to_mesh(), "preview.png")            # matplotlib 3D preview

# ---- INTERACTIVE VIEWER  (needs a display; main thread only) ---------------
picogk.show(part)                     # window: left-drag orbit, scroll zoom, Esc close
picogk.render_png(part, "render.png", size=(1920, 1080))   # offscreen one-shot
from picogk import Viewer
with Viewer() as viewer:              # 'with' guarantees cleanup (close() is required)
    viewer.add(part).set_group_material(0, (0.35, 0.6, 0.95))
    viewer.screenshot("shot.png")
    # viewer.run()                    # blocks until the window closes

# ---- RELIABILITY -----------------------------------------------------------
# Native errors are CATCHABLE, never fatal:
from picogk._errors import PicoGKError, InvalidHandleError
try:
    part.intersect_implicit_(lambda x, y, z: x)   # sharp edge: prefer composing in render_implicit_
except PicoGKError as e:
    print(e)
# NaN/inf geometry inputs raise ValueError up front (would otherwise segfault/hang).
# Wrong-typed args raise TypeError. Using a closed object raises InvalidHandleError.

# ---- LIFETIME --------------------------------------------------------------
part.close()                          # optional for most objects (GC frees them);
                                      # REQUIRED for Viewer (use 'with' or run()).
picogk.shutdown()                     # end the session (also runs atexit)
```

For deeper, narrative coverage see the tutorials:
[Novice](novice/01-setup.md) · [Intermediate](intermediate/01-implicit-modeling.md) ·
[Advanced](advanced/01-performance.md).
