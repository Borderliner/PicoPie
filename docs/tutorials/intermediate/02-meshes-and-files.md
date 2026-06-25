# Intermediate 2 — Meshes, lattices, and file I/O

This tutorial covers moving geometry in and out of PicoPie: meshes (STL/OBJ),
lattices (beam/strut structures), and OpenVDB files.

## Meshes ↔ Voxels

Go from a volume to a mesh and back:

```python
import numpy as np
import picogk
from picogk import Voxels, Mesh

picogk.init(voxel_size_mm=0.3)

part = Voxels.sphere(radius=10)
mesh = part.to_mesh()                 # marching-cubes surface

verts = mesh.vertices                 # (N, 3) float32 NumPy array (mm)
tris  = mesh.triangles                # (M, 3) int32 vertex indices
print(verts.shape, tris.shape, mesh.bounding_box().size)

solid = Voxels.from_mesh(mesh)        # voxelize a (closed) mesh back into a volume
```

`mesh.vertices` / `mesh.triangles` are plain NumPy arrays — use them with any
analysis or export code you like.

### Import an STL/OBJ and voxelize it

```python
imported = Mesh.load_stl("bracket.stl")     # binary or ASCII; also Mesh.load_obj(...)
vox = Voxels.from_mesh(imported)            # now you can offset/shell/boolean it
vox.shell_(1.2)
vox.to_mesh().save_stl("bracket_hollow.stl")
```

### Build a mesh from arrays

```python
verts = np.array([(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)], np.float32)
tris  = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], np.int32)
tet   = Mesh.from_arrays(verts, tris)       # vertices de-duplicated automatically
```

## Lattices — beams and nodes

A `Lattice` is a set of struts (beams) and spheres, voxelized into a solid — ideal
for trusses and scaffolds:

```python
from picogk import Lattice

lat = Lattice()
lat.add_sphere((-10, 0, 0), 2.0)                       # node
lat.add_sphere(( 10, 0, 0), 2.0)
lat.add_beam((-10, 0, 0), (10, 0, 0), 1.0, 1.0)        # strut: start, end, r_start, r_end
beams = lat.to_voxels()                                # -> Voxels (then boolean/mesh it)
```

`add_beam(start, end, radius_start, radius_end=None, round_cap=True)` — a tapered
strut if the two radii differ.

## OpenVDB files — save and load whole models

`save_vdb` / `load_vdb` persist named voxels **and** fields in one `.vdb` file
(readable by other OpenVDB tools, and by C# PicoGK):

```python
from picogk import save_vdb, load_vdb, ScalarField

part = Voxels.sphere(radius=10)
heat = ScalarField.from_voxels(part)            # (fields covered in the next tutorial)

save_vdb("model.vdb", body=part, heat=heat)     # keyword name -> object

objs = load_vdb("model.vdb")                     # -> {"body": Voxels, "heat": ScalarField}
body = objs["body"]
print(body.calculate_properties()[0], "mm³")
```

For finer control (inspecting field names/types before loading) there's the
`VdbFile` class and the `FieldType` enum:

```python
from picogk import VdbFile, FieldType

f = VdbFile.load("model.vdb")
for name, ftype in f.fields():                   # [("body", FieldType.VOXELS), ...]
    print(name, ftype)
part = f.get_voxels(0)                            # type-checked accessor
f.close()
```

Next: [scalar/vector fields and metadata →](03-fields-and-metadata.md)
