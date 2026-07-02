# Intermediate 3 — Fields and metadata

Voxels describe *shape*. Often you also want *data over space* (a temperature, a
flow vector) or *annotations* (material, units, parameters). PicoPie has fields and
metadata for these — and both travel with your geometry into `.vdb` files.

## Scalar fields

A `ScalarField` stores one float per active voxel — think a heat map or a density.

```python
import numpy as np
import picopie
from picopie import Voxels, ScalarField

picopie.init(voxel_size_mm=0.3)

part  = Voxels.sphere(radius=10)
field = ScalarField.from_voxels(part)        # a field defined where `part` has data

# Single point access:
field.set((0, 0, 0), 100.0)
print(field.get((0, 0, 0)))                  # 100.0   (None if that voxel is inactive)

# Bulk access (fast — uses the compiled path):
pts  = np.array([(0, 0, 0), (5, 0, 0), (0, 5, 0)], np.float32)
vals = np.array([10.0, 20.0, 30.0], np.float32)
field.set_many(pts, vals)

out, found = field.get_many(pts)             # out: (N,) float32, found: (N,) bool
print(out, found)
```

- `get(p)` returns `None` for an inactive voxel; `get_many` returns a parallel
  `found` mask instead.
- `set_many` / `get_many` take `(N, 3)` position arrays — always prefer them over a
  Python loop of `set`/`get` (see the [performance tutorial](../advanced/01-performance.md)).

You can also seed a field from a volume's signed distance:

```python
near = ScalarField.build_from_voxels(part, value=1.0, sd_threshold=0.5)
```

## Vector fields

A `VectorField` stores a 3-vector per voxel (e.g. a direction or displacement):

```python
from picopie import VectorField

vf = VectorField.from_voxels(part)
vf.set((0, 0, 0), (1.0, 0.0, 0.0))
print(vf.get((0, 0, 0)))                     # array([1., 0., 0.], dtype=float32)
```

## Metadata

`Metadata` attaches typed key/value annotations (string, float, or 3-vector) to a
`Voxels` or field — perfect for units, material, and parameters. It uses dict-like
syntax:

```python
from picopie import Metadata

md = Metadata.from_voxels(part)
md["material"] = "Ti-6Al-4V"                 # string
md["wall_mm"]  = 1.25                        # float
md["build_dir"] = (0, 0, 1)                  # 3-vector

print(md["material"], md["wall_mm"])
print(md.to_dict())                          # {'material': 'Ti-6Al-4V', 'wall_mm': 1.25, ...}
print("material" in md, "nope" in md)        # True False
```

> Names beginning with `PicoGK.` are **reserved** for the runtime's internal
> bookkeeping — writing one raises `ValueError`, so you can't corrupt grid state.

## It all persists together

Fields and metadata round-trip through `.vdb` alongside the geometry:

```python
from picopie import save_vdb, load_vdb

save_vdb("annotated.vdb", body=part, heat=field)
objs = load_vdb("annotated.vdb")             # {"body": Voxels, "heat": ScalarField}
```

You've now covered the whole modeling data model. The **Advanced** tutorials go into
performance, the reliability guarantees, and the interactive viewer.

→ [Advanced: performance](../advanced/01-performance.md)
