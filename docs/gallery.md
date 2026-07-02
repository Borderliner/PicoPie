# Gallery

A selection of shapes built with [`picopie.shapes`](tutorials/shapes/01-parametric-shapes.md)
and rendered with `render_png` / the interactive viewer. Every image is produced
by the runnable [`examples/shapekernel/gallery.py`](https://github.com/Borderliner/PicoPie/blob/main/examples/shapekernel/gallery.py)
(a port of LEAP 71's ShapeKernel examples):

```bash
python examples/shapekernel/gallery.py out_dir   # render every scene (needs a display)
python examples/shapekernel/gallery.py --show sphere
```

## Base shapes

| Sphere (modulated radii) | Ring (modulated tube) | Lens (curved profiles) |
|---|---|---|
| ![sphere](images/gallery/sphere.png) | ![ring](images/gallery/ring.png) | ![lens](images/gallery/lens.png) |

## Pipes & segments

| Pipe (transformed + modulated) | Pipe segments (angular slices) |
|---|---|
| ![pipe](images/gallery/pipe.png) | ![pipe segment](images/gallery/pipe_segment.png) |

## Lattices & implicits

| Lattice manifold (printable tips) | Gyroid sphere | Gyroid genus |
|---|---|---|
| ![lattice manifold](images/gallery/lattice_manifold.png) | ![gyroid sphere](images/gallery/gyroid_sphere.png) | ![gyroid genus](images/gallery/gyroid_genus.png) |

## Implicits & painter

| Super-ellipsoids | Mesh painter (overhang angle) |
|---|---|
| ![superellipsoid](images/gallery/superellipsoid.png) | ![mesh painter](images/gallery/mesh_painter.png) |
