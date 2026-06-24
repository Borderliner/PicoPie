"""Headless visualization: render voxel slices to PNG and preview meshes.

These helpers need optional dependencies (install with ``pip install picopie[viz]``):
PNG output uses Pillow; the 3D mesh preview uses matplotlib (Agg backend, no
display required). Imports are lazy so the core package never depends on them.
"""

from __future__ import annotations

import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PNG output needs Pillow. Install with: pip install 'picopie[viz]'"
        ) from e
    return Image


def colorize(slice_arr: np.ndarray, mode: str = "sdf",
             scale: float | None = None) -> np.ndarray:
    """Map a signed-distance slice (mm, <=0 inside) to a uint8 image.

    - ``mask``: inside (<=0) white, outside black -> (H, W) grayscale.
    - ``sdf``:  diverging blue(inside)-white(surface)-red(outside) -> (H, W, 3).
    - ``gray``: signed distance normalized to 0..255 -> (H, W) grayscale.
    """
    a = np.asarray(slice_arr, dtype=np.float32)
    if mode == "mask":
        return np.where(a <= 0.0, 255, 0).astype(np.uint8)
    if mode == "sdf":
        s = scale if scale else max(float(np.abs(a).max()), 1e-6)
        t = np.clip(a / s, -1.0, 1.0)
        pos = np.clip(t, 0.0, 1.0)        # outside
        neg = np.clip(-t, 0.0, 1.0)       # inside
        r = 1.0 - neg * 0.8
        g = 1.0 - (pos + neg) * 0.8
        b = 1.0 - pos * 0.8
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    if mode == "gray":
        lo, hi = float(a.min()), float(a.max())
        rng = hi - lo if hi > lo else 1.0
        return (((a - lo) / rng) * 255).astype(np.uint8)
    raise ValueError(f"unknown mode {mode!r} (use 'mask', 'sdf', or 'gray')")


def _slicer(voxels, axis: str):
    return {"x": voxels.slice_x, "y": voxels.slice_y, "z": voxels.slice_z}[axis]


def save_slice_png(voxels, path: str, axis: str = "z",
                   index: int | None = None, mode: str = "sdf") -> str:
    """Render one slice of ``voxels`` to a PNG. Defaults to the middle slice."""
    if axis not in _AXES:
        raise ValueError(f"axis must be x/y/z, got {axis!r}")
    Image = _require_pillow()
    _, size = voxels.voxel_dimensions()
    n = int(size[_AXES[axis]])
    if n == 0:
        raise ValueError("voxels are empty (no active slices)")
    if index is None:
        index = n // 2
    img = colorize(_slicer(voxels, axis)(index), mode)
    Image.fromarray(img).save(path)
    return path


def save_slice_sheet(voxels, path: str, axis: str = "z", count: int = 16,
                     mode: str = "mask", cols: int = 4, pad: int = 2) -> str:
    """Render a montage of ``count`` evenly spaced slices to one PNG."""
    if axis not in _AXES:
        raise ValueError(f"axis must be x/y/z, got {axis!r}")
    Image = _require_pillow()
    _, size = voxels.voxel_dimensions()
    n = int(size[_AXES[axis]])
    if n == 0:
        raise ValueError("voxels are empty (no active slices)")
    count = max(1, min(count, n))
    idxs = np.unique(np.linspace(0, n - 1, count).round().astype(int))
    slicer = _slicer(voxels, axis)
    tiles = [colorize(slicer(int(i)), mode) for i in idxs]

    h, w = tiles[0].shape[:2]
    ch = tiles[0].shape[2] if tiles[0].ndim == 3 else 1
    rows = int(np.ceil(len(tiles) / cols))
    fill = 40
    shape = (rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad)
    grid = np.full(shape + ((ch,) if ch == 3 else ()), fill, dtype=np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y0, x0 = r * (h + pad), c * (w + pad)
        grid[y0:y0 + h, x0:x0 + w] = t
    Image.fromarray(grid).save(path)
    return path


def mesh_preview(mesh, path: str | None = None, elev: float = 25.0,
                 azim: float = -60.0, color: str = "#6699cc", size: int = 800):
    """Render a 3D preview of a mesh with matplotlib (Agg, headless).

    Saves a PNG if ``path`` is given; returns the matplotlib Figure either way.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "mesh_preview needs matplotlib. Install with: pip install 'picopie[viz]'"
        ) from e

    verts = mesh.vertices
    tris = mesh.triangles
    if len(tris) == 0:
        raise ValueError("mesh has no triangles to preview")

    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    polys = Poly3DCollection(verts[tris], alpha=1.0)
    polys.set_facecolor(color)
    polys.set_edgecolor((0, 0, 0, 0.08))
    ax.add_collection3d(polys)

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2 or 1.0
    for setlim, c in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setlim(ctr[c] - rad, ctr[c] + rad)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    return fig
