"""Per-property mesh colouring (port of ShapeKernel ``MeshPainter``).

Split a mesh into coloured sub-meshes by overhang angle or a custom triangle
property, each coloured via an :mod:`~picopie.shapes.colors` scale. The split
functions return ``[(Mesh, rgb), ...]`` (display-independent); :func:`preview`
shows them in the interactive viewer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..mesh import Mesh


def _group(verts, tris, values, scale, n_classes):
    lo, hi = scale.min_value, scale.max_value
    step = (hi - lo) / (n_classes - 1)
    ratio = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    idx = (ratio * (n_classes - 1)).astype(int)
    groups = []
    for k in range(n_classes):
        mask = idx == k
        if not mask.any():
            continue
        groups.append((Mesh.from_arrays(verts, tris[mask]), scale.color(lo + k * step)))
    return groups


def split_by_overhang_angle(mesh: Mesh, scale, n_classes: int = 30,
                            only_down_facing: bool = False):
    """Split by triangle overhang angle (deg: 0 = vertical wall, 90 = horizontal).

    Returns ``[(sub_mesh, rgb), ...]``. ``scale`` provides the min/max angle and
    the colour mapping.
    """
    verts = mesh.vertices.astype(np.float64)
    tris = mesh.triangles
    tv = verts[tris]
    a, b, c = tv[:, 0], tv[:, 1], tv[:, 2]
    normal = np.cross(a - b, c - b)
    norms = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, norms, out=np.zeros_like(normal), where=norms > 0)
    d_r = np.hypot(normal[:, 0], normal[:, 1])
    d_z = np.abs(normal[:, 2])
    angle = np.clip(np.degrees(np.arctan2(d_z, d_r)), 0.0, 90.0)
    if only_down_facing:
        angle = np.where(normal[:, 2] < 0, 0.0, angle)
    return _group(verts, tris, angle, scale, n_classes)


def split_by_property(mesh: Mesh, scale,
                      color_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                      n_classes: int = 30):
    """Split by a custom per-triangle value.

    ``color_func(a, b, c)`` receives the three corner arrays ``(M, 3)`` and
    returns ``(M,)`` values (numpy-vectorised).
    """
    verts = mesh.vertices.astype(np.float64)
    tris = mesh.triangles
    tv = verts[tris]
    values = np.asarray(color_func(tv[:, 0], tv[:, 1], tv[:, 2]), dtype=np.float64)
    return _group(verts, tris, values, scale, n_classes)


def preview(groups, *, title: str = "PicoPie", size: tuple[int, int] = (1280, 800)):
    """Show coloured ``(mesh, rgb)`` groups in the interactive viewer (blocks)."""
    from ..viewer import Viewer
    with Viewer(title=title, size=size) as v:
        for i, (obj, rgb) in enumerate(groups):
            v.add(obj, group=i)
            v.set_group_material(i, rgb)
        v.run()
