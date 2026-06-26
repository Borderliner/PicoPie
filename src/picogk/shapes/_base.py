"""Abstract base class and meshing helpers for parametric shapes.

A Pythonic port of LEAP 71 ShapeKernel's ``BaseShape``. A concrete shape
implements :meth:`BaseShape.to_mesh` (sampling its parametric surface into a
triangle :class:`~picogk.Mesh`); :meth:`BaseShape.to_voxels` then rasterises
that mesh into the kernel via :meth:`Voxels.from_mesh`.

C# equivalence: ``mshConstruct()`` -> ``to_mesh()``, ``voxConstruct()`` ->
``to_voxels()``, ``SetTransformation(fn)`` -> the ``transform`` property.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..mesh import Mesh
from ..voxels import Voxels

#: A point-wise vertex transform: takes and returns an ``(N, 3)`` array.
VertexTransform = Callable[[np.ndarray], np.ndarray]


class BaseShape:
    """Base class for parametric shapes (port of ShapeKernel ``BaseShape``)."""

    def __init__(self, transform: VertexTransform | None = None) -> None:
        self._transform = transform

    @property
    def transform(self) -> VertexTransform | None:
        """Optional vectorised vertex transform applied during construction.

        Unlike C#'s point-wise ``fnVertexTransformation`` delegate, this takes
        an ``(N, 3)`` array and must return one of the same shape (numpy-fast).
        """
        return self._transform

    @transform.setter
    def transform(self, fn: VertexTransform | None) -> None:
        self._transform = fn

    def _apply_transform(self, pts: np.ndarray) -> np.ndarray:
        """Apply :attr:`transform` to an ``(N, 3)`` array (identity if unset)."""
        if self._transform is None:
            return pts
        out = np.asarray(self._transform(pts), dtype=np.float64)
        if out.shape != pts.shape:
            raise ValueError(
                f"transform must return an array shaped like its input "
                f"{pts.shape}, got {out.shape}")
        return out

    def to_mesh(self) -> Mesh:
        """Sample the shape's surface into a triangle mesh."""
        raise NotImplementedError

    def to_voxels(self) -> Voxels:
        """Construct the shape as voxels (renders :meth:`to_mesh`)."""
        return Voxels.from_mesh(self.to_mesh())


def quad_grid_to_mesh(grid: np.ndarray) -> Mesh:
    """Triangulate an ``(A, B, 3)`` grid of surface points into a :class:`Mesh`.

    Each cell ``(i, j)`` becomes two triangles ``(p0, p1, p2)`` and
    ``(p0, p2, p3)`` with ShapeKernel's winding, where ``p0..p3`` are the cell
    corners ``[i, j] [i+1, j] [i+1, j+1] [i, j+1]``. Vertices are emitted
    per-triangle (no dedup): ``Voxels.from_mesh`` rasterises geometry, so
    duplicate vertices don't change the result.
    """
    if grid.ndim != 3 or grid.shape[2] != 3:
        raise ValueError(f"grid must be (A, B, 3), got {grid.shape}")
    p0 = grid[:-1, :-1].reshape(-1, 3)
    p1 = grid[1:, :-1].reshape(-1, 3)
    p2 = grid[1:, 1:].reshape(-1, 3)
    p3 = grid[:-1, 1:].reshape(-1, 3)
    nc = p0.shape[0]
    verts = np.empty((nc * 6, 3), dtype=np.float32)
    verts[0::6], verts[1::6], verts[2::6] = p0, p1, p2
    verts[3::6], verts[4::6], verts[5::6] = p0, p2, p3
    tris = np.arange(nc * 6, dtype=np.int32).reshape(-1, 3)
    return Mesh.from_arrays(verts, tris)
