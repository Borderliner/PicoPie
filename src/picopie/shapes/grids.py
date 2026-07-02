"""2-D point-grid helpers (port of ShapeKernel ``GridOperations``).

A grid is an ``(rows, cols, 3)`` numpy array (rows = x, cols = y), matching
ShapeKernel's ``List<List<Vector3>>``. Functions return new arrays.
"""

from __future__ import annotations

import numpy as np


def _grid(g) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 3 or g.shape[2] != 3:
        raise ValueError(f"grid must be (rows, cols, 3), got {g.shape}")
    return g


def inverse(grid) -> np.ndarray:
    """Swap rows (x) and columns (y) — a transpose of the first two axes."""
    return np.swapaxes(_grid(grid), 0, 1).copy()


def add_row_x(grid, points) -> np.ndarray:
    """Append a row (in x) at the end."""
    grid = _grid(grid)
    row = np.asarray(points, dtype=np.float64).reshape(1, -1, 3)
    return np.concatenate([grid, row], axis=0)


def remove_row_x(grid, index: int) -> np.ndarray:
    """Remove the row (in x) at ``index``."""
    return np.delete(_grid(grid), int(index), axis=0)


def row_x(grid, index: int) -> np.ndarray:
    """The row (in x) at ``index`` as an ``(cols, 3)`` array."""
    return _grid(grid)[int(index)].copy()


def add_col_y(grid, points) -> np.ndarray:
    """Append a column (in y) at the end."""
    grid = _grid(grid)
    col = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    return np.concatenate([grid, col], axis=1)


def remove_col_y(grid, index: int) -> np.ndarray:
    """Remove the column (in y) at ``index``."""
    return np.delete(_grid(grid), int(index), axis=1)


def col_y(grid, index: int) -> np.ndarray:
    """The column (in y) at ``index`` as an ``(rows, 3)`` array."""
    return _grid(grid)[:, int(index)].copy()
