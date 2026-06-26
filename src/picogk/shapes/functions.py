"""Convenience builders (port of ShapeKernel ``Sh.latFrom*`` + ``CylUtility`` /
``RectUtility``).

Lattice builders turn point lists / splines / grids into a
:class:`~picogk.Lattice`; the ``*_between_z`` helpers make a solid spanning two
heights along an axis. (The obsolete ``Sh.vox*`` delegators are omitted —
PicoPie's :class:`~picogk.Voxels` already exposes offset/boolean/shell/implicit
operations directly.)
"""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice
from ..voxels import Voxels
from .box import Box
from .cylinder import Cone, Cylinder
from .frames import LocalFrame
from .pipe import Pipe


def _pts(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


# --- lattice builders ----------------------------------------------------------
def lat_from_line(points, beam: float) -> Lattice:
    """A lattice of beams connecting consecutive points (a tube along a spline)."""
    return add_line(Lattice(), points, beam)


def add_line(lattice: Lattice, points, beam: float) -> Lattice:
    """Append a connected beam line to ``lattice`` (returns it for chaining)."""
    pts = _pts(points)
    for i in range(1, len(pts)):
        lattice.add_beam(pts[i - 1], pts[i], beam, beam, True)
    return lattice


def lat_from_points(points, beam: float) -> Lattice:
    """A node-only lattice (unconnected spheres) from a point cloud."""
    lat = Lattice()
    for p in _pts(points):
        lat.add_sphere(p, beam)
    return lat


def lat_from_point(point, beam: float) -> Lattice:
    """A single-sphere lattice."""
    return Lattice().add_sphere(np.asarray(point, dtype=np.float64).reshape(3), beam)


def lat_from_edges(edges, beam: float) -> Lattice:
    """A lattice from several point lists / splines."""
    lat = Lattice()
    for edge in edges:
        add_line(lat, edge, beam)
    return lat


def lat_from_grid(grid, beam: float) -> Lattice:
    """A lattice connecting an ``(rows, cols, 3)`` grid along both directions.

    (Upstream's ``latFromGrid`` has a bug that drops all but the last row; this
    connects every row *and* every column as intended.)
    """
    g = np.asarray(grid, dtype=np.float64)
    lat = Lattice()
    for r in range(g.shape[0]):
        add_line(lat, g[r], beam)
    for c in range(g.shape[1]):
        add_line(lat, g[:, c], beam)
    return lat


def lat_from_beam(p1, p2, beam1: float, beam2=None, rounded: bool = True) -> Lattice:
    """A single-beam lattice (constant or tapered radius)."""
    b2 = beam1 if beam2 is None else beam2
    return Lattice().add_beam(np.asarray(p1, dtype=np.float64).reshape(3),
                              np.asarray(p2, dtype=np.float64).reshape(3),
                              beam1, b2, rounded)


# --- z-axis solid helpers ------------------------------------------------------
def _z_frame(start_z: float, frame: LocalFrame | None) -> LocalFrame:
    if frame is None:
        return LocalFrame((0.0, 0.0, start_z))
    return frame.translated(start_z * frame.local_z)


def box_between_z(start_z: float, end_z: float, x_size: float, y_size: float, *,
                  frame: LocalFrame | None = None) -> Voxels:
    """A box spanning ``start_z..end_z`` along Z (or a frame's local Z)."""
    return Box(_z_frame(start_z, frame), end_z - start_z, x_size, y_size).to_voxels()


def cylinder_between_z(start_z: float, end_z: float, radius: float, *,
                       frame: LocalFrame | None = None) -> Voxels:
    """A cylinder spanning ``start_z..end_z`` along Z (or a frame's local Z)."""
    return Cylinder(_z_frame(start_z, frame), end_z - start_z, radius).to_voxels()


def cone_between_z(start_z: float, end_z: float, start_radius: float, end_radius: float,
                   *, frame: LocalFrame | None = None) -> Voxels:
    """A cone spanning ``start_z..end_z`` along Z (or a frame's local Z)."""
    return Cone(_z_frame(start_z, frame), end_z - start_z, start_radius, end_radius).to_voxels()


def pipe_between_z(start_z: float, end_z: float, inner_radius: float, outer_radius: float,
                   *, frame: LocalFrame | None = None) -> Voxels:
    """A pipe spanning ``start_z..end_z`` along Z (or a frame's local Z)."""
    return Pipe(_z_frame(start_z, frame), end_z - start_z, inner_radius, outer_radius).to_voxels()
