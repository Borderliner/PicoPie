"""Mesh construction / transformation helpers (port of ShapeKernel ``MeshUtility``).

Build meshes from point grids/quads, transform a mesh's vertices, move a mesh
between frames, and append meshes. Functions return new :class:`~picogk.Mesh`
objects (the vertex transform is vectorised: it takes/returns an ``(N, 3)`` array).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..mesh import Mesh
from ..voxels import Voxels
from .frames import LocalFrame


def _v(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).reshape(3)


def mesh_from_quad(p1, p2, p3, p4) -> Mesh:
    """A mesh of two triangles spanning a quad (ShapeKernel winding)."""
    p1, p2, p3, p4 = _v(p1), _v(p2), _v(p3), _v(p4)
    verts = np.array([p4, p1, p2, p2, p3, p4])
    return Mesh.from_arrays(verts, np.arange(6, dtype=np.int32).reshape(2, 3))


def mesh_from_grid(grid) -> Mesh:
    """A mesh triangulating an ``(rows, cols, 3)`` point grid (ShapeKernel winding)."""
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 3 or g.shape[2] != 3:
        raise ValueError(f"grid must be (rows, cols, 3), got {g.shape}")
    p1 = g[:-1, :-1].reshape(-1, 3)
    p2 = g[:-1, 1:].reshape(-1, 3)
    p3 = g[1:, 1:].reshape(-1, 3)
    p4 = g[1:, :-1].reshape(-1, 3)
    nc = p1.shape[0]
    verts = np.empty((nc * 6, 3), dtype=np.float64)
    verts[0::6], verts[1::6], verts[2::6] = p4, p1, p2     # triangle (p4,p1,p2)
    verts[3::6], verts[4::6], verts[5::6] = p2, p3, p4     # triangle (p2,p3,p4)
    return Mesh.from_arrays(verts, np.arange(nc * 6, dtype=np.int32).reshape(-1, 3))


def add_quad(mesh: Mesh, p1, p2, p3, p4) -> Mesh:
    """Append a quad (two triangles) to ``mesh`` in place; returns it."""
    i4 = mesh.add_vertex(_v(p4))
    i1 = mesh.add_vertex(_v(p1))
    i2 = mesh.add_vertex(_v(p2))
    i3 = mesh.add_vertex(_v(p3))
    mesh.add_triangle(i4, i1, i2)
    mesh.add_triangle(i2, i3, i4)
    return mesh


def apply_transformation(mesh: Mesh, transform: Callable[[np.ndarray], np.ndarray]) -> Mesh:
    """A new mesh with ``transform`` applied to every vertex (vectorised ``(N,3)``)."""
    verts = mesh.vertices.astype(np.float64)
    out = np.asarray(transform(verts), dtype=np.float64).reshape(-1, 3)
    if out.shape != verts.shape:
        raise ValueError("transform must return an array shaped like its input")
    return Mesh.from_arrays(out, mesh.triangles)


def vox_apply_transformation(voxels: Voxels,
                             transform: Callable[[np.ndarray], np.ndarray]) -> Voxels:
    """Mesh ``voxels``, transform its vertices, and re-voxelise."""
    return Voxels.from_mesh(apply_transformation(voxels.to_mesh(), transform))


def translate_mesh_onto_frame(mesh: Mesh, input_frame: LocalFrame,
                              output_frame: LocalFrame) -> Mesh:
    """Move a mesh from one local frame to another (express in input, place in output)."""
    verts = mesh.vertices.astype(np.float64)
    rel = verts - np.asarray(input_frame.position, dtype=np.float64)
    local = np.column_stack([rel @ np.asarray(input_frame.local_x, dtype=np.float64),
                             rel @ np.asarray(input_frame.local_y, dtype=np.float64),
                             rel @ np.asarray(input_frame.local_z, dtype=np.float64)])
    world = (np.asarray(output_frame.position, dtype=np.float64)
             + local[:, 0:1] * np.asarray(output_frame.local_x, dtype=np.float64)
             + local[:, 1:2] * np.asarray(output_frame.local_y, dtype=np.float64)
             + local[:, 2:3] * np.asarray(output_frame.local_z, dtype=np.float64))
    return Mesh.from_arrays(world, mesh.triangles)


def append(mesh1: Mesh, mesh2: Mesh) -> Mesh:
    """A new mesh combining the triangles of both inputs."""
    v1, v2 = mesh1.vertices, mesh2.vertices
    verts = np.concatenate([v1, v2], axis=0)
    tris = np.concatenate([mesh1.triangles, mesh2.triangles + len(v1)], axis=0)
    return Mesh.from_arrays(verts, tris)
