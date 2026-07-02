"""Measurement utilities (port of ShapeKernel ``Measure``).

Volume, surface area, centre of gravity, and the inertia tensor of a voxel
field. Volume/area match C# exactly (same mesh). Centre-of-gravity and inertia
are integrated analytically over the *meshed solid* (a closed-mesh divergence /
tetrahedron integration) rather than C#'s narrow-band voxel sampling — so they
are exact for the meshed shape and validated against analytic formulas.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ..voxels import Voxels

# canonical tetra (0,e1,e2,e3) covariance integral of x_i x_j dV
_C_CANON = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 120.0


def volume(voxels: Voxels) -> float:
    """Volume of the voxel field in mm^3 (cleaned, via ``calculate_properties``)."""
    return voxels.calculate_properties()[0]


def triangle_area(a, b, c) -> float:
    """Area (mm^2) of a single triangle."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def surface_area(obj) -> float:
    """Total surface area (mm^2) of a :class:`Voxels` or :class:`Mesh`."""
    mesh = obj.to_mesh() if isinstance(obj, Voxels) else obj
    tri = mesh.vertices.astype(np.float64)[mesh.triangles]      # (M, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * float(np.linalg.norm(cross, axis=1).sum())


def _solid_tetra_data(mesh: Mesh, verts: np.ndarray):
    """Per-triangle signed tetra (origin,a,b,c): returns (det, A) arrays."""
    tri = verts[mesh.triangles]                                # (M, 3, 3)
    a_mat = np.transpose(tri, (0, 2, 1))                        # columns = a,b,c
    det = np.linalg.det(a_mat)                                 # = 6 * signed vol
    return det, a_mat


def centre_of_gravity(voxels: Voxels) -> np.ndarray:
    """Centroid (mm) of the solid, integrated over the meshed volume."""
    mesh = voxels.to_mesh()
    verts = mesh.vertices.astype(np.float64)
    tri = verts[mesh.triangles]                                # (M, 3, 3)
    det = np.linalg.det(np.transpose(tri, (0, 2, 1)))          # 6*signed vol
    total = det.sum()
    if total == 0:
        raise ValueError("cannot compute centre of gravity of an empty volume")
    # tetra centroid = (0 + a + b + c)/4; weight by signed volume
    return (det[:, None] * tri.sum(axis=1)).sum(axis=0) / (4.0 * total)


def moment_of_inertia(voxels: Voxels, ref_frame, density: float) -> np.ndarray:
    """Inertia tensor (kg*mm^2) about ``ref_frame``'s origin, in its local axes.

    ``density`` is in kg/m^3 and assumed homogeneous. The result is integrated
    over the meshed solid (exact for the mesh), unlike C#'s voxel sampling.
    """
    mesh = voxels.to_mesh()
    verts = mesh.vertices.astype(np.float64)
    # express vertices in the reference frame's local coordinates
    rel = verts - np.asarray(ref_frame.position, dtype=np.float64)
    local = np.column_stack([rel @ np.asarray(ref_frame.local_x, dtype=np.float64),
                             rel @ np.asarray(ref_frame.local_y, dtype=np.float64),
                             rel @ np.asarray(ref_frame.local_z, dtype=np.float64)])
    det, a_mat = _solid_tetra_data(mesh, local)
    # covariance C = sum det * A @ Ccanon @ A.T  (signed -> closed-mesh interior)
    cov = (det[:, None, None] * (a_mat @ _C_CANON @ np.transpose(a_mat, (0, 2, 1)))).sum(axis=0)
    rho = density * 1.0e-9                                     # kg/mm^3
    return rho * (np.trace(cov) * np.eye(3) - cov)
