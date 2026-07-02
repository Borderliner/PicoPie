"""Vector utilities (port of ShapeKernel ``VecOperations``).

Pure-Python/numpy reimplementation of ShapeKernel's vector helpers — cylindrical
and spherical coordinates, rotations, alignment, and arc interpolations. All
functions take and return ``float64`` ``(3,)`` numpy arrays (point/direction
inputs may be any 3-sequence). The rotation matches C#'s
``Quaternion.CreateFromAxisAngle`` + ``Vector3.Transform`` exactly.

C# names map by dropping the Hungarian prefix: ``vecGetCylPoint`` ->
``cyl_point``, ``fGetRadius`` -> ``planar_radius``, ``vecRotateAroundAxis`` ->
``rotate_around_axis``, etc.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid an import cycle (frames imports this module)
    from .frames import LocalFrame


def _v(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(3)


def safe_normalized(v) -> np.ndarray:
    """Unit vector, or the zero vector if the input has zero length."""
    v = _v(v)
    n = float(np.linalg.norm(v))
    return v / n if n != 0.0 else np.zeros(3)


# --- cylindrical / spherical coordinates ---------------------------------------
def planar_radius(pt) -> float:
    """The XY (planar) radius of a point about the absolute Z axis."""
    pt = _v(pt)
    return float(math.hypot(pt[0], pt[1]))


def phi(pt) -> float:
    """Planar polar angle ``atan2(y, x)`` (radians), cylindrical convention."""
    pt = _v(pt)
    return float(math.atan2(pt[1], pt[0]))


def theta(pt) -> float:
    """Azimuthal angle ``atan2(z, planar_radius)`` (radians), spherical convention."""
    pt = _v(pt)
    return float(math.atan2(pt[2], planar_radius(pt)))


def cyl_point(radius: float, phi_: float, z: float) -> np.ndarray:
    """Cartesian point from cylindrical coordinates (angle in radians)."""
    return np.array([radius * math.cos(phi_), radius * math.sin(phi_), z])


def sph_point(radius: float, phi_: float, theta_: float) -> np.ndarray:
    """Cartesian point from spherical coordinates (angles in radians).

    Matches ShapeKernel's convention: ``(r cosφ cosθ, r sinφ cosθ, r sinθ)``.
    """
    return np.array([
        radius * math.cos(phi_) * math.cos(theta_),
        radius * math.sin(phi_) * math.cos(theta_),
        radius * math.sin(theta_),
    ])


def with_radius(pt, new_radius: float) -> np.ndarray:
    """Same Z/phi, new planar radius."""
    pt = _v(pt)
    return cyl_point(new_radius, phi(pt), pt[2])


def with_phi(pt, new_phi: float) -> np.ndarray:
    """Same Z/radius, new polar angle."""
    pt = _v(pt)
    return cyl_point(planar_radius(pt), new_phi, pt[2])


def with_z(pt, new_z: float) -> np.ndarray:
    """Same radius/phi, new height."""
    pt = _v(pt)
    return cyl_point(planar_radius(pt), phi(pt), new_z)


def shift_radius(pt, d_radius: float) -> np.ndarray:
    pt = _v(pt)
    return cyl_point(planar_radius(pt) + d_radius, phi(pt), pt[2])


def shift_phi(pt, d_phi: float) -> np.ndarray:
    pt = _v(pt)
    return cyl_point(planar_radius(pt), phi(pt) + d_phi, pt[2])


def shift_z(pt, d_z: float) -> np.ndarray:
    pt = _v(pt)
    return cyl_point(planar_radius(pt), phi(pt), pt[2] + d_z)


def planar_dir(pt) -> np.ndarray:
    """Normalised planar (XY) direction from the Z axis to the point."""
    pt = _v(pt)
    return safe_normalized(np.array([pt[0], pt[1], 0.0]))


# --- alignment -----------------------------------------------------------------
def flip_for_alignment(direction, target) -> np.ndarray:
    """``direction`` or its negation, whichever points more towards ``target``."""
    direction = _v(direction)
    target = _v(target)
    return direction if np.dot(target, direction) >= np.dot(target, -direction) else -direction


def is_aligned(direction, target) -> bool:
    """True if ``direction`` points more towards ``target`` than away."""
    direction = _v(direction)
    target = _v(target)
    return bool(np.dot(target, direction) >= np.dot(target, -direction))


# --- rotations -----------------------------------------------------------------
def rotate_around_axis(pt, angle: float, axis, origin=None) -> np.ndarray:
    """Rotate ``pt`` by ``angle`` (radians) about ``axis`` through ``origin``.

    Replicates ``Quaternion.CreateFromAxisAngle(axis, angle)`` followed by
    ``Vector3.Transform`` (so it matches C# even for a non-unit axis).
    """
    pt = _v(pt)
    axis = _v(axis)
    origin = np.zeros(3) if origin is None else _v(origin)
    rel = pt - origin

    half = 0.5 * angle
    s = math.sin(half)
    qx, qy, qz = axis * s
    qw = math.cos(half)

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    wx2, wy2, wz2 = qw * x2, qw * y2, qw * z2
    xx2, xy2, xz2 = qx * x2, qx * y2, qx * z2
    yy2, yz2, zz2 = qy * y2, qy * z2, qz * z2
    vx, vy, vz = rel
    rot = np.array([
        vx * (1 - yy2 - zz2) + vy * (xy2 - wz2) + vz * (xz2 + wy2),
        vx * (xy2 + wz2) + vy * (1 - xx2 - zz2) + vz * (yz2 - wx2),
        vx * (xz2 - wy2) + vy * (yz2 + wx2) + vz * (1 - xx2 - yy2),
    ])
    return rot + origin


def rotate_around_z(pt, d_phi: float, origin=None) -> np.ndarray:
    """Rotate a point about the absolute Z axis through ``origin``."""
    pt = _v(pt)
    origin = np.zeros(3) if origin is None else _v(origin)
    diff = pt - origin
    return origin + with_phi(diff, phi(diff) + d_phi)


def orthogonal_dir(direction) -> np.ndarray:
    """An arbitrary unit vector orthogonal to ``direction``.

    Matches ShapeKernel: cross with UnitX unless nearly parallel
    (``|dot| > 0.95``), else UnitY. The threshold uses ``direction`` as given
    (not pre-normalised), matching C#.
    """
    direction = _v(direction)
    non_parallel = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, non_parallel))) > 0.95:
        non_parallel = np.array([0.0, 1.0, 0.0])
    return safe_normalized(np.cross(direction, non_parallel))


def angle_between(a, b) -> float:
    """Minimum unsigned angle (radians) between two vectors."""
    a = safe_normalized(a)
    b = safe_normalized(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = math.acos(dot)
    if math.isnan(angle) and abs(dot) == 1.0:
        return math.pi
    return angle


def signed_angle_between(a, b, ref_normal) -> float:
    """Signed minimum angle (radians); sign follows the order via ``ref_normal``."""
    a = _v(a)
    b = _v(b)
    ref_normal = _v(ref_normal)
    err = 1e-20
    if (np.dot(a, a) < err or np.dot(b, b) < err or np.dot(ref_normal, ref_normal) < err):
        raise ValueError("vector with zero length")
    a = safe_normalized(a)
    b = safe_normalized(b)
    ref_normal = safe_normalized(ref_normal)
    normal = flip_for_alignment(np.cross(a, b), ref_normal)
    t = abs(angle_between(a, b))
    pos_dot = float(np.dot(a, rotate_around_axis(b, t, normal)))
    neg_dot = float(np.dot(a, rotate_around_axis(b, -t, normal)))
    return -t if neg_dot > pos_dot else t


# --- interpolation -------------------------------------------------------------
def lerp(p1, p2, ratio: float) -> np.ndarray:
    """Linear interpolation between two points."""
    p1 = _v(p1)
    p2 = _v(p2)
    return p1 + ratio * (p2 - p1)


def cylindrical_interpolation(p1, p2, ratio: float, origin=None) -> np.ndarray:
    """Interpolate between two points along a cylindrical arc about Z."""
    p1 = _v(p1)
    p2 = _v(p2)
    origin = np.zeros(3) if origin is None else _v(origin)
    origin = with_z(origin, 0.0)
    min_angle = angle_between(p1, p2)
    side1 = safe_normalized(p1 - origin)

    dist_pos = float(np.linalg.norm(p2 - rotate_around_z(p1, min_angle)))
    dist_neg = float(np.linalg.norm(p2 - rotate_around_z(p1, -min_angle)))
    sense = -1 if dist_neg < dist_pos else 1

    radius1 = planar_radius(p1 - origin)
    radius2 = planar_radius(p2 - origin)
    inter_radius = radius1 + ratio * (radius2 - radius1)
    inter_z = p1[2] + ratio * (p2[2] - p1[2])

    inter = rotate_around_z(inter_radius * side1, sense * ratio * min_angle)
    inter = with_z(inter, inter_z)
    return inter + origin


def spherical_interpolation(p1, p2, ratio: float, origin=None) -> np.ndarray:
    """Interpolate between two points along a spherical arc."""
    p1 = _v(p1)
    p2 = _v(p2)
    origin = np.zeros(3) if origin is None else _v(origin)
    min_angle = angle_between(p1, p2)
    side1 = safe_normalized(p1 - origin)
    side2 = safe_normalized(p2 - origin)
    normal = np.cross(side1, side2)

    dist_pos = float(np.linalg.norm(p2 - rotate_around_axis(p1, min_angle, normal)))
    dist_neg = float(np.linalg.norm(p2 - rotate_around_axis(p1, -min_angle, normal)))
    sense = -1 if dist_neg < dist_pos else 1

    radius1 = float(np.linalg.norm(p1 - origin))
    radius2 = float(np.linalg.norm(p2 - origin))
    inter_radius = radius1 + ratio * (radius2 - radius1)

    inter = rotate_around_axis(inter_radius * side1, sense * ratio * min_angle, normal)
    return inter + origin


# --- frame-relative helpers ----------------------------------------------------
def point_to_world(frame: LocalFrame, local_pt) -> np.ndarray:
    """Express a point given in a frame's local axes in world coordinates."""
    local_pt = _v(local_pt)
    return (frame.position
            + local_pt[0] * frame.local_x
            + local_pt[1] * frame.local_y
            + local_pt[2] * frame.local_z)


def point_to_local(frame: LocalFrame, world_pt) -> np.ndarray:
    """Express a world point in a frame's local coordinates."""
    rel = _v(world_pt) - frame.position
    return np.array([
        float(np.dot(rel, frame.local_x)),
        float(np.dot(rel, frame.local_y)),
        float(np.dot(rel, frame.local_z)),
    ])


def radius_to_axis(frame: LocalFrame, world_pt) -> float:
    """Planar radius of a point about a frame's local Z axis."""
    rel = _v(world_pt) - frame.position
    fx = float(np.dot(rel, frame.local_x))
    fy = float(np.dot(rel, frame.local_y))
    return float(math.hypot(fx, fy))


def phi_to_axis(frame: LocalFrame, world_pt) -> float:
    """Polar angle of a point about a frame's local Z axis (radians)."""
    rel = _v(world_pt) - frame.position
    fx = float(np.dot(rel, frame.local_x))
    fy = float(np.dot(rel, frame.local_y))
    return float(math.atan2(fy, fx))


def direction_to_axis(frame: LocalFrame, world_pt) -> np.ndarray:
    """Normalised in-plane direction from a frame's Z axis to a point."""
    rel = _v(world_pt) - frame.position
    fx = float(np.dot(rel, frame.local_x))
    fy = float(np.dot(rel, frame.local_y))
    return safe_normalized(fx * frame.local_x + fy * frame.local_y)
