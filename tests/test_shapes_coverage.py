"""Coverage for shape-layer branches the final gap audit flagged as untested:
the post-construction step-bump fix, spherical frame alignment, the cylindrical
control-spline tangential step, closed control surfaces, lattice-manifold
double-sided extension, the down-facing painter split, and mesh STL export.
All CI-offline.
"""

import numpy as np

import picopie
from picopie.shapes import (
    Box,
    Cone,
    ControlPointSurface,
    Cylinder,
    CylindricalControlSpline,
    Frames,
    LatticeManifold,
    LocalFrame,
    Pipe,
    io_utils,
    painter,
)
from picopie.shapes.colors import ColorScale3D, rainbow_spectrum


# --- the setter step-bump fix (0.4.2) ----------------------------------------
def test_box_setter_re_bumps_step_counts():
    b = Box(LocalFrame(), 20, 10, 10)                 # constant -> coarse
    assert b.width_steps == 5 and b.length_steps == 5
    b.width = lambda lr: 10 - 2 * np.cos(6 * lr)      # modulate AFTER construction
    assert b.width_steps == 500 and b.length_steps == 500
    assert Box(LocalFrame(), 20, 10, 10, length_steps=12).length_steps == 12  # override wins


def test_cylinder_and_pipe_setters_re_bump():
    c = Cylinder(LocalFrame(), 20, 8)
    assert c.length_steps == 5
    c.radius = lambda phi, lr: 8 - np.cos(8 * lr)
    assert c.length_steps == 500
    assert Cone(LocalFrame(), 20, 8, 2).length_steps == 500   # linear radius -> bumped

    p = Pipe(LocalFrame(), 20, 6, 10)
    assert p.length_steps == 5
    p.outer_radius = lambda phi, lr: 10 - np.cos(8 * lr)
    assert p.length_steps == 500


# --- previously-untested shape branches --------------------------------------
def test_frames_aligned_spherical():
    pts = [[0, 0, 0], [0, 10, 0], [5, 10, 10], [10, 10, 20]]
    fr = Frames.aligned(pts, "spherical")
    lz = fr.local_z_at(0.5)
    assert np.isfinite(lz).all() and abs(np.linalg.norm(lz) - 1.0) < 1e-5


def test_cylindrical_control_spline_tangential_step():
    s = CylindricalControlSpline((10.0, 0.0, 0.0))
    s.add_relative_step("tangential", 5.0).add_relative_step("z", 4.0)
    pts = s.points(50)
    assert pts.shape[1] == 3 and len(pts) > 0 and np.isfinite(pts).all()


def test_control_point_surface_closed_u():
    ring = [[np.cos(t), np.sin(t), 0.0] for t in np.linspace(0, 2 * np.pi, 5)]
    grid = np.array([np.add(ring, [0, 0, z]) for z in range(3)]) * 10.0   # (3, 5, 3)
    surf = ControlPointSurface(grid, closed_u=True)
    p = surf.point_at(0.5, 0.5)
    assert p.shape == (3,) and np.isfinite(p).all()


def test_lattice_manifold_extend_both_sides():
    lm = LatticeManifold(LocalFrame((0, 0, 0), local_z=(0, 1, 0)), 50, 5, 45,
                         extend_both_sides=True)
    assert lm.to_voxels().calculate_properties()[0] > 0


def test_painter_only_down_facing():
    mesh = picopie.Voxels.sphere(radius=20).to_mesh()
    scale = ColorScale3D(rainbow_spectrum(), 0.0, 90.0)
    scene = painter.split_by_overhang_angle(mesh, scale, n_classes=20, only_down_facing=True)
    assert isinstance(scene, list) and len(scene) > 0
    for _obj, rgb in scene:
        assert len(rgb) == 3


def test_export_mesh_to_stl(tmp_path):
    out = tmp_path / "m.stl"
    io_utils.export_mesh_to_stl(picopie.Voxels.sphere(radius=5).to_mesh(), str(out))
    assert out.exists() and out.stat().st_size > 0
