"""Assert the input-validation guards actually RAISE.

The fuzz campaign *feeds* NaN/inf but wraps each call in ``suppress(Exception)``,
so it only proves "no process abort" — it never asserts the guard layer rejects
bad input. These do: ``require_finite`` / ``to_vec3`` / bounds checks must raise a
clear Python exception. All CI-offline (the conftest session is at 0.5 mm); the
lifecycle cases that need a *fresh* (uninitialized) process use a subprocess.
"""

import subprocess
import sys

import numpy as np
import pytest

import picopie
from picopie import Lattice, Mesh, ScalarField, VectorField, Voxels
from picopie import library as _lib
from picopie.types import require_finite, to_vec3

NONFINITE = [float("nan"), float("inf"), -float("inf")]


@pytest.mark.parametrize("bad", NONFINITE)
def test_require_finite_rejects(bad):
    with pytest.raises(ValueError, match="finite"):
        require_finite("x", bad)


@pytest.mark.parametrize("bad", NONFINITE)
def test_to_vec3_rejects_nonfinite(bad):
    with pytest.raises(ValueError):
        to_vec3((bad, 0.0, 0.0))


def test_voxels_size_args_reject_nonfinite():
    with pytest.raises(ValueError):
        Voxels.sphere(radius=float("nan"))
    with pytest.raises(ValueError):
        Voxels.capsule((0, 0, 0), (1, 0, 0), float("inf"), 1.0)
    with pytest.raises(ValueError):
        Voxels.sphere(radius=5).offset_(float("nan"))
    with pytest.raises(ValueError):
        Voxels.sphere(radius=5).double_offset_(1.0, float("inf"))
    with pytest.raises(ValueError):
        Voxels.sphere(radius=5).triple_offset_(float("nan"))
    with pytest.raises(ValueError):
        Voxels.sphere(radius=5).shell_(float("nan"))
    with pytest.raises(ValueError):
        Voxels.sphere(radius=5).project_z_slice_(float("nan"), 1.0)


def test_mesh_shell_rejects_nonfinite_radius():
    m = Voxels.sphere(radius=5).to_mesh()
    with pytest.raises(ValueError, match="finite"):
        Voxels.mesh_shell(m, float("nan"))


def test_mesh_from_arrays_rejects_nonfinite_vertices():
    verts = np.array([[0, 0, 0], [1, 0, 0], [float("nan"), 1, 0]], np.float32)
    tris = np.array([[0, 1, 2]], np.int32)
    with pytest.raises(ValueError):
        Mesh.from_arrays(verts, tris)


def test_mesh_add_triangle_bounds():
    m = Mesh()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        m.add_vertex(p)
    with pytest.raises(IndexError):
        m.add_triangle(0, 1, 7)          # out of range high
    with pytest.raises(IndexError):
        m.add_triangle(0, -1, 2)         # negative
    assert m.add_triangle(0, 1, 2) >= 0  # valid indices pass


def test_field_bulk_reject_nonfinite_positions():
    bad = np.array([[float("nan"), 0, 0]], np.float32)
    sf = ScalarField.from_voxels(Voxels.sphere(radius=5))
    with pytest.raises(ValueError, match="finite"):
        sf.set_many(bad, np.array([1.0], np.float32))
    with pytest.raises(ValueError, match="finite"):
        sf.get_many(bad)
    vf = VectorField.from_voxels(Voxels.sphere(radius=5))
    with pytest.raises(ValueError, match="finite"):
        vf.set_many(bad, np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError, match="finite"):
        vf.get_many(bad)


def test_scalarfield_slice_bounds():
    f = ScalarField.from_voxels(Voxels.sphere(radius=5))
    with pytest.raises(IndexError):
        f.slice(-1)
    with pytest.raises(IndexError):
        f.slice(10**6)


def test_lattice_add_reject_nonfinite():
    lat = Lattice()
    with pytest.raises(ValueError):
        lat.add_sphere((0, 0, 0), float("nan"))
    with pytest.raises(ValueError):
        lat.add_beam((0, 0, 0), (1, 0, 0), float("inf"), 1.0)


def test_init_rejects_nonpositive_voxel_size():
    # the <= 0 check fires before the session guard, so this is safe with a live
    # session and doesn't mutate it.
    with pytest.raises(ValueError):
        picopie.init(voxel_size_mm=0.0)
    with pytest.raises(ValueError):
        picopie.init(voxel_size_mm=-1.0)


# --- lifecycle that needs a *fresh* (uninitialized) process -> subprocess -------
def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_not_initialized_error_in_fresh_process():
    r = _run(
        "import picopie\n"
        "try:\n"
        "    picopie.Voxels.sphere(radius=5)\n"
        "    print('NORAISE')\n"
        "except picopie.NotInitializedError:\n"
        "    print('RAISED')\n")
    assert "RAISED" in r.stdout, r.stdout + r.stderr


def test_session_context_manager_in_fresh_process():
    r = _run(
        "import picopie\n"
        "from picopie import library\n"
        "with picopie.session(0.5):\n"
        "    v = picopie.Voxels.sphere(radius=5)\n"
        "    print('VOL', v.volume_mm3() > 0)\n"
        "print('AFTER', library.is_initialized())\n")
    assert "VOL True" in r.stdout and "AFTER False" in r.stdout, r.stdout + r.stderr


def test_uninitialized_accessor_name_is_exported():
    assert _lib.NotInitializedError is picopie.NotInitializedError
