"""Non-finite input validation (the B1 fix from the geometry review).

Shape dimensions / modulations flow through ``Mesh.from_arrays``, and lattice
radii through ``Lattice.add_*``; both now reject NaN/inf before any native call
(consistently across the fast and fallback mesh paths), so degenerate inputs
raise a clean ``ValueError`` instead of producing silent-garbage voxels.
"""

import numpy as np
import pytest

import picogk._fast as _fast
from picogk import Lattice, Mesh
from picogk.shapes import (
    Box,
    Cone,
    Cylinder,
    Frames,
    LatticePipe,
    Lens,
    LocalFrame,
    Pipe,
    PipeSegment,
    Revolve,
    Ring,
    Sphere,
)

# these tests feed NaN/inf on purpose; numpy's "invalid value" notices are expected
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize("make", [
    lambda: Sphere(radius=NAN),
    lambda: Sphere(radius=INF),
    lambda: Sphere(radius=lambda phi, theta: NAN),          # modulation output
    lambda: Box(length=10, width=INF, depth=10),
    lambda: Cylinder(length=10, radius=NAN),
    lambda: Cone(LocalFrame(), 10, 5, NAN),
    lambda: Ring(ring_radius=NAN, radius=5),
    lambda: Ring(ring_radius=30, radius=INF),
    lambda: Lens(height=NAN, inner_radius=0, outer_radius=5),
    lambda: Pipe(length=10, inner_radius=NAN, outer_radius=5),
    lambda: PipeSegment(length=10, inner_radius=2, outer_radius=5, start=NAN, end=1),
    lambda: Revolve(LocalFrame(), Frames.extrude(10, LocalFrame()), 0, NAN),
    lambda: LatticePipe(length=10, radius=NAN),
])
def test_shape_with_nonfinite_dimension_raises(make):
    with pytest.raises(ValueError):
        make().to_voxels()


def test_mesh_from_arrays_rejects_nonfinite_both_paths():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [NAN, 0, 0]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)
    with pytest.raises(ValueError):                          # fast path
        Mesh.from_arrays(verts, tris)
    saved = _fast.lib
    _fast.lib = None                                         # force fallback path
    try:
        with pytest.raises(ValueError):
            Mesh.from_arrays(verts, tris)
    finally:
        _fast.lib = saved


def test_mesh_from_arrays_accepts_finite():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=int)
    assert Mesh.from_arrays(verts, tris).triangle_count() == 1


def test_lattice_rejects_nonfinite_radius():
    with pytest.raises(ValueError):
        Lattice().add_sphere((0, 0, 0), NAN)
    with pytest.raises(ValueError):
        Lattice().add_beam((0, 0, 0), (0, 0, 1), NAN, 1.0)
    with pytest.raises(ValueError):
        Lattice().add_beam((0, 0, 0), (0, 0, 1), 1.0, INF)
