"""Voxel-size sweep tripwire.

The 0.3.1 ``IntersectImplicit`` bug and the 0.3.2 ``ProjectZSlice`` bug were the
same class -- a *millimetre* value used as an integer *voxel count* -- and both
only fired below ~0.33 / ~0.167 mm. The whole suite runs at a single voxel size
(``tests/conftest.py`` -> 0.5 mm), so neither was caught for two releases.

These tests exercise the resolution-sensitive native ops across fine->coarse
voxel sizes and assert the results stay valid (non-empty) and -- where the result
should be voxel-size-independent -- consistent with the 0.5 mm baseline. This is
the deterministic tripwire; ``scripts/fuzz_abort.py`` adds the same voxel-size
dimension to the (randomized, abort-hunting) campaign.

Each test re-inits the global library per size and restores the session's voxel
size at the end (init with a different size raises while a session is live).
"""

import pytest

import picogk
from picogk import Voxels

SIZES = [0.1, 0.2, 0.5, 1.0]   # fine..coarse; 0.05 is left to fuzz_abort.py (slow)


@pytest.fixture
def restore_session():
    prev = picogk.voxel_size()
    picogk.shutdown()
    yield
    picogk.shutdown()
    picogk.init(voxel_size_mm=prev)


def _vol(v: Voxels) -> float:
    return v.calculate_properties()[0]


def _at(vs: float, build):
    """Init at ``vs``, run ``build`` (which must return a *scalar* -- the grid is
    invalidated on shutdown), then shut down."""
    picogk.init(voxel_size_mm=vs)
    try:
        return build()
    finally:
        picogk.shutdown()


def test_intersect_implicit_across_sizes(restore_session):
    # 0.3.1 regression: aborted below ~0.33 mm (narrow band truncated to 0).
    def build():
        v = Voxels.sphere(radius=6.0)
        v.intersect_implicit_(lambda x, y, z: x)        # keep the x<=0 hemisphere
        return _vol(v)
    for vs in SIZES:
        assert _at(vs, build) > 0, f"intersect_implicit_ empty at {vs} mm"


def test_project_z_slice_watertight_across_sizes(restore_session):
    # 0.3.2 regression: the end-cap seal loop iterated (int)(0.5 + background_mm)
    # slices, so below ~0.167 mm the bottom cap was never sealed and the solid
    # came out non-watertight -> re-voxelising yielded a near-zero volume. The
    # sphere sits above the slice end so the projection sweeps into empty space
    # (where the cap matters). The result must be ~voxel-size-independent.
    def build():
        v = Voxels.sphere(center=(0, 0, 10), radius=4.0)
        v.project_z_slice_(8.0, -8.0)
        return _vol(v)
    vols = {vs: _at(vs, build) for vs in SIZES}
    base = vols[0.5]
    for vs, vol in vols.items():
        assert vol == pytest.approx(base, rel=0.25), \
            f"project_z_slice_ at {vs} mm = {vol:.1f} mm^3 vs 0.5 mm baseline {base:.1f}"


def test_core_ops_nonempty_across_sizes(restore_session):
    # The offset family + booleans + mesh round-trip + implicit render must all
    # stay non-empty at every resolution (sub-voxel features aside).
    def battery():
        return {
            "offset": _vol(Voxels.sphere(radius=5.0).offset_(1.0)),
            "double_offset": _vol(Voxels.sphere(radius=5.0).double_offset_(2.0, -2.0)),
            "triple_offset": _vol(Voxels.sphere(radius=5.0).triple_offset_(1.0)),
            "bool_intersect": _vol(Voxels.sphere(radius=5.0)
                                   & Voxels.sphere(center=(4, 0, 0), radius=5.0)),
            "mesh_roundtrip": _vol(Voxels.from_mesh(Voxels.sphere(radius=5.0).to_mesh())),
            "mesh_shell": _vol(Voxels.mesh_shell(Voxels.sphere(radius=5.0).to_mesh(), 1.5)),
            "render_implicit": _vol(Voxels().render_implicit_(
                lambda x, y, z: (x * x + y * y + z * z) ** 0.5 - 5.0,
                ((-7, -7, -7), (7, 7, 7)))),
        }
    for vs in SIZES:
        res = _at(vs, battery)
        for name, vol in res.items():
            assert vol > 0, f"{name} empty at {vs} mm"
