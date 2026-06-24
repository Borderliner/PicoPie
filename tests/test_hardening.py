"""Phase 5.5 hardening: native calls that used to abort the process must now
raise catchable Python exceptions instead."""

import ctypes as C

import pytest

import picogk
from picogk import ScalarField, VdbFile, Voxels, library
from picogk._errors import PicoGKError
from picogk._native import ctypes_types


# --- H2: out-of-range VDB field index must raise, not abort -------------------
@pytest.mark.parametrize("method", [
    "field_name", "field_type", "get_voxels", "get_scalar_field", "get_vector_field",
])
def test_vdbfile_bad_index_raises(method):
    f = VdbFile()
    try:
        with pytest.raises(IndexError):
            getattr(f, method)(99)
        with pytest.raises(IndexError):
            getattr(f, method)(-1)
    finally:
        f.close()


def test_vdbfile_get_bad_index_raises():
    f = VdbFile()
    try:
        with pytest.raises(IndexError):
            f.get(99)
    finally:
        f.close()


def test_vdbfile_valid_index_still_works(tmp_path):
    v = Voxels.sphere(radius=5)
    p = tmp_path / "x.vdb"
    picogk.save_vdb(str(p), body=v)
    f = VdbFile.load(str(p))
    try:
        assert f.field_name(0) == "body"          # in range: fine
    finally:
        f.close()


# --- H1: using an object after shutdown() must raise, not abort ---------------
# Runs in a subprocess: shutdown() is global and would disturb the shared
# session fixture, and we want to assert the process does NOT abort (exit 0).
def test_operation_after_shutdown_raises(tmp_path):
    import subprocess
    import sys
    code = (
        "import picogk\n"
        "from picogk import Voxels\n"
        "picogk.init(0.5)\n"
        "v = Voxels.sphere(radius=5)\n"
        "picogk.shutdown()\n"
        "try:\n"
        "    v.volume_mm3()\n"
        "    print('NO_RAISE')\n"
        "except picogk.InvalidHandleError:\n"
        "    print('RAISED')\n"
        "v.close()  # must be a safe no-op, not an abort\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"process aborted: rc={r.returncode}\n{r.stderr}"
    assert "RAISED" in r.stdout, r.stdout


# --- H3: an SDF callback that raises must propagate, not silently corrupt -----
def test_render_implicit_propagates_callback_error():
    def bad_sdf(x, y, z):
        raise ValueError("boom in user sdf")

    v = Voxels()
    with pytest.raises(ValueError, match="boom in user sdf"):
        v.render_implicit_(bad_sdf, ((-2, -2, -2), (2, 2, 2)))


def test_render_implicit_nonfinite_is_safe():
    # returning NaN/inf must not abort or inject a 0.0 surface distance
    import math as _m
    v = Voxels()
    v.render_implicit_(lambda x, y, z: _m.nan, ((-2, -2, -2), (2, 2, 2)))
    assert v.is_empty()      # all voxels treated as "outside"


# --- Phase 6: runtime version gate -------------------------------------------
def test_runtime_version_matches_expected():
    # the pinned/bundled runtime must report the major.minor this binding targets
    assert library.version().startswith(library._EXPECTED_RUNTIME_VERSION)


@pytest.mark.parametrize("bad", ["27.0.0", "26.3.0", "26.20.0", "25.2.0", "", "garbage"])
def test_version_gate_rejects_mismatch(monkeypatch, bad):
    monkeypatch.setattr(library, "_info_string", lambda fn: bad)
    with pytest.raises(PicoGKError, match="version mismatch"):
        library._check_runtime_version()


@pytest.mark.parametrize("ok", ["26.2", "26.2.0", "26.2.5", "26.2.99"])
def test_version_gate_accepts_patch_levels(monkeypatch, ok):
    monkeypatch.setattr(library, "_info_string", lambda fn: ok)
    library._check_runtime_version()             # must not raise


# --- Phase 6: ABI struct-layout self-check -----------------------------------
def test_struct_layout_matches_packed_abi():
    ctypes_types._assert_struct_layout()         # must not raise
    assert C.sizeof(ctypes_types.PKVector3) == 12
    assert C.sizeof(ctypes_types.PKBBox3) == 24
    assert C.sizeof(ctypes_types.PKMatrix4x4) == 64


# --- Phase 7: the one confirmed hard-abort + typed-getter guards --------------
def _solid(r=4.0):
    v = Voxels()
    v.render_implicit_(lambda x, y, z: (x * x + y * y + z * z) ** 0.5 - r,
                       ((-r - 1, -r - 1, -r - 1), (r + 1, r + 1, r + 1)))
    return v


def test_double_intersect_implicit_raises_not_aborts():
    # A second implicit intersect aborts the process in OpenVDB; we must raise.
    v = _solid()
    v.intersect_implicit_(lambda x, y, z: x)         # first is fine
    with pytest.raises(PicoGKError, match="twice"):
        v.intersect_implicit_(lambda x, y, z: y)


def test_intersect_implicit_bad_state_follows_copy():
    v = _solid()
    v.intersect_implicit_(lambda x, y, z: x)
    with pytest.raises(PicoGKError):
        v.copy().intersect_implicit_(lambda x, y, z: y)


def test_vdb_typed_getter_rejects_wrong_field_type():
    v = _solid()
    sf = ScalarField.from_voxels(v)
    f = VdbFile()
    f.add_voxels("part", v)                           # field 0 -> VOXELS
    f.add_scalar_field("field", sf)                   # field 1 -> SCALAR
    assert f.field_count() == 2
    try:
        # wrong-type accessors on the voxels field must raise (silent-data footgun)
        with pytest.raises(TypeError, match="VOXELS"):
            f.get_scalar_field(0)
        with pytest.raises(TypeError, match="VOXELS"):
            f.get_vector_field(0)
        # wrong-type accessor on the scalar field
        with pytest.raises(TypeError, match="SCALAR"):
            f.get_voxels(1)
        # correct-type access and the auto-dispatching get() still work
        assert isinstance(f.get_voxels(0), Voxels)
        assert isinstance(f.get_scalar_field(1), ScalarField)
        assert isinstance(f.get(0), Voxels)
    finally:
        f.close()
