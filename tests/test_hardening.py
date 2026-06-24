"""Phase 5.5 hardening: native calls that used to abort the process must now
raise catchable Python exceptions instead."""

import pytest

import picogk
from picogk import VdbFile, Voxels


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
