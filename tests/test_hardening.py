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
