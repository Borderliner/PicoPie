"""The compiled active-voxel collector must match the pure-Python fallback
exactly (same coords, values, dtype, shape, and ordering) and handle the empty
field. We exercise both paths by toggling ``_fast.lib`` so the test is meaningful
whether or not the extension is compiled in this environment."""

import numpy as np
import pytest

import picopie._fast as _fast
from picopie import ScalarField, VectorField, Voxels


def _both_paths(call):
    """Return (fast_result, python_result) for a zero-arg callable, forcing each
    path by toggling the compiled lib. If the extension isn't built, the 'fast'
    result is just the fallback again (still a valid self-consistency check)."""
    saved = _fast.lib
    try:
        fast = call()          # compiled if available
        _fast.lib = None
        py = call()            # pure-Python fallback
    finally:
        _fast.lib = saved
    return fast, py


@pytest.fixture(scope="module")
def scalar_field():
    return ScalarField.from_voxels(Voxels.sphere(radius=12))


@pytest.fixture(scope="module")
def vector_field():
    return VectorField.from_voxels(Voxels.sphere(radius=12))


def test_scalar_active_values_matches_fallback(scalar_field):
    (c1, v1), (c2, v2) = _both_paths(scalar_field.active_values)
    assert c1.shape == c2.shape and c1.shape[1] == 3
    assert c1.dtype == np.float32 and v1.dtype == np.float32
    assert v1.shape == (c1.shape[0],)
    assert np.array_equal(c1, c2)      # identical coords, same order
    assert np.array_equal(v1, v2)      # identical values
    assert len(v1) > 1000              # the sphere's band is non-trivial


def test_vector_active_values_matches_fallback(vector_field):
    (c1, v1), (c2, v2) = _both_paths(vector_field.active_values)
    assert c1.shape == c2.shape and v1.shape == v2.shape
    assert c1.shape[1] == 3 and v1.shape[1] == 3
    assert c1.dtype == np.float32 and v1.dtype == np.float32
    assert np.array_equal(c1, c2)
    assert np.array_equal(v1, v2)


def test_scalar_active_values_values_correct():
    sf = ScalarField()
    sf.set((1.0, 0.0, 0.0), 9.0)
    (c1, v1), (c2, v2) = _both_paths(sf.active_values)
    assert c1.shape == (1, 3) and v1.shape == (1,)
    assert v1[0] == pytest.approx(9.0)
    assert np.array_equal(c1, c2) and np.array_equal(v1, v2)


def test_vector_active_values_values_correct():
    vf = VectorField()
    vf.set((0, 0, 0), (1, 2, 3))
    (c1, v1), (c2, v2) = _both_paths(vf.active_values)
    assert c1.shape == (1, 3) and v1.shape == (1, 3)
    assert np.allclose(v1[0], [1, 2, 3])
    assert np.array_equal(c1, c2) and np.array_equal(v1, v2)


def test_empty_field_active_values():
    (sc, sv), _ = _both_paths(ScalarField().active_values)
    assert sc.shape == (0, 3) and sv.shape == (0,)
    assert sc.dtype == np.float32 and sv.dtype == np.float32
    (vc, vv), _ = _both_paths(VectorField().active_values)
    assert vc.shape == (0, 3) and vv.shape == (0, 3)


@pytest.mark.skipif(not _fast.available(), reason="compiled collector not built")
def test_compiled_collector_is_reentrant_after_call(scalar_field):
    # two sequential compiled calls must not corrupt each other via the globals
    c1, v1 = scalar_field.active_values()
    c2, v2 = scalar_field.active_values()
    assert np.array_equal(c1, c2) and np.array_equal(v1, v2)
