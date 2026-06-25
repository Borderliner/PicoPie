"""Phase 11b: property fuzzing of the abort surface.

The property under test, across the whole API: **for any input, an operation
either succeeds or raises a catchable ``Exception`` -- it never aborts the
process.** With the Phase-11a never-abort runtime guard in place, a C++/OpenVDB
error surfaces as ``PicoGKError`` (an ``Exception``); if any input slipped past
the guard it would ``SIGABRT`` and crash this whole pytest process -- a loud,
unmissable failure. So each test feeds adversarial inputs (NaN/inf/huge/negative,
out-of-range indices, degenerate geometry) and just asserts the call is
*containable*.

We deliberately target the ABORT surface, not resource limits: parameters that
*size a voxel grid* (radius, centre, bbox, offsets, mesh coords) are kept to a
bounded range (NaN/degenerate still included) so a huge-but-finite value can't
turn this into an OOM/hang test. Non-sizing inputs (SDF return values, query
points, field values/positions) use the full adversarial set.

Marked ``fuzz`` -> excluded from the per-wheel CI test-command (slow); run locally
(`pytest -m fuzz`) and, for deep campaigns, via scripts/fuzz_abort.py.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from picogk import Mesh, Metadata, ScalarField, VdbFile, Voxels

pytestmark = pytest.mark.fuzz

# non-sizing adversarial scalar: NaN, +-inf, huge, tiny, negative, zero
weird = st.floats(allow_nan=True, allow_infinity=True, width=32)
wpt = st.tuples(weird, weird, weird)
# grid-sizing: bounded magnitude, but still NaN / negative / zero / degenerate.
# (Hypothesis forbids allow_nan with min/max, so NaN is mixed in via one_of.)
small = st.one_of(st.sampled_from([float("nan"), 0.0]),
                  st.floats(min_value=-12, max_value=12, allow_infinity=False, width=32))
spt = st.tuples(small, small, small)
# render bbox kept tiny: the SDF callback runs once per voxel (pure Python)
bx = st.one_of(st.sampled_from([float("nan")]),
               st.floats(min_value=-8, max_value=8, allow_infinity=False, width=32))
bpt = st.tuples(bx, bx, bx)
# query points: bounded + finite. NaN/inf coords make the lookups/ray-marcher
# *hang* (unbounded native loops) rather than abort -- a resource concern, out of
# scope for the never-abort property -- so they're excluded here.
qf = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False, width=32)
qpt = st.tuples(qf, qf, qf)

FUZZ = settings(max_examples=120, deadline=None)
FUZZ_RENDER = settings(max_examples=40, deadline=None)


def _containable(fn) -> None:
    """Call fn; any Exception is acceptable. Survival == the property holds."""
    with contextlib.suppress(Exception):
        fn()


# --- constructors / unary ops ------------------------------------------------
@FUZZ
@given(r=small, c=spt)
def test_fuzz_sphere(r, c):
    _containable(lambda: Voxels.sphere(center=c, radius=r))


@FUZZ
@given(a=spt, b=spt, r=small, r2=small)
def test_fuzz_capsule(a, b, r, r2):
    _containable(lambda: Voxels.capsule(a, b, r, r2))


@FUZZ_RENDER  # fewer examples: each op grows a fresh grid; chaining would balloon it
@given(d=small, d2=small)
def test_fuzz_offsets(d, d2):
    _containable(lambda: Voxels.sphere(radius=6.0).offset_(d))
    _containable(lambda: Voxels.sphere(radius=6.0).double_offset_(d, d2))
    _containable(lambda: Voxels.sphere(radius=6.0).shell_(d))


# --- booleans (two fuzzed operands) ------------------------------------------
@FUZZ
@given(r1=small, r2=small,
       op=st.sampled_from(["bool_add_", "bool_subtract_", "bool_intersect_"]))
def test_fuzz_booleans(r1, r2, op):
    def run():
        a = Voxels.sphere(radius=r1)
        b = Voxels.sphere(center=(3, 0, 0), radius=r2)
        getattr(a, op)(b)
    _containable(run)


# --- implicit modeling (SDF return values + bbox both fuzzed) ----------------
@FUZZ_RENDER
@given(ret=weird, lo=bpt, hi=bpt)
def test_fuzz_render_implicit(ret, lo, hi):
    v = Voxels()
    _containable(lambda: v.render_implicit_(lambda x, y, z: ret, (lo, hi)))


@FUZZ
@given(ret=weird)
def test_fuzz_intersect_implicit(ret):
    v = Voxels.sphere(radius=6.0)
    _containable(lambda: v.intersect_implicit_(lambda x, y, z: ret))


# --- meshes: arbitrary vertex/index arrays -----------------------------------
@FUZZ
@given(
    verts=st.lists(spt, max_size=12),
    tris=st.lists(st.tuples(st.integers(-5, 20), st.integers(-5, 20), st.integers(-5, 20)),
                  max_size=12),
)
def test_fuzz_mesh_from_arrays(verts, tris):
    def build():
        m = Mesh.from_arrays(np.array(verts or [(0, 0, 0)], dtype=np.float32),
                             np.array(tris or [(0, 0, 0)], dtype=np.int32))
        Voxels.from_mesh(m)        # voxelize whatever we built
    _containable(build)


# --- fields: bulk + single, fuzzed positions/values --------------------------
@FUZZ
@given(pos=st.lists(wpt, max_size=64), val=weird)
def test_fuzz_scalar_field(pos, val):
    f = ScalarField.from_voxels(Voxels.sphere(radius=5.0))
    arr = np.array(pos or [(0, 0, 0)], dtype=np.float32)
    vals = np.full(len(arr), val, dtype=np.float32)
    _containable(lambda: f.set_many(arr, vals))
    _containable(lambda: f.get_many(arr))


# --- point queries -----------------------------------------------------------
@FUZZ
@given(p=qpt, d=qpt)
def test_fuzz_point_queries(p, d):
    v = Voxels.sphere(radius=5.0)
    _containable(lambda: v.is_inside(p))
    _containable(lambda: v.closest_point(p))
    _containable(lambda: v.surface_normal(p))
    # ray_cast from a bounded origin along a fixed valid direction (a degenerate
    # NaN/zero direction marches unboundedly -- a hang, not an abort).
    _containable(lambda: v.ray_cast(p, (1.0, 0.0, 0.0)))


# --- slices, by arbitrary index ----------------------------------------------
@FUZZ
@given(i=st.integers(-1000, 1000))
def test_fuzz_slices(i):
    v = Voxels.sphere(radius=5.0)
    _containable(lambda: v.slice_z(i))
    _containable(lambda: v.slice_x(i))


# --- metadata: arbitrary names/values ----------------------------------------
@FUZZ
@given(name=st.text(max_size=40), s=st.text(max_size=40), f=weird, vec=wpt)
def test_fuzz_metadata(name, s, f, vec):
    md = Metadata.from_voxels(Voxels.sphere(radius=4.0))
    _containable(lambda: md.set_string(name, s))
    _containable(lambda: md.set_float(name, f))
    _containable(lambda: md.set_vector(name, vec))
    _containable(lambda: md.get(name))


# --- vdb field access, arbitrary index ---------------------------------------
@FUZZ
@given(i=st.integers(-100, 100))
def test_fuzz_vdb_get(i):
    f = VdbFile()
    f.add_voxels("v", Voxels.sphere(radius=4.0))
    _containable(lambda: f.get(i))
    _containable(lambda: f.field_type(i))
