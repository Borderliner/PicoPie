"""Direct coverage for thin public wrappers that had no dedicated test
(Tier-2 of the final gap audit). All CI-offline."""

import importlib

import numpy as np
import pytest

import picogk
from picogk import Lattice, Metadata, ScalarField, VdbFile, VectorField, Voxels
from picogk._errors import PicoGKError


def test_voxels_voxel_size_mm_matches_session():
    assert Voxels.sphere(radius=5).voxel_size_mm() == pytest.approx(picogk.voxel_size())


def test_voxels_bounding_box_is_nondegenerate_grid_extent():
    bb = Voxels.sphere(radius=10).bounding_box()
    assert np.all(np.asarray(bb.size) > 0)               # looser than the surface bbox
    assert np.allclose(bb.center, [0, 0, 0], atol=1.0)


def test_lattice_is_valid():
    lat = Lattice()
    lat.add_sphere((0, 0, 0), 3.0)
    assert lat.is_valid() is True


def test_vdbfile_is_valid():
    f = VdbFile()
    f.add_voxels("v", Voxels.sphere(radius=4))
    assert f.is_valid() is True


def test_field_memory_bytes_positive():
    assert ScalarField.from_voxels(Voxels.sphere(radius=5)).memory_bytes() > 0
    assert VectorField.from_voxels(Voxels.sphere(radius=5)).memory_bytes() > 0


def test_metadata_to_dict():
    md = Metadata.from_voxels(Voxels.sphere(radius=4))
    md["material"] = "Ti"
    md.set_float("k", 1.5)
    d = md.to_dict()
    assert d["material"] == "Ti"
    assert d["k"] == pytest.approx(1.5)


def test_vdbfile_save_failure_raises():
    f = VdbFile()
    f.add_voxels("v", Voxels.sphere(radius=4))
    with pytest.raises(PicoGKError):
        f.save("/no_such_dir_picopie_xyz/cannot/write.vdb")


def test_public_names_all_import():
    # a broken re-export in picogk.__init__ would slip through everything else
    mod = importlib.import_module("picogk")
    for name in mod.__all__:
        assert hasattr(mod, name), f"picogk.{name} is exported but missing"
