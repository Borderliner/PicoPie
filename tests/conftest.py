import pytest

import picopie

VOXEL_SIZE_MM = 0.5


@pytest.fixture(scope="session", autouse=True)
def _session():
    picopie.init(voxel_size_mm=VOXEL_SIZE_MM)
    yield
    picopie.shutdown()
