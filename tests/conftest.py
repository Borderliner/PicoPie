import pytest

import picogk

VOXEL_SIZE_MM = 0.5


@pytest.fixture(scope="session", autouse=True)
def _session():
    picogk.init(voxel_size_mm=VOXEL_SIZE_MM)
    yield
    picogk.shutdown()
