"""PicoPie -- a Pythonic binding for the PicoGK computational-geometry kernel.

Quick start::

    import picogk
    from picogk import Voxels

    picogk.init(voxel_size_mm=0.2)

    body = Voxels.sphere(radius=10)
    hole = Voxels.sphere(center=(5, 0, 0), radius=6)
    part = body - hole                 # boolean subtract -> new Voxels
    part.shell_(1.0)                   # 1 mm wall, in place

    print("volume:", part.volume_mm3(), "mm^3")
    part.to_mesh().save_stl("part.stl")
"""

from __future__ import annotations

from ._errors import InvalidHandleError, NotInitializedError, PicoGKError
from .lattice import Lattice
from .library import (
    build_info, init, instance, is_initialized, name, session, shutdown,
    total_memory_bytes, version, voxel_size,
)
from .mesh import Mesh
from .types import BBox3
from .voxels import Voxels

__all__ = [
    "init", "shutdown", "session", "is_initialized", "voxel_size",
    "name", "version", "build_info", "instance", "total_memory_bytes",
    "Voxels", "Mesh", "Lattice", "BBox3",
    "PicoGKError", "NotInitializedError", "InvalidHandleError",
]

__version__ = "0.1.0.dev0"
