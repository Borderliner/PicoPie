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
from .fields import ScalarField, VectorField
from .lattice import Lattice
from .library import (
    build_info,
    init,
    instance,
    is_initialized,
    name,
    session,
    shutdown,
    total_memory_bytes,
    version,
    voxel_size,
)
from .mesh import Mesh
from .metadata import Metadata, MetaType
from .polyline import PolyLine
from .types import BBox3
from .vdb import FieldType, VdbFile, load_vdb, save_vdb
from .viewer import Viewer, render_png, show
from .viz import colorize, mesh_preview, save_slice_png, save_slice_sheet
from .voxels import Voxels

__all__ = [
    "BBox3",
    "FieldType",
    "InvalidHandleError",
    "Lattice",
    "Mesh",
    "MetaType",
    "Metadata",
    "NotInitializedError",
    "PicoGKError",
    "PolyLine",
    "ScalarField",
    "VdbFile",
    "VectorField",
    "Viewer",
    "Voxels",
    "build_info",
    "colorize",
    "init",
    "instance",
    "is_initialized",
    "load_vdb",
    "mesh_preview",
    "name",
    "render_png",
    "save_slice_png",
    "save_slice_sheet",
    "save_vdb",
    "session",
    "show",
    "shutdown",
    "total_memory_bytes",
    "version",
    "voxel_size",
]

__version__ = "0.3.2"
