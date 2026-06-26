"""Parametric shapes — a Pythonic port of LEAP 71's `ShapeKernel
<https://github.com/leap71/LEAP71_ShapeKernel>`_, built on the PicoPie core.

ShapeKernel is a higher-level, parametric geometry layer (boxes, spheres,
pipes, lenses, lattices, ...) defined by local frames and *modulations* and
rasterised into voxels. This package re-implements it in pure Python on top of
``Voxels.from_mesh`` / ``Mesh.from_arrays`` — no native code beyond the core.

The API is Pythonic (snake_case; ``to_voxels()`` / ``to_mesh()`` instead of
``voxConstruct()`` / ``mshConstruct()``); see the tutorials for the full
C#-equivalence table.

Example::

    import picogk
    from picogk.shapes import Sphere, LocalFrame

    picogk.init(0.5)
    part = Sphere(LocalFrame(position=(0, 0, 0)), radius=10).to_voxels()
"""

from ._base import BaseShape
from .frames import LocalFrame
from .modulations import LineModulation, SurfaceModulation
from .sphere import Sphere

__all__ = [
    "BaseShape",
    "LineModulation",
    "LocalFrame",
    "Sphere",
    "SurfaceModulation",
]
