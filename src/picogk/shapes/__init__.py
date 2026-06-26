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

The math/support layer lives in submodules: :mod:`~picogk.shapes.vectors`,
:mod:`~picogk.shapes.formulas`, :mod:`~picogk.shapes.lists`,
:mod:`~picogk.shapes.grids`, :mod:`~picogk.shapes.spline_ops`,
:mod:`~picogk.shapes.decimation`.
"""

from . import (
    decimation,
    formulas,
    functions,
    grids,
    io_utils,
    lists,
    measure,
    mesh_utils,
    spline_ops,
    vectors,
)
from ._base import BaseShape
from .bisection import Bisection, BisectionError
from .box import Box
from .cylinder import Cone, Cylinder
from .frames import Frames, LocalFrame
from .implicits import (
    ImplicitGenus,
    ImplicitGyroid,
    ImplicitSphere,
    ImplicitSuperEllipsoid,
)
from .io_utils import CsvWriter
from .lattice_shapes import LatticeManifold, LatticePipe
from .lens import Lens
from .logobox import LogoBox
from .modulations import (
    Distribution,
    GenericContour,
    LineModulation,
    SurfaceModulation,
)
from .pipe import Pipe, PipeSegment
from .revolve import Revolve
from .ring import Ring
from .sphere import Sphere
from .splines import (
    ControlPointSpline,
    ControlPointSurface,
    CylindricalControlSpline,
    ISpline,
    TangentialControlSpline,
)

__all__ = [
    "BaseShape",
    "Bisection",
    "BisectionError",
    "Box",
    "Cone",
    "ControlPointSpline",
    "ControlPointSurface",
    "CsvWriter",
    "Cylinder",
    "CylindricalControlSpline",
    "Distribution",
    "Frames",
    "GenericContour",
    "ISpline",
    "ImplicitGenus",
    "ImplicitGyroid",
    "ImplicitSphere",
    "ImplicitSuperEllipsoid",
    "LatticeManifold",
    "LatticePipe",
    "Lens",
    "LineModulation",
    "LocalFrame",
    "LogoBox",
    "Pipe",
    "PipeSegment",
    "Revolve",
    "Ring",
    "Sphere",
    "SurfaceModulation",
    "TangentialControlSpline",
    "decimation",
    "formulas",
    "functions",
    "grids",
    "io_utils",
    "lists",
    "measure",
    "mesh_utils",
    "spline_ops",
    "vectors",
]
