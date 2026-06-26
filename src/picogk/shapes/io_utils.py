"""Export helpers + CSV writer (port of ShapeKernel ``ShExportFunctions`` /
``CSVWriter``).

Thin wrappers over PicoPie's existing I/O (``Mesh.save_stl``, ``save_vdb``) plus
a small line-based CSV writer and an export-path helper. (CLI export is omitted
— the bundled runtime doesn't expose ``SaveToCliFile``.)
"""

from __future__ import annotations

import os

from ..mesh import Mesh
from ..vdb import save_vdb
from ..voxels import Voxels

#: Recognised export kinds for :func:`export_path` (lowercased file extension).
EXPORT_KINDS = ("stl", "obj", "tga", "png", "csv", "vdb")


def export_path(kind: str, filename: str, folder: str = ".") -> str:
    """``folder/filename.kind`` (kind is a file-extension stub, e.g. ``"stl"``)."""
    return os.path.join(folder, f"{filename}.{kind.lower()}")


def export_mesh_to_stl(mesh: Mesh, path: str) -> None:
    """Write a mesh to a binary STL file."""
    mesh.save_stl(path)


def export_voxels_to_stl(voxels: Voxels, path: str) -> None:
    """Mesh a voxel field and write it to STL."""
    voxels.to_mesh().save_stl(path)


def export_voxels_to_vdb(voxels: Voxels, path: str, name: str = "voxels") -> None:
    """Write a voxel field to an OpenVDB file under grid ``name``."""
    save_vdb(path, **{name: voxels})


class CsvWriter:
    """A simple line-based CSV writer (port of ``CSVWriter``).

    Usable as a context manager::

        with CsvWriter("out.csv") as w:
            w.add_line("a,b,c")
    """

    def __init__(self, filename: str):
        self.filename = filename
        self._lines: list[str] = []

    def add_line(self, line: str) -> None:
        self._lines.append(line)

    def export(self) -> None:
        """Flush all buffered lines to the file."""
        with open(self.filename, "w", newline="") as f:
            f.write("\n".join(self._lines))
            if self._lines:
                f.write("\n")

    def __enter__(self) -> CsvWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.export()
