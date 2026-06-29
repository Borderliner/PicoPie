"""Browser-based 3D viewer — three.js rendered inside an `anywidget`.

Geometry is computed in Python and streamed to the browser as binary buffers, so
it renders where the desktop GLFW :class:`picogk.Viewer` can't: JupyterLab, VS
Code notebooks, Google Colab, and remote/SSH sessions (compute-in-Python,
render-in-browser — *not* a WASM port of the kernel).

Requires the optional ``[web]`` extra::

    pip install "picopie[web]"

Usage (in a notebook cell)::

    import picogk
    from picogk.web import show
    picogk.init(0.5)
    show(picogk.Voxels.sphere(radius=10))            # renders inline

    # or, mirroring the desktop Viewer:
    from picogk.web import WebViewer
    v = WebViewer(background=(0.1, 0.1, 0.12))
    v.add(part, group=0)
    v.set_group_material(0, (0.4, 0.6, 0.9), metallic=0.2, roughness=0.4)
    v                                                # last expression renders it

The API mirrors :class:`picogk.Viewer` (``add`` / ``set_group_material`` /
``set_background`` / ``show``). three.js is loaded from a CDN the first time
(needs internet; the browser caches it afterwards).
"""
from __future__ import annotations

import base64
import json
import pathlib

import numpy as np

try:
    import anywidget
    import traitlets
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise ModuleNotFoundError(
        "picogk.web needs anywidget — install the web extra:\n"
        '    pip install "picopie[web]"'
    ) from exc

from ..mesh import Mesh
from ..polyline import PolyLine
from ..voxels import Voxels

__all__ = ["WebViewer", "export_html", "show"]

_JS = pathlib.Path(__file__).resolve().parent.parent / "_assets" / "web_viewer.js"

# Default per-group colors (mirrors picogk.viewer._PALETTE so the web and desktop
# viewers color a multi-object scene the same way).
_PALETTE: list[tuple[float, float, float]] = [
    (0.35, 0.60, 0.95), (0.95, 0.55, 0.30), (0.45, 0.80, 0.45),
    (0.85, 0.45, 0.75), (0.90, 0.80, 0.35), (0.50, 0.75, 0.90),
]


def _rgb(color) -> list[float]:
    return [float(c) for c in tuple(color)[:3]]


_IDENTITY16 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _mat16(matrix) -> list[float]:
    """A 4x4 row-major transform -> flat 16 floats (accepts a (4, 4) array or a
    length-16 sequence)."""
    arr = np.asarray(matrix, dtype=float).reshape(-1)
    if arr.size != 16:
        raise ValueError("matrix must be 4x4 (16 values, row-major)")
    return [float(x) for x in arr]


def _mesh_item(mesh: Mesh) -> dict:
    verts = np.ascontiguousarray(mesh.vertices, dtype="<f4")
    tris = np.ascontiguousarray(mesh.triangles, dtype="<u4").reshape(-1)
    return {"kind": "mesh", "positions": verts.tobytes(), "indices": tris.tobytes()}


def _line_item(pl: PolyLine) -> dict:
    pts = np.ascontiguousarray(pl.vertices, dtype="<f4")
    return {"kind": "line", "positions": pts.tobytes(), "line_color": _rgb(pl.color())}


class WebViewer(anywidget.AnyWidget):
    """An interactive three.js viewer that renders inline in a notebook.

    Add :class:`~picogk.Voxels`, :class:`~picogk.Mesh`, or :class:`~picogk.PolyLine`
    objects with :meth:`add`; display by making the widget the last expression in
    a cell (or call ``IPython.display.display(viewer)``).
    """

    _esm = _JS
    geometry = traitlets.List().tag(sync=True)  # type: ignore[var-annotated]
    background = traitlets.List(traitlets.Float()).tag(sync=True)
    width = traitlets.Int(720).tag(sync=True)
    height = traitlets.Int(480).tag(sync=True)
    _fit = traitlets.Int(0).tag(sync=True)          # bump -> re-fit camera
    _grab = traitlets.Int(0).tag(sync=True)         # bump -> front-end sends a PNG back
    _wireframe = traitlets.Int(0).tag(sync=True)    # bump -> toggle wireframe

    def __init__(self, *, width: int = 720, height: int = 480,
                 background=(0.16, 0.16, 0.2)):
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.background = _rgb(background)
        self._items: list[dict] = []
        self._materials: dict[int, dict] = {}
        self._hidden: set[int] = set()
        self._group_matrices: dict[int, list[float]] = {}
        self._next_id = 0
        self._pending_screenshot: str | None = None
        self.on_msg(self._handle_msg)

    # --- content -------------------------------------------------------------
    def add(self, obj, group: int = 0, color=None) -> WebViewer:
        """Add a ``Voxels`` | ``Mesh`` | ``PolyLine`` to ``group``.

        ``color`` (an ``(r, g, b)`` in 0..1) overrides the group/palette color
        for this object only.
        """
        if isinstance(obj, Voxels):
            item = _mesh_item(obj.to_mesh())
        elif isinstance(obj, Mesh):
            item = _mesh_item(obj)
        elif isinstance(obj, PolyLine):
            item = _line_item(obj)
        else:
            raise TypeError(
                f"WebViewer.add expects Voxels | Mesh | PolyLine, got "
                f"{type(obj).__name__}")
        item["group"] = int(group)
        item["id"] = self._next_id
        item["_obj"] = obj
        self._next_id += 1
        if color is not None:
            item["color_override"] = _rgb(color)
        self._items.append(item)
        self._sync()
        return self

    def remove(self, obj) -> WebViewer:
        """Remove every previously-added object that is ``obj``."""
        self._items = [it for it in self._items if it["_obj"] is not obj]
        self._sync()
        return self

    def remove_all(self) -> WebViewer:
        """Remove every object."""
        self._items = []
        self._sync()
        return self

    # --- styling -------------------------------------------------------------
    def set_group_material(self, group: int, color, metallic: float = 0.1,
                           roughness: float = 0.6) -> WebViewer:
        """Set the PBR material of every object in ``group``."""
        self._materials[int(group)] = {
            "color": _rgb(color), "metallic": float(metallic),
            "roughness": float(roughness)}
        self._sync()
        return self

    def set_group_visible(self, group: int, visible: bool = True) -> WebViewer:
        """Show or hide a group."""
        self._hidden.discard(int(group)) if visible else self._hidden.add(int(group))
        self._sync()
        return self

    def set_background(self, color) -> WebViewer:
        """Set the viewport background color (``(r, g, b)`` in 0..1)."""
        self.background = _rgb(color)
        return self

    # --- transforms ----------------------------------------------------------
    def set_group_matrix(self, group: int, matrix) -> WebViewer:
        """Apply a 4x4 row-major transform (mm) to every object in ``group``
        (matches :meth:`picogk.Viewer.set_group_matrix`)."""
        self._group_matrices[int(group)] = _mat16(matrix)
        self._sync()
        return self

    def set_object_matrix(self, obj, matrix) -> WebViewer:
        """Apply a 4x4 row-major transform to a specific added object (overrides
        its group's matrix)."""
        m = _mat16(matrix)
        for it in self._items:
            if it["_obj"] is obj:
                it["matrix_override"] = m
        self._sync()
        return self

    # --- camera / capture ----------------------------------------------------
    def reset_view(self) -> WebViewer:
        """Re-fit the camera to the scene (like pressing ``F`` over the viewport)."""
        self._fit += 1
        return self

    def toggle_wireframe(self) -> WebViewer:
        """Toggle wireframe rendering (like pressing ``W``)."""
        self._wireframe += 1
        return self

    def screenshot(self, path: str) -> str:
        """Save a PNG of the current view to ``path``.

        The browser renders the image and sends it back to the kernel, so the
        file is written **asynchronously** -- it appears once the front-end
        responds (the widget must be displayed and the kernel processing
        messages). Pressing ``S`` over the viewport downloads a PNG directly.
        """
        self._pending_screenshot = str(path)
        self._grab += 1
        return path

    def _handle_msg(self, _widget, content, buffers) -> None:
        if (isinstance(content, dict) and content.get("type") == "screenshot"
                and buffers and self._pending_screenshot is not None):
            pathlib.Path(self._pending_screenshot).write_bytes(bytes(buffers[0]))
            self._pending_screenshot = None

    # --- internal ------------------------------------------------------------
    def _color_for(self, item: dict) -> list[float]:
        if "color_override" in item:
            return item["color_override"]
        mat = self._materials.get(item["group"])
        if mat is not None:
            return mat["color"]
        if item["kind"] == "line":
            return item.get("line_color", [1.0, 1.0, 1.0])
        return list(_PALETTE[item["group"] % len(_PALETTE)])

    def _matrix_for(self, item: dict) -> list[float]:
        if "matrix_override" in item:
            return item["matrix_override"]
        return self._group_matrices.get(item["group"], _IDENTITY16)

    def _sync(self) -> None:
        out = []
        for it in self._items:
            mat = self._materials.get(it["group"], {})
            entry = {
                "id": it["id"],
                "kind": it["kind"],
                "positions": it["positions"],
                "color": self._color_for(it),
                "metallic": mat.get("metallic", 0.1),
                "roughness": mat.get("roughness", 0.6),
                "visible": it["group"] not in self._hidden,
                "matrix": self._matrix_for(it),
            }
            if it["kind"] == "mesh":
                entry["indices"] = it["indices"]
            out.append(entry)
        self.geometry = out


def show(*objects, width: int = 720, height: int = 480,
         background=(0.16, 0.16, 0.2)) -> WebViewer:
    """Convenience one-liner: build a :class:`WebViewer`, add each object to its
    own group with an auto-assigned palette color, and return it (display it by
    making it the last expression in a notebook cell)."""
    viewer = WebViewer(width=width, height=height, background=background)
    for i, obj in enumerate(objects):
        viewer.add(obj, group=i)
    return viewer


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>html,body{margin:0;height:100%;overflow:hidden;background:#28282e}
#app{width:100vw;height:100vh}
#hint{position:fixed;left:12px;bottom:10px;color:#9aa;pointer-events:none;
  font:12px system-ui,sans-serif;text-shadow:0 1px 2px #000}</style></head>
<body><div id="app"></div>
<div id="hint">drag orbit &nbsp; scroll zoom &nbsp; F fit &nbsp; W wireframe &nbsp; S save PNG</div>
<script type="module">
__JS__

const DATA = __DATA__;
function b64(s, Ctor){const bin=atob(s);const u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new Ctor(u.buffer);}
for(const it of DATA.geometry){it.positions=b64(it.positions,Float32Array);
  if(it.indices)it.indices=b64(it.indices,Uint32Array);}
const el=document.getElementById("app");
const v=createViewer(el,{width:el.clientWidth,height:el.clientHeight,background:DATA.background});
v.setBackground(DATA.background);
v.setGeometry(DATA.geometry);
addEventListener("resize",()=>v.resize(el.clientWidth,el.clientHeight));
</script></body></html>
"""


def export_html(objects, path: str, *, background=(0.16, 0.16, 0.2),
                title: str = "PicoPie") -> str:
    """Write a **self-contained, double-click-to-open** HTML file rendering
    ``objects`` (a single object or a list) — no Jupyter required, just a browser
    with internet (three.js loads from a CDN). Returns ``path``.

    The same three.js renderer as the notebook widget is inlined, and the
    geometry is embedded as base64 buffers.
    """
    objs = list(objects) if isinstance(objects, (list, tuple)) else [objects]
    viewer = show(*objs, background=background)
    items = []
    for e in viewer.geometry:
        item = {k: e[k] for k in ("kind", "color", "metallic", "roughness", "visible")}
        item["positions"] = base64.b64encode(e["positions"]).decode("ascii")
        if "indices" in e:
            item["indices"] = base64.b64encode(e["indices"]).decode("ascii")
        items.append(item)
    data = json.dumps({"geometry": items, "background": _rgb(background)})
    html = (_HTML.replace("__JS__", _JS.read_text(encoding="utf-8"))
                 .replace("__DATA__", data)
                 .replace("__TITLE__", title))
    pathlib.Path(path).write_text(html, encoding="utf-8")
    return path
