# Advanced 4 — The web viewer

The [desktop viewer](03-viewer.md) needs a local window + OpenGL. The **web
viewer** renders the same geometry with [three.js](https://threejs.org) inside an
[anywidget](https://anywidget.dev), so it works where GLFW can't: **JupyterLab,
VS Code notebooks, Google Colab, and remote/SSH sessions**. Geometry is computed
in Python and streamed to the browser as binary buffers (compute-in-Python,
render-in-browser — not a WASM port of the kernel).

It needs the optional `[web]` extra and a browser with internet (three.js loads
from a CDN the first time, then the browser caches it):

```bash
pip install "picopie[web]"
```

## The one-liner (in a notebook)

```python
import picopie
from picopie.web import show

picopie.init(0.2)
part = picopie.Voxels.sphere(radius=12) - picopie.Voxels.sphere(center=(8, 0, 0), radius=7)
part.shell_(1.5)

show(part)                       # renders inline; make it the last line in a cell
show(body, holes, struts)        # several objects, each auto-colored
```

Over the viewport: **drag** orbits, **scroll** zooms, **right/middle-drag** pans;
**F** re-fits the camera, **W** toggles wireframe, **S** downloads a PNG.

## No notebook? Export a self-contained HTML

`export_html` writes a single file you can **double-click open in any browser** —
the three.js renderer is inlined and the geometry is embedded, so no Jupyter and
no server are needed:

```python
from picopie.web import export_html

export_html(part, "part.html")            # one object
export_html([body, holes], "scene.html")  # a list -> palette colors
```

There's a ready-to-run demo: `python examples/web/demo.py` (writes `web_demo.html`
and opens it).

## Driving the `WebViewer` — same API as the desktop `Viewer`

```python
from picopie.web import WebViewer

v = WebViewer(width=900, height=600, background=(0.1, 0.1, 0.12))
v.add(part, group=0)
v.set_group_material(0, (0.35, 0.6, 0.95), metallic=0.2, roughness=0.4)
v.set_object_matrix(part, translation)    # 4x4 row-major (mm), like the desktop viewer
v.reset_view()
v                                         # last expression in a cell -> renders
```

It mirrors `picopie.Viewer`: `add(obj, group=, color=)`, `remove(obj)`,
`remove_all()`, `set_group_material`, `set_group_visible`,
`set_group_matrix(group, mat4)` / `set_object_matrix(obj, mat4)`,
`set_background`, `reset_view()`, `toggle_wireframe()`, and `screenshot(path)`.
It accepts `Voxels`, `Mesh`, and `PolyLine` — exactly like the desktop viewer.

```python
v.screenshot("shot.png")   # the browser renders a PNG and sends it back to the
                           # kernel, so the file is written asynchronously
```

## Desktop vs. web — which to use

| | Desktop (`picopie.Viewer`) | Web (`picopie.web`) |
|---|---|---|
| Needs | local display + OpenGL | a browser (notebook **or** an exported HTML) |
| Works over SSH / in Colab | ✗ | ✓ |
| Event loop | native, blocks (`run()`) | the browser's |
| Screenshot | synchronous file | `S` to download, or async `screenshot()` |
| Shareable artifact | a PNG | a self-contained interactive HTML |

The two share the same scene API, so moving a script from one to the other is
mostly swapping `picopie.show` for `picopie.web.show`.

For a terse cheat sheet of the whole library, see **[QuickLearn](../QuickLearn.md)**.
