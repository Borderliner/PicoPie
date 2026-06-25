# Advanced 3 — The interactive viewer

PicoPie binds PicoGK's native GLFW/OpenGL viewer for real-time, PBR-lit display.
It needs a window system + OpenGL — on a headless machine (CI, a server) creating
one raises `PicoGKError`, so use the [headless PNG helpers](#headless-use-the-png-helpers-instead)
(below) there.

![A rendered part](../../images/viewer_example.png)

## The one-liner

```python
import picogk
from picogk import Voxels

picogk.init(0.2)
part = Voxels.sphere(radius=12) - Voxels.sphere(center=(8, 0, 0), radius=7)
part.shell_(1.5)

picogk.show(part)                      # opens a window, runs until you close it
picogk.show(body, holes, struts)       # several objects, each auto-colored
```

In the window: **left-drag** orbits, **right/middle-drag** (or **Shift+left**) pans,
**scroll** zooms; **Esc/Q** close, **F** re-fits, **S** screenshots.

## Offscreen render (no event loop)

To render straight to an image (handy in scripts that also need a display):

```python
picogk.render_png(part, "part.png", size=(1920, 1080))
picogk.render_png([body, holes], "scene.png")     # a list -> palette colors
```

## Driving the `Viewer` directly

For control over groups, materials, and the loop:

```python
from picogk import Viewer

with Viewer(title="part", size=(1280, 800)) as v:     # 'with' guarantees cleanup
    v.add(part, group=0)
    v.set_group_material(0, (0.35, 0.6, 0.95), metallic=0.1, roughness=0.4)
    v.set_background((0.16, 0.16, 0.2))
    v.screenshot("part.png")            # render to an image (native TGA -> PNG via Pillow)
    v.run()                             # blocks until the window closes
```

Useful methods: `add(obj, group=)`, `remove(obj)`, `remove_all()`,
`set_group_material`, `set_group_visible`, `set_group_matrix(group, mat4)` /
`set_object_matrix(obj, mat4)` (a 4×4 transform), `reset_view()`, `is_valid()`.

## Rules that matter

- **Main thread only.** The window and its event loop must live on the process main
  thread (a GLFW requirement, mandatory on macOS). The `Viewer` enforces this and
  raises if you create or poll it elsewhere.
- **Always release it.** Use `with Viewer(...)`, `run()`, or `render_png()` (all
  close for you), or call `close()` yourself. Using a viewer after close raises
  `InvalidHandleError` — it won't crash.
- **One per process** is the typical pattern.

## Headless? Use the PNG helpers instead

On CI/servers there's no display, so prefer the headless tools:

```python
from picogk import save_slice_png, save_slice_sheet, mesh_preview

save_slice_png(part, "z.png", axis="z", mode="sdf")   # a cross-section
save_slice_sheet(part, "sheet.png", count=16)          # a montage of slices
mesh_preview(part.to_mesh(), "preview.png")            # a shaded 3D preview (matplotlib)
```

These need the `[viz]` extra (`pip install "picopie[viz]"`) and work entirely
without a display.

That's the full tour. For a terse, copy-pasteable cheat sheet of the whole API, see
**[QuickLearn](../QuickLearn.md)**.
