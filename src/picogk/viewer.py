"""Interactive 3D viewer (optional; requires a window system + OpenGL).

PicoPie is headless-first, but this binds PicoGK's native GLFW/OpenGL viewer for
interactive display. On a machine with no display (e.g. CI) creating a
:class:`Viewer` raises :class:`PicoGKError`; for headless image output use the
PNG helpers in :mod:`picogk.viz`, or :func:`screenshot` after a few render frames.

    import picogk
    from picogk import Voxels, Viewer
    picogk.init(0.5)
    v = Viewer()
    v.add(Voxels.sphere(radius=10))
    v.screenshot("sphere.png")     # render to an image
    v.run()                        # or open the interactive window (main thread)

GLFW requires the window + poll loop to live on the main thread.
"""
from __future__ import annotations

import ctypes as C
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import library
from ._errors import PicoGKError
from ._native.ctypes_types import (
    PKBBox3,
    PKColorFloat,
    PKFInfo,
    PKMatrix4x4,
    PKPFKeyPressed,
    PKPFMouseButton,
    PKPFMouseMoved,
    PKPFScrollWheel,
    PKPFUpdateRequested,
    PKPFWindowSize,
    PKVector2,
    PKVector3,
    PKVector4,
)
from .types import to_color

_ASSET = Path(__file__).parent / "_assets" / "viewer_environment.zip"
_UP = np.array([0.0, 0.0, 1.0], np.float32)   # PicoGK is Z-up


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """System.Numerics-compatible right-handed view matrix (row-major)."""
    z = eye - target
    z = z / (np.linalg.norm(z) or 1.0)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) or 1.0)
    y = np.cross(z, x)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 0], m[2, 0] = x
    m[0, 1], m[1, 1], m[2, 1] = y
    m[0, 2], m[1, 2], m[2, 2] = z
    m[3, 0] = -float(x @ eye)
    m[3, 1] = -float(y @ eye)
    m[3, 2] = -float(z @ eye)
    m[3, 3] = 1.0
    return m


def _perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    """System.Numerics-compatible perspective matrix (row-major)."""
    ys = 1.0 / math.tan(fovy * 0.5)
    xs = ys / max(aspect, 1e-6)
    p = np.zeros((4, 4), np.float32)
    p[0, 0] = xs
    p[1, 1] = ys
    p[2, 2] = far / (near - far)
    p[2, 3] = -1.0
    p[3, 2] = near * far / (near - far)
    return p


class Viewer:
    """A native interactive viewer window. One per process is typical."""

    _FOV_Y = math.radians(35.0)

    def __init__(self, title: str = "PicoPie", size: tuple[int, int] = (1280, 800)):
        self._lib = library.lib()
        self._inst = library.instance()
        self._closed = True
        # camera state (made interactive in a later phase)
        self._target = np.zeros(3, np.float32)
        self._radius = 10.0
        self._azimuth = math.radians(45.0)
        self._elevation = math.radians(25.0)
        self._zoom = 1.0
        self._autofit = True
        self._bg = PKColorFloat(0.16, 0.16, 0.20, 1.0)
        self._cb_error: list[BaseException] = []

        # CFUNCTYPE instances MUST be retained for the viewer's lifetime.
        self._cbs = (
            PKFInfo(self._on_info),
            PKPFUpdateRequested(self._on_update),
            PKPFKeyPressed(self._on_key),
            PKPFMouseMoved(self._on_mouse_moved),
            PKPFMouseButton(self._on_mouse_button),
            PKPFScrollWheel(self._on_scroll),
            PKPFWindowSize(self._on_window_size),
        )
        sz = PKVector2(float(size[0]), float(size[1]))
        self._h = self._lib.Viewer_hCreate(title.encode(), C.byref(sz), *self._cbs)
        if not self._h:
            raise PicoGKError(
                "could not create a viewer window (no display / OpenGL available?). "
                "Use picogk.viz for headless PNG output.")
        self._closed = False
        self._load_lights()

    # --- setup ---------------------------------------------------------------
    def _load_lights(self) -> None:
        """Load the bundled image-based-lighting environment (else the scene is dark)."""
        if not _ASSET.exists():
            return
        with zipfile.ZipFile(_ASSET) as z:
            diff = z.read("Diffuse.dds")
            spec = z.read("Specular.dds")
        self._lib.Viewer_bLoadLightSetup(self._h, diff, len(diff), spec, len(spec))

    # --- content -------------------------------------------------------------
    def add(self, obj, group: int = 0) -> Viewer:
        """Add a ``Voxels`` / ``Mesh`` / ``PolyLine`` to a display group."""
        from .mesh import Mesh
        from .polyline import PolyLine
        from .voxels import Voxels
        if isinstance(obj, Voxels):
            self._lib.Viewer_AddVoxels(self._inst, self._h, int(group), obj.handle)
        elif isinstance(obj, Mesh):
            self._lib.Viewer_AddMesh(self._inst, self._h, int(group), obj.handle)
        elif isinstance(obj, PolyLine):
            self._lib.Viewer_AddPolyLine(self._inst, self._h, int(group), obj.handle)
        else:
            raise TypeError(f"cannot add {type(obj).__name__} to a Viewer")
        return self

    def remove_all(self) -> Viewer:
        self._lib.Viewer_RemoveAllObjects(self._h)
        return self

    def set_group_material(self, group: int, color, metallic: float = 0.1,
                           roughness: float = 0.5) -> Viewer:
        c = to_color(color)
        self._lib.Viewer_SetGroupMaterial(self._h, int(group), C.byref(c),
                                          float(metallic), float(roughness))
        return self

    def set_group_visible(self, group: int, visible: bool = True) -> Viewer:
        self._lib.Viewer_SetGroupVisible(self._h, int(group), bool(visible))
        return self

    def set_background(self, color) -> Viewer:
        self._bg = to_color(color)
        self._request_update()
        return self

    # --- rendering / loop ----------------------------------------------------
    def _request_update(self) -> None:
        self._lib.Viewer_RequestUpdate(self._h)

    def poll(self) -> bool:
        """Process one batch of window events + render. Returns False once closed."""
        alive = bool(self._lib.Viewer_bPoll(self._h))
        if self._cb_error:
            err = self._cb_error.pop(0)
            raise err
        return alive

    def _render_frames(self, n: int) -> None:
        for _ in range(n):
            self._request_update()
            if not self.poll():
                break

    def run(self) -> None:
        """Open the window and pump events until it is closed (main thread)."""
        self._request_update()
        while self.poll():
            pass
        self.close()

    def screenshot(self, path: str, frames: int = 12) -> str:
        """Render and save a screenshot. ``.png``/``.jpg`` are converted (Pillow);
        any other suffix keeps the native TGA. Returns the written path."""
        self._render_frames(frames)
        tmp = Path(tempfile.mkdtemp()) / "shot.tga"
        self._lib.Viewer_RequestScreenShot(self._h, str(tmp).encode())
        self._render_frames(frames)          # flush the queued screenshot
        out = Path(path)
        if out.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            from PIL import Image  # picopie[viz]
            Image.open(tmp).convert("RGB").save(out)
        else:
            shutil.move(str(tmp), str(out))
        return str(out)

    # --- camera --------------------------------------------------------------
    def _view_projection(self, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        dist = self._radius / math.sin(self._FOV_Y * 0.5) * 1.1 * self._zoom
        ce, se = math.cos(self._elevation), math.sin(self._elevation)
        ca, sa = math.cos(self._azimuth), math.sin(self._azimuth)
        eye = self._target + np.array([ce * ca, ce * sa, se], np.float32) * dist
        near = max(dist * 0.01, 1e-2)
        far = dist * 10.0 + 1000.0
        return _look_at(eye, self._target, _UP) @ _perspective(self._FOV_Y, aspect, near, far), eye

    # --- callbacks (must never raise into native code) -----------------------
    def _on_info(self, msg, fatal):
        pass

    def _on_update(self, viewer, vp, bg, mvp, eye):
        try:
            if self._autofit:
                box = PKBBox3()
                self._lib.Viewer_GetBoundingBox(self._h, C.byref(box))
                lo = np.array([box.vecMin.X, box.vecMin.Y, box.vecMin.Z], np.float32)
                hi = np.array([box.vecMax.X, box.vecMax.Y, box.vecMax.Z], np.float32)
                if np.all(hi >= lo):
                    self._target = (lo + hi) * 0.5
                    self._radius = max(float(np.linalg.norm(hi - lo) * 0.5), 1e-3)
            aspect = vp[0].X / max(1.0, vp[0].Y)
            m, e = self._view_projection(aspect)
            mvp[0] = PKMatrix4x4(*(PKVector4(*m[r]) for r in range(4)))
            eye[0] = PKVector3(*e)
            bg[0] = self._bg
        except BaseException as exc:
            self._cb_error.append(exc)

    def _on_key(self, viewer, key, scancode, action, mods):
        # GLFW: action 1 == press. 256 == Escape, 81 == 'Q', 83 == 'S'.
        if action == 1 and key in (256, 81):
            self._lib.Viewer_RequestClose(self._h)

    def _on_mouse_moved(self, viewer, pos, shift, ctrl, alt, sup):
        pass

    def _on_mouse_button(self, viewer, button, action, mods, pos):
        pass

    def _on_scroll(self, viewer, offset, pos, shift, ctrl, alt, sup):
        pass

    def _on_window_size(self, viewer, size):
        self._request_update()

    # --- lifetime ------------------------------------------------------------
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._lib.Viewer_Destroy(self._h)

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
