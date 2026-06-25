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
import threading
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
        # GLFW requires the window + its event loop on the process main thread
        # (mandatory on macOS); enforce it for portable behaviour.
        if threading.current_thread() is not threading.main_thread():
            raise PicoGKError("Viewer must be created on the main thread")
        self._thread = threading.get_ident()
        self._lib = library.lib()
        self._inst = library.instance()
        self._closed = True
        # orbit-camera state (driven by the mouse/scroll callbacks below)
        self._target = np.zeros(3, np.float32)
        self._radius = 10.0
        self._azimuth = math.radians(45.0)
        self._elevation = math.radians(25.0)
        self._zoom = 1.0
        self._autofit = True
        # interaction state
        self._drag_button: int | None = None
        self._mouse = (0.0, 0.0)
        self._orbit_speed = 0.008
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

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise PicoGKError("Viewer must be polled/closed on the thread that created it")

    def poll(self) -> bool:
        """Process one batch of window events + render. Returns False once closed."""
        self._check_thread()
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
    def _distance(self) -> float:
        return self._radius / math.sin(self._FOV_Y * 0.5) * 1.1 * self._zoom

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(view direction target->eye, camera right, camera up) for the orbit."""
        ce, se = math.cos(self._elevation), math.sin(self._elevation)
        ca, sa = math.cos(self._azimuth), math.sin(self._azimuth)
        d = np.array([ce * ca, ce * sa, se], np.float32)        # = z axis in look_at
        right = np.cross(_UP, d)
        right = right / (np.linalg.norm(right) or 1.0)
        up_cam = np.cross(d, right)
        return d, right, up_cam

    def _view_projection(self, aspect: float) -> tuple[np.ndarray, np.ndarray]:
        dist = self._distance()
        d, _, _ = self._basis()
        eye = self._target + d * dist
        near = max(dist * 0.01, 1e-2)
        far = dist * 10.0 + 1000.0
        return _look_at(eye, self._target, _UP) @ _perspective(self._FOV_Y, aspect, near, far), eye

    def reset_view(self) -> Viewer:
        """Re-enable auto-framing of the scene at the default angle."""
        self._autofit = True
        self._zoom = 1.0
        self._azimuth, self._elevation = math.radians(45.0), math.radians(25.0)
        self._request_update()
        return self

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
        # GLFW key codes: 256 Esc, 81 Q, 70 F, 83 S; action 1 == press.
        if action != 1:
            return
        if key in (256, 81):                 # Esc / Q -> close
            self._lib.Viewer_RequestClose(self._h)
        elif key == 70:                      # F -> re-fit the view
            self.reset_view()
        elif key == 83:                      # S -> screenshot
            try:
                self.screenshot("picopie_screenshot.png")
            except BaseException as exc:
                self._cb_error.append(exc)

    def _on_mouse_moved(self, viewer, pos, shift, ctrl, alt, sup):
        px, py = float(pos[0].X), float(pos[0].Y)
        dx, dy = px - self._mouse[0], py - self._mouse[1]
        self._mouse = (px, py)
        if self._drag_button is None:
            return
        self._autofit = False                # user took control of the camera
        pan = self._drag_button in (1, 2) or (self._drag_button == 0 and shift)
        if pan:
            _, right, up_cam = self._basis()
            scale = self._distance() * 0.0015
            self._target = self._target + (-dx * right + dy * up_cam) * np.float32(scale)
        else:                                # orbit
            self._azimuth -= dx * self._orbit_speed
            lim = math.pi / 2 - 1e-3
            self._elevation = max(-lim, min(lim, self._elevation + dy * self._orbit_speed))
        self._request_update()

    def _on_mouse_button(self, viewer, button, action, mods, pos):
        if action == 1:                      # press
            self._drag_button = int(button)
            self._mouse = (float(pos[0].X), float(pos[0].Y))
        else:                                # release
            self._drag_button = None

    def _on_scroll(self, viewer, offset, pos, shift, ctrl, alt, sup):
        self._zoom = max(0.05, min(20.0, self._zoom * (1.0 - 0.1 * float(offset[0].Y))))
        self._request_update()

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


# A few distinct, readable colors for show()'s default groups.
_PALETTE = [
    (0.35, 0.60, 0.95), (0.95, 0.55, 0.30), (0.45, 0.80, 0.45),
    (0.85, 0.45, 0.75), (0.90, 0.80, 0.35), (0.50, 0.75, 0.90),
]


def show(*objects, title: str = "PicoPie", size: tuple[int, int] = (1280, 800),
         background=None, block: bool = True) -> Viewer:
    """Open an interactive viewer for one or more objects (a one-liner).

    Each object goes in its own group with a color from a default palette::

        picogk.show(part)
        picogk.show(body, holes)          # two objects, two colors

    With ``block=True`` (default) this runs the window's event loop until it is
    closed (main thread); pass ``block=False`` to drive :meth:`Viewer.poll`
    yourself. Returns the :class:`Viewer`. Requires a display -- for headless
    image output use :meth:`Viewer.screenshot` or :mod:`picogk.viz`.
    """
    v = Viewer(title=title, size=size)
    if background is not None:
        v.set_background(background)
    for i, obj in enumerate(objects):
        v.add(obj, group=i)
        v.set_group_material(i, _PALETTE[i % len(_PALETTE)])
    if block:
        v.run()
    return v


def render_png(obj, path: str, *, size: tuple[int, int] = (1280, 800),
               background=None, frames: int = 12) -> str:
    """Render object(s) to an image in one shot, without an interactive loop.

    ``obj`` is a single ``Voxels``/``Mesh``/``PolyLine`` or a list of them
    (each gets a palette color)::

        picogk.render_png(part, "part.png")
        picogk.render_png([body, holes], "scene.png", size=(1920, 1080))

    Still needs a display + OpenGL (raises :class:`PicoGKError` otherwise); for
    truly headless image output use :mod:`picogk.viz`. Returns the written path.
    """
    objs = list(obj) if isinstance(obj, (list, tuple)) else [obj]
    v = Viewer(size=size)
    try:
        if background is not None:
            v.set_background(background)
        for i, o in enumerate(objs):
            v.add(o, group=i)
            v.set_group_material(i, _PALETTE[i % len(_PALETTE)])
        return v.screenshot(str(path), frames=frames)
    finally:
        v.close()
