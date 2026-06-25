"""Interactive 3D viewer (optional; requires a window system + OpenGL).

PicoPie is headless-first, but this binds PicoGK's native GLFW/OpenGL viewer for
interactive display. On a machine with no display (e.g. CI) creating a
:class:`Viewer` raises :class:`PicoGKError`; for headless image output use the
PNG helpers in :mod:`picogk.viz`, or :func:`render_png` / :meth:`Viewer.screenshot`.

    import picogk
    from picogk import Voxels, Viewer
    picogk.init(0.5)
    with Viewer() as v:                # close() is mandatory -> use 'with' or run()
        v.add(Voxels.sphere(radius=10))
        v.screenshot("sphere.png")
        v.run()

The window + event loop must live on the **main thread** (a GLFW requirement,
mandatory on macOS); the Viewer enforces this.
"""
from __future__ import annotations

import contextlib
import ctypes as C
import math
import tempfile
import threading
import warnings
import zipfile
from pathlib import Path

import numpy as np

from . import library
from ._errors import InvalidHandleError, PicoGKError
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
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


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
    far = max(far, near + 1e-3)                 # guard against near==far -> NaN
    ys = 1.0 / math.tan(fovy * 0.5)
    xs = ys / max(aspect, 1e-6)
    p = np.zeros((4, 4), np.float32)
    p[0, 0] = xs
    p[1, 1] = ys
    p[2, 2] = far / (near - far)
    p[2, 3] = -1.0
    p[3, 2] = near * far / (near - far)
    return p


def _to_mat4(matrix) -> PKMatrix4x4:
    """A 4x4 array-like (row-major, System.Numerics convention) -> PKMatrix4x4."""
    a = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    return PKMatrix4x4(*(PKVector4(*a[r]) for r in range(4)))


class Viewer:
    """A native interactive viewer window. One per process is typical.

    Must be created, polled, and closed on the main thread. Always release it
    with :meth:`close` (or use it as a context manager / call :meth:`run`).
    """

    _FOV_Y = math.radians(35.0)

    def __init__(self, title: str = "PicoPie", size: tuple[int, int] = (1280, 800)):
        if threading.current_thread() is not threading.main_thread():
            raise PicoGKError("Viewer must be created on the main thread")
        self._thread = threading.get_ident()
        self._lib = library.lib()
        self._inst = library.instance()
        self._closed = True
        # orbit-camera state (driven by the mouse/scroll callbacks)
        self._target = np.zeros(3, np.float32)
        self._radius = 10.0
        self._azimuth = math.radians(45.0)
        self._elevation = math.radians(25.0)
        self._zoom = 1.0
        self._autofit = True
        # interaction / loop state
        self._drag_button: int | None = None
        self._mouse = (0.0, 0.0)
        self._orbit_speed = 0.008
        self._bg = PKColorFloat(0.16, 0.16, 0.20, 1.0)
        self._cb_error: list[BaseException] = []
        self._pending_screenshot: str | None = None
        self._in_screenshot = False

        # CFUNCTYPE instances MUST be retained for the viewer's lifetime; each is
        # wrapped so a Python exception is deferred, never unwound into native C.
        self._cbs = (
            PKFInfo(self._safe(self._on_info)),
            PKPFUpdateRequested(self._safe(self._on_update)),
            PKPFKeyPressed(self._safe(self._on_key)),
            PKPFMouseMoved(self._safe(self._on_mouse_moved)),
            PKPFMouseButton(self._safe(self._on_mouse_button)),
            PKPFScrollWheel(self._safe(self._on_scroll)),
            PKPFWindowSize(self._safe(self._on_window_size)),
        )
        sz = PKVector2(float(size[0]), float(size[1]))
        self._h = self._lib.Viewer_hCreate(title.encode(), C.byref(sz), *self._cbs)
        if not self._h:
            raise PicoGKError(
                "could not create a viewer window (no display / OpenGL available?). "
                "Use picogk.viz for headless PNG output.")
        self._closed = False
        library._register(self)        # so shutdown() invalidates us before destroying the instance
        self._load_lights()

    # --- guards --------------------------------------------------------------
    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise PicoGKError("Viewer must be used on the thread that created it")

    def _alive(self) -> None:
        if self._closed:
            raise InvalidHandleError("operation on a closed Viewer")
        self._check_thread()

    def _safe(self, fn):
        """Wrap a callback so exceptions are captured (re-raised from poll())."""
        def wrapper(*args):
            try:
                fn(*args)
            except BaseException as exc:
                self._cb_error.append(exc)
        return wrapper

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
        self._alive()
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

    def remove(self, obj) -> Viewer:
        """Remove a previously added ``Voxels`` / ``Mesh`` / ``PolyLine``."""
        self._alive()
        from .mesh import Mesh
        from .polyline import PolyLine
        from .voxels import Voxels
        if isinstance(obj, Voxels):
            self._lib.Viewer_RemoveVoxels(self._inst, self._h, obj.handle)
        elif isinstance(obj, Mesh):
            self._lib.Viewer_RemoveMesh(self._inst, self._h, obj.handle)
        elif isinstance(obj, PolyLine):
            self._lib.Viewer_RemovePolyLine(self._inst, self._h, obj.handle)
        else:
            raise TypeError(f"cannot remove {type(obj).__name__} from a Viewer")
        return self

    def remove_all(self) -> Viewer:
        self._alive()
        self._lib.Viewer_RemoveAllObjects(self._h)
        return self

    def set_group_material(self, group: int, color, metallic: float = 0.1,
                           roughness: float = 0.5) -> Viewer:
        self._alive()
        c = to_color(color)
        self._lib.Viewer_SetGroupMaterial(self._h, int(group), C.byref(c),
                                          float(metallic), float(roughness))
        return self

    def set_group_visible(self, group: int, visible: bool = True) -> Viewer:
        self._alive()
        self._lib.Viewer_SetGroupVisible(self._h, int(group), bool(visible))
        return self

    def set_group_matrix(self, group: int, matrix) -> Viewer:
        """Apply a 4x4 transform (row-major) to a whole display group."""
        self._alive()
        m = _to_mat4(matrix)
        self._lib.Viewer_SetGroupMatrix(self._h, int(group), C.byref(m))
        return self

    def set_object_matrix(self, obj, matrix) -> Viewer:
        """Apply a 4x4 transform (row-major) to a single added object."""
        self._alive()
        from .mesh import Mesh
        from .polyline import PolyLine
        from .voxels import Voxels
        m = _to_mat4(matrix)
        if isinstance(obj, Voxels):
            self._lib.Viewer_SetVoxelsMatrix(self._inst, self._h, obj.handle, C.byref(m))
        elif isinstance(obj, Mesh):
            self._lib.Viewer_SetMeshMatrix(self._inst, self._h, obj.handle, C.byref(m))
        elif isinstance(obj, PolyLine):
            self._lib.Viewer_SetPolyLineMatrix(self._inst, self._h, obj.handle, C.byref(m))
        else:
            raise TypeError(f"cannot transform {type(obj).__name__}")
        return self

    def set_background(self, color) -> Viewer:
        self._alive()
        self._bg = to_color(color)
        self._lib.Viewer_RequestUpdate(self._h)
        return self

    def enable_experimental(self, enable: bool = True) -> Viewer:
        self._alive()
        self._lib.Viewer_EnableExperimental(self._h, bool(enable))
        return self

    def is_valid(self) -> bool:
        """Whether the underlying native viewer is still valid (False once closed)."""
        if self._closed:
            return False
        self._check_thread()
        return bool(self._lib.Viewer_bIsValid(self._h))

    # --- rendering / loop ----------------------------------------------------
    def _request_update(self) -> None:
        self._lib.Viewer_RequestUpdate(self._h)

    def _pump(self, n: int) -> None:
        """Internal: force n render frames (no guards/recursion; callers guard)."""
        for _ in range(n):
            self._request_update()
            if not self._lib.Viewer_bPoll(self._h):
                break

    def poll(self) -> bool:
        """Process one batch of window events + render. Returns False once closed."""
        self._alive()
        alive = bool(self._lib.Viewer_bPoll(self._h))
        if self._cb_error:
            raise self._cb_error.pop(0)
        if self._pending_screenshot and not self._in_screenshot:
            path, self._pending_screenshot = self._pending_screenshot, None
            self._take_screenshot(path)
        return alive

    def run(self) -> None:
        """Open the window and pump events until it is closed (main thread)."""
        self._alive()
        self._request_update()
        try:
            while self.poll():
                pass
        finally:
            self.close()

    def screenshot(self, path: str, frames: int = 12) -> str:
        """Render and save a screenshot. ``.png``/``.jpg``/``.bmp`` are converted
        via Pillow; any other suffix keeps the native TGA. Returns the path."""
        self._alive()
        return self._take_screenshot(path, frames)

    def _take_screenshot(self, path: str, frames: int = 12) -> str:
        out = Path(path)
        self._in_screenshot = True            # so a reentrant poll() doesn't recurse
        try:
            self._pump(frames)                # ensure the scene is rendered first
            if out.suffix.lower() in _IMAGE_SUFFIXES:
                from PIL import Image  # picopie[viz]
                with tempfile.TemporaryDirectory() as d:
                    tga = Path(d) / "shot.tga"
                    self._lib.Viewer_RequestScreenShot(self._h, str(tga).encode())
                    self._pump(frames)        # flush the queued capture
                    Image.open(tga).convert("RGB").save(out)
            else:
                self._lib.Viewer_RequestScreenShot(self._h, str(out).encode())
                self._pump(frames)
        finally:
            self._in_screenshot = False
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
        far = max(dist * 10.0 + 1000.0, near + 1e-3)
        return _look_at(eye, self._target, _UP) @ _perspective(self._FOV_Y, aspect, near, far), eye

    def reset_view(self) -> Viewer:
        """Re-enable auto-framing of the scene at the default angle."""
        self._autofit = True
        self._zoom = 1.0
        self._azimuth, self._elevation = math.radians(45.0), math.radians(25.0)
        self._request_update()
        return self

    # pure camera-state mutations (no native calls -> unit-testable)
    def _apply_orbit(self, dx: float, dy: float) -> None:
        self._autofit = False
        self._azimuth -= dx * self._orbit_speed
        lim = math.pi / 2 - 1e-3
        self._elevation = max(-lim, min(lim, self._elevation + dy * self._orbit_speed))

    def _apply_pan(self, dx: float, dy: float) -> None:
        self._autofit = False
        _, right, up_cam = self._basis()
        scale = self._distance() * 0.0015
        delta = (-dx * right + dy * up_cam) * scale
        self._target = (self._target + delta).astype(np.float32)

    def _apply_zoom(self, amount: float) -> None:
        self._zoom = max(0.05, min(20.0, self._zoom * (1.0 - 0.1 * amount)))

    # --- callbacks (wrapped by _safe; never raise into native code) ----------
    def _on_info(self, msg, fatal):
        pass

    def _on_update(self, viewer, vp, bg, mvp, eye):
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

    def _on_key(self, viewer, key, scancode, action, mods):
        # GLFW key codes: 256 Esc, 81 Q, 70 F, 83 S; action 1 == press.
        if action != 1:
            return
        if key in (256, 81):                 # Esc / Q -> close
            self._lib.Viewer_RequestClose(self._h)
        elif key == 70:                      # F -> re-fit the view
            self.reset_view()
        elif key == 83:                      # S -> screenshot (deferred out of the callback)
            self._pending_screenshot = "picopie_screenshot.png"

    def _on_mouse_moved(self, viewer, pos, shift, ctrl, alt, sup):
        px, py = float(pos[0].X), float(pos[0].Y)
        dx, dy = px - self._mouse[0], py - self._mouse[1]
        self._mouse = (px, py)
        if self._drag_button is None:
            return
        if self._drag_button in (1, 2) or (self._drag_button == 0 and shift):
            self._apply_pan(dx, dy)
        else:
            self._apply_orbit(dx, dy)
        self._request_update()

    def _on_mouse_button(self, viewer, button, action, mods, pos):
        if action == 1:                      # press
            self._drag_button = int(button)
            self._mouse = (float(pos[0].X), float(pos[0].Y))
        else:                                # release
            self._drag_button = None

    def _on_scroll(self, viewer, offset, pos, shift, ctrl, alt, sup):
        self._apply_zoom(float(offset[0].Y))
        self._request_update()

    def _on_window_size(self, viewer, size):
        self._request_update()

    # --- lifetime ------------------------------------------------------------
    def close(self) -> None:
        """Destroy the native window. Idempotent; safe to call more than once."""
        if not self._closed:
            self._closed = True
            self._lib.Viewer_Destroy(self._h)
            self._h = None

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        # Best-effort: GLFW teardown must be on the creating thread, and a GC
        # finalizer can run anywhere -- so only destroy if we're on that thread.
        with contextlib.suppress(Exception):
            if not self._closed:
                if threading.get_ident() == self._thread:
                    self.close()
                else:
                    warnings.warn(
                        "Viewer garbage-collected without close(); use 'with "
                        "Viewer(...)' or call close() on the main thread",
                        ResourceWarning, stacklevel=1)


# A few distinct, readable colors for show()/render_png's default groups.
_PALETTE = [
    (0.35, 0.60, 0.95), (0.95, 0.55, 0.30), (0.45, 0.80, 0.45),
    (0.85, 0.45, 0.75), (0.90, 0.80, 0.35), (0.50, 0.75, 0.90),
]


def _populate(v: Viewer, objects, background) -> None:
    if background is not None:
        v.set_background(background)
    for i, obj in enumerate(objects):
        v.add(obj, group=i)
        v.set_group_material(i, _PALETTE[i % len(_PALETTE)])


def show(*objects, title: str = "PicoPie", size: tuple[int, int] = (1280, 800),
         background=None, block: bool = True) -> Viewer:
    """Open an interactive viewer for one or more objects (a one-liner).

    Each object goes in its own group with a color from a default palette::

        picogk.show(part)
        picogk.show(body, holes)          # two objects, two colors

    With ``block=True`` (default) this runs the window's event loop until it is
    closed (main thread); pass ``block=False`` to drive :meth:`Viewer.poll`
    yourself. Returns the :class:`Viewer`. Requires a display -- for headless
    image output use :func:`render_png` or :mod:`picogk.viz`.
    """
    v = Viewer(title=title, size=size)
    _populate(v, objects, background)
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
    with Viewer(size=size) as v:
        _populate(v, objs, background)
        return v.screenshot(str(path), frames=frames)
