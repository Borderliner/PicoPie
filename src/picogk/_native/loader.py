"""Locate and load the native PicoGK runtime shared library.

Resolution order:
  1. ``$PICOGK_RUNTIME`` env var (full path to the shared library), if set.
  2. A copy bundled inside the wheel at ``picogk/_lib/`` (populated at build time).
  3. Developer build trees under the repo's ``native/`` directory.
  4. The OS default search path (``picogk`` / ``libpicogk``).

The native CMake build sets an empty library prefix, so on Linux the file is
named ``picogk.so`` (not ``libpicogk.so``); we therefore search broadly.
"""

from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path

_LIB: C.CDLL | None = None


def _platform_names() -> list[str]:
    if sys.platform == "darwin":
        return ["picogk.dylib", "libpicogk.dylib", "picogk.26.2.dylib"]
    if sys.platform == "win32":
        return ["picogk.dll", "picogk.26.2.dll"]
    return ["picogk.so", "libpicogk.so", "libpicogk.so.26.2.0"]


def _candidate_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    dirs: list[Path] = []
    # 2. bundled in wheel
    dirs.append(here.parent.parent / "_lib")
    # 3. developer build trees (repo root is .../src/picogk/_native -> up 4)
    repo_root = here.parents[3]
    native = repo_root / "native" / "PicoGKRuntime"
    dirs += [native / "Dist", native / "build" / "lib", native / "build" / "Dist",
             native / "build" / "bin"]  # Windows VS generator output
    return dirs


def _glob_match(d: Path) -> Path | None:
    if not d.is_dir():
        return None
    for name in _platform_names():
        p = d / name
        if p.exists():
            return p
    # last resort: any picogk*.{so,dylib,dll} in the dir
    for pat in ("picogk*.so", "libpicogk*.so", "picogk*.dylib",
                "libpicogk*.dylib", "picogk*.dll"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def find_runtime() -> str:
    """Return the path to the native runtime, raising if it cannot be found."""
    env = os.environ.get("PICOGK_RUNTIME")
    if env:
        if not Path(env).exists():
            raise FileNotFoundError(
                f"PICOGK_RUNTIME points to a missing file: {env}")
        return env

    for d in _candidate_dirs():
        hit = _glob_match(d)
        if hit:
            return str(hit)

    # 4. let the OS resolver try
    for name in _platform_names():
        try:
            C.CDLL(name)
            return name
        except OSError:
            continue

    searched = "\n  ".join(str(d) for d in _candidate_dirs())
    raise FileNotFoundError(
        "Could not locate the PicoGK runtime shared library.\n"
        "Set $PICOGK_RUNTIME to its full path, or build it under native/.\n"
        f"Searched:\n  {searched}")


def load_runtime() -> C.CDLL:
    """Load (once) and return the native runtime as a ctypes CDLL."""
    global _LIB
    if _LIB is None:
        path = find_runtime()
        _LIB = C.CDLL(path)
        _LIB._picogk_path = path  # type: ignore[attr-defined]
    return _LIB
