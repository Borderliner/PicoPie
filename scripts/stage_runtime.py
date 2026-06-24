#!/usr/bin/env python3
"""Copy the built native runtime into ``src/picogk/_lib/`` so it gets bundled
into the wheel. Run after building PicoGKRuntime and before ``python -m build``.

Resolution: ``$PICOGK_RUNTIME`` if set, else the runtime build tree under
``native/PicoGKRuntime/``. The destination dir is wiped first so stale binaries
never linger in a wheel.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "picogk" / "_lib"

_NAMES = {
    "linux": ["picogk.so", "libpicogk.so"],
    "darwin": ["picogk.dylib", "libpicogk.dylib"],
    "win32": ["picogk.dll", "picogk.26.2.dll"],
}


def _platform_names() -> list[str]:
    if sys.platform.startswith("linux"):
        return _NAMES["linux"]
    if sys.platform == "darwin":
        return _NAMES["darwin"]
    if sys.platform == "win32":
        return _NAMES["win32"]
    raise RuntimeError(f"unsupported platform: {sys.platform}")


def _search_dirs() -> list[Path]:
    native = ROOT / "native" / "PicoGKRuntime"
    b = native / "build"
    return [
        native / "Dist",
        b / "lib", b / "Dist",
        b / "bin",                       # Windows: VS generator puts the .dll here
        b / "bin" / "Release", b / "lib" / "Release",  # VS multi-config subdirs
    ]


def find_runtime() -> Path:
    env = os.environ.get("PICOGK_RUNTIME")
    if env:
        p = Path(env)
        if not p.exists():
            sys.exit(f"PICOGK_RUNTIME points to a missing file: {env}")
        return p
    names = _platform_names()
    for d in _search_dirs():
        for n in names:
            p = d / n
            if p.exists():
                return p
        if d.is_dir():  # any picogk*.{so,dylib,dll}
            for pat in ("picogk*.so", "libpicogk*.so", "picogk*.dylib",
                        "libpicogk*.dylib", "picogk*.dll"):
                hits = sorted(d.glob(pat))
                if hits:
                    return hits[0]
    sys.exit("could not find a built runtime; set $PICOGK_RUNTIME or build it "
             "(scripts/build_runtime.sh)")


def _stage_windows_deps() -> None:
    """Bundle the runtime's dependency DLLs (tbb/blosc/boost/zlib/...) next to it.

    picogk.dll is dlopen'd, so Windows can't find its deps via the import graph
    the way auditwheel/delocate handle it on Linux/macOS. We copy the vcpkg bin
    DLLs into ``_lib`` and the loader adds that dir to the DLL search path.
    """
    vbin = (ROOT / "native" / "PicoGKRuntime" / "Install_Dependencies"
            / "vcpkg" / "installed" / "x64-windows" / "bin")
    if not vbin.is_dir():
        print(f"  (no vcpkg bin at {vbin}; skipping dep bundling)")
        return
    n = 0
    for dll in sorted(vbin.glob("*.dll")):
        target = DEST / dll.name
        if not target.exists():
            shutil.copy2(dll, target)
            n += 1
    print(f"  bundled {n} dependency DLL(s) from {vbin}")


def main() -> int:
    src = find_runtime()
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    dst = DEST / src.name
    shutil.copy2(src, dst)
    print(f"staged {src}  ->  {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
    if sys.platform == "win32":
        _stage_windows_deps()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
