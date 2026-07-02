"""Build hook for the optional compiled fast-loop extension.

Project metadata lives in pyproject.toml; this file only declares the Cython
extension. It prefers the ``.pyx`` (cythonized at build time) but falls back to
a pre-generated ``.c`` when present (e.g. an sdist built without Cython). If
neither a source nor a compiler is available, the build proceeds without the
extension and PicoPie uses its pure-Python transfer loops.
"""
import sys
from pathlib import Path

from setuptools import Extension, setup

NAME = "picopie._fastloop"
PYX = "src/picopie/_fastloop.pyx"
CSRC = "src/picopie/_fastloop.c"

ext_modules = []
try:
    if Path(PYX).exists():
        from Cython.Build import cythonize

        ext_modules = cythonize(
            [Extension(NAME, [PYX])],
            compiler_directives={"language_level": "3"},
        )
    elif Path(CSRC).exists():
        # sdist shipped the generated C source; build it without needing Cython.
        ext_modules = [Extension(NAME, [CSRC])]
    else:
        print("[picopie] no _fastloop source found; using pure-Python fallback",
              file=sys.stderr)
except Exception as exc:  # noqa: BLE001 - any failure -> pure-Python fallback
    print(f"[picopie] building without _fastloop extension ({exc}); "
          f"using pure-Python fallback", file=sys.stderr)

setup(ext_modules=ext_modules)
