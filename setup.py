"""Build hook for the optional compiled fast-loop extension.

Project metadata lives in pyproject.toml; this file only declares the Cython
extension. If Cython or a C compiler is unavailable, the build proceeds without
the extension and PicoPie falls back to its pure-Python transfer loops.
"""
import sys

from setuptools import Extension, setup

ext_modules = []
try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [Extension("picogk._fastloop", ["src/picogk/_fastloop.pyx"])],
        compiler_directives={"language_level": "3"},
    )
except Exception as exc:  # noqa: BLE001 - any failure -> pure-Python fallback
    print(f"[picopie] building without _fastloop extension ({exc}); "
          f"using pure-Python fallback", file=sys.stderr)

setup(ext_modules=ext_modules)
