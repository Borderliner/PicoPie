#!/usr/bin/env python3
"""Patch the (pinned) PicoGKRuntime at build time. Three independent fixes:

1. **Never-abort guard** (``PicoGKLibrary.cpp``): uncaught C++/OpenVDB exceptions
   become a *settable error* at the C ABI instead of ``std::terminate``-ing the
   whole process. Handles cross the ABI as ``extern "C"`` functions; a C++
   exception escaping one of them aborts the process (uncatchable from Python).
   We wrap every ``PICOGK_API`` body in try/catch that records the message and
   returns a type-appropriate sentinel. The Python side
   (src/picopie/_native/runtime.py) reads the flag after each call and raises
   ``PicoPieError`` -- so *any* native error becomes an ordinary catchable one.

2. **CSG narrow-band fix** (``PicoGKVdbVoxels.h``): upstream's
   ``Voxels::IntersectImplicit`` builds its temp grid with
   ``Voxels oVox(oVoxelSize(), fBackgroundMM())`` -- but that ctor's 2nd arg is an
   ``int nNarrowBand`` while ``fBackgroundMM()`` returns the background in *mm*
   (``narrowBand * voxel_size``, a float). For voxel sizes below ~0.33 mm it
   truncates to ``int 0`` -> a grid with background 0 -> OpenVDB's
   ``csgIntersection`` aborts ("expected grid A outside value > 0, got 0"). We
   pass the source's narrow band instead, so it works at any voxel size.

3. **ProjectZSlice narrow-band fix** (``PicoGKVdbVoxels.h``): the end-cap seal
   loops in ``ProjectZSliceDn`` / ``ProjectZSliceUp`` iterate
   ``(int)(0.5f + background_mm)`` slices -- but the background is in *mm*
   (``narrowBand * voxel_size``), not a voxel count. Below ~0.167 mm it truncates
   to 0 so the cap is never sealed (a non-watertight result, silently produced);
   above ~1.5 mm it over-iterates. We use the narrow band directly.

All are applied at build time to the cloned runtime (build_runtime.sh and the
Windows .ps1) by invoking this script on ``PicoGKLibrary.cpp``; the sibling
header is patched automatically. The runtime stays vendored-unmodified in git.
Idempotent. Should be upstreamed to leap71/PicoGKRuntime so we eventually don't
carry these.

Usage: python scripts/patch_runtime.py [path/to/PicoGKLibrary.cpp]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "PICOGK_NEVER_ABORT_GUARD"

INFRA = f"""
// ---- {MARKER} (injected by PicoPie scripts/patch_runtime.py) --------------
#include <string>
#include <cstring>
#include <cmath>
#include <exception>
static std::string g_pkLastError;
extern "C" {{
#if defined(_WIN32)
__declspec(dllexport) int g_pkLastErrorFlag = 0;
#else
__attribute__((visibility("default"))) int g_pkLastErrorFlag = 0;
#endif
}}
PICOGK_API void PicoGK_SetError(const char* psz) {{
    g_pkLastError = (psz != nullptr) ? psz : "PicoGK: unknown native error";
    g_pkLastErrorFlag = 1;
}}
PICOGK_API int PicoGK_nGetLastError(char* psz, int nMax) {{
    int n = (int) g_pkLastError.size();
    if (psz != nullptr && nMax > 0) {{
        int c = (n < nMax - 1) ? n : nMax - 1;
        std::memcpy(psz, g_pkLastError.data(), (size_t) c);
        psz[c] = 0;
    }}
    return n;
}}
#define PICOGK_GUARD_TRY g_pkLastErrorFlag = 0; try {{
#define PICOGK_GUARD_CATCH(s) }} \\
    catch (const std::exception& e) {{ PicoGK_SetError(e.what()); return s; }} \\
    catch (...) {{ PicoGK_SetError("PicoGK: unknown native error"); return s; }}
#define PICOGK_GUARD_CATCH_VOID }} \\
    catch (const std::exception& e) {{ PicoGK_SetError(e.what()); return; }} \\
    catch (...) {{ PicoGK_SetError("PicoGK: unknown native error"); return; }}
// ---------------------------------------------------------------------------
"""

# A PICOGK_API *function* definition: return type, name, then '('. (The exported
# variable `g_pkLastErrorFlag` has no '(' so it never matches.)
SIG_RE = re.compile(r"^PICOGK_API\s+([A-Za-z0-9_:<>\*\s]+?)\s+([A-Za-z_]\w*)\s*\(", re.M)

# our own injected API helpers must not be wrapped
_SKIP = {"PicoGK_SetError", "PicoGK_nGetLastError"}


def _sentinel(ret: str) -> str | None:
    ret = ret.strip()
    if ret == "void":
        return None                     # -> CATCH_VOID (bare `return;`)
    if ret == "bool":
        return "false"
    if ret in ("float", "double"):
        return "NAN"
    return "0"                          # int*/int64/handles/pointers


def _body_braces(text: str, sig_end: int) -> tuple[int, int]:
    """(index of body '{', index of its matching '}') starting from sig_end."""
    open_i = text.index("{", sig_end)
    depth = 0
    i = open_i
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return open_i, i
        i += 1
    raise RuntimeError("unbalanced braces")


def patch(src: str) -> tuple[str, int]:
    if MARKER in src:
        return src, -1                  # already patched
    first = SIG_RE.search(src)
    if not first:
        raise SystemExit("no PICOGK_API functions found -- wrong file?")
    inject_at = src.rfind("\n", 0, first.start()) + 1
    src = src[:inject_at] + INFRA + "\n" + src[inject_at:]

    # Wrap from the end so insertions don't shift earlier match offsets.
    wrapped = 0
    for m in reversed(list(SIG_RE.finditer(src))):
        if m.group(2) in _SKIP:
            continue
        open_i, close_i = _body_braces(src, m.end())
        sent = _sentinel(m.group(1))
        catch = "PICOGK_GUARD_CATCH_VOID" if sent is None else f"PICOGK_GUARD_CATCH({sent})"
        src = src[:close_i] + f"\n    {catch}\n" + src[close_i:]
        src = src[:open_i + 1] + "\n    PICOGK_GUARD_TRY" + src[open_i + 1:]
        wrapped += 1
    return src, wrapped


# Fix 2: the IntersectImplicit narrow-band truncation bug (see module docstring).
_CSG_BUG = "Voxels oVox(oVoxelSize(), fBackgroundMM());"
_CSG_FIX = "Voxels oVox(oVoxelSize(), m_nSdfNarrowBand);"


def patch_csg_narrowband(path: Path) -> int:
    """Patch ``PicoGKVdbVoxels.h`` in place. Idempotent. Returns 1 if patched, 0
    if already fixed. Raises if the expected upstream line is gone (a pin bump
    that touched it must be re-checked rather than silently shipping the bug)."""
    text = path.read_text(encoding="utf-8")
    if _CSG_FIX in text:
        return 0                        # already patched
    if _CSG_BUG not in text:
        raise SystemExit(
            f"{path}: neither the buggy nor the fixed IntersectImplicit line found "
            "-- upstream changed; re-check the CSG narrow-band fix.")
    path.write_text(text.replace(_CSG_BUG, _CSG_FIX), encoding="utf-8")
    return 1


# Fix 3: the ProjectZSliceDn/Up end-cap seal loops (see module docstring). The
# substring appears once in each of the two functions.
_PROJ_BUG = "(int) (0.5f + m_roGrid->background())"
_PROJ_FIX = "m_nSdfNarrowBand"
_PROJ_MARK = "// Close the last slice, and update the background"


def patch_projectz_narrowband(path: Path) -> int:
    """Patch ``ProjectZSliceDn`` / ``ProjectZSliceUp`` in PicoGKVdbVoxels.h.
    Idempotent. Returns the number of occurrences fixed (2 on a fresh clone, 0 if
    already patched). Raises if the seal loops are gone (a pin bump touched them
    -> re-check rather than silently shipping the bug)."""
    text = path.read_text(encoding="utf-8")
    n = text.count(_PROJ_BUG)
    if n == 0:
        if _PROJ_MARK in text:
            return 0                    # loops present, bug pattern gone -> already patched
        raise SystemExit(
            f"{path}: ProjectZSlice end-cap seal loops not found -- upstream "
            "changed; re-check the ProjectZSlice narrow-band fix.")
    path.write_text(text.replace(_PROJ_BUG, _PROJ_FIX), encoding="utf-8")
    return n


def main(argv: list[str]) -> int:
    path = (Path(argv[1]) if len(argv) > 1
            else Path("native/PicoGKRuntime/Source/PicoGKLibrary.cpp"))
    # The runtime source is UTF-8 (curly quotes in its license header); be explicit
    # so this doesn't fall back to a locale codec (cp1252) on Windows.
    out, n = patch(path.read_text(encoding="utf-8"))
    if n < 0:
        print(f"already patched: {path}")
    else:
        path.write_text(out, encoding="utf-8")
        print(f"patched {path}: wrapped {n} PICOGK_API functions")

    # Also fix the CSG narrow-band bug in the sibling header (all build scripts
    # invoke this on PicoGKLibrary.cpp, so this reaches every platform for free).
    hdr = path.parent / "PicoGKVdbVoxels.h"
    if hdr.exists():
        m = patch_csg_narrowband(hdr)
        print(f"{'patched' if m else 'already patched'}: {hdr} (CSG narrow-band)")
        p = patch_projectz_narrowband(hdr)
        tag = f"patched {p} occ" if p else "already patched"
        print(f"{tag}: {hdr} (ProjectZSlice narrow-band)")
    else:
        print(f"skipped CSG / ProjectZSlice fixes: {hdr} not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
