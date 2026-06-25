#!/usr/bin/env python3
"""Generate a CycloneDX SBOM (sbom.cdx.json) for PicoPie (Phase 11f).

The PicoPie wheel bundles a deep native dependency tree (the PicoGK runtime +
OpenVDB/GLFW/Boost/TBB/Blosc/zlib). This inventories those components -- with
versions, source refs, and licenses -- so the wheel's contents are auditable
(security scanning, license/compliance) without unpacking it.

Because every component is *pinned* (see build_runtime.sh / the cibw_* scripts /
homebrew-core pin), this is derived from those pins. **Update the NATIVE list
below when a pin changes.** Stdlib-only, so CI can run it with no install.

  python scripts/gen_sbom.py            # write sbom.cdx.json
  python scripts/gen_sbom.py --print    # write the timestamp to stderr, JSON to stdout
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

# Bundled native components -- pinned in build_runtime.sh / cibw_before_all_*.
# Versions can differ per platform (noted); the source pin is authoritative.
NATIVE = [
    {"name": "PicoGKRuntime", "version": "26.2.0", "license": "Apache-2.0",
     "purl": "pkg:github/leap71/PicoGKRuntime@PicoGK-v2.2.0",
     "ref": "https://github.com/leap71/PicoGKRuntime",
     "note": "commit 0f26321c18ed878a7820ef769c38fd5d49d39242"},
    {"name": "OpenVDB", "version": "13.0.0", "license": "MPL-2.0",
     "purl": "pkg:github/AcademySoftwareFoundation/openvdb@v13.0.0",
     "ref": "https://github.com/AcademySoftwareFoundation/openvdb"},
    {"name": "GLFW", "version": "3.4", "license": "Zlib",
     "purl": "pkg:github/glfw/glfw@3.4",
     "ref": "https://github.com/glfw/glfw", "note": "submodule commit b00e6a8a"},
    {"name": "Boost", "version": "1.86.0", "license": "BSL-1.0",
     "purl": "pkg:generic/boost@1.86.0",
     "ref": "https://www.boost.org",
     "note": "Linux: 1.86.0 (source); macOS: brew (homebrew-core pin); Windows: vcpkg"},
    {"name": "oneTBB", "version": "varies", "license": "Apache-2.0",
     "purl": "pkg:github/oneapi-src/oneTBB",
     "ref": "https://github.com/oneapi-src/oneTBB",
     "note": "Linux: distro tbb-devel; macOS: brew 2023.0.0; Windows: vcpkg 2026.06.01"},
    {"name": "c-blosc", "version": "1.21.6", "license": "BSD-3-Clause",
     "purl": "pkg:github/Blosc/c-blosc@v1.21.6",
     "ref": "https://github.com/Blosc/c-blosc",
     "note": "Linux/macOS 1.21.6; Windows: vcpkg 2026.06.01"},
    {"name": "zlib", "version": "varies", "license": "Zlib",
     "purl": "pkg:generic/zlib", "ref": "https://zlib.net",
     "note": "distro / brew / vcpkg provided"},
]

# SPDX-ish license id guesses for the Python deps (best-effort; informational).
_PY_LICENSE = {"numpy": "BSD-3-Clause", "setuptools": "MIT", "wheel": "MIT",
               "Cython": "Apache-2.0", "packaging": "Apache-2.0"}


def _component(name, version, license_id, purl=None, ref=None, note=None, ctype="library"):
    c: dict = {"type": ctype, "name": name, "version": version}
    if purl:
        c["purl"] = purl
    if license_id:
        c["licenses"] = [{"license": {"id": license_id}}]
    refs = []
    if ref:
        refs.append({"type": "website", "url": ref})
    if refs:
        c["externalReferences"] = refs
    if note:
        c["properties"] = [{"name": "picopie:note", "value": note}]
    return c


def build(timestamp: str) -> dict:
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text())
    proj = pp["project"]
    version = proj["version"]

    components = [_component(n["name"], n["version"], n["license"],
                            n.get("purl"), n.get("ref"), n.get("note")) for n in NATIVE]
    # Python runtime + build dependencies (constraint strings recorded as-is).
    for dep in proj.get("dependencies", []):
        nm = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
        components.append(_component(nm, dep, _PY_LICENSE.get(nm, ""),
                                     purl=f"pkg:pypi/{nm.lower()}"))
    for dep in pp["build-system"]["requires"]:
        nm = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
        components.append(_component(nm, dep, _PY_LICENSE.get(nm, ""),
                                     purl=f"pkg:pypi/{nm.lower()}", ctype="application"))

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [{"name": "picopie gen_sbom.py"}],
            "component": {
                "type": "library", "name": proj["name"], "version": version,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "purl": f"pkg:pypi/{proj['name']}@{version}",
            },
        },
        "components": components,
    }


def main(argv: list[str]) -> int:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sbom = build(ts)
    text = json.dumps(sbom, indent=2)
    if "--print" in argv:
        print(f"generated SBOM @ {ts} ({len(sbom['components'])} components)", file=sys.stderr)
        print(text)
    else:
        out = ROOT / "sbom.cdx.json"
        out.write_text(text + "\n")
        print(f"wrote {out} ({len(sbom['components'])} components)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
