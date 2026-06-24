#!/usr/bin/env bash
# Reproducible build of the native PicoGK runtime (Linux), then stage it into
# the package so a wheel can bundle it.
#
# System deps (Debian/Ubuntu/Mint):
#   sudo apt-get install -y --no-install-recommends \
#     libboost-all-dev libblosc-dev libtbb-dev extra-cmake-modules \
#     xorg-dev mesa-common-dev libgl1-mesa-dev ninja-build cmake g++ git
#
# Usage:  scripts/build_runtime.sh [--clean]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE="$ROOT/native/PicoGKRuntime"
BUILD="$NATIVE/build"

# Pin the upstream runtime for reproducible builds. Tag PicoGK-v2.2.0 -> the
# runtime reports version "26.2.0" (kept in sync with _EXPECTED_RUNTIME_VERSION
# in src/picogk/library.py). Pinning the superproject commit also pins its
# submodules (OpenVDB/GLFW) via their recorded gitlinks.
PICOGK_RUNTIME_REF="${PICOGK_RUNTIME_REF:-0f26321c18ed878a7820ef769c38fd5d49d39242}"  # PicoGK-v2.2.0

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "$BUILD" "$NATIVE/Dist"
fi

if [[ ! -d "$NATIVE/.git" ]]; then
  echo ">> cloning PicoGKRuntime @ $PICOGK_RUNTIME_REF"
  git clone https://github.com/leap71/PicoGKRuntime "$NATIVE"
  git -C "$NATIVE" checkout --quiet "$PICOGK_RUNTIME_REF"
  git -C "$NATIVE" submodule update --init --recursive --jobs 8
fi

echo ">> configuring (Release, OpenVDB static core only)"
EXTRA_CMAKE=()
if [[ "$OSTYPE" == darwin* ]]; then
  # TBB's C++17 aligned new/delete requires a modern deployment target (>= 10.13);
  # honor cibuildwheel's MACOSX_DEPLOYMENT_TARGET, default 11.0 (also covers arm64).
  EXTRA_CMAKE+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.0}")
fi

cmake -S "$NATIVE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOPENVDB_BUILD_BINARIES=OFF \
  -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
  -DOPENVDB_BUILD_UNITTESTS=OFF \
  -DOPENVDB_BUILD_AX=OFF \
  -DOPENVDB_BUILD_NANOVDB=OFF \
  -DOPENVDB_CORE_SHARED=OFF -DOPENVDB_CORE_STATIC=ON \
  -DUSE_BLOSC=ON -DUSE_ZLIB=ON \
  -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF \
  "${EXTRA_CMAKE[@]}"

echo ">> building"
cmake --build "$BUILD" -j "$(nproc)"

echo ">> staging runtime into the package"
python3 "$ROOT/scripts/stage_runtime.py"

echo ">> done"
