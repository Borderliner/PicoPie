#!/usr/bin/env bash
# cibuildwheel "before-all" for macOS: install OpenVDB deps via Homebrew, then
# build + stage the native runtime. LEAP 71 also ships a prebuilt mac runtime;
# point $PICOGK_RUNTIME at it to skip the source build.
#
# NOTE: builds for the host arch. For cross-arch (x86_64 on Apple Silicon) you
# must build the runtime for the target arch too (set CMAKE_OSX_ARCHITECTURES).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

brew install boost tbb c-blosc cmake ninja || true
bash "$HERE/build_runtime.sh"
