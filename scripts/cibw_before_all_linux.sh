#!/usr/bin/env bash
# cibuildwheel "before-all" for Linux: runs ONCE inside the manylinux container
# (AlmaLinux-based) to install OpenVDB's deps and build + stage the native runtime.
#
# NOTE: this is the least-tested path. manylinux package names/versions vary by
# image; Boost must be new enough for OpenVDB (>= 1.73). Use a recent manylinux
# (manylinux_2_34 = AlmaLinux 9) so dnf Boost is acceptable. Expect to iterate.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dnf install -y epel-release || true
dnf install -y \
  cmake ninja-build gcc-c++ git make \
  boost-devel tbb-devel zlib-devel bzip2-devel xz-devel \
  mesa-libGL-devel libX11-devel libXrandr-devel libXinerama-devel \
  libXcursor-devel libXi-devel wayland-devel libxkbcommon-devel \
  || { echo "dnf install failed -- adjust packages for this manylinux image"; exit 1; }

# Blosc from source (matches the runtime's reference Dockerfile)
if ! ldconfig -p | grep -q libblosc; then
  git clone --depth 1 https://github.com/Blosc/c-blosc.git /tmp/blosc
  cmake -S /tmp/blosc -B /tmp/blosc/build -DCMAKE_INSTALL_PREFIX=/usr/local
  cmake --build /tmp/blosc/build --target install -j"$(nproc)"
  ldconfig
fi

bash "$HERE/build_runtime.sh"
