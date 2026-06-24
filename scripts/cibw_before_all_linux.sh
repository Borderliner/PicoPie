#!/usr/bin/env bash
# cibuildwheel "before-all" for Linux: runs ONCE inside the manylinux container
# (AlmaLinux 9 / manylinux_2_34) to install OpenVDB's deps and build + stage the
# native runtime.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dnf install -y epel-release || true
dnf install -y \
  cmake ninja-build gcc-c++ git make curl bzip2 which \
  tbb-devel zlib-devel bzip2-devel xz-devel \
  mesa-libGL-devel libX11-devel libXrandr-devel libXinerama-devel \
  libXcursor-devel libXi-devel wayland-devel libxkbcommon-devel \
  || { echo "dnf install failed -- adjust packages for this manylinux image"; exit 1; }

# OpenVDB 13 requires Boost >= 1.82, but AlmaLinux 9 only ships 1.75 -> build the
# needed components from source into /usr/local (iostreams pulls zlib/bz2/lzma).
if [[ ! -f /usr/local/lib/cmake/Boost-*/BoostConfig.cmake ]] 2>/dev/null; then
  BVER=1.86.0; BDIR=boost_1_86_0
  curl -fsSL -o /tmp/boost.tar.bz2 \
    "https://archives.boost.io/release/${BVER}/source/${BDIR}.tar.bz2"
  tar -xf /tmp/boost.tar.bz2 -C /tmp
  ( cd "/tmp/${BDIR}"
    ./bootstrap.sh --with-libraries=iostreams,system,regex --prefix=/usr/local
    ./b2 -j"$(nproc)" --with-iostreams --with-system --with-regex install )
  ldconfig
fi

# Blosc from source (matches the runtime's reference Dockerfile)
if ! ldconfig -p | grep -q libblosc; then
  git clone --branch v1.21.6 --depth 1 https://github.com/Blosc/c-blosc.git /tmp/blosc
  cmake -S /tmp/blosc -B /tmp/blosc/build -DCMAKE_INSTALL_PREFIX=/usr/local
  cmake --build /tmp/blosc/build --target install -j"$(nproc)"
  ldconfig
fi

bash "$HERE/build_runtime.sh"
