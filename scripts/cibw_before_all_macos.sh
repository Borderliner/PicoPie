#!/usr/bin/env bash
# cibuildwheel "before-all" for macOS: install OpenVDB deps via Homebrew, then
# build + stage the native runtime. LEAP 71 also ships a prebuilt mac runtime;
# point $PICOGK_RUNTIME at it to skip the source build.
#
# NOTE: builds for the host arch. For cross-arch (x86_64 on Apple Silicon) you
# must build the runtime for the target arch too (set CMAKE_OSX_ARCHITECTURES).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reproducible Homebrew deps. We pin homebrew-core to a commit: each formula
# there carries its exact version + bottle SHA256s, so `brew install` resolves to
# fixed versions from fixed bottles (the "pin via bottle hashes" approach). This
# is the last rolling input in the build. Pinning is best-effort -- if the tap
# can't be checked out we fall back to the runner image's formulae (a loud
# warning) rather than break the build; the version assertion below guards drift
# either way. NO_AUTO_UPDATE stops mid-run formula churn regardless.
export HOMEBREW_NO_AUTO_UPDATE=1
HOMEBREW_CORE_REF="3cedcd7aaefa1ffda99a25a1b87e41f5e85a3975"  # 2026-06-25; boost 1.90, tbb 2023.0.0, c-blosc 1.21.6

if HOMEBREW_NO_INSTALL_FROM_API=1 brew tap homebrew/core >/dev/null 2>&1 \
   && git -C "$(brew --repo homebrew/core)" checkout -q "$HOMEBREW_CORE_REF" 2>/dev/null; then
  export HOMEBREW_NO_INSTALL_FROM_API=1   # make `brew install` use the pinned tap
  echo ">> homebrew-core pinned @ $HOMEBREW_CORE_REF"
else
  echo ">> WARNING: could not pin homebrew-core; using the runner image's formulae"
fi

brew install boost tbb c-blosc cmake ninja

echo ">> resolved dependency versions:"
brew list --versions boost tbb c-blosc
# OpenVDB 13 requires Boost >= 1.82; fail loudly if a drift slips an older one in.
bver="$(brew list --versions boost | awk '{print $2}')"
bkey="$(echo "$bver" | awk -F. '{printf "%d%02d", $1, $2}')"
if [ "${bkey:-0}" -lt 182 ]; then
  echo "ERROR: Boost $bver is too old for OpenVDB 13 (need >= 1.82)"; exit 1
fi

bash "$HERE/build_runtime.sh"
