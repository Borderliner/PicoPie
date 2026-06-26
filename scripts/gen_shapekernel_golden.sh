#!/usr/bin/env bash
# Generate the C# ShapeKernel golden parity values
# (tests/golden/shapekernel_parity.json). Requires: dotnet, the C# PicoGK clone
# (native/PicoGK_csharp), the pinned ShapeKernel clone
# (native/LEAP71_ShapeKernel @ tag ShapeKernel-v2.1.0), and a built native
# runtime. Run after scripts/build_runtime.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

[[ -d "$ROOT/native/LEAP71_ShapeKernel/ShapeKernel" ]] || {
  echo "ShapeKernel clone missing; run:"
  echo "  git clone --branch ShapeKernel-v2.1.0 --depth 1 \\"
  echo "    https://github.com/leap71/LEAP71_ShapeKernel.git native/LEAP71_ShapeKernel"
  exit 1
}

SO="$(find "$ROOT/native/PicoGKRuntime/Dist" "$ROOT/src/picogk/_lib" \
        -name 'picogk*.so' 2>/dev/null | head -1)"
[[ -n "$SO" ]] || { echo "no built runtime found; run scripts/build_runtime.sh"; exit 1; }

# Build quietly, then run the produced assembly so only JSON hits stdout.
dotnet build -c Release "$ROOT/parity-shapekernel" >/dev/null
DLL="$(find "$ROOT/parity-shapekernel/bin/Release" -name PicoGKShapeKernelParity.dll | head -1)"

mkdir -p "$ROOT/tests/golden"
# The C# Library prints log lines on teardown; keep only the JSON line.
dotnet "$DLL" "$SO" 0.5 | grep -m1 '^{' > "$ROOT/tests/golden/shapekernel_parity.json"
echo "wrote tests/golden/shapekernel_parity.json:"
cat "$ROOT/tests/golden/shapekernel_parity.json"
echo
