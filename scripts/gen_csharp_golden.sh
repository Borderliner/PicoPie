#!/usr/bin/env bash
# Generate the C# PicoGK golden parity values (tests/golden/csharp_parity.json).
# Requires: dotnet, the C# PicoGK clone (native/PicoGK_csharp), and a built
# native runtime. Run after scripts/build_runtime.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

SO="$(find "$ROOT/native/PicoGKRuntime/Dist" "$ROOT/src/picopie/_lib" \
        -name 'picogk*.so' 2>/dev/null | head -1)"
[[ -n "$SO" ]] || { echo "no built runtime found; run scripts/build_runtime.sh"; exit 1; }

# Build quietly, then run the produced assembly so only JSON hits stdout.
dotnet build -c Release "$ROOT/parity" >/dev/null
DLL="$(find "$ROOT/parity/bin/Release" -name PicoGKParity.dll | head -1)"

mkdir -p "$ROOT/tests/golden"
VDB="$ROOT/tests/golden/csharp_sphere.vdb"
# The C# Library prints "Disposing Library" on teardown; keep only the JSON line.
dotnet "$DLL" "$SO" 0.5 "$VDB" | grep -m1 '^{' > "$ROOT/tests/golden/csharp_parity.json"
echo "wrote tests/golden/csharp_parity.json:"
cat "$ROOT/tests/golden/csharp_parity.json"
echo; echo "wrote $VDB ($(du -h "$VDB" 2>/dev/null | cut -f1))"
