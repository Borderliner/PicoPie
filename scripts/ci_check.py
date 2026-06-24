#!/usr/bin/env python3
"""Post-install wheel smoke check, run by cibuildwheel's test step.

Asserts the two things a from-wheel install must guarantee (and which a
source-tree run can hide): the native runtime is bundled *inside the package*,
and the compiled _fastloop extension built. Then exercises real geometry.
"""
import math
import sys

import picogk
from picogk import Voxels, _fast
from picogk._native.loader import find_runtime


def main() -> int:
    rt = find_runtime()
    if "_lib" not in rt.replace("\\", "/"):
        print(f"FAIL: runtime not bundled in the wheel (loaded {rt})")
        return 1
    if not _fast.available():
        print("FAIL: compiled _fastloop extension missing from the wheel")
        return 1

    picogk.init(voxel_size_mm=0.5)
    vol = Voxels.sphere(radius=10).volume_mm3()
    ideal = 4 / 3 * math.pi * 10**3
    if abs(vol - ideal) / ideal > 0.02:
        print(f"FAIL: sphere volume {vol} off from {ideal}")
        return 1

    print(f"OK  PicoGK {picogk.version()}  runtime={rt}  fastloop={_fast.available()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
