# Novice 1 — Set up a project and install PicoPie

This first tutorial gets you from nothing to a running PicoPie script using
[uv](https://docs.astral.sh/uv/), the fast Python project manager. (If you prefer
plain `pip`/`venv`, the install command is the same — just `pip install picopie`.)

## 1. Install uv (once)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Create a project

```bash
uv init my-parts          # creates my-parts/ with pyproject.toml + a venv on first use
cd my-parts
```

`uv init` scaffolds a project. uv manages a virtual environment for you — you don't
have to activate anything; `uv run` does it.

## 3. Add PicoPie from PyPI

```bash
uv add picopie            # core: modeling, meshing, file I/O
uv add "picopie[viz]"     # also the headless PNG helpers (Pillow + matplotlib)
```

That's it — PicoPie ships **self-contained wheels**: the native PicoGK runtime
(OpenVDB) and all its C++ dependencies are bundled. No .NET, no compiler, no system
libraries, no separate runtime to install. Wheels are prebuilt for CPython
3.10–3.13 on Linux (x86-64), macOS (Apple Silicon), and Windows (x64).

## 4. Your first script

Create `main.py`:

```python
import picogk
from picogk import Voxels

# Every session starts by choosing a voxel size (the modeling resolution, in mm).
# Smaller = finer detail but more memory/time. 0.2 mm is a good starting point.
picogk.init(voxel_size_mm=0.2)

print("PicoGK runtime:", picogk.version())     # e.g. "26.2.0"

ball = Voxels.sphere(radius=10)                # a 10 mm-radius sphere
vol, bbox = ball.calculate_properties()        # accurate volume + bounding box
print(f"volume = {vol:.1f} mm³")
print("bbox size =", bbox.size.round(2), "mm")
```

Run it:

```bash
uv run main.py
```

You should see something like:

```
PicoGK runtime: 26.2.0
volume = 4188.8 mm³
bbox size = [19.97 19.97 19.97] mm
```

## What just happened

- **`picogk.init(voxel_size_mm=...)`** starts a *session* — one native library
  instance with a fixed voxel size. Call it once before you create geometry. (Call
  `picogk.shutdown()` to end it, or use `with picogk.session(0.2): ...`.)
- **`Voxels`** is the core object: a signed-distance / level-set volume. Everything
  you model is a `Voxels`.
- **`calculate_properties()`** returns `(volume_mm³, bounding_box)` measured
  accurately (the same way C# PicoGK's `CalculateProperties` does).

Next: [making and combining shapes →](02-first-shapes.md)
