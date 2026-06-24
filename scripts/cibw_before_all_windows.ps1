# cibuildwheel "before-all" for Windows: build OpenVDB deps with vcpkg (as the
# upstream runtime expects), then build + stage the native runtime DLL.
#
# NOTE: least-tested path. The runtime's CMake looks for vcpkg under
# Install_Dependencies/vcpkg (see its CMakeLists BOOTSTRAP_PATH). delvewheel may
# need --add-path to vendor the DLL deps of the dlopen'd picogk.dll.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$native = Join-Path $root "native\PicoGKRuntime"

# Pinned for reproducibility (see scripts/build_runtime.sh). Tag PicoGK-v2.2.0
# -> runtime "26.2.0"; pinning the superproject commit pins its submodules too.
$runtimeRef = "0f26321c18ed878a7820ef769c38fd5d49d39242"  # PicoGK-v2.2.0
$vcpkgRef   = "2026.06.01"

if (-not (Test-Path "$native\.git")) {
    git clone https://github.com/leap71/PicoGKRuntime $native
    git -C $native checkout --quiet $runtimeRef
    git -C $native submodule update --init --recursive --jobs 8
}

# vcpkg deps for OpenVDB. boost-interprocess (header-only) is needed by OpenVDB's
# delayed-loading io/Archive.cc (file_mapping); vcpkg installs boost per-component,
# so unlike a full Boost source tree it must be requested explicitly.
$vcpkg = Join-Path $native "Install_Dependencies\vcpkg"
if (-not (Test-Path $vcpkg)) {
    git clone --branch $vcpkgRef --depth 1 https://github.com/microsoft/vcpkg $vcpkg
    & "$vcpkg\bootstrap-vcpkg.bat"
}
& "$vcpkg\vcpkg.exe" install boost-iostreams:x64-windows boost-interprocess:x64-windows boost-system:x64-windows tbb:x64-windows blosc:x64-windows zlib:x64-windows

# Never-abort guard: wrap the C ABI so OpenVDB exceptions become a settable error
# instead of std::terminate (idempotent; see scripts/patch_runtime.py).
python "$root\scripts\patch_runtime.py" "$native\Source\PicoGKLibrary.cpp"

# Let CMake auto-detect the installed Visual Studio (the runner's VS version
# changes over time -- it is currently VS 18); don't hardcode the generator.
# NB: do NOT set CMAKE_CXX_FLAGS here -- it would clobber CMake's default MSVC
# flags (incl. /D_WINDOWS), making PicoGK.h take its non-Windows __attribute__
# branch. (--parallel below gives cross-project parallelism safely.)
cmake -S $native -B "$native\build" `
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
    -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_CORE_SHARED=OFF -DOPENVDB_CORE_STATIC=ON `
    -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF
cmake --build "$native\build" --config Release --parallel

python "$root\scripts\stage_runtime.py"
