# cibuildwheel "before-all" for Windows: build OpenVDB deps with vcpkg (as the
# upstream runtime expects), then build + stage the native runtime DLL.
#
# NOTE: least-tested path. The runtime's CMake looks for vcpkg under
# Install_Dependencies/vcpkg (see its CMakeLists BOOTSTRAP_PATH). delvewheel may
# need --add-path to vendor the DLL deps of the dlopen'd picogk.dll.
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..").Path
$native = Join-Path $root "native\PicoGKRuntime"

if (-not (Test-Path "$native\.git")) {
    git clone --recurse-submodules --jobs 8 https://github.com/leap71/PicoGKRuntime $native
}

# vcpkg deps for OpenVDB. boost-interprocess (header-only) is needed by OpenVDB's
# delayed-loading io/Archive.cc (file_mapping); vcpkg installs boost per-component,
# so unlike a full Boost source tree it must be requested explicitly.
$vcpkg = Join-Path $native "Install_Dependencies\vcpkg"
if (-not (Test-Path $vcpkg)) {
    git clone https://github.com/microsoft/vcpkg $vcpkg
    & "$vcpkg\bootstrap-vcpkg.bat"
}
& "$vcpkg\vcpkg.exe" install boost-iostreams:x64-windows boost-interprocess:x64-windows boost-system:x64-windows tbb:x64-windows blosc:x64-windows zlib:x64-windows

# Let CMake auto-detect the installed Visual Studio (the runner's VS version
# changes over time -- it is currently VS 18); don't hardcode the generator.
cmake -S $native -B "$native\build" `
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
    -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_CORE_SHARED=OFF -DOPENVDB_CORE_STATIC=ON `
    -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF
cmake --build "$native\build" --config Release

python "$root\scripts\stage_runtime.py"
