// Generates golden values from the reference C# ShapeKernel (pinned), run
// headlessly against our locally-built native runtime. Output: JSON on stdout.
//
//   args[0] = absolute path to picogk.so/.dylib/.dll  (the runtime to bind)
//   args[1] = (optional) voxel size in mm, default 0.5

using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using PicoGK;
using Leap71.ShapeKernel;

string soPath = args.Length > 0
    ? args[0]
    : throw new ArgumentException("pass the path to the native runtime as arg 0");
float vs = args.Length > 1 ? float.Parse(args[1]) : 0.5f;

// Route PicoGK's DllImport("picogk.26.2") to our built runtime.
NativeLibrary.SetDllImportResolver(typeof(Library).Assembly,
    (name, asm, path) => name.Contains("picogk")
        ? NativeLibrary.Load(soPath)
        : IntPtr.Zero);

using var lib = new Library(vs);          // headless: constructor, no viewer
// ShapeKernel uses global-object constructors (new Mesh(), new Voxels(mesh)),
// which resolve the library via Library.oLibrary(). Register ours so they work
// without Library.Go() (which would spawn a viewer/task thread).
Library.RegisterGlobalLibrary(lib);
var r = new Dictionary<string, object>();
r["voxel_size_mm"] = vs;

static float[] Bb(BBox3 b) => new[]
    { b.vecMin.X, b.vecMin.Y, b.vecMin.Z, b.vecMax.X, b.vecMax.Y, b.vecMax.Z };
static float[] V3(Vector3 v) => new[] { v.X, v.Y, v.Z };

// --- BaseSphere (origin frame, radius 10), default 360/180 tessellation ---
var oFrame  = new LocalFrame(new Vector3(0, 0, 0));
var oSphere = new BaseSphere(oFrame, 10f);

Mesh sphMsh = oSphere.mshConstruct();
r["sphere_mesh_tris"] = sphMsh.nTriangleCount();

Voxels sphVox = oSphere.voxConstruct();
sphVox.CalculateProperties(out float sphVol, out BBox3 sphBox);
r["sphere_volume"] = sphVol;
r["sphere_bbox"] = Bb(sphBox);

// --- VecOperations (Phase 12b math core) ---
r["vec_rotate_axis"] = V3(VecOperations.vecRotateAroundAxis(
    new Vector3(1, 0, 0), MathF.PI / 3f, new Vector3(0, 0, 1)));
r["vec_orthogonal_dir"] = V3(VecOperations.vecGetOrthogonalDir(new Vector3(0.3f, 0.4f, 0.5f)));
r["vec_angle_between"] = VecOperations.fGetAngleBetween(
    new Vector3(1, 0, 0), new Vector3(1, 1, 0));
r["vec_signed_angle"] = VecOperations.fGetSignedAngleBetween(
    new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1));
r["vec_cyl_interp"] = V3(VecOperations.vecCylindricalInterpolation(
    new Vector3(5, 0, 0), new Vector3(0, 5, 2), 0.3f));
r["vec_sph_interp"] = V3(VecOperations.vecSphericalInterpolation(
    new Vector3(5, 0, 0), new Vector3(0, 5, 2), 0.3f));
r["vec_cyl_point"] = V3(VecOperations.vecGetCylPoint(5f, MathF.PI / 4f, 2f));
r["vec_sph_point"] = V3(VecOperations.vecGetSphPoint(5f, MathF.PI / 4f, MathF.PI / 6f));

// --- LocalFrame axes for a tilted local Z (uses vecGetOrthogonalDir) ---
var oTiltFrame = new LocalFrame(new Vector3(0, 0, 0), new Vector3(0.3f, 0.4f, 0.5f));
r["frame_local_x"] = V3(oTiltFrame.vecGetLocalX());
r["frame_local_y"] = V3(oTiltFrame.vecGetLocalY());
r["frame_local_z"] = V3(oTiltFrame.vecGetLocalZ());

// --- LineModulation discrete-point interpolation (values=Y over axis=X) ---
var aLinePts = new List<Vector3>() {
    new Vector3(0f, 0f, 0f), new Vector3(0.5f, 2f, 0f), new Vector3(1f, 1f, 0f) };
var oLine = new LineModulation(aLinePts, LineModulation.ECoord.Y, LineModulation.ECoord.X);
r["line_mod_samples"] = new[] { 0f, 0.25f, 0.5f, 0.75f, 1f }
    .Select(t => oLine.fGetModulation(t)).ToArray();

// --- Bisection: solve x^2 = 2 on [0, 2] ---
var oBis = new Bisection((x) => x * x, 0f, 2f, 2f, 1e-4f);
r["bisection_sqrt2"] = oBis.fFindOptimalInput();

Console.WriteLine(JsonSerializer.Serialize(r));
