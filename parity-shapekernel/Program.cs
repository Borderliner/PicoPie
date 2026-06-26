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

// --- BaseSphere (origin frame, radius 10), default 360/180 tessellation ---
var oFrame  = new LocalFrame(new Vector3(0, 0, 0));
var oSphere = new BaseSphere(oFrame, 10f);

Mesh sphMsh = oSphere.mshConstruct();
r["sphere_mesh_tris"] = sphMsh.nTriangleCount();

Voxels sphVox = oSphere.voxConstruct();
sphVox.CalculateProperties(out float sphVol, out BBox3 sphBox);
r["sphere_volume"] = sphVol;
r["sphere_bbox"] = Bb(sphBox);

Console.WriteLine(JsonSerializer.Serialize(r));
