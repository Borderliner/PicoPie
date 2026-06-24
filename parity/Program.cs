// Generates golden values from the reference C# PicoGK wrapper, run headlessly
// against our locally-built native runtime. Output: JSON on stdout.
//
//   args[0] = absolute path to picogk.so/.dylib/.dll  (the runtime to bind)
//   args[1] = (optional) voxel size in mm, default 0.5

using System.Numerics;
using System.Runtime.InteropServices;
using System.Text.Json;
using PicoGK;

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
var r = new Dictionary<string, object>();
r["voxel_size_mm"] = vs;

static float[] Bb(BBox3 b) => new[]
    { b.vecMin.X, b.vecMin.Y, b.vecMin.Z, b.vecMax.X, b.vecMax.Y, b.vecMax.Z };

// --- sphere: volume, voxel dims, mesh counts + bbox ---
var sph = Voxels.voxSphere(lib, new Vector3(0, 0, 0), 10f);
sph.CalculateProperties(out float sphVol, out _);
r["sphere_volume"] = sphVol;
sph.GetVoxelDimensions(out int ox, out int oy, out int oz,
                       out int sx, out int sy, out int sz);
r["sphere_voxeldims"] = new[] { ox, oy, oz, sx, sy, sz };
var msh = new Mesh(sph);
r["sphere_mesh_verts"] = msh.nVertexCount();
r["sphere_mesh_tris"] = msh.nTriangleCount();
r["sphere_mesh_bbox"] = Bb(msh.oBoundingBox());

// --- boolean subtract / intersect ---
var sphB = Voxels.voxSphere(lib, new Vector3(6, 0, 0), 6f);
(sph - sphB).CalculateProperties(out float subVol, out _);
r["bool_subtract_volume"] = subVol;

var sphC = Voxels.voxSphere(lib, new Vector3(6, 0, 0), 10f);
(sph & sphC).CalculateProperties(out float interVol, out _);
r["bool_intersect_volume"] = interVol;

// --- offset ---
sph.voxOffset(2f).CalculateProperties(out float offVol, out _);
r["offset_volume"] = offVol;

// --- capsule ---
var cap = Voxels.voxLatticeBeam(lib, new Vector3(-10, 0, 0), 3f,
                                     new Vector3(10, 0, 0), 3f);
cap.CalculateProperties(out float capVol, out _);
r["capsule_volume"] = capVol;

// --- lattice (beam + sphere nodes) -> voxels ---
var lat = new Lattice(lib);
lat.AddSphere(new Vector3(-8, 0, 0), 2f);
lat.AddSphere(new Vector3(8, 0, 0), 2f);
lat.AddBeam(new Vector3(-8, 0, 0), new Vector3(8, 0, 0), 1f, 1f, true);
new Voxels(lat).CalculateProperties(out float latVol, out _);
r["lattice_volume"] = latVol;

Console.WriteLine(JsonSerializer.Serialize(r));
