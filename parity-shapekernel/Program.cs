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

// --- Splines (Phase 12c) ---
var oCps = new ControlPointSpline(new List<Vector3>() {
    new Vector3(0, 0, 0), new Vector3(5, 10, 0),
    new Vector3(10, 0, 0), new Vector3(15, 10, 0) });
r["cps_samples"] = new[] { 0f, 0.25f, 0.5f, 0.75f, 1f }
    .Select(t => V3(oCps.vecGetPointAt(t))).ToArray();

var oCpsClosed = new ControlPointSpline(new List<Vector3>() {
    new Vector3(0, 0, 0), new Vector3(10, 0, 0),
    new Vector3(10, 10, 0), new Vector3(0, 10, 0) },
    2, ControlPointSpline.EEnds.CLOSED);
r["cps_closed_samples"] = new[] { 0f, 0.25f, 0.5f, 0.75f, 1f }
    .Select(t => V3(oCpsClosed.vecGetPointAt(t))).ToArray();

var aGrid = new List<List<Vector3>>();
for (int u = 0; u < 3; u++)
{
    var row = new List<Vector3>();
    for (int v = 0; v < 3; v++)
        row.Add(new Vector3(u * 5f, v * 5f, u + v));
    aGrid.Add(row);
}
r["surf_point"] = V3(new ControlPointSurface(aGrid).vecGetPointAt(0.3f, 0.7f));

var oTcs = new TangentialControlSpline(
    new Vector3(0, 0, 0), new Vector3(10, 0, 0),
    new Vector3(0, 1, 0), new Vector3(0, -1, 0));
r["tcs_samples"] = oTcs.aGetPoints(5).Select(p => V3(p)).ToArray();

var aRaw = new List<Vector3>() {
    new Vector3(0, 0, 0), new Vector3(10, 0, 0), new Vector3(10, 10, 0) };
var aRep = SplineOperations.aGetReparametrizedSpline(aRaw, (uint)8);
r["reparam_len"] = aRep.Count;
r["reparam_pts"] = aRep.Select(p => V3(p)).ToArray();

// --- Frames: straight extrude (exact; const axes) ---
var oFrames = new Frames(20f, new LocalFrame(new Vector3(0, 0, 0)), 1f);
r["frames_spine_05"] = V3(oFrames.vecGetSpineAlongLength(0.5f));
r["frames_z_05"] = V3(oFrames.vecGetLocalZAlongLength(0.5f));
r["frames_x_05"] = V3(oFrames.vecGetLocalXAlongLength(0.5f));

// --- frame-based shapes (Phase 12d): volume + surface bbox ---
var oOriginFrame = new LocalFrame(new Vector3(0, 0, 0));

new BaseBox(oOriginFrame, 20f, 10f, 8f).voxConstruct()
    .CalculateProperties(out float boxVol, out BBox3 boxBox);
r["box_volume"] = boxVol; r["box_bbox"] = Bb(boxBox);

new BaseCylinder(oOriginFrame, 20f, 10f).voxConstruct()
    .CalculateProperties(out float cylVol, out BBox3 cylBox);
r["cyl_volume"] = cylVol; r["cyl_bbox"] = Bb(cylBox);

new BaseCone(oOriginFrame, 20f, 10f, 0f).voxConstruct()
    .CalculateProperties(out float coneVol, out BBox3 coneBox);
r["cone_volume"] = coneVol; r["cone_bbox"] = Bb(coneBox);

new BaseRing(oOriginFrame, 30f, 5f).voxConstruct()
    .CalculateProperties(out float ringVol, out BBox3 ringBox);
r["ring_volume"] = ringVol; r["ring_bbox"] = Bb(ringBox);

new BaseLens(oOriginFrame, 4f, 0f, 10f).voxConstruct()
    .CalculateProperties(out float lensVol, out BBox3 lensBox);
r["lens_volume"] = lensVol; r["lens_bbox"] = Bb(lensBox);

// --- spined / revolve shapes (Phase 12e) ---
new BasePipe(oOriginFrame, 20f, 5f, 10f).voxConstruct()
    .CalculateProperties(out float pipeVol, out BBox3 pipeBox);
r["pipe_volume"] = pipeVol; r["pipe_bbox"] = Bb(pipeBox);

new BasePipeSegment(oOriginFrame, 20f, 5f, 10f,
    new LineModulation(0f), new LineModulation(MathF.PI / 2f),
    BasePipeSegment.EMethod.START_END).voxConstruct()
    .CalculateProperties(out float segVol, out BBox3 segBox);
r["seg_volume"] = segVol; r["seg_bbox"] = Bb(segBox);

var oRevFrames = new Frames(20f, new LocalFrame(new Vector3(0, 0, 0)));
new BaseRevolve(new LocalFrame(new Vector3(0, 0, 0)), oRevFrames, 0f, 5f).voxConstruct()
    .CalculateProperties(out float revVol, out BBox3 revBox);
r["revolve_volume"] = revVol; r["revolve_bbox"] = Bb(revBox);

// --- lattice shapes (Phase 12f) ---
new LatticePipe(oOriginFrame, 20f, 5f).voxConstruct()
    .CalculateProperties(out float latPipeVol, out BBox3 latPipeBox);
r["latpipe_volume"] = latPipeVol; r["latpipe_bbox"] = Bb(latPipeBox);

new LatticeManifold(oOriginFrame, 20f, 5f, 45f).voxConstruct()
    .CalculateProperties(out float latManVol, out BBox3 latManBox);
r["latman_volume"] = latManVol; r["latman_bbox"] = Bb(latManBox);

// --- implicit signed-distance values ---
var aImpPts = new[] { new Vector3(0, 0, 0), new Vector3(3, 1, 2), new Vector3(5, 5, 5) };
var gyr = new ImplicitGyroid(5f, 0.3f);
r["imp_gyroid"] = aImpPts.Select(p => gyr.fSignedDistance(p)).ToArray();
var impSphere = new ImplicitSphere(new Vector3(0, 0, 0), 8f);
r["imp_sphere"] = aImpPts.Select(p => impSphere.fSignedDistance(p)).ToArray();
var gen = new ImplicitGenus(0.5f);
r["imp_genus"] = aImpPts.Select(p => gen.fSignedDistance(p)).ToArray();
var se = new ImplicitSuperEllipsoid(new Vector3(0, 0, 0), 5f, 5f, 5f, 1f, 1f);
r["imp_superellipsoid"] = aImpPts.Select(p => se.fSignedDistance(p)).ToArray();

// --- Measure: surface area (deterministic, same mesh as Python) ---
r["sphere_surface_area"] = Measure.fGetSurfaceArea(
    Voxels.voxSphere(lib, new Vector3(0, 0, 0), 10f));

// --- Frames alignment modes (Phase 12j) ---
var aSpinePts = new List<Vector3>() {
    new Vector3(0, 0, 0), new Vector3(10, 0, 0),
    new Vector3(10, 10, 0), new Vector3(20, 10, 0) };
var oFramesCyl = new Frames(aSpinePts, Frames.EFrameType.CYLINDRICAL);
r["frames_cyl_spine05"] = V3(oFramesCyl.vecGetSpineAlongLength(0.5f));
r["frames_cyl_z05"] = V3(oFramesCyl.vecGetLocalZAlongLength(0.5f));
r["frames_cyl_x05"] = V3(oFramesCyl.vecGetLocalXAlongLength(0.5f));
var oFramesMin = new Frames(aSpinePts, Frames.EFrameType.MIN_ROTATION);
r["frames_min_x05"] = V3(oFramesMin.vecGetLocalXAlongLength(0.5f));
var oFramesTx = new Frames(aSpinePts, new Vector3(0, 0, 1));   // targetX = +Z
r["frames_tx_spine05"] = V3(oFramesTx.vecGetSpineAlongLength(0.5f));
r["frames_tx_x05"] = V3(oFramesTx.vecGetLocalXAlongLength(0.5f));

// --- PipeSegment (mid_range method) ---
new BasePipeSegment(oOriginFrame, 20f, 5f, 10f,
    new LineModulation(0.5f * MathF.PI), new LineModulation(0.5f * MathF.PI),
    BasePipeSegment.EMethod.MID_RANGE).voxConstruct()
    .CalculateProperties(out float segMrVol, out BBox3 segMrBox);
r["seg_midrange_volume"] = segMrVol; r["seg_midrange_bbox"] = Bb(segMrBox);

// --- CylindricalControlSpline ---
var oCcs = new CylindricalControlSpline(new Vector3(5, 0, 0));
oCcs.AddRelativeStep(CylindricalControlSpline.EDirection.Z, 10f);
oCcs.AddRelativeStep(CylindricalControlSpline.EDirection.RADIAL, 5f);
r["ccs_samples"] = oCcs.aGetPoints(5).Select(p => V3(p)).ToArray();

// --- transform= on a box (vectorised vs C# point-wise) ---
var oTfBox = new BaseBox(oOriginFrame, 20f, 10f, 8f);
oTfBox.SetTransformation((p) => p + new Vector3(10, 0, 0));
oTfBox.voxConstruct().CalculateProperties(out float tfVol, out BBox3 tfBox);
r["box_xform_volume"] = tfVol; r["box_xform_bbox"] = Bb(tfBox);

// --- modulated-radius cylinder (the 500-step tessellation path) ---
var oModCyl = new BaseCylinder(oOriginFrame, 20f, 12f);
oModCyl.SetRadius(new SurfaceModulation((float phi, float lr) => 12f + 3f * MathF.Cos(5f * phi)));
oModCyl.voxConstruct().CalculateProperties(out float modCylVol, out _);
r["mod_cyl_volume"] = modCylVol;

// --- supershape / polygon radii ---
float[] aPhis = { 0f, 0.5f, 1.2f, 3.0f };
r["supershape_custom"] = aPhis.Select(ph => Uf.fGetSuperShapeRadius(ph, 6f, 2f, 1.2f, 1.2f)).ToArray();
r["supershape_hex"] = aPhis.Select(ph => Uf.fGetSuperShapeRadius(ph, Uf.ESuperShape.HEX)).ToArray();
r["polygon_custom"] = aPhis.Select(ph => Uf.fGetPolygonRadius(ph, 6u)).ToArray();
r["polygon_tri"] = aPhis.Select(ph => Uf.fGetPolygonRadius(ph, Uf.EPolygon.TRI)).ToArray();

Console.WriteLine(JsonSerializer.Serialize(r));
