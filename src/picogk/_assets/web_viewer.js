// PicoPie web viewer front-end (three.js).
//
// Geometry is computed in Python and handed to `createViewer` as two lists:
//   - geometry: heavy, per-id {id, kind, positions, indices} buffers (changes
//     only when objects are added/removed)
//   - style:    light, per-id {id, color, metallic, roughness, visible, matrix}
//     (changes on every recolor/transform/visibility toggle)
// Splitting them means a style tweak does a cheap in-place update — no buffer
// resend, no scene rebuild, and no camera re-fit.
//
// The same `createViewer` powers both the anywidget (notebook) path via the
// default `render` export and the self-contained HTML export
// (picogk.web.export_html).
//
// three.js is loaded from the esm.sh CDN (needs internet the first time; the
// browser caches it afterwards).

import * as THREE from "https://esm.sh/three@0.160.0";
import { OrbitControls } from "https://esm.sh/three@0.160.0/addons/controls/OrbitControls.js";

// A synced bytes field arrives as a DataView; the HTML export passes typed arrays.
function asFloat32(b) {
  if (b instanceof Float32Array) return b;
  const buf = b.buffer ? b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
                       : b.slice(0);
  return new Float32Array(buf);
}
function asUint32(b) {
  if (b instanceof Uint32Array) return b;
  const buf = b.buffer ? b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
                       : b.slice(0);
  return new Uint32Array(buf);
}

function applyMatrixTo(obj, m16) {
  obj.matrixAutoUpdate = false;          // we drive the local matrix directly
  if (m16 && m16.length === 16) obj.matrix.set(...m16);   // Matrix4.set is row-major
  else obj.matrix.identity();
  obj.matrixWorldNeedsUpdate = true;
}

export function createViewer(el, opts = {}) {
  let width = opts.width || 720;
  let height = opts.height || 480;

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(width, height);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.borderRadius = "6px";
  renderer.domElement.style.outline = "none";
  renderer.domElement.tabIndex = 0;
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1e6);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 0.35));
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
  keyLight.position.set(1, 1.4, 1.2);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
  fillLight.position.set(-1, -0.6, -0.8);
  scene.add(fillLight);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const content = new THREE.Group();
  scene.add(content);
  const meshes = new Map();              // id -> Object3D
  let wireframe = false;
  let autofit = true;                    // first load auto-frames; user interaction stops it
  controls.addEventListener("start", () => { autofit = false; });

  function disposeContent() {
    for (const obj of meshes.values()) {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
      content.remove(obj);
    }
    meshes.clear();
  }

  function frameCamera(box) {
    if (!box || box.isEmpty() || !isFinite(box.min.x) || !isFinite(box.max.x)) {
      camera.position.set(40, 30, 40);
      camera.near = 0.1; camera.far = 1e5;
      camera.updateProjectionMatrix();
      camera.lookAt(0, 0, 0);
      controls.target.set(0, 0, 0);
      controls.update();
      return;
    }
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 1e-3) * 0.5;
    const dist = (radius / Math.tan((camera.fov * Math.PI) / 360)) * 1.7;
    camera.near = Math.max(dist - radius * 4, radius * 1e-3);
    camera.far = dist + radius * 20;
    camera.position.set(center.x + dist * 0.7, center.y + dist * 0.45, center.z + dist * 0.7);
    camera.updateProjectionMatrix();
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
  }

  function visibleBox() {
    content.updateMatrixWorld(true);
    const box = new THREE.Box3();
    for (const obj of meshes.values()) {
      if (obj.visible) box.expandByObject(obj);
    }
    return box;
  }

  function fit() { frameCamera(visibleBox()); }

  function buildOne(item) {
    const positions = asFloat32(item.positions);
    let obj;
    if (item.kind === "line") {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      obj = new THREE.Line(geo, new THREE.LineBasicMaterial());
    } else {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      if (item.indices) geo.setIndex(new THREE.BufferAttribute(asUint32(item.indices), 1));
      geo.computeVertexNormals();
      obj = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ wireframe }));
    }
    return obj;
  }

  // Light, in-place: recolor / re-material / show-hide / transform by id.
  function setStyle(styles) {
    for (const s of styles || []) {
      const obj = meshes.get(s.id);
      if (!obj) continue;
      obj.visible = s.visible !== false;
      const c = s.color || [0.8, 0.8, 0.8];
      if (obj.material && obj.material.color) obj.material.color.setRGB(c[0], c[1], c[2]);
      if (obj.material && "metalness" in obj.material) obj.material.metalness = s.metallic ?? 0.1;
      if (obj.material && "roughness" in obj.material) obj.material.roughness = s.roughness ?? 0.6;
      applyMatrixTo(obj, s.matrix);
    }
    content.updateMatrixWorld(true);     // styles can change transforms -- but DON'T re-fit
  }

  // Heavy: rebuild geometry from buffers, then apply style; re-fit only if autofit.
  function setGeometry(items, styles) {
    disposeContent();
    for (const item of items || []) {
      const obj = buildOne(item);
      meshes.set(item.id, obj);
      content.add(obj);
    }
    setStyle(styles);
    if (autofit) frameCamera(visibleBox());
  }

  function setBackground(c) {
    const bg = c || [0.16, 0.16, 0.2];
    scene.background = new THREE.Color(bg[0], bg[1], bg[2]);
  }

  function resize(w, h) {
    width = w; height = h;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function toggleWireframe() {
    wireframe = !wireframe;
    for (const obj of meshes.values()) {
      if (obj.material && "wireframe" in obj.material) obj.material.wireframe = wireframe;
    }
  }

  function pngBlob(cb) {
    renderer.render(scene, camera);
    renderer.domElement.toBlob((blob) => cb(blob), "image/png");
  }

  function downloadPNG(name = "picogk.png") {
    pngBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = name; a.click();
      URL.revokeObjectURL(url);
    });
  }

  renderer.domElement.addEventListener("pointerenter", () => renderer.domElement.focus());
  renderer.domElement.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (k === "f") fit();
    else if (k === "w") toggleWireframe();
    else if (k === "s") downloadPNG();
  });

  let raf = 0;
  (function animate() {
    raf = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();

  function dispose() {
    cancelAnimationFrame(raf);
    controls.dispose();
    disposeContent();
    renderer.dispose();
    el.innerHTML = "";
  }

  return {
    setGeometry, setStyle, setBackground, resize, fit, toggleWireframe,
    downloadPNG, pngBlob, dispose, renderer, scene, camera,
  };
}

// anywidget entry point (notebook path).
function render({ model, el }) {
  const v = createViewer(el, {
    width: model.get("width"),
    height: model.get("height"),
    background: model.get("background"),
  });
  v.setBackground(model.get("background"));
  v.setGeometry(model.get("geometry"), model.get("style"));
  model.on("change:geometry", () => v.setGeometry(model.get("geometry"), model.get("style")));
  model.on("change:style", () => v.setStyle(model.get("style")));
  model.on("change:background", () => v.setBackground(model.get("background")));
  model.on("change:width", () => v.resize(model.get("width"), model.get("height")));
  model.on("change:height", () => v.resize(model.get("width"), model.get("height")));
  model.on("change:_fit", () => v.fit());
  model.on("change:_wireframe", () => v.toggleWireframe());
  model.on("change:_grab", () => {
    v.pngBlob((blob) => blob.arrayBuffer().then(
      (ab) => model.send({ type: "screenshot" }, [ab])));
  });
  return () => v.dispose();
}

export default { render };
