// PicoPie web viewer front-end (three.js).
//
// Geometry is computed in Python and handed to `createViewer` as a list of items
// (positions/indices as typed arrays or DataViews, plus color/material/matrix).
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

// Build a viewer inside `el`. Returns handles used by both front-ends.
export function createViewer(el, opts = {}) {
  let width = opts.width || 720;
  let height = opts.height || 480;

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(width, height);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.borderRadius = "6px";
  renderer.domElement.style.outline = "none";
  renderer.domElement.tabIndex = 0;          // so it can receive keyboard shortcuts
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1e6);

  // Soft ambient + hemisphere fill + two directionals so the shape reads from
  // any orbit angle (loosely mirrors the desktop viewer's lighting).
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 0.35));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(1, 1.4, 1.2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(-1, -0.6, -0.8);
  scene.add(fill);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  let content = new THREE.Group();
  scene.add(content);
  let lastBox = new THREE.Box3();
  let wireframe = false;

  function disposeContent() {
    content.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    scene.remove(content);
    content = new THREE.Group();
    scene.add(content);
  }

  function frameCamera(box) {
    if (!box || box.isEmpty()) {
      camera.position.set(40, 30, 40);
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

  function applyMatrix(obj, m16) {
    if (m16 && m16.length === 16) {
      const m = new THREE.Matrix4();
      m.set(...m16);                          // Matrix4.set takes row-major args
      obj.applyMatrix4(m);
    }
  }

  function setGeometry(items) {
    disposeContent();
    for (const it of items || []) {
      if (it.visible === false) continue;
      const positions = asFloat32(it.positions);
      const col = it.color || [0.8, 0.8, 0.8];
      const color = new THREE.Color(col[0], col[1], col[2]);
      let obj;
      if (it.kind === "line") {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        obj = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
      } else {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        if (it.indices) geo.setIndex(new THREE.BufferAttribute(asUint32(it.indices), 1));
        geo.computeVertexNormals();
        obj = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
          color, metalness: it.metallic ?? 0.1, roughness: it.roughness ?? 0.6,
          wireframe,
        }));
      }
      applyMatrix(obj, it.matrix);
      content.add(obj);
    }
    content.updateMatrixWorld(true);
    lastBox = new THREE.Box3().setFromObject(content);   // world bbox (honors matrices)
    frameCamera(lastBox);
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

  function fit() { frameCamera(lastBox); }

  function toggleWireframe() {
    wireframe = !wireframe;
    content.traverse((o) => { if (o.material && "wireframe" in o.material) o.material.wireframe = wireframe; });
  }

  function pngBlob(cb) {
    renderer.render(scene, camera);                      // ensure the buffer is current
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
    setGeometry, setBackground, resize, fit, toggleWireframe,
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
  v.setGeometry(model.get("geometry"));
  model.on("change:geometry", () => v.setGeometry(model.get("geometry")));
  model.on("change:background", () => v.setBackground(model.get("background")));
  model.on("change:width", () => v.resize(model.get("width"), model.get("height")));
  model.on("change:height", () => v.resize(model.get("width"), model.get("height")));
  model.on("change:_fit", () => v.fit());
  model.on("change:_wireframe", () => v.toggleWireframe());
  // programmatic screenshot: Python bumps _grab; send the PNG back as a buffer.
  model.on("change:_grab", () => {
    v.pngBlob((blob) => blob.arrayBuffer().then(
      (ab) => model.send({ type: "screenshot" }, [ab])));
  });
  return () => v.dispose();
}

export default { render };
