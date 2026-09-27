"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type CloudPoint = { e?: number; n?: number; u?: number; r?: number; g?: number; b?: number };
export type ReconCamera = {
  e: number;
  n: number;
  u: number;
  rotation: number[];
  fx: number;
  width: number;
  height: number;
};
export type MeshPayload = { positions: number[]; colors: number[]; indices: number[] };

type Props = {
  points: CloudPoint[];
  cameras: ReconCamera[];
  mesh: MeshPayload | null;
};

function toThree(e: number, n: number, u: number) {
  return new THREE.Vector3(e, u, -n);
}

export function ReconViewport({ points, cameras, mesh }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    cloud: THREE.Points | null;
    frustums: THREE.Group;
    mesh: THREE.Mesh | null;
    fitted: boolean;
  } | null>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#1b1d21");
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
    camera.position.set(40, 30, 40);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const sun = new THREE.DirectionalLight(0xffffff, 0.7);
    sun.position.set(40, 80, 20);
    scene.add(sun);
    const grid = new THREE.GridHelper(200, 40, 0x3a3f48, 0x2a2e35);
    grid.position.y = 0;
    scene.add(grid);
    const frustums = new THREE.Group();
    scene.add(frustums);
    const fit = () => {
      const w = el.clientWidth || 800;
      const h = el.clientHeight || 600;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      controls.update();
      renderer.render(scene, camera);
    };
    loop();
    api.current = { scene, camera, renderer, controls, cloud: null, frustums, mesh: null, fitted: false };
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      api.current = null;
    };
  }, []);

  useEffect(() => {
    const view = api.current;
    if (!view) return;
    if (view.cloud) {
      view.scene.remove(view.cloud);
      view.cloud.geometry.dispose();
      (view.cloud.material as THREE.Material).dispose();
      view.cloud = null;
    }
    const usable = points.filter((p) => Number.isFinite(p.e) && Number.isFinite(p.n) && Number.isFinite(p.u));
    if (!usable.length) return;
    const pos = new Float32Array(usable.length * 3);
    const col = new Float32Array(usable.length * 3);
    usable.forEach((p, i) => {
      const v = toThree(p.e!, p.n!, p.u!);
      pos[i * 3] = v.x;
      pos[i * 3 + 1] = v.y;
      pos[i * 3 + 2] = v.z;
      col[i * 3] = (p.r ?? 220) / 255;
      col[i * 3 + 1] = (p.g ?? 220) / 255;
      col[i * 3 + 2] = (p.b ?? 220) / 255;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const cloud = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.18, vertexColors: true, sizeAttenuation: true }));
    view.scene.add(cloud);
    view.cloud = cloud;
    if (!view.fitted) {
      geo.computeBoundingBox();
      const box = geo.boundingBox;
      if (box) {
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const span = Math.max(size.x, size.y, size.z, 8);
        view.controls.target.copy(center);
        view.camera.position.copy(center).add(new THREE.Vector3(span * 0.7, span * 0.45, span * 0.7));
        view.fitted = true;
      }
    }
  }, [points]);

  useEffect(() => {
    const view = api.current;
    if (!view) return;
    view.frustums.clear();
    const mat = new THREE.LineBasicMaterial({ color: 0xf59e0b });
    for (const cam of cameras) {
      const depth = 6;
      const fx = cam.fx || 800;
      const hw = (cam.width / 2 / fx) * depth;
      const hh = (cam.height / 2 / fx) * depth;
      const R = cam.rotation;
      const apply = (x: number, y: number, z: number) => {
        const e = R[0] * x + R[1] * y + R[2] * z + cam.e;
        const n = R[3] * x + R[4] * y + R[5] * z + cam.n;
        const u = R[6] * x + R[7] * y + R[8] * z + cam.u;
        return toThree(e, n, u);
      };
      const apex = toThree(cam.e, cam.n, cam.u);
      const corners = [
        apply(-hw, -hh, depth),
        apply(hw, -hh, depth),
        apply(hw, hh, depth),
        apply(-hw, hh, depth),
      ];
      const verts: number[] = [];
      const push = (a: THREE.Vector3, b: THREE.Vector3) => verts.push(a.x, a.y, a.z, b.x, b.y, b.z);
      corners.forEach((c) => push(apex, c));
      for (let i = 0; i < 4; i++) push(corners[i], corners[(i + 1) % 4]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      view.frustums.add(new THREE.LineSegments(geo, mat));
    }
  }, [cameras]);

  useEffect(() => {
    const view = api.current;
    if (!view) return;
    if (view.mesh) {
      view.scene.remove(view.mesh);
      view.mesh.geometry.dispose();
      (view.mesh.material as THREE.Material).dispose();
      view.mesh = null;
    }
    if (!mesh || mesh.positions.length < 9 || mesh.indices.length < 3) return;
    const src = mesh.positions;
    const pos = new Float32Array(src.length);
    for (let i = 0; i < src.length; i += 3) {
      const v = toThree(src[i], src[i + 1], src[i + 2]);
      pos[i] = v.x;
      pos[i + 1] = v.y;
      pos[i + 2] = v.z;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(mesh.indices);
    if (mesh.colors.length === src.length) {
      const col = new Float32Array(mesh.colors.length);
      for (let i = 0; i < mesh.colors.length; i++) col[i] = mesh.colors[i] / 255;
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    }
    geo.computeVertexNormals();
    const solid = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({ color: 0xd7dde6, vertexColors: geo.hasAttribute("color"), metalness: 0.05, roughness: 0.85, wireframe: false, side: THREE.DoubleSide }),
    );
    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.35 }),
    );
    solid.add(wire);
    view.scene.add(solid);
    view.mesh = solid;
  }, [mesh]);

  return <div ref={host} className="h-full min-h-[520px] w-full bg-[#1b1d21]" />;
}
