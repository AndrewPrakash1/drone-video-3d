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
export type ViewMode = "points" | "mesh" | "neural";

type SplatObject = THREE.Object3D & {
  dispose: () => Promise<void>;
  addSplatScene: (path: string, options?: Record<string, unknown>) => Promise<unknown>;
};

type Props = {
  points: CloudPoint[];
  cameras: ReconCamera[];
  mesh: MeshPayload | null;
  viewMode: ViewMode;
  splatUrl: string | null;
  onSplatError?: (message: string) => void;
};

function toThree(e: number, n: number, u: number) {
  return new THREE.Vector3(e, u, -n);
}

function syncScene(
  view: {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    cloud: THREE.Points | null;
    frustums: THREE.Group;
    mesh: THREE.Mesh | null;
    splats: SplatObject | null;
    fitted: boolean;
    fittedSpan: number;
  },
  data: { points: CloudPoint[]; cameras: ReconCamera[]; mesh: MeshPayload | null; viewMode: ViewMode },
) {
  if (view.cloud) {
    view.scene.remove(view.cloud);
    view.cloud.geometry.dispose();
    (view.cloud.material as THREE.Material).dispose();
    view.cloud = null;
  }
  const usable = data.points.filter((p) => Number.isFinite(p.e) && Number.isFinite(p.n) && Number.isFinite(p.u));
  if (usable.length) {
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
    const cloud = new THREE.Points(
      geo,
      new THREE.PointsMaterial({ size: 5, vertexColors: true, sizeAttenuation: false }),
    );
    cloud.visible = data.viewMode === "points";
    view.scene.add(cloud);
    view.cloud = cloud;
    geo.computeBoundingBox();
    const box = geo.boundingBox;
    if (box && !box.isEmpty()) {
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const span = Math.max(size.x, size.y, size.z, 8);
      if (!view.fitted || span > view.fittedSpan * 1.2) {
        view.controls.target.copy(center);
        view.camera.position.copy(center).add(new THREE.Vector3(span * 0.9, span * 0.62, span * 0.9));
        view.camera.near = Math.max(span / 2000, 0.05);
        view.camera.far = Math.max(span * 30, 800);
        view.camera.updateProjectionMatrix();
        view.fitted = true;
        view.fittedSpan = span;
      }
    }
  }

  view.frustums.clear();
  const mat = new THREE.LineBasicMaterial({ color: 0xf59e0b });
  for (const cam of data.cameras) {
    const depth = 6;
    const fx = cam.fx || 800;
    const hw = (cam.width / 2 / fx) * depth;
    const hh = (cam.height / 2 / fx) * depth;
    const R = cam.rotation || [];
    const apply = (x: number, y: number, z: number) => {
      const e = (R[0] ?? 1) * x + (R[1] ?? 0) * y + (R[2] ?? 0) * z + cam.e;
      const n = (R[3] ?? 0) * x + (R[4] ?? 1) * y + (R[5] ?? 0) * z + cam.n;
      const u = (R[6] ?? 0) * x + (R[7] ?? 0) * y + (R[8] ?? 1) * z + cam.u;
      return toThree(e, n, u);
    };
    const apex = toThree(cam.e, cam.n, cam.u);
    const corners = [apply(-hw, -hh, depth), apply(hw, -hh, depth), apply(hw, hh, depth), apply(-hw, hh, depth)];
    const verts: number[] = [];
    const push = (a: THREE.Vector3, b: THREE.Vector3) => verts.push(a.x, a.y, a.z, b.x, b.y, b.z);
    corners.forEach((c) => push(apex, c));
    for (let i = 0; i < 4; i++) push(corners[i], corners[(i + 1) % 4]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    view.frustums.add(new THREE.LineSegments(geo, mat));
  }

  if (view.mesh) {
    view.scene.remove(view.mesh);
    view.mesh.geometry.dispose();
    (view.mesh.material as THREE.Material).dispose();
    view.mesh = null;
  }
  const mesh = data.mesh;
  if (mesh && mesh.positions.length >= 9 && mesh.indices.length >= 3) {
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
      new THREE.MeshStandardMaterial({
        color: 0xd7dde6,
        vertexColors: geo.hasAttribute("color"),
        metalness: 0.05,
        roughness: 0.85,
        side: THREE.DoubleSide,
      }),
    );
    solid.visible = data.viewMode === "mesh";
    view.scene.add(solid);
    view.mesh = solid;
  }
  if (view.splats) view.splats.visible = data.viewMode === "neural";
}

export function ReconViewport({ points, cameras, mesh, viewMode, splatUrl, onSplatError }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef({ points, cameras, mesh, viewMode });
  latest.current = { points, cameras, mesh, viewMode };
  const onError = useRef(onSplatError);
  onError.current = onSplatError;
  const api = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    cloud: THREE.Points | null;
    frustums: THREE.Group;
    mesh: THREE.Mesh | null;
    splats: SplatObject | null;
    fitted: boolean;
    fittedSpan: number;
    draw: () => void;
  } | null>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#14161a");
    const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 8000);
    camera.position.set(40, 30, 40);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1);
    const canvas = renderer.domElement;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    el.appendChild(canvas);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 0.8);
    sun.position.set(40, 80, 20);
    scene.add(sun);
    const grid = new THREE.GridHelper(200, 20, 0x6b7280, 0x3f4652);
    grid.position.y = 0;
    scene.add(grid);
    const frustums = new THREE.Group();
    scene.add(frustums);
    const fit = () => {
      const w = Math.max(el.clientWidth, 2);
      const h = Math.max(el.clientHeight, 2);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    const view = {
      scene,
      camera,
      renderer,
      controls,
      cloud: null as THREE.Points | null,
      frustums,
      mesh: null as THREE.Mesh | null,
      splats: null as SplatObject | null,
      fitted: false,
      fittedSpan: 0,
      draw: () => {
        return;
      },
    };
    view.draw = () => syncScene(view, latest.current);
    api.current = view;
    view.draw();
    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      controls.update();
      renderer.render(scene, camera);
    };
    loop();
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      canvas.remove();
      api.current = null;
    };
  }, []);

  useEffect(() => {
    api.current?.draw();
  }, [points, cameras, mesh, viewMode]);

  useEffect(() => {
    const url = splatUrl;
    if (!url) return;
    let dead = false;
    const drop = async () => {
      const view = api.current;
      if (!view) return;
      const mod = await import("@mkkellogg/gaussian-splats-3d");
      const DropInViewer = mod.DropInViewer;
      if (!DropInViewer || dead) return;
      const splats = new DropInViewer({
        sharedMemoryForWorkers: false,
        gpuAcceleratedSort: false,
        integerBasedSort: false,
        sphericalHarmonicsDegree: 0,
      }) as SplatObject;
      splats.visible = latest.current.viewMode === "neural";
      // ENU (east, north, up) -> Three.js (east, up, -north), a -90° rotation about X.
      await splats.addSplatScene(url, {
        showLoadingUI: false,
        position: [0, 0, 0],
        rotation: [-Math.SQRT1_2, 0, 0, Math.SQRT1_2],
        scale: [1, 1, 1],
        splatAlphaRemovalThreshold: 1,
      });
      if (dead || !api.current) {
        await splats.dispose();
        return;
      }
      api.current.scene.add(splats);
      api.current.splats = splats;
      api.current.draw();
    };
    drop().catch((err: unknown) => {
      const message = err instanceof Error ? err.message : "Neural render failed to load";
      onError.current?.(message);
    });
    return () => {
      dead = true;
      const current = api.current?.splats;
      if (!current) return;
      api.current?.scene.remove(current);
      api.current!.splats = null;
      void current.dispose();
    };
  }, [splatUrl]);

  return <div ref={host} className="absolute inset-0 bg-[#14161a]" />;
}
