"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type CloudPoint = { e?: number; n?: number; u?: number; r?: number; g?: number; b?: number; conf?: number };
export type ColorMode = "confidence" | "rgb" | "height";
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
export type SpatialPayload = {
  seen: number[];
  summary: { faces: number; seen_faces: number; seen_fraction: number; largest_unseen_patch: number; faultlines: number };
  faultlines: number[][];
};
export type ChiselTag = "road" | "facade" | "ground" | "fill";
export type ChiselBlock = { id: string; tag: ChiselTag; min: [number, number, number]; max: [number, number, number] };
export type AgentFlag = { e: number; n: number; u: number; kind: string };
export type AgentRoute = { path: number[][]; length_m: number; flags: AgentFlag[]; reason?: string };
export type EnuPoint = { e: number; n: number; u: number };

type SplatObject = THREE.Object3D & {
  dispose: () => Promise<void>;
  addSplatScene: (path: string, options?: Record<string, unknown>) => Promise<unknown>;
};

const CHISEL_COLOR: Record<ChiselTag, number> = {
  road: 0xf59e0b,
  facade: 0xfb7185,
  ground: 0x34d399,
  fill: 0xa78bfa,
};

type Props = {
  points: CloudPoint[];
  cameras: ReconCamera[];
  mesh: MeshPayload | null;
  viewMode: ViewMode;
  colorMode: ColorMode;
  splatUrl: string | null;
  onSplatError?: (message: string) => void;
  showOcclusion?: boolean;
  spatial?: SpatialPayload | null;
  chisel?: ChiselBlock[];
  route?: AgentRoute | null;
  agentMarkers?: { start?: EnuPoint; goal?: EnuPoint };
  pickActive?: boolean;
  onSurfaceClick?: (point: EnuPoint) => void;
};

let disc: THREE.Texture | null = null;

function discTexture(): THREE.Texture {
  if (disc) return disc;
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff";
  g.beginPath();
  g.arc(32, 32, 30, 0, Math.PI * 2);
  g.fill();
  disc = new THREE.CanvasTexture(c);
  return disc;
}

function toThree(e: number, n: number, u: number) {
  return new THREE.Vector3(e, u, -n);
}

function pointColor(p: CloudPoint, mode: ColorMode, uMin: number, uMax: number): [number, number, number] {
  if (mode === "rgb") return [(p.r ?? 220) / 255, (p.g ?? 220) / 255, (p.b ?? 220) / 255];
  if (mode === "height") {
    const t = uMax > uMin ? ((p.u ?? 0) - uMin) / (uMax - uMin) : 0.5;
    return [0.3 + 0.65 * t, 0.78 - 0.15 * t, 0.95 - 0.7 * t];
  }
  const c = p.conf ?? 0.4;
  if (c >= 0.72) return [0.204, 0.827, 0.6];
  if (c >= 0.42) return [0.984, 0.749, 0.141];
  return [0.973, 0.443, 0.443];
}

function syncScene(
  view: {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    cloud: THREE.Points | null;
    frustums: THREE.Group;
    grid: THREE.GridHelper;
    spatial: THREE.Group;
    mesh: THREE.Mesh | null;
    splats: SplatObject | null;
    fitted: boolean;
    fittedSpan: number;
  },
  data: {
    points: CloudPoint[];
    cameras: ReconCamera[];
    mesh: MeshPayload | null;
    viewMode: ViewMode;
    colorMode: ColorMode;
    showOcclusion: boolean;
    spatial: SpatialPayload | null;
    chisel: ChiselBlock[];
    route: AgentRoute | null;
    agentMarkers: { start?: EnuPoint; goal?: EnuPoint };
  },
) {
  if (view.cloud) {
    view.scene.remove(view.cloud);
    view.cloud.geometry.dispose();
    (view.cloud.material as THREE.Material).dispose();
    view.cloud = null;
  }
  const usable = data.points.filter((p) => Number.isFinite(p.e) && Number.isFinite(p.n) && Number.isFinite(p.u));
  if (usable.length) {
    let uMin = Infinity;
    let uMax = -Infinity;
    for (const p of usable) {
      const u = p.u ?? 0;
      if (u < uMin) uMin = u;
      if (u > uMax) uMax = u;
    }
    const pos = new Float32Array(usable.length * 3);
    const col = new Float32Array(usable.length * 3);
    usable.forEach((p, i) => {
      const v = toThree(p.e!, p.n!, p.u!);
      pos[i * 3] = v.x;
      pos[i * 3 + 1] = v.y;
      pos[i * 3 + 2] = v.z;
      const [r, g, b] = pointColor(p, data.colorMode, uMin, uMax);
      col[i * 3] = r;
      col[i * 3 + 1] = g;
      col[i * 3 + 2] = b;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    // Outliers stretch a plain bounding box, so frame on the 2nd–98th percentile.
    const lo = new THREE.Vector3();
    const hi = new THREE.Vector3();
    for (let axis = 0; axis < 3; axis++) {
      const vals = new Float32Array(usable.length);
      for (let i = 0; i < usable.length; i++) vals[i] = pos[i * 3 + axis];
      vals.sort();
      lo.setComponent(axis, vals[Math.floor(vals.length * 0.02)]);
      hi.setComponent(axis, vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.98))]);
    }
    const size = hi.clone().sub(lo);
    const footprint = Math.max(size.x * size.z, 1);
    const pointSize = THREE.MathUtils.clamp(1.5 * Math.sqrt(footprint / usable.length), 0.04, 6);
    const cloud = new THREE.Points(
      geo,
      new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: true, map: discTexture(), alphaTest: 0.5 }),
    );
    cloud.visible = data.viewMode === "points";
    view.scene.add(cloud);
    view.cloud = cloud;
    const center = lo.clone().add(hi).multiplyScalar(0.5);
    const span = Math.max(size.x, size.y, size.z, 8);
    if (!view.fitted || span > view.fittedSpan * 1.2) {
      view.controls.target.copy(center);
      const offset = new THREE.Vector3(span * 0.9, span * 0.62, span * 0.9);
      if (data.cameras.length) {
        // Look from roughly the capture side (best-covered surfaces), swung
        // off-axis and raised so the frustum trail doesn't fill the view.
        const eye = new THREE.Vector3();
        for (const cam of data.cameras) eye.add(toThree(cam.e, cam.n, cam.u));
        eye.divideScalar(data.cameras.length).sub(center);
        eye.y = 0;
        if (eye.lengthSq() > 1e-6) {
          eye.normalize().applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(40));
          offset.copy(eye.multiplyScalar(span * 0.75)).setY(span * 0.7);
        }
      }
      view.camera.position.copy(center).add(offset);
      view.camera.near = Math.max(span / 2000, 0.05);
      view.camera.far = Math.max(span * 30, 800);
      view.camera.updateProjectionMatrix();
      view.fitted = true;
      view.fittedSpan = span;
    }
    const gridSize = Math.max(Math.ceil(Math.max(size.x, size.z) * 1.3 / 10) * 10, 20);
    if (view.grid.userData.size !== gridSize) {
      view.scene.remove(view.grid);
      view.grid.geometry.dispose();
      (view.grid.material as THREE.Material).dispose();
      view.grid = new THREE.GridHelper(gridSize, 20, 0x6b7280, 0x3f4652);
      view.grid.userData.size = gridSize;
      view.scene.add(view.grid);
    }
    view.grid.position.set(center.x, lo.y - span * 0.02, center.z);
  }

  view.frustums.clear();
  const mat = new THREE.LineBasicMaterial({ color: 0xf59e0b });
  for (const cam of data.cameras) {
    const depth = Math.max(6, view.fittedSpan * 0.025);
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
    const faceCount = mesh.indices.length / 3;
    const paintOcclusion = data.showOcclusion && data.spatial && data.spatial.seen.length === faceCount;
    if (paintOcclusion && data.spatial) {
      const seen = data.spatial.seen;
      const acc = new Float32Array(src.length / 3);
      const weight = new Float32Array(src.length / 3);
      for (let f = 0; f < faceCount; f++) {
        const covered = seen[f] > 0 ? 1 : 0;
        for (let k = 0; k < 3; k++) {
          const vi = mesh.indices[f * 3 + k];
          acc[vi] += covered;
          weight[vi] += 1;
        }
      }
      const col = new Float32Array(src.length);
      for (let i = 0; i < acc.length; i++) {
        const t = weight[i] ? acc[i] / weight[i] : 0;
        col[i * 3] = 0.9 - 0.7 * t;
        col[i * 3 + 1] = 0.25 + 0.55 * t;
        col[i * 3 + 2] = 0.22 + 0.28 * t;
      }
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    } else if (mesh.colors.length === src.length) {
      const col = new Float32Array(mesh.colors.length);
      for (let i = 0; i < mesh.colors.length; i++) col[i] = mesh.colors[i] / 255;
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    }
    geo.computeVertexNormals();
    const solid = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({
        color: geo.hasAttribute("color") ? 0xffffff : 0xd7dde6,
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

  clearGroup(view.spatial);
  if (data.showOcclusion && data.spatial) {
    const verts: number[] = [];
    for (const seg of data.spatial.faultlines) {
      if (seg.length < 6) continue;
      const a = toThree(seg[0], seg[1], seg[2]);
      const b = toThree(seg[3], seg[4], seg[5]);
      verts.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    if (verts.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      view.spatial.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xf43f5e })));
    }
  }
  for (const block of data.chisel) {
    const sx = Math.max(block.max[0] - block.min[0], 0.05);
    const sy = Math.max(block.max[2] - block.min[2], 0.05);
    const sz = Math.max(block.max[1] - block.min[1], 0.05);
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(sx, sy, sz),
      new THREE.MeshStandardMaterial({ color: CHISEL_COLOR[block.tag], transparent: true, opacity: 0.38, roughness: 0.6 }),
    );
    const center = toThree(
      (block.min[0] + block.max[0]) / 2,
      (block.min[1] + block.max[1]) / 2,
      (block.min[2] + block.max[2]) / 2,
    );
    box.position.copy(center);
    view.spatial.add(box);
  }
  const path = data.route?.path ?? [];
  if (path.length) {
    const verts: number[] = [];
    for (const p of path) {
      const v = toThree(p[0], p[1], p[2]);
      verts.push(v.x, v.y, v.z);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    view.spatial.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x22d3ee })));
  }
  const markerRadius = Math.max((view.fittedSpan || 40) * 0.012, 0.4);
  for (const flag of data.route?.flags ?? []) {
    const mark = new THREE.Mesh(
      new THREE.SphereGeometry(markerRadius, 10, 8),
      new THREE.MeshBasicMaterial({ color: flag.kind === "faultline" ? 0xf43f5e : 0xfbbf24 }),
    );
    mark.position.copy(toThree(flag.e, flag.n, flag.u));
    view.spatial.add(mark);
  }
  const dropMarker = (point: EnuPoint | undefined, color: number) => {
    if (!point) return;
    const mark = new THREE.Mesh(new THREE.SphereGeometry(markerRadius * 1.35, 12, 8), new THREE.MeshBasicMaterial({ color }));
    mark.position.copy(toThree(point.e, point.n, point.u));
    view.spatial.add(mark);
  };
  dropMarker(data.agentMarkers.start, 0x34d399);
  dropMarker(data.agentMarkers.goal, 0x38bdf8);
  if (view.splats) view.splats.visible = data.viewMode === "neural";
}

function clearGroup(group: THREE.Group) {
  for (const child of [...group.children]) {
    group.remove(child);
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) material.forEach((m) => m.dispose());
    else material?.dispose();
  }
}

export function ReconViewport({
  points, cameras, mesh, viewMode, colorMode, splatUrl, onSplatError,
  showOcclusion = false, spatial = null, chisel = [], route = null, agentMarkers = {},
  pickActive = false, onSurfaceClick,
}: Props) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef({ points, cameras, mesh, viewMode, colorMode, showOcclusion, spatial, chisel, route, agentMarkers });
  latest.current = { points, cameras, mesh, viewMode, colorMode, showOcclusion, spatial, chisel, route, agentMarkers };
  const onError = useRef(onSplatError);
  onError.current = onSplatError;
  const pick = useRef({ active: pickActive, onClick: onSurfaceClick });
  pick.current = { active: pickActive, onClick: onSurfaceClick };
  const api = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    controls: OrbitControls;
    cloud: THREE.Points | null;
    frustums: THREE.Group;
    grid: THREE.GridHelper;
    spatial: THREE.Group;
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
    const spatialGroup = new THREE.Group();
    scene.add(spatialGroup);
    let down: { x: number; y: number } | null = null;
    const onDown = (ev: PointerEvent) => {
      down = { x: ev.clientX, y: ev.clientY };
    };
    const onUp = (ev: PointerEvent) => {
      if (!down) return;
      const dx = ev.clientX - down.x;
      const dy = ev.clientY - down.y;
      down = null;
      if (dx * dx + dy * dy > 16 || !pick.current.active || !pick.current.onClick) return;
      const current = api.current;
      if (!current?.mesh) return;
      const rect = canvas.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((ev.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1,
        -((ev.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1,
      );
      const ray = new THREE.Raycaster();
      ray.setFromCamera(ndc, current.camera);
      const hit = ray.intersectObject(current.mesh, false)[0];
      if (!hit) return;
      pick.current.onClick({ e: hit.point.x, n: -hit.point.z, u: hit.point.y });
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
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
      grid,
      spatial: spatialGroup,
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
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      controls.dispose();
      renderer.dispose();
      canvas.remove();
      api.current = null;
    };
  }, []);

  useEffect(() => {
    api.current?.draw();
  }, [points, cameras, mesh, viewMode, colorMode, showOcclusion, spatial, chisel, route, agentMarkers]);

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
