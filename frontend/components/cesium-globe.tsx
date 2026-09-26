"use client";

import { useEffect, useRef, useState } from "react";
import { apiUrl, type GeoPoint, type Reference } from "@/lib/api";

declare global {
  interface Window {
    CESIUM_BASE_URL?: string;
    Cesium?: any;
  }
}

export type ColorMode = "confidence" | "rgb" | "height";

type Pose = { lat: number; lon: number; alt: number; heading?: number };

type Props = {
  points: GeoPoint[];
  pose: Pose | NonePose;
  trajectory: Pose[];
  colorMode: ColorMode;
  measuring: boolean;
  reference: Reference | null;
  onPick: (pt: GeoPoint) => void;
};

type NonePose = null;

function loadCesium(): Promise<any> {
  if (typeof window === "undefined") return Promise.reject();
  if (window.Cesium) return Promise.resolve(window.Cesium);
  window.CESIUM_BASE_URL = "/cesium/";
  return new Promise((resolve, reject) => {
    const cssId = "cesium-widgets-css";
    if (!document.getElementById(cssId)) {
      const link = document.createElement("link");
      link.id = cssId;
      link.rel = "stylesheet";
      link.href = "/cesium/Widgets/widgets.css";
      document.head.appendChild(link);
    }
    const script = document.createElement("script");
    script.src = "/cesium/Cesium.js";
    script.async = true;
    script.onload = () => resolve(window.Cesium);
    script.onerror = () => reject(new Error("Failed to load Cesium"));
    document.body.appendChild(script);
  });
}

function lookOblique(viewer: any, Cesium: any, lat: number, lon: number) {
  // Camera sits above the map surface (ellipsoid) so building walls are visible.
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon - 0.00115, lat - 0.0017, 85),
    orientation: {
      heading: Cesium.Math.toRadians(28),
      pitch: Cesium.Math.toRadians(-28),
      roll: 0,
    },
    duration: 1.1,
  });
}

async function addOsmBuildings(viewer: any, Cesium: any, lat: number, lon: number) {
  const key = `${lat.toFixed(3)},${lon.toFixed(3)}`;
  if (viewer.__osmKey === key) return;
  viewer.__osmKey = key;
  const stale = viewer.entities.values.filter((e: any) => e.osmBuilding);
  stale.forEach((e: any) => viewer.entities.remove(e));
  const res = await fetch(apiUrl(`/osm/buildings?lat=${lat}&lon=${lon}`));
  if (!res.ok) return;
  const data = await res.json();
  const buildings = data.buildings || [];
  for (const b of buildings) {
    const ring: number[][] = b.ring || [];
    if (ring.length < 4) continue;
    const flat: number[] = [];
    for (const pair of ring) flat.push(pair[0], pair[1]);
    const h = Math.max(Number(b.height) || 9, 8);
    const positions = Cesium.Cartesian3.fromDegreesArray(flat);
    viewer.entities.add({
      osmBuilding: true,
      wall: {
        positions,
        minimumHeights: ring.map(() => 0),
        maximumHeights: ring.map(() => h),
        material: Cesium.Color.fromCssColorString("#f4f5f7"),
      },
    });
    viewer.entities.add({
      osmBuilding: true,
      polygon: {
        hierarchy: positions,
        height: h,
        material: Cesium.Color.WHITE,
        outline: false,
      },
    });
  }
  lookOblique(viewer, Cesium, lat, lon);
}

function colorFor(p: GeoPoint, mode: ColorMode, Cesium: any) {
  if (mode === "rgb" && p.r != null) {
    return Cesium.Color.fromBytes(p.r, p.g ?? 0, p.b ?? 0, 220);
  }
  if (mode === "height") {
    const t = Math.max(0, Math.min(1, (p.u ?? 0) / 12));
    return Cesium.Color.fromCssColorString(t > 0.55 ? "#f4d35e" : "#4cc9f0");
  }
  const c = p.conf ?? 0.4;
  if (c >= 0.72) return Cesium.Color.fromCssColorString("#34d399");
  if (c >= 0.42) return Cesium.Color.fromCssColorString("#fbbf24");
  return Cesium.Color.fromCssColorString("#f87171");
}

export function CesiumGlobe({
  points,
  pose,
  trajectory,
  colorMode,
  measuring,
  reference,
  onPick,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const primitivesRef = useRef<any>(null);
  const drawnRef = useRef(0);
  const colorRef = useRef(colorMode);
  const pickRef = useRef(onPick);
  const referenceRef = useRef(reference);
  const groundRef = useRef(216);
  const flewAway = useRef(false);
  const [cesiumError, setCesiumError] = useState<string | null>(null);
  colorRef.current = colorMode;
  pickRef.current = onPick;
  referenceRef.current = reference;

  useEffect(() => {
    let canceled = false;
    let viewer: any;
    loadCesium()
      .then((Cesium) => {
        if (canceled || !containerRef.current) return;
        Cesium.Ion.defaultAccessToken = "";
        viewer = new Cesium.Viewer(containerRef.current, {
          animation: false,
          timeline: false,
          geocoder: false,
          homeButton: false,
          sceneModePicker: false,
          baseLayerPicker: false,
          navigationHelpButton: false,
          fullscreenButton: false,
          infoBox: false,
          selectionIndicator: false,
          baseLayer: false,
          terrainProvider: new Cesium.EllipsoidTerrainProvider(),
        });
        viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#0b1220");
        viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#070b14");
        viewer.scene.globe.enableLighting = true;
        viewer.scene.globe.depthTestAgainstTerrain = false;
        viewer.scene.skyBox.show = false;
        viewer.scene.skyAtmosphere.show = true;
        viewer.scene.screenSpaceCameraController.enableTilt = true;
        viewer.scene.screenSpaceCameraController.enableLook = true;
        viewer.imageryLayers.addImageryProvider(
          new Cesium.UrlTemplateImageryProvider({
            url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            credit: "© OpenStreetMap",
          }),
        );
        const collection = new Cesium.PointPrimitiveCollection();
        viewer.scene.primitives.add(collection);
        primitivesRef.current = collection;
        viewerRef.current = viewer;

        const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
        handler.setInputAction((click: any) => {
          const picked = viewer.scene.pick(click.position);
          const ref = referenceRef.current;
          const entityId = picked?.id?.id ?? picked?.id;
          if (entityId === "ref-a" && ref) {
            pickRef.current({ lat: ref.a.lat, lon: ref.a.lon, height: ref.a.height });
            return;
          }
          if (entityId === "ref-b" && ref) {
            pickRef.current({ lat: ref.b.lat, lon: ref.b.lon, height: ref.b.height });
            return;
          }
          if (picked?.id && picked.id.onepass) {
            pickRef.current(picked.id.onepass);
            return;
          }
          const cartesian = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid);
          if (!cartesian) return;
          const carto = Cesium.Cartographic.fromCartesian(cartesian);
          pickRef.current({
            lat: Cesium.Math.toDegrees(carto.latitude),
            lon: Cesium.Math.toDegrees(carto.longitude),
            height: carto.height,
          });
        }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

        lookOblique(viewer, Cesium, 28.5448, 77.1924);
        addOsmBuildings(viewer, Cesium, 28.5448, 77.1924).catch((err) => console.error(err));
        requestAnimationFrame(() => {
          if (!viewer.isDestroyed()) viewer.resize();
        });
      })
      .catch((err) => {
        console.error(err);
        setCesiumError("Cesium failed to load. Run: cd frontend && node scripts/copy-cesium.mjs");
      });

    return () => {
      canceled = true;
      if (viewer && !viewer.isDestroyed()) viewer.destroy();
      viewerRef.current = null;
      primitivesRef.current = null;
      drawnRef.current = 0;
    };
  }, []);

  useEffect(() => {
    const Cesium = window.Cesium;
    const viewer = viewerRef.current;
    if (!Cesium || !viewer || points.length < 30) return;
    const hs = points.map((p) => p.height).filter((h) => Number.isFinite(h)).sort((a, b) => a - b);
    const ground = hs[Math.floor(hs.length * 0.08)] ?? groundRef.current;
    if (Math.abs(ground - groundRef.current) > 8) {
      groundRef.current = ground;
      if (primitivesRef.current) {
        primitivesRef.current.removeAll();
        drawnRef.current = 0;
      }
    }
    const lat = points[0].lat;
    const lon = points[0].lon;
    if (!flewAway.current && (Math.abs(lat - 28.5448) > 0.05 || Math.abs(lon - 77.1924) > 0.05)) {
      flewAway.current = true;
      lookOblique(viewer, Cesium, lat, lon);
    }
    addOsmBuildings(viewer, Cesium, lat, lon).catch((err) => console.error(err));
  }, [points]);

  useEffect(() => {
    const Cesium = window.Cesium;
    const viewer = viewerRef.current;
    if (!Cesium || !viewer) return;
    if (colorMode && primitivesRef.current) {
      primitivesRef.current.removeAll();
      drawnRef.current = 0;
    }
  }, [colorMode]);

  useEffect(() => {
    const Cesium = window.Cesium;
    const collection = primitivesRef.current;
    if (!Cesium || !collection) return;
    const start = drawnRef.current;
    for (let i = start; i < points.length; i++) {
      const p = points[i];
      collection.add({
        position: Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.height - groundRef.current),
        color: colorFor(p, colorRef.current, Cesium),
        pixelSize: 5,
        id: { onepass: p },
      });
    }
    drawnRef.current = points.length;
  }, [points]);

  useEffect(() => {
    const Cesium = window.Cesium;
    const viewer = viewerRef.current;
    if (!Cesium || !viewer) return;
    viewer.entities.removeById("uav");
    viewer.entities.removeById("traj");
    viewer.entities.removeById("ref-a");
    viewer.entities.removeById("ref-b");
    viewer.entities.removeById("ref-line");
    if (trajectory.length > 1) {
      viewer.entities.add({
        id: "traj",
        polyline: {
          positions: trajectory.map((p) =>
            Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt - groundRef.current),
          ),
          width: 2.5,
          material: Cesium.Color.fromCssColorString("#67e8f9"),
        },
      });
    }
    if (pose) {
      viewer.entities.add({
        id: "uav",
        position: Cesium.Cartesian3.fromDegrees(pose.lon, pose.lat, pose.alt - groundRef.current),
        point: { pixelSize: 14, color: Cesium.Color.fromCssColorString("#22d3ee"), outlineColor: Cesium.Color.WHITE, outlineWidth: 2 },
        label: {
          text: "UAV",
          font: "12px Inter, sans-serif",
          fillColor: Cesium.Color.WHITE,
          pixelOffset: new Cesium.Cartesian2(0, -18),
        },
      });
    }
    if (reference) {
      const a = Cesium.Cartesian3.fromDegrees(reference.a.lon, reference.a.lat, reference.a.height - groundRef.current);
      const b = Cesium.Cartesian3.fromDegrees(reference.b.lon, reference.b.lat, reference.b.height - groundRef.current);
      viewer.entities.add({
        id: "ref-line",
        polyline: { positions: [a, b], width: 3, material: Cesium.Color.fromCssColorString("#c4b5fd") },
      });
      viewer.entities.add({
        id: "ref-a",
        position: a,
        point: { pixelSize: 10, color: Cesium.Color.fromCssColorString("#a78bfa") },
        label: { text: "A 20.0 m eave", fillColor: Cesium.Color.WHITE, font: "11px sans-serif", pixelOffset: new Cesium.Cartesian2(12, 0) },
      });
      viewer.entities.add({
        id: "ref-b",
        position: b,
        point: { pixelSize: 10, color: Cesium.Color.fromCssColorString("#a78bfa") },
      });
    }
  }, [pose, trajectory, reference]);

  return (
    <div className="relative h-[70vh] min-h-[520px] w-full overflow-hidden rounded-xl border border-white/10 bg-[#0b1220]">
      <div ref={containerRef} className="absolute inset-0" />
      {cesiumError ? (
        <div className="absolute inset-0 flex items-center justify-center bg-[#070b14] p-6 text-center text-sm text-rose-200">
          {cesiumError}
        </div>
      ) : null}
      {measuring ? (
        <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/70 px-3 py-1.5 text-xs text-cyan-100">
          Measure mode — click two points on the model
        </div>
      ) : null}
    </div>
  );
}
