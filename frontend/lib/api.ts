export type GeoPoint = {
  lat: number;
  lon: number;
  height: number;
  e?: number;
  n?: number;
  u?: number;
  r?: number;
  g?: number;
  b?: number;
  conf?: number;
  band?: "high" | "medium" | "low";
};

export type Challenge = {
  id: string;
  label: string;
  active: boolean;
  response: string;
};

export type Reference = {
  name: string;
  true_length_m: number;
  a: { lat: number; lon: number; height: number };
  b: { lat: number; lon: number; height: number };
  a_enu: number[];
  b_enu: number[];
  tolerance_m: number;
  tolerance_pct: number;
  note?: string;
};

export type MeasureResult = {
  measured_m: number;
  true_length_m?: number;
  abs_error_m?: number;
  pct_error?: number;
  tolerance_m?: number;
  pass?: boolean | null;
};

export async function startDemo(): Promise<{ id: string }> {
  const res = await fetch("/api/jobs/demo", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startUpload(video: File, telemetry: File): Promise<{ id: string }> {
  const body = new FormData();
  body.append("video", video);
  body.append("telemetry", telemetry);
  const res = await fetch("/api/jobs", { method: "POST", body });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchReference(): Promise<Reference> {
  const res = await fetch("/api/demo/reference");
  if (!res.ok) throw new Error("reference unavailable");
  return res.json();
}

export async function measurePoints(
  origin: { lat: number; lon: number; alt: number },
  a: GeoPoint,
  b: GeoPoint,
  trueLength?: number,
): Promise<MeasureResult> {
  const res = await fetch("/api/metric/measure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      origin,
      a: { lat: a.lat, lon: a.lon, height: a.height },
      b: { lat: b.lat, lon: b.lon, height: b.height },
      true_length_m: trueLength ?? 0,
      tolerance_m: 1,
      tolerance_pct: 5,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
