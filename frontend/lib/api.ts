/** Direct backend URL avoids Next.js proxy issues with SSE on Windows. */
const DIRECT_CANDIDATES = [
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, ""),
  "http://127.0.0.1:8765",
  "http://localhost:8765",
].filter(Boolean) as string[];

let resolvedBase = DIRECT_CANDIDATES[0] ?? "http://127.0.0.1:8765";

export function getApiBase(): string {
  return resolvedBase;
}

async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const bases = [...new Set(DIRECT_CANDIDATES)];
  let lastError: unknown;
  for (const base of bases) {
    try {
      const res = await fetch(`${base}${path}`, init);
      if (res.ok) {
        resolvedBase = base;
        return res;
      }
      if (res.status >= 400 && res.status < 500) return res;
    } catch (err) {
      lastError = err;
    }
  }
  try {
    const res = await fetch(`/api${path}`, init);
    if (res.ok) return res;
    return res;
  } catch (err) {
    lastError = err;
  }
  throw lastError instanceof Error ? lastError : new Error("API unreachable on port 8765");
}

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

export type JobSnapshot = {
  id: string;
  kind: string;
  status: string;
  error?: string | null;
  message?: string;
  progress?: number;
  origin?: { lat: number; lon: number; alt: number };
  reference?: Reference;
  adapters?: { vggt: unknown; colmap: boolean };
  mode?: string;
  points: GeoPoint[];
  pose?: { lat: number; lon: number; alt: number; heading?: number; hdop?: number };
  trajectory?: { lat: number; lon: number; alt: number; heading?: number; hdop?: number }[];
  stats?: { points: number; high: number; medium: number; low: number };
  challenges?: Challenge[];
  result?: { metric?: MeasureResult };
};

export async function checkHealth(): Promise<{ ok: boolean; vggt?: unknown; colmap?: boolean }> {
  const res = await apiFetch("/health", { cache: "no-store" });
  if (!res.ok) throw new Error(`API unreachable (${res.status})`);
  return res.json();
}

export async function startDemo(): Promise<{ id: string }> {
  const res = await apiFetch("/jobs/demo", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startUpload(video: File, telemetry: File): Promise<{ id: string }> {
  const body = new FormData();
  body.append("video", video);
  body.append("telemetry", telemetry);
  const res = await apiFetch("/jobs", { method: "POST", body });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchReference(): Promise<Reference> {
  const res = await apiFetch("/demo/reference");
  if (!res.ok) throw new Error("reference unavailable");
  return res.json();
}

export async function fetchJobSnapshot(jobId: string): Promise<JobSnapshot> {
  const res = await apiFetch(`/jobs/${jobId}/snapshot`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function measurePoints(
  origin: { lat: number; lon: number; alt: number },
  a: GeoPoint,
  b: GeoPoint,
  trueLength?: number,
): Promise<MeasureResult> {
  const res = await apiFetch("/metric/measure", {
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

