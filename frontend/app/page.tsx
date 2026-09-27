"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Download,
  Loader2,
  MapPin,
  Radar,
  Ruler,
  TriangleAlert,
  Upload,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  checkHealth,
  fetchJobSnapshot,
  fetchReference,
  apiUrl,
  getApiBase,
  measurePoints,
  startBrighton,
  startDemo,
  startUpload,
  type Challenge,
  type GeoPoint,
  type JobSnapshot,
  type MeasureResult,
  type Reference,
} from "@/lib/api";
import type { ColorMode } from "@/components/cesium-globe";
import { ReconViewport, type MeshPayload, type ReconCamera } from "@/components/recon-viewport";

type AdapterState = {
  vggt: boolean | { available?: boolean; cuda?: boolean; package?: boolean; device?: string | null; reason?: string | null };
  colmap: boolean;
};

function vggtReady(v: AdapterState["vggt"]): boolean {
  if (typeof v === "boolean") return v;
  return Boolean(v?.available);
}

function vggtLabel(v: AdapterState["vggt"]): string {
  if (vggtReady(v)) {
    const device = typeof v === "object" ? v.device : null;
    return device ? `ready on ${device}` : "ready";
  }
  if (typeof v === "object" && v?.reason) return `CPU fallback (${v.reason})`;
  return "CPU fallback";
}

type Pose = { lat: number; lon: number; alt: number; heading?: number; hdop?: number };
type StageView = { stage: string; label: string; status: string; stats?: Record<string, unknown> };
type JobState = "idle" | "running" | "done" | "error";

function applySnapshot(
  snap: JobSnapshot,
  setters: {
    setMessage: (m: string) => void;
    setProgress: (p: number) => void;
    setPoints: (p: GeoPoint[]) => void;
    setPose: (p: Pose | null) => void;
    setTrajectory: (t: Pose[]) => void;
    setStats: (s: { points: number; high: number; medium: number; low: number }) => void;
    setChallenges: (c: Challenge[]) => void;
    setOrigin: (o: { lat: number; lon: number; alt: number } | null) => void;
    setReference: (r: Reference | null) => void;
    setAdapters: (a: AdapterState) => void;
    setReconMode: (m: string) => void;
    setMetric: (m: MeasureResult | null) => void;
    setState: (s: JobState) => void;
    setError: (e: string | null) => void;
    setCameras: (c: ReconCamera[]) => void;
    setStages: (s: Record<string, StageView>) => void;
  },
) {
  if (snap.message) setters.setMessage(snap.message);
  if (snap.progress) setters.setProgress(snap.progress);
  if (snap.origin) setters.setOrigin(snap.origin);
  if (snap.reference) setters.setReference(snap.reference);
  if (snap.adapters) {
    setters.setAdapters({
      vggt: (snap.adapters.vggt as AdapterState["vggt"]) ?? false,
      colmap: Boolean(snap.adapters.colmap),
    });
  }
  if (snap.mode) setters.setReconMode(snap.mode);
  if (snap.points?.length) setters.setPoints(snap.points);
  if (snap.pose) setters.setPose(snap.pose);
  if (snap.trajectory?.length) setters.setTrajectory(snap.trajectory);
  if (snap.stats) setters.setStats(snap.stats);
  if (snap.challenges?.length) setters.setChallenges(snap.challenges);
  if (snap.cameras?.length) setters.setCameras(snap.cameras);
  if (snap.stages) setters.setStages(snap.stages);
  if (snap.result?.metric) setters.setMetric(snap.result.metric);
  if (snap.status === "done") {
    setters.setState("done");
    setters.setProgress(100);
    setters.setError(null);
  } else if (snap.status === "error") {
    setters.setState("error");
    setters.setError(snap.error || "Reconstruction failed");
  } else if (snap.status === "running") {
    setters.setState("running");
  }
}

export default function MissionPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [state, setState] = useState<JobState>("idle");
  const [message, setMessage] = useState("No mission. Start the proxy flyby or upload a single-pass clip.");
  const [progress, setProgress] = useState(0);
  const [points, setPoints] = useState<GeoPoint[]>([]);
  const [pose, setPose] = useState<Pose | null>(null);
  const [trajectory, setTrajectory] = useState<Pose[]>([]);
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [stats, setStats] = useState({ points: 0, high: 0, medium: 0, low: 0 });
  const [colorMode, setColorMode] = useState<ColorMode>("confidence");
  const [measuring, setMeasuring] = useState(false);
  const [picks, setPicks] = useState<GeoPoint[]>([]);
  const [metric, setMetric] = useState<MeasureResult | null>(null);
  const [reference, setReference] = useState<Reference | null>(null);
  const [origin, setOrigin] = useState<{ lat: number; lon: number; alt: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [adapters, setAdapters] = useState<AdapterState>({ vggt: false, colmap: false });
  const [reconMode, setReconMode] = useState<string>("cpu");
  const [timeline, setTimeline] = useState<{ t: number; score: number; keep: boolean }[]>([]);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [apiBase, setApiBase] = useState("http://127.0.0.1:8765");
  const [cameras, setCameras] = useState<ReconCamera[]>([]);
  const [stages, setStages] = useState<Record<string, StageView>>({});
  const [mesh, setMesh] = useState<MeshPayload | null>(null);
  const pointsReceivedRef = useRef(0);

  const snapshotSetters = useMemo(
    () => ({
      setMessage,
      setProgress,
      setPoints,
      setPose,
      setTrajectory,
      setStats,
      setChallenges,
      setOrigin,
      setReference,
      setAdapters,
      setReconMode,
      setMetric,
      setState,
      setError,
      setCameras,
      setStages,
    }),
    [],
  );

  const refreshHealth = useCallback(() => {
    checkHealth()
      .then((h) => {
        setApiOk(Boolean(h.ok));
        setApiBase(getApiBase());
        if (h?.vggt || h?.colmap != null) {
          setAdapters({ vggt: (h.vggt as AdapterState["vggt"]) ?? false, colmap: Boolean(h.colmap) });
        }
      })
      .catch(() => {
        setApiOk(false);
        setError("Cannot reach API on port 8765. Start uvicorn in the backend folder.");
      });
  }, []);

  useEffect(() => {
    fetchReference().then(setReference).catch(() => undefined);
    refreshHealth();
  }, [refreshHealth]);

  const onPick = useCallback((pt: GeoPoint) => {
    setPicks((prev) => {
      if (!measuring) return prev;
      const next = prev.length >= 2 ? [pt] : [...prev, pt];
      return next.slice(-2);
    });
  }, [measuring]);

  useEffect(() => {
    if (picks.length !== 2 || !origin) return;
    measurePoints(origin, picks[0], picks[1], reference?.true_length_m).then(setMetric).catch((e) => setError(String(e)));
  }, [picks, origin, reference]);

  useEffect(() => {
    if (!jobId) return;
    pointsReceivedRef.current = 0;
    const base = getApiBase();
    setApiBase(base || "/api");
    const es = new EventSource(apiUrl(`/jobs/${jobId}/events`));

    const handleEvent = (data: Record<string, unknown>) => {
      if (data.type === "end") return;
      if (data.type === "status") {
        setMessage(String(data.message || ""));
        setProgress(Number(data.progress ?? 8));
      }
      if (data.type === "meta") {
        const meta = data as {
          origin?: { lat: number; lon: number; alt: number };
          reference?: Reference;
          adapters?: AdapterState;
          mode?: string;
          frames?: { timeline?: { t: number; score: number; keep: boolean }[] };
        };
        if (meta.origin) setOrigin(meta.origin);
        if (meta.reference) setReference(meta.reference);
        if (meta.adapters) setAdapters(meta.adapters);
        if (meta.mode) setReconMode(meta.mode);
        if (meta.frames?.timeline) setTimeline(meta.frames.timeline);
      }
      if (data.type === "stage") {
        const stage = data as StageView;
        setStages((prev) => ({ ...prev, [stage.stage]: stage }));
        setMessage(`${stage.label} — ${stage.status}`);
        setProgress(Number(data.progress ?? progress));
      }
      if (data.type === "pose" && data.pose) {
        setPose(data.pose as Pose);
        setTrajectory((t) => t.concat(data.pose as Pose));
      }
      if (data.type === "sparse" || data.type === "dense" || data.type === "chunk") {
        const chunkPts = (data.points as GeoPoint[]) || [];
        if (chunkPts.length) {
          pointsReceivedRef.current += chunkPts.length;
          setPoints((p) => p.concat(chunkPts));
        }
        if (data.pose) {
          setPose(data.pose as Pose);
          setTrajectory((t) => t.concat(data.pose as Pose));
        }
        if (data.stats) setStats(data.stats as typeof stats);
        if (data.challenges) setChallenges(data.challenges as Challenge[]);
        if (data.cameras) setCameras(data.cameras as ReconCamera[]);
        setProgress(Number(data.progress ?? progress));
        if (data.type === "sparse") setMessage(`Sparse cloud · ${data.count ?? ""} points`);
        else if (data.type === "dense") setMessage(`Dense cloud · ${data.count ?? ""} points`);
        else setMessage(`Chunk ${Number(data.index) + 1}/${data.total} fused (${data.source || "geo"})`);
      }
      if (data.type === "done") {
        setState("done");
        setProgress(100);
        setMessage(String(data.message || "Reconstruction complete"));
        const result = data.result as { metric?: MeasureResult } | undefined;
        if (result?.metric) setMetric(result.metric);
        fetch(apiUrl(`/jobs/${jobId}/mesh.json`))
          .then((r) => (r.ok ? r.json() : null))
          .then((m) => { if (m?.positions) setMesh(m as MeshPayload); })
          .catch(() => undefined);
      }
      if (data.type === "error") {
        setState("error");
        setError(String(data.message || "Reconstruction failed"));
        setMessage(String(data.message || "Reconstruction failed"));
      }
    };

    es.onmessage = (ev) => {
      try {
        handleEvent(JSON.parse(ev.data));
      } catch {
        /* ignore malformed */
      }
    };
    es.onerror = async () => {
      try {
        const snap = await fetchJobSnapshot(jobId);
        applySnapshot(snap, snapshotSetters);
        if (snap.points?.length) es.close();
      } catch {
        setError(`Lost live stream from ${base}. Click Run proxy mission again.`);
      }
    };

    const poll = window.setInterval(async () => {
      try {
        const snap = await fetchJobSnapshot(jobId);
        if (snap.status === "done" || snap.status === "error") {
          applySnapshot(snap, snapshotSetters);
          window.clearInterval(poll);
          es.close();
          return;
        }
        if (pointsReceivedRef.current === 0 && snap.points?.length) {
          applySnapshot(snap, snapshotSetters);
        }
      } catch {
        /* still running */
      }
    }, 2000);

    return () => {
      window.clearInterval(poll);
      es.close();
    };
  }, [jobId, progress, snapshotSetters]);

  async function onDemo() {
    setError(null);
    setPoints([]);
    setCameras([]);
    setStages({});
    setMesh(null);
    setTrajectory([]);
    setPicks([]);
    setMetric(null);
    setProgress(4);
    setState("running");
    setMessage("Starting proxy single-pass mission…");
    try {
      refreshHealth();
      const job = await startDemo();
      setJobId(job.id);
    } catch (e) {
      setState("error");
      setError(e instanceof Error ? e.message : "Could not start demo");
    }
  }

  async function onBrighton() {
    setError(null);
    setPoints([]);
    setCameras([]);
    setStages({});
    setMesh(null);
    setTrajectory([]);
    setPicks([]);
    setMetric(null);
    setReference(null);
    setProgress(4);
    setState("running");
    setMessage("Brighton Beach — real DJI survey, syncing GPS and reconstructing…");
    try {
      const job = await startBrighton();
      setJobId(job.id);
    } catch (e) {
      setState("error");
      setError(e instanceof Error ? e.message : "Could not start Brighton Beach");
    }
  }

  async function onUpload() {
    if (!videoFile || !csvFile) {
      setError("Choose a video file and a telemetry CSV.");
      return;
    }
    setError(null);
    setPoints([]);
    setCameras([]);
    setStages({});
    setMesh(null);
    setTrajectory([]);
    setPicks([]);
    setMetric(null);
    setReference(null);
    setProgress(2);
    setState("running");
    setMessage("Uploading clip and synchronizing telemetry…");
    try {
      const job = await startUpload(videoFile, csvFile);
      setJobId(job.id);
    } catch (e) {
      setState("error");
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  }

  const statusBadge = useMemo(() => {
    if (state === "running") return <Badge className="bg-cyan-500/20 text-cyan-200">Reconstructing</Badge>;
    if (state === "done") return <Badge className="bg-emerald-500/20 text-emerald-200">Complete</Badge>;
    if (state === "error") return <Badge variant="destructive">Failed</Badge>;
    return <Badge variant="secondary">Idle</Badge>;
  }, [state]);

  const showEmptyHint = state === "idle" && points.length === 0;

  return (
    <div className="flex min-h-screen flex-col bg-[#070b14] text-slate-100">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-400/80">SIH26158 · NTRO</p>
          <h1 className="text-lg font-semibold tracking-tight">OnePass — single-pass drone video to 3D</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {apiOk === false ? (
            <Badge variant="destructive">API offline — start uvicorn on 8765</Badge>
          ) : apiOk === true ? (
            <Badge className="bg-emerald-500/20 text-emerald-200">API connected</Badge>
          ) : null}
          {statusBadge}
          <Button onClick={onDemo} disabled={state === "running"} className="bg-cyan-500 text-slate-950 hover:bg-cyan-400">
            {state === "running" ? <Loader2 className="animate-spin" /> : <Radar />}
            Run proxy mission
          </Button>
          <Button onClick={onBrighton} disabled={state === "running"} variant="outline">
            Real DJI flight
          </Button>
        </div>
      </header>

      {apiOk === false ? (
        <div className="border-b border-rose-500/30 bg-rose-950/40 px-4 py-2 text-sm text-rose-100">
          Dashboard cannot reach the API at <code className="rounded bg-black/30 px-1">{apiBase}</code>. In a second PowerShell window run:
          <code className="ml-2 rounded bg-black/30 px-2 py-0.5 text-xs">cd backend; $env:PYTHONPATH=&quot;.&quot;; python -m uvicorn app.main:app --host 127.0.0.1 --port 8765</code>
          <Button size="sm" variant="outline" className="ml-3" onClick={refreshHealth}>Retry</Button>
        </div>
      ) : null}

      <main className="grid flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="flex min-h-[520px] flex-col gap-3">
          <div className="relative min-h-[480px] flex-1">
            <ReconViewport points={points} cameras={cameras} mesh={mesh} />
            {showEmptyHint ? (
              <div className="pointer-events-none absolute bottom-3 left-3 right-3 flex justify-center">
                <div className="pointer-events-auto flex max-w-xl items-center gap-3 rounded-xl border border-white/10 bg-black/70 px-4 py-3 shadow-xl">
                  <MapPin className="h-5 w-5 shrink-0 text-amber-400" />
                  <p className="text-sm text-slate-200">
                    GPU viewport. Run the real DJI flight to reconstruct a point cloud, camera frustums, and Poisson mesh.
                  </p>
                  <Button onClick={onBrighton} disabled={state === "running"} className="shrink-0">
                    Start flight
                  </Button>
                </div>
              </div>
            ) : null}
            {state === "running" && points.length === 0 ? (
              <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/70 px-3 py-1.5 text-xs text-cyan-100">
                Reconstructing… points will appear in a few seconds
              </div>
            ) : null}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-400">Color</span>
            {(["confidence", "rgb", "height"] as ColorMode[]).map((m) => (
              <Button key={m} size="sm" variant={colorMode === m ? "default" : "outline"} onClick={() => setColorMode(m)}>
                {m}
              </Button>
            ))}
            <Button size="sm" variant={measuring ? "default" : "outline"} onClick={() => { setMeasuring((v) => !v); setPicks([]); }}>
              <Ruler /> Measure
            </Button>
            <div className="ml-auto flex gap-3 text-xs text-slate-400">
              <span className="text-emerald-400">High {stats.high}</span>
              <span className="text-amber-300">Med {stats.medium}</span>
              <span className="text-rose-400">Low {stats.low}</span>
              <span>{stats.points} points</span>
            </div>
          </div>
        </section>

        <aside className="flex flex-col gap-3">
          <Card className="border-white/10 bg-slate-900/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Mission ingest</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-xs text-slate-400">{message}</p>
              <Progress value={progress} />
              {error ? (
                <p className="flex items-start gap-2 text-xs text-rose-300">
                  <TriangleAlert className="mt-0.5 h-3.5 w-3.5" /> {error}
                </p>
              ) : null}
              <Separator />
              <div className="grid gap-2">
                <Label htmlFor="video">Drone video (1080p/4K)</Label>
                <Input id="video" type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)} />
                <Label htmlFor="csv">Telemetry CSV</Label>
                <Input id="csv" type="file" accept=".csv,text/csv" onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)} />
                <Button variant="outline" onClick={onUpload} disabled={state === "running"}>
                  <Upload /> Reconstruct upload
                </Button>
              </div>
              <p className="text-[11px] text-slate-500">
                API: {apiBase}. CSV columns: timestamp, lat, lon, alt, heading, speed, hdop. Mode: {reconMode}. COLMAP {adapters.colmap ? "on PATH — live SfM fusion" : "not on PATH"}; VGGT {vggtLabel(adapters.vggt)}.
              </p>
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-slate-900/50">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Activity className="h-4 w-4 text-cyan-400" /> Live UAV
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-xs font-mono text-slate-300">
              {pose ? (
                <>
                  <div>lat {pose.lat.toFixed(6)}</div>
                  <div>lon {pose.lon.toFixed(6)}</div>
                  <div>alt {pose.alt.toFixed(1)} m</div>
                  {pose.hdop != null ? <div>HDOP {pose.hdop}</div> : null}
                </>
              ) : (
                <p className="font-sans text-slate-500">Waiting for first pose.</p>
              )}
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-slate-900/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Metric check (judges)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p className="text-xs text-slate-400">
                Known eave on the proxy building is <strong className="text-slate-200">20.0 m</strong>. Click Measure, then the two violet markers.
              </p>
              {metric ? (
                <div className="rounded-md border border-white/10 bg-black/30 p-2 font-mono text-xs">
                  <div>measured {metric.measured_m.toFixed(3)} m</div>
                  {metric.true_length_m != null ? <div>true {metric.true_length_m.toFixed(3)} m</div> : null}
                  {metric.abs_error_m != null ? <div>error {metric.abs_error_m.toFixed(3)} m ({metric.pct_error}%)</div> : null}
                  {metric.pass === true ? <div className="text-emerald-400">PASS within tolerance</div> : null}
                  {metric.pass === false ? <div className="text-rose-400">FAIL — do not overclaim accuracy</div> : null}
                </div>
              ) : (
                <p className="text-xs text-slate-500">No measurement yet.</p>
              )}
              {picks.length > 0 ? <p className="text-[11px] text-slate-500">{picks.length}/2 points picked</p> : null}
              {reference && origin ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    measurePoints(
                      origin,
                      { lat: reference.a.lat, lon: reference.a.lon, height: reference.a.height },
                      { lat: reference.b.lat, lon: reference.b.lon, height: reference.b.height },
                      reference.true_length_m,
                    ).then(setMetric);
                  }}
                >
                  Snap measure 20 m eave
                </Button>
              ) : null}
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-slate-900/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Photogrammetry</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {([
                ["features", "Feature extraction (SIFT)"],
                ["matching", "KNN + RANSAC"],
                ["sfm", "Sparse SfM"],
                ["ba", "Bundle adjustment"],
                ["mvs", "Dense multi-view stereo"],
                ["mesh", "Poisson surface"],
                ["vggt", "VGGT neural (GPU)"],
              ] as const).map(([id, fallback]) => {
                const stage = stages[id];
                return (
                  <div key={id} className="flex items-start justify-between gap-2 text-xs">
                    <div>
                      <div className="text-slate-200">{stage?.label ?? fallback}</div>
                      {stage?.stats ? (
                        <div className="text-slate-500">
                          {Object.entries(stage.stats)
                            .filter(([k, v]) => ["keypoints", "ransac_inliers", "outliers_removed", "cameras", "points", "dense_points", "rmse_after_px", "method", "triangles", "chunks", "device", "reason"].includes(k) && v != null)
                            .map(([k, v]) => `${k} ${v}`)
                            .join(" · ")}
                        </div>
                      ) : null}
                    </div>
                    <Badge variant={stage?.status === "done" ? "default" : "secondary"} className="shrink-0">
                      {stage?.status ?? "idle"}
                    </Badge>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <Card className="border-white/10 bg-slate-900/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Eight challenges</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {challenges.length === 0 ? (
                <p className="text-xs text-slate-500">They populate as the adaptive layer runs.</p>
              ) : (
                challenges.map((c) => (
                  <div key={c.id} className="flex items-start justify-between gap-2 text-xs">
                    <div>
                      <div className="text-slate-200">{c.label}</div>
                      <div className="text-slate-500">{c.response}</div>
                    </div>
                    <Badge variant={c.active ? "default" : "secondary"} className="shrink-0">
                      {c.active ? "on" : "idle"}
                    </Badge>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          {timeline.length > 0 ? (
            <Card className="border-white/10 bg-slate-900/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Frame quality</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex h-12 items-end gap-px">
                  {timeline.map((f, i) => (
                    <div
                      key={i}
                      title={`${f.t.toFixed(2)}s score ${f.score.toFixed(2)}`}
                      className={f.keep ? "bg-cyan-400" : "bg-slate-600"}
                      style={{ width: `${100 / timeline.length}%`, height: `${Math.max(8, f.score * 100)}%` }}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : null}

          {jobId && state === "done" ? (
            <a href={apiUrl(`/jobs/${jobId}/cloud.ply`)} className={buttonVariants({ variant: "outline" })}>
              <Download /> Download PLY
            </a>
          ) : null}
        </aside>
      </main>
    </div>
  );
}
