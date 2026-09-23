"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
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
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  fetchReference,
  measurePoints,
  startDemo,
  startUpload,
  type Challenge,
  type GeoPoint,
  type MeasureResult,
  type Reference,
} from "@/lib/api";
import type { ColorMode } from "@/components/cesium-globe";

const CesiumGlobe = dynamic(
  () => import("@/components/cesium-globe").then((m) => m.CesiumGlobe),
  { ssr: false, loading: () => <div className="flex h-full items-center justify-center text-sm text-muted-foreground">Loading globe…</div> },
);

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
type JobState = "idle" | "running" | "done" | "error";

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

  useEffect(() => {
    fetchReference().then(setReference).catch(() => undefined);
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        if (h?.vggt || h?.colmap != null) {
          setAdapters({ vggt: h.vggt ?? false, colmap: Boolean(h.colmap) });
        }
      })
      .catch(() => undefined);
  }, []);

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
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    es.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === "end") {
        es.close();
        return;
      }
      if (data.type === "status") {
        setMessage(data.message);
        setProgress(data.progress ?? 8);
      }
      if (data.type === "meta") {
        setOrigin(data.origin);
        if (data.reference) setReference(data.reference);
        if (data.adapters) setAdapters(data.adapters);
        if (data.mode) setReconMode(data.mode);
        if (data.frames?.timeline) setTimeline(data.frames.timeline);
      }
      if (data.type === "chunk") {
        setProgress(data.progress ?? progress);
        setPoints((p) => p.concat(data.points || []));
        if (data.pose) {
          setPose(data.pose);
          setTrajectory((t) => t.concat(data.pose));
        }
        if (data.stats) setStats(data.stats);
        if (data.challenges) setChallenges(data.challenges);
        setMessage(`Chunk ${data.index + 1}/${data.total} fused (${data.source || "geo"})`);
      }
      if (data.type === "done") {
        setState("done");
        setProgress(100);
        setMessage(data.message);
        if (data.result?.metric) setMetric(data.result.metric);
      }
      if (data.type === "error") {
        setState("error");
        setError(data.message);
        setMessage(data.message);
      }
    };
    es.onerror = () => {
      /* keep open; proxy may stall briefly */
    };
    return () => es.close();
  }, [jobId]);

  async function onDemo() {
    setError(null);
    setPoints([]);
    setTrajectory([]);
    setPicks([]);
    setMetric(null);
    setProgress(2);
    setState("running");
    setMessage("Starting proxy single-pass mission…");
    try {
      const job = await startDemo();
      setJobId(job.id);
    } catch (e) {
      setState("error");
      setError(e instanceof Error ? e.message : "Could not start demo");
    }
  }

  async function onUpload() {
    if (!videoFile || !csvFile) {
      setError("Choose a video file and a telemetry CSV.");
      return;
    }
    setError(null);
    setPoints([]);
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

  return (
    <div className="flex min-h-screen flex-col bg-[#070b14] text-slate-100">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.18em] text-cyan-400/80">SIH26158 · NTRO</p>
          <h1 className="text-lg font-semibold tracking-tight">OnePass — single-pass drone video to 3D</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {statusBadge}
          <Button onClick={onDemo} disabled={state === "running"} className="bg-cyan-500 text-slate-950 hover:bg-cyan-400">
            {state === "running" ? <Loader2 className="animate-spin" /> : <Radar />}
            Run proxy mission
          </Button>
        </div>
      </header>

      <main className="grid flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-[minmax(0,1fr)_340px]">
        <section className="flex min-h-[520px] flex-col gap-3">
          <div className="min-h-[480px] flex-1">
            {state === "idle" && points.length === 0 ? (
              <Card className="flex h-full min-h-[480px] items-center justify-center border-white/10 bg-slate-900/40">
                <CardContent className="max-w-md space-y-3 text-center">
                  <MapPin className="mx-auto h-10 w-10 text-cyan-400" />
                  <h2 className="text-xl font-medium">No reconstruction yet</h2>
                  <p className="text-sm text-slate-400">
                    Event data arrives only at the hackathon. Run the South Delhi proxy flyby to reconstruct a building with a
                    known 20.0 m rooftop, then measure it. Or upload your own 1080p/4K clip plus GPS CSV.
                  </p>
                  <Button onClick={onDemo}>Start proxy flyby</Button>
                </CardContent>
              </Card>
            ) : (
              <CesiumGlobe
                points={points}
                pose={pose}
                trajectory={trajectory}
                colorMode={colorMode}
                measuring={measuring}
                reference={state !== "idle" ? reference : null}
                onPick={onPick}
              />
            )}
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
                CSV columns: timestamp, lat, lon, alt, heading, speed, hdop. Mode: {reconMode}. COLMAP {adapters.colmap ? "on PATH — live SfM fusion" : "not on PATH"}; VGGT {vggtLabel(adapters.vggt)}.
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
            <Button variant="outline" asChild>
              <a href={`/api/jobs/${jobId}/cloud.ply`}>
                <Download /> Download PLY
              </a>
            </Button>
          ) : null}
        </aside>
      </main>
    </div>
  );
}
