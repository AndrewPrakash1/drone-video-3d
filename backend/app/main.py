from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .adapters.colmap import colmap_available
from .adapters.vggt import vggt_available, vggt_status
from .demo_scene import write_demo_files
from .geo import Origin, geodetic_to_enu, make_origin
from .jobs import create_job, get_job, job_snapshot, sse_stream
from .metric import evaluate
from .osm_buildings import fetch_buildings
from .pipeline import run_demo, run_upload

ROOT = Path(__file__).resolve().parents[2]
DATA_DEMO = ROOT / "data" / "demo"

app = FastAPI(title="OnePass", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    write_demo_files(DATA_DEMO)


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "vggt": vggt_status(),
        "colmap": colmap_available(),
    }


@app.get("/osm/buildings")
def osm_buildings(lat: float = 28.5448, lon: float = 77.1924, radius_m: float = 1100.0) -> dict:
    try:
        buildings = fetch_buildings(lat, lon, radius_m)
    except Exception as exc:
        raise HTTPException(502, f"OSM buildings unavailable: {exc}") from exc
    return {"lat": lat, "lon": lon, "count": len(buildings), "buildings": buildings}


@app.get("/demo/reference")
def demo_reference() -> dict:
    path = DATA_DEMO / "reference.json"
    if not path.exists():
        write_demo_files(DATA_DEMO)
    import json

    return json.loads(path.read_text())


@app.post("/jobs/demo")
async def start_demo() -> dict:
    job = create_job("demo")
    asyncio.create_task(run_demo(job, DATA_DEMO))
    return {"id": job.id, "kind": job.kind}


@app.post("/jobs")
async def start_upload(
    video: UploadFile = File(...),
    telemetry: UploadFile | None = File(None),
    telemetry_text: str | None = Form(None),
) -> dict:
    job = create_job("upload")
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    video_path = job.artifact_dir / f"input{suffix}"
    video_path.write_bytes(await video.read())
    text = telemetry_text or ""
    if telemetry is not None:
        text = (await telemetry.read()).decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(400, "Provide telemetry CSV (file or telemetry_text)")
    (job.artifact_dir / "telemetry.csv").write_text(text, encoding="utf-8")
    asyncio.create_task(run_upload(job, video_path, text))
    return {"id": job.id, "kind": job.kind}


@app.get("/jobs/{jid}")
def job_status(jid: str) -> dict:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "error": job.error,
        "result": job.result,
        "events": len(job.events),
    }


@app.get("/jobs/{jid}/snapshot")
def job_snapshot_route(jid: str) -> dict:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    return job_snapshot(job)


@app.get("/jobs/{jid}/events")
async def job_events(jid: str) -> StreamingResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    return StreamingResponse(sse_stream(job), media_type="text/event-stream")


@app.get("/jobs/{jid}/cloud.ply")
def job_cloud(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "cloud.ply"
    if not path.exists():
        raise HTTPException(404, "cloud not ready")
    return FileResponse(path, filename="cloud.ply")


@app.get("/jobs/{jid}/mesh.ply")
def job_mesh(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "mesh.ply"
    if not path.exists():
        raise HTTPException(404, "mesh not ready")
    return FileResponse(path, filename="mesh.ply")


@app.get("/jobs/{jid}/trajectory.geojson")
def job_traj(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "trajectory.geojson"
    if not path.exists():
        raise HTTPException(404, "trajectory not ready")
    return FileResponse(path, filename="trajectory.geojson")


@app.post("/metric/measure")
async def metric_measure(body: dict) -> dict:
    try:
        a = body["a"]
        b = body["b"]
        origin = body["origin"]
        true_length_m = float(body.get("true_length_m") or 0)
        o: Origin = make_origin(float(origin["lat"]), float(origin["lon"]), float(origin.get("alt", 0)))
        ae = geodetic_to_enu(float(a["lat"]), float(a["lon"]), float(a.get("height", 0)), o)
        be = geodetic_to_enu(float(b["lat"]), float(b["lon"]), float(b.get("height", 0)), o)
        measured = float(((ae - be) ** 2).sum() ** 0.5)
        if true_length_m <= 0:
            return {"measured_m": round(measured, 3), "pass": None}
        return evaluate(
            measured,
            true_length_m,
            body.get("tolerance_m"),
            float(body.get("tolerance_pct") or 5.0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
