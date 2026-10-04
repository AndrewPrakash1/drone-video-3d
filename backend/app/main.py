from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .adapters.colmap import colmap_available
from .adapters.vggt import vggt_available, vggt_status
from .splats.train import splat_status
from .demo_scene import write_demo_files
from .geo import Origin, geodetic_to_enu, make_origin
from .jobs import Job, create_job, emit, get_job, job_snapshot, sse_stream
from .metric import evaluate
from .osm_buildings import fetch_buildings
from .pipeline import run_demo, run_upload
from .scene import read_scene_manifest
from .spatial import load_mesh, normalize_blocks, plan_route, write_spatial
from .spatial_model import spatial_model_status

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


def _spawn(job: Job, coro: Coroutine) -> None:
    """Run a job coroutine in the background and surface any unhandled error.

    Without this, an exception outside the runner's own try/except is dropped by
    the event loop and leaves job.status stuck at "running" forever.
    """

    async def _runner() -> None:
        try:
            await coro
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            await emit(job, {"type": "error", "message": job.error})

    asyncio.create_task(_runner())


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "vggt": vggt_status(),
        "colmap": colmap_available(),
        "splat": splat_status(),
        "spatial_model": spatial_model_status(),
    }


@app.get("/osm/buildings")
def osm_buildings(lat: float = 28.5448, lon: float = 77.1924, radius_m: float = 1100.0) -> dict:
    try:
        buildings = fetch_buildings(lat, lon, radius_m)
    except Exception:
        buildings = []
    return {"lat": lat, "lon": lon, "count": len(buildings), "buildings": buildings}


@app.get("/demo/reference")
def demo_reference() -> dict:
    path = DATA_DEMO / "reference.json"
    if not path.exists():
        write_demo_files(DATA_DEMO)
    import json

    return json.loads(path.read_text())


BRIGHTON = ROOT / "data" / "brighton"


@app.post("/jobs/brighton")
async def start_brighton() -> dict:
    video = BRIGHTON / "flight.mp4"
    telem = BRIGHTON / "telemetry.csv"
    if not video.exists() or not telem.exists():
        raise HTTPException(404, "Brighton Beach flight is not in data/brighton")
    job = create_job("brighton")
    _spawn(job, run_upload(job, video, telem.read_text(encoding="utf-8")))
    return {"id": job.id, "kind": "brighton", "dataset": "OpenDroneMap Brighton Beach"}


@app.post("/jobs/demo")
async def start_demo() -> dict:
    job = create_job("demo")
    _spawn(job, run_demo(job, DATA_DEMO))
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
    _spawn(job, run_upload(job, video_path, text))
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


@app.get("/jobs/{jid}/cloud.json")
def job_cloud_json(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "cloud.json"
    if not path.exists():
        raise HTTPException(404, "cloud not ready")
    return FileResponse(path, media_type="application/json")


@app.get("/jobs/{jid}/scene.json")
def job_scene(jid: str) -> dict:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    scene = read_scene_manifest(job.artifact_dir)
    if scene is None:
        raise HTTPException(404, "scene not ready")
    return scene


@app.get("/jobs/{jid}/spatial.json")
def job_spatial(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "spatial.json"
    if not path.exists():
        mesh = job.artifact_dir / "mesh.ply"
        cameras_path = job.artifact_dir / "cameras.json"
        if not mesh.exists() or not cameras_path.exists():
            raise HTTPException(404, "spatial not ready")
        cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
        write_spatial(mesh, cameras, path)
    return FileResponse(path, media_type="application/json")


@app.get("/jobs/{jid}/chisel.json")
def job_chisel(jid: str) -> dict:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "chisel.json"
    if not path.exists():
        return {"blocks": []}
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/jobs/{jid}/chisel")
def job_chisel_save(jid: str, body: dict) -> dict:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    try:
        blocks = normalize_blocks(body.get("blocks") or [])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    payload = {"blocks": blocks}
    (job.artifact_dir / "chisel.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


@app.post("/jobs/{jid}/route")
def job_route(jid: str, body: dict) -> dict:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    mesh = job.artifact_dir / "mesh.ply"
    if not mesh.exists():
        raise HTTPException(404, "mesh not ready")
    try:
        start = [float(v) for v in body["start"]]
        goal = [float(v) for v in body["goal"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, "start and goal must be ENU triples") from exc
    if len(start) != 3 or len(goal) != 3:
        raise HTTPException(400, "start and goal must be ENU triples")
    if "blocks" in body:
        try:
            blocks = normalize_blocks(body.get("blocks") or [])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        (job.artifact_dir / "chisel.json").write_text(json.dumps({"blocks": blocks}), encoding="utf-8")
    else:
        chisel_path = job.artifact_dir / "chisel.json"
        blocks = json.loads(chisel_path.read_text(encoding="utf-8")).get("blocks") or [] if chisel_path.exists() else []
    verts, faces = load_mesh(mesh)
    seen = None
    faultlines: list = []
    spatial_path = job.artifact_dir / "spatial.json"
    if spatial_path.exists():
        payload = json.loads(spatial_path.read_text(encoding="utf-8"))
        raw_seen = payload.get("seen") or []
        if len(raw_seen) == len(faces):
            seen = np.asarray(raw_seen, dtype=np.int32)
        faultlines = payload.get("faultlines") or []
    return plan_route(verts, faces, seen, start, goal, blocks, faultlines)


@app.get("/jobs/{jid}/spatial_model.json")
def job_spatial_model(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "spatial_model.json"
    if not path.exists():
        raise HTTPException(404, "spatial model status not ready")
    return FileResponse(path, media_type="application/json")


@app.get("/jobs/{jid}/mesh.json")
def job_mesh_json(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "mesh.json"
    if not path.exists():
        raise HTTPException(404, "mesh not ready")
    return FileResponse(path, media_type="application/json")


@app.get("/jobs/{jid}/splat.ply")
def job_splat(jid: str) -> FileResponse:
    job = get_job(jid)
    if not job:
        raise HTTPException(404, "unknown job")
    path = job.artifact_dir / "splat.ply"
    if not path.exists():
        raise HTTPException(404, "splat not ready")
    return FileResponse(path, filename="splat.ply", media_type="application/octet-stream")


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
