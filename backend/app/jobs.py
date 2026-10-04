"""In-memory reconstruction jobs with SSE event logs."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

WORKDIR = Path(tempfile.gettempdir()) / "onepass-jobs"
WORKDIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    waiters: list[asyncio.Future] = field(default_factory=list)
    artifact_dir: Path = field(default_factory=Path)
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)


JOBS: dict[str, Job] = {}


def create_job(kind: str) -> Job:
    jid = uuid.uuid4().hex[:12]
    job = Job(id=jid, kind=kind, artifact_dir=WORKDIR / jid)
    job.artifact_dir.mkdir(parents=True, exist_ok=True)
    JOBS[jid] = job
    return job


def get_job(jid: str) -> Job | None:
    job = JOBS.get(jid)
    if job is not None or not re.fullmatch(r"[0-9a-f]{12}", jid):
        return job
    # Finished jobs survive an API restart through their artifact folder.
    folder = WORKDIR / jid
    if not (folder / "cloud.json").exists():
        return None
    result: dict[str, Any] = {}
    try:
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    job = Job(id=jid, kind=str(result.get("kind") or "upload"), status="done", artifact_dir=folder, result=result)
    JOBS[jid] = job
    return job


def _save_result(job: Job) -> None:
    try:
        payload = {**job.result, "kind": job.kind}
        (job.artifact_dir / "result.json").write_text(json.dumps(payload, default=str), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def job_snapshot(job: Job) -> dict[str, Any]:
    """Rebuild dashboard state from stored SSE events or cloud.json."""
    points: list[dict[str, Any]] = []
    meta: dict[str, Any] | None = None
    pose: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    challenges: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    stages: dict[str, dict[str, Any]] = {}
    message = ""
    progress = 0
    result = job.result or None
    scene: dict[str, Any] | None = None
    scene_path = job.artifact_dir / "scene.json"
    if scene_path.exists():
        try:
            raw_scene = json.loads(scene_path.read_text(encoding="utf-8"))
            if isinstance(raw_scene, dict):
                scene = raw_scene
        except (OSError, json.JSONDecodeError):
            pass

    for event in job.events:
        et = event.get("type")
        if et == "meta":
            meta = event
        elif et == "status":
            message = str(event.get("message") or message)
            progress = int(event.get("progress") or progress)
        elif et == "stage":
            stages[str(event.get("stage"))] = event
            message = f"{event.get('label')} — {event.get('status')}"
            progress = int(event.get("progress") or progress)
        elif et == "pose" and event.get("pose"):
            pose = event["pose"]
            trajectory.append(event["pose"])
        elif et in {"chunk", "sparse", "dense"}:
            points.extend(event.get("points") or [])
            if event.get("cameras"):
                cameras = event["cameras"]
            if event.get("pose"):
                pose = event["pose"]
                trajectory.append(event["pose"])
            if event.get("stats"):
                stats = event["stats"]
            if event.get("challenges"):
                challenges = event["challenges"]
            progress = int(event.get("progress") or progress)
        elif et == "stats":
            stats = event.get("stats") or stats
            challenges = event.get("challenges") or challenges
        elif et == "done":
            message = str(event.get("message") or message)
            progress = 100
            result = event.get("result") or result
        elif et == "error":
            message = str(event.get("message") or job.error or message)

    if job.status == "error":
        for key, stage in list(stages.items()):
            if stage.get("status") == "running":
                stages[key] = {**stage, "status": "failed"}
        if job.error:
            message = "Reconstruction failed"

    cloud_path = job.artifact_dir / "cloud.json"
    if cloud_path.exists():
        try:
            points = json.loads(cloud_path.read_text(encoding="utf-8")).get("points") or points
        except json.JSONDecodeError:
            pass
    cameras_path = job.artifact_dir / "cameras.json"
    if not cameras and cameras_path.exists():
        try:
            cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if job.status == "done" and not stages and result:
        mesh = result.get("mesh") or {}
        stages["mesh"] = {
            "type": "stage",
            "stage": "mesh",
            "label": "Surface mesh",
            "status": "done" if mesh.get("status") == "ok" else "skipped",
            "stats": {k: v for k, v in mesh.items() if k != "path"},
        }
        message = message or "Reconstruction complete"
        progress = 100

    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "error": job.error,
        "message": message,
        "progress": progress,
        "origin": (meta or {}).get("origin") or (result or {}).get("origin"),
        "reference": (meta or {}).get("reference"),
        "adapters": (meta or {}).get("adapters") or (result or {}).get("adapters"),
        "mode": (meta or {}).get("mode") or (result or {}).get("mode"),
        "points": points,
        "pose": pose,
        "trajectory": trajectory,
        "stats": stats or {
            "points": len(points),
            "high": sum(1 for p in points if (p.get("conf") or 0) >= 0.72),
            "medium": sum(1 for p in points if 0.42 <= (p.get("conf") or 0) < 0.72),
            "low": sum(1 for p in points if (p.get("conf") or 0) < 0.42),
        },
        "challenges": challenges,
        "cameras": cameras,
        "stages": stages,
        "result": result,
        "scene": scene or (result or {}).get("scene"),
    }


async def emit(job: Job, event: dict[str, Any]) -> None:
    job.events.append(event)
    if event.get("type") == "done":
        _save_result(job)
    pending = job.waiters
    job.waiters = []
    for fut in pending:
        if not fut.done():
            fut.set_result(event)


async def wait_event(job: Job, after: int) -> dict[str, Any] | None:
    if after < len(job.events):
        return job.events[after]
    if job.status in {"done", "error"}:
        return None
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    job.waiters.append(fut)
    try:
        return await asyncio.wait_for(fut, timeout=25.0)
    except TimeoutError:
        return {"type": "ping"}


async def sse_stream(job: Job) -> AsyncIterator[str]:
    idx = 0
    while True:
        event = await wait_event(job, idx)
        if event is None:
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            break
        if event.get("type") != "ping" or idx >= len(job.events):
            if event.get("type") != "ping":
                idx += 1
        yield f"data: {json.dumps(event)}\n\n"
        if event.get("type") in {"done", "error"}:
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            break
