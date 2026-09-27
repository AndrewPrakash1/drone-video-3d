"""In-memory reconstruction jobs with SSE event logs."""

from __future__ import annotations

import asyncio
import json
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
    return JOBS.get(jid)


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

    cloud_path = job.artifact_dir / "cloud.json"
    if cloud_path.exists():
        try:
            points = json.loads(cloud_path.read_text(encoding="utf-8")).get("points") or points
        except json.JSONDecodeError:
            pass

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
    }


async def emit(job: Job, event: dict[str, Any]) -> None:
    job.events.append(event)
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
