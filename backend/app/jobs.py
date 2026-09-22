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
