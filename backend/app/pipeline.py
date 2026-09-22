"""Job runners: demo playback and uploaded video reconstruction."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np

from .adapters.colmap import colmap_available, run_sfm
from .adapters.vggt import vggt_available
from .confidence import classify
from .demo_scene import (
    ORIGIN_ALT,
    chunk_cloud,
    generate_cloud,
    generate_telemetry,
    origin,
    reference_segment,
    write_demo_files,
)
from .frames import ScoredFrame, select_frames
from .geo import Origin, enu_to_geodetic, make_origin
from .jobs import Job, emit
from .mesh import mesh_from_points, write_ply
from .metric import evaluate_segment
from .reconstruct import ReconPoint, triangulate_pair, voxel_downsample
from .telemetry import TelemetryTrack, parse_telemetry_csv


def _point_payload(p: ReconPoint, o: Origin, _ellipsoidal_origin_alt: float = 0.0) -> dict:
    lat, lon, alt = enu_to_geodetic(p.e, p.n, p.u, o)
    return {
        "lat": lat,
        "lon": lon,
        "height": alt,
        "e": p.e,
        "n": p.n,
        "u": p.u,
        "r": p.r,
        "g": p.g,
        "b": p.b,
        "conf": p.conf,
        "band": classify(p.conf),
    }


def _challenge_state(
    *,
    blur_dropped: int,
    sparse: bool,
    gps_downweighted: bool,
    illumination: int,
    dynamic: bool,
    occlusions: bool,
    near_rt: bool,
    metric: bool,
) -> list[dict]:
    return [
        {"id": "angles", "label": "Limited viewing angles", "active": sparse, "response": "AI/nadir path on sparse overlap"},
        {"id": "blur", "label": "Motion blur / compression", "active": blur_dropped > 0, "response": f"Dropped {blur_dropped} blurred frames"},
        {"id": "illum", "label": "Variable illumination", "active": illumination > 0, "response": f"Gated {illumination} washed/dark frames"},
        {"id": "dynamic", "label": "Dynamic objects", "active": dynamic, "response": "Short-lifetime tracks filtered"},
        {"id": "gps", "label": "GPS noise", "active": gps_downweighted, "response": "Telemetry weight reduced"},
        {"id": "realtime", "label": "Near-real-time", "active": near_rt, "response": "Chunked 2–5 fps updates"},
        {"id": "occlude", "label": "Occlusions", "active": occlusions, "response": "Low-confidence courtyard / far terrain"},
        {"id": "metric", "label": "Metric without GCPs", "active": metric, "response": "GPS-constrained scale + reference length check"},
    ]


async def run_demo(job: Job, data_root: Path) -> None:
    job.status = "running"
    write_demo_files(data_root)
    o = origin()
    cloud = generate_cloud()
    chunks = chunk_cloud(cloud, chunks=9)
    telem = generate_telemetry()
    ref = evaluate_segment(reference_segment())
    adapters = {"vggt": vggt_available(), "colmap": colmap_available()}

    await emit(job, {"type": "status", "message": "Synchronizing proxy video timestamps with GPS/IMU", "progress": 4})
    await emit(
        job,
        {
            "type": "meta",
            "origin": {"lat": o.lat, "lon": o.lon, "alt": ORIGIN_ALT},
            "reference": json.loads((data_root / "reference.json").read_text()),
            "adapters": adapters,
            "mode": "hybrid-demo",
        },
    )

    fused: list[ReconPoint] = []
    n_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        fused.extend(chunk)
        fused = voxel_downsample(fused, voxel=0.7)
        sample = telem[min(int((i + 1) / n_chunks * (len(telem) - 1)), len(telem) - 1)]
        gps_noisy = sample["hdop"] > 2.5
        payload_pts = [_point_payload(p, o, ORIGIN_ALT) for p in chunk]
        await emit(
            job,
            {
                "type": "chunk",
                "index": i,
                "total": n_chunks,
                "progress": int(12 + 80 * (i + 1) / n_chunks),
                "pose": {
                    "lat": sample["lat"],
                    "lon": sample["lon"],
                    "alt": sample["alt"],
                    "heading": sample["heading"],
                    "hdop": sample["hdop"],
                },
                "points": payload_pts,
                "stats": {
                    "points": len(fused),
                    "high": sum(1 for p in fused if p.conf >= 0.72),
                    "medium": sum(1 for p in fused if 0.42 <= p.conf < 0.72),
                    "low": sum(1 for p in fused if p.conf < 0.42),
                },
                "challenges": _challenge_state(
                    blur_dropped=7,
                    sparse=True,
                    gps_downweighted=gps_noisy,
                    illumination=3,
                    dynamic=True,
                    occlusions=True,
                    near_rt=True,
                    metric=True,
                ),
            },
        )
        await asyncio.sleep(0.85)

    write_ply(job.artifact_dir / "cloud.ply", fused)
    mesh_info = mesh_from_points(fused, job.artifact_dir / "mesh.ply")
    traj = [
        {"lat": r["lat"], "lon": r["lon"], "alt": r["alt"]}
        for r in telem
    ]
    (job.artifact_dir / "trajectory.geojson").write_text(
        json.dumps(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"], p["alt"]] for p in traj]},
                "properties": {"name": "uav"},
            }
        ),
        encoding="utf-8",
    )
    job.result = {
        "points": len(fused),
        "mesh": mesh_info,
        "metric": ref,
        "adapters": adapters,
        "origin": {"lat": o.lat, "lon": o.lon, "alt": ORIGIN_ALT},
    }
    job.status = "done"
    await emit(
        job,
        {
            "type": "done",
            "progress": 100,
            "message": "Proxy reconstruction complete. Measure the 20 m rooftop eave.",
            "result": job.result,
        },
    )


def _read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


async def run_upload(job: Job, video_path: Path, telemetry_text: str) -> None:
    job.status = "running"
    try:
        track = parse_telemetry_csv(telemetry_text)
    except ValueError as exc:
        job.status = "error"
        job.error = str(exc)
        await emit(job, {"type": "error", "message": str(exc)})
        return

    await emit(
        job,
        {
            "type": "status",
            "message": f"Telemetry reliability {track.reliability:.2f} — extracting geometrically useful frames",
            "progress": 6,
        },
    )

    try:
        scored = select_frames(str(video_path))
    except ValueError as exc:
        job.status = "error"
        job.error = str(exc)
        await emit(job, {"type": "error", "message": str(exc)})
        return

    kept = [f for f in scored if f.keep]
    blur_dropped = sum(1 for f in scored if "motion_blur" in f.reasons)
    illum = sum(1 for f in scored if "illumination" in f.reasons)
    first = track.samples[0]
    o = make_origin(first.lat, first.lon, 0.0)
    origin_alt = first.alt

    await emit(
        job,
        {
            "type": "meta",
            "origin": {"lat": o.lat, "lon": o.lon, "alt": origin_alt},
            "reference": None,
            "adapters": {"vggt": vggt_available(), "colmap": colmap_available()},
            "mode": "cpu" if not vggt_available() else "hybrid",
            "frames": {
                "scanned": len(scored),
                "kept": len(kept),
                "timeline": [
                    {
                        "t": f.t,
                        "score": f.score,
                        "keep": f.keep,
                        "sharpness": f.sharpness,
                        "features": f.features,
                        "reasons": f.reasons,
                    }
                    for f in scored
                ],
            },
            "telemetry_reliability": track.reliability,
        },
    )

    cap = cv2.VideoCapture(str(video_path))
    fused: list[ReconPoint] = []
    gps_down = track.reliability < 0.55

    pairs = list(zip(kept, kept[1:])) or [(kept[0], kept[0])] if kept else []
    if not kept:
        job.status = "error"
        job.error = "No usable frames after quality gating"
        await emit(job, {"type": "error", "message": job.error})
        cap.release()
        return

    n_pairs = max(len(pairs), 1)
    for i, (fa, fb) in enumerate(pairs):
        img_a = _read_frame(cap, fa.index)
        img_b = _read_frame(cap, fb.index)
        if img_a is None:
            continue
        if img_b is None:
            img_b = img_a
        pose_a = track.interpolate(fa.t)
        pose_b = track.interpolate(fb.t)
        chunk = triangulate_pair(img_a, img_b, pose_a, pose_b, o, track.reliability)
        fused.extend(chunk)
        fused = voxel_downsample(fused, voxel=0.9)
        sample = pose_b
        await emit(
            job,
            {
                "type": "chunk",
                "index": i,
                "total": n_pairs,
                "progress": int(10 + 80 * (i + 1) / n_pairs),
                "pose": {
                    "lat": sample.lat,
                    "lon": sample.lon,
                    "alt": sample.alt,
                    "heading": sample.heading,
                    "hdop": sample.hdop,
                },
                "points": [_point_payload(p, o, 0.0) for p in chunk[:450]],
                "stats": {
                    "points": len(fused),
                    "high": sum(1 for p in fused if p.conf >= 0.72),
                    "medium": sum(1 for p in fused if 0.42 <= p.conf < 0.72),
                    "low": sum(1 for p in fused if p.conf < 0.42),
                },
                "challenges": _challenge_state(
                    blur_dropped=blur_dropped,
                    sparse=len(chunk) < 80,
                    gps_downweighted=gps_down,
                    illumination=illum,
                    dynamic=True,
                    occlusions=True,
                    near_rt=True,
                    metric=True,
                ),
            },
        )
        await asyncio.sleep(0.05)

    cap.release()

    # Optional COLMAP on dumped keyframes
    colmap_info = None
    if colmap_available() and kept:
        img_dir = job.artifact_dir / "keyframes"
        img_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        for f in kept[:40]:
            fr = _read_frame(cap, f.index)
            if fr is not None:
                cv2.imwrite(str(img_dir / f"frame_{f.index:06d}.jpg"), fr)
        cap.release()
        colmap_info = run_sfm(img_dir, job.artifact_dir / "colmap")

    write_ply(job.artifact_dir / "cloud.ply", fused)
    mesh_info = mesh_from_points(fused, job.artifact_dir / "mesh.ply")
    traj = [{"lat": s.lat, "lon": s.lon, "alt": s.alt} for s in track.samples]
    (job.artifact_dir / "trajectory.geojson").write_text(
        json.dumps(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"], p["alt"]] for p in traj]},
            }
        ),
        encoding="utf-8",
    )
    job.result = {
        "points": len(fused),
        "mesh": mesh_info,
        "colmap": colmap_info,
        "adapters": {"vggt": vggt_available(), "colmap": colmap_available()},
        "telemetry_reliability": track.reliability,
        "frames_kept": len(kept),
        "origin": {"lat": o.lat, "lon": o.lon, "alt": origin_alt},
    }
    job.status = "done"
    await emit(
        job,
        {
            "type": "done",
            "progress": 100,
            "message": "Upload reconstruction finished. Measure a known length if you have a reference.",
            "result": job.result,
        },
    )
