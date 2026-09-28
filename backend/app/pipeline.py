"""Job runners: demo playback and uploaded video reconstruction."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .adapters.colmap import colmap_available, reconstruct_aligned
from .adapters.vggt import reconstruct_chunk as vggt_reconstruct_chunk
from .adapters.vggt import vggt_available, vggt_status
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
from .frames import select_frames
from .geo import Origin, enu_to_geodetic, make_origin
from .jobs import Job, emit
from .mesh import mesh_from_points, write_ply
from .metric import evaluate_segment
from .photogrammetry.run import run_photogrammetry
from .reconstruct import ReconPoint, triangulate_pair, voxel_downsample
from .splats.train import splat_status, train_splats
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


def _stats(fused: list[ReconPoint]) -> dict:
    return {
        "points": len(fused),
        "high": sum(1 for p in fused if p.conf >= 0.72),
        "medium": sum(1 for p in fused if 0.42 <= p.conf < 0.72),
        "low": sum(1 for p in fused if p.conf < 0.42),
    }


def _recon_mode() -> str:
    if vggt_available() and colmap_available():
        return "hybrid-vggt-colmap"
    if vggt_available():
        return "vggt"
    if colmap_available():
        return "cpu-colmap"
    return "cpu"



async def run_demo(job: Job, data_root: Path) -> None:
    job.status = "running"
    write_demo_files(data_root)
    o = origin()
    cloud = generate_cloud()
    chunks = chunk_cloud(cloud, chunks=9)
    telem = generate_telemetry()
    ref = evaluate_segment(reference_segment())
    adapters = {"vggt": vggt_status(), "colmap": colmap_available()}

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
    (job.artifact_dir / "cloud.json").write_text(
        json.dumps({"points": [_point_payload(p, o, ORIGIN_ALT) for p in fused]}),
        encoding="utf-8",
    )
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

    await emit(job, {"type": "status", "message": f"Telemetry reliability {track.reliability:.2f} — selecting frames", "progress": 3})
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
    gps_down = track.reliability < 0.55
    use_vggt = vggt_available()
    mode = "photogrammetry" + ("+vggt" if use_vggt else "")

    await emit(job, {
        "type": "meta",
        "origin": {"lat": o.lat, "lon": o.lon, "alt": origin_alt},
        "reference": None,
        "adapters": {"vggt": vggt_status(), "colmap": colmap_available()},
        "mode": mode,
        "frames": {"scanned": len(scored), "kept": len(kept), "timeline": [{"t": f.t, "score": f.score, "keep": f.keep} for f in scored]},
        "telemetry_reliability": track.reliability,
    })
    if not kept:
        job.status = "error"
        job.error = "No usable frames after quality gating"
        await emit(job, {"type": "error", "message": job.error})
        return

    max_frames = max(6, int(os.environ.get("ONEPASS_MAX_FRAMES", "24")))
    if len(kept) > max_frames:
        idx = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
        kept = [kept[i] for i in sorted(set(idx.tolist()))]

    cap = cv2.VideoCapture(str(video_path))
    max_w = int(os.environ.get("ONEPASS_MAX_WIDTH", "1280"))
    frames = []
    samples = []
    for fr in kept:
        img = _read_frame(cap, fr.index)
        if img is None:
            continue
        if img.shape[1] > max_w:
            s = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        frames.append((fr.index, fr.t, img))
        samples.append(track.interpolate(fr.t))
    cap.release()
    if len(frames) < 2:
        job.status = "error"
        job.error = "Need at least two readable frames"
        await emit(job, {"type": "error", "message": job.error})
        return

    for i, sample in enumerate(samples):
        await emit(job, {"type": "pose", "index": i, "pose": {"lat": sample.lat, "lon": sample.lon, "alt": sample.alt, "heading": sample.heading, "hdop": sample.hdop}})

    try:
        pg = await run_photogrammetry(job, frames, samples, o, track.reliability, progress_base=8)
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        await emit(job, {"type": "error", "message": str(exc)})
        return
    if pg.get("status") != "ok":
        job.status = "error"
        job.error = pg.get("reason", "photogrammetry failed")
        await emit(job, {"type": "error", "message": job.error})
        return
    fused: list[ReconPoint] = list(pg["fused"])

    status = vggt_status()
    vggt_points = 0
    if use_vggt:
        chunk = max(2, int(os.environ.get("VGGT_CHUNK_FRAMES", "3")))
        windows = [frames[i : i + chunk] for i in range(0, len(frames), max(chunk - 1, 1))]
        windows = [w for w in windows if len(w) >= 2]
        await emit(job, {
            "type": "stage",
            "stage": "vggt",
            "label": "VGGT neural reconstruction",
            "status": "running",
            "progress": 91,
            "stats": {"device": status.get("device"), "chunks": len(windows), "points": 0},
        })
        for wi, window in enumerate(windows):
            poses = [samples[frames.index(f)] for f in window]
            neural = await asyncio.to_thread(
                vggt_reconstruct_chunk, [f[2] for f in window], poses, o, track.reliability
            )
            if neural:
                vggt_points += len(neural)
                fused.extend(neural)
                await emit(job, {
                    "type": "dense",
                    "source": "vggt",
                    "progress": 91 + int(4 * (wi + 1) / max(len(windows), 1)),
                    "points": [_point_payload(p, o, 0.0) for p in neural[:2500]],
                    "count": vggt_points,
                })
        fused = voxel_downsample(fused, voxel=0.12)
        await emit(job, {
            "type": "stage",
            "stage": "vggt",
            "label": "VGGT neural reconstruction",
            "status": "done" if vggt_points else "failed",
            "progress": 95,
            "stats": {"device": status.get("device"), "chunks": len(windows), "points": vggt_points},
        })
    else:
        await emit(job, {
            "type": "stage",
            "stage": "vggt",
            "label": "VGGT neural reconstruction",
            "status": "skipped",
            "progress": 91,
            "stats": {"reason": status.get("reason") or "needs NVIDIA CUDA + vggt package", "cuda": status.get("cuda"), "package": status.get("package")},
        })

    write_ply(job.artifact_dir / "cloud.ply", fused)
    (job.artifact_dir / "cloud.json").write_text(json.dumps({"points": [_point_payload(p, o, 0.0) for p in fused[:50000]]}), encoding="utf-8")
    splat_info = await _run_splats(job, pg.get("gs_dir"), fused)
    traj = [{"lat": s.lat, "lon": s.lon, "alt": s.alt} for s in track.samples]
    (job.artifact_dir / "trajectory.geojson").write_text(json.dumps({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"], p["alt"]] for p in traj]}}), encoding="utf-8")
    challenges = _challenge_state(blur_dropped=blur_dropped, sparse=pg["dense_points"] < 2000, gps_downweighted=gps_down, illumination=illum, dynamic=True, occlusions=True, near_rt=True, metric=True)
    job.result = {
        "points": len(fused),
        "sparse_points": pg["sparse_points"],
        "dense_points": pg["dense_points"],
        "vggt_points": vggt_points,
        "cameras": pg["cameras"],
        "mesh": pg["mesh"],
        "mesh_json": pg["mesh_json"],
        "bundle_adjustment": pg["ba"],
        "gps_scale": pg["gps_scale"],
        "gps_rmse_m": pg["gps_rmse_m"],
        "focal_px": pg["focal_px"],
        "adapters": {"vggt": vggt_status(), "colmap": colmap_available()},
        "mode": mode,
        "telemetry_reliability": track.reliability,
        "frames_kept": len(frames),
        "origin": {"lat": o.lat, "lon": o.lon, "alt": origin_alt},
        "challenges": challenges,
        "splat": {k: v for k, v in splat_info.items() if k != "path"},
    }
    await emit(job, {"type": "stats", "stats": _stats(fused), "challenges": challenges})
    job.status = "done"
    neural = ""
    if splat_info.get("status") == "ok":
        neural = f" Neural render ready ({splat_info.get('gaussians', 0)} Gaussians)."
    await emit(job, {"type": "done", "progress": 100, "message": f"Photogrammetry finished: {pg['cameras']} cameras, {pg['sparse_points']} sparse, {pg['dense_points']} dense points.{neural}", "result": job.result})


def _splat_event(status: str, progress: int, **stats) -> dict:
    return {
        "type": "stage",
        "stage": "splat",
        "label": "Gaussian splatting",
        "status": status,
        "progress": progress,
        "stats": {k: v for k, v in stats.items() if v is not None and k != "path"},
    }


async def _run_splats(job: Job, gs_dir: str | None, points: list[ReconPoint]) -> dict:
    """Train the appearance layer after the metric cloud is final. Never fails the job."""
    status = splat_status()
    dataset = Path(gs_dir) if gs_dir else None
    if dataset is None or not (dataset / "cameras.json").exists():
        info = {"status": "skipped", "reason": "no posed frames"}
    elif not status["available"]:
        info = {"status": "skipped", "reason": status.get("reason") or "CUDA gsplat unavailable", "cuda": status.get("cuda"), "package": status.get("package"), "device": status.get("device")}
    else:
        await emit(job, _splat_event("running", 96, device=status.get("device"), frames=0))
        loop = asyncio.get_running_loop()

        def on_step(step: int, total: int, loss: float) -> None:
            asyncio.run_coroutine_threadsafe(
                emit(job, _splat_event("running", 96 + int(3 * step / max(total, 1)), step=step, steps=total, loss=round(loss, 4), device=status.get("device"))),
                loop,
            )

        info = await asyncio.to_thread(train_splats, dataset, points, job.artifact_dir / "splat.ply", on_step)
    final = "done" if info.get("status") == "ok" else str(info.get("status") or "failed")
    await emit(job, _splat_event(final, 99, **info))
    await emit(job, {"type": "splat", "status": info.get("status"), "reason": info.get("reason"), "gaussians": info.get("gaussians")})
    return info
