"""Job runners: demo playback and uploaded video reconstruction."""

from __future__ import annotations

import asyncio
import json
import logging
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
from .geo import Origin, enu_to_geodetic, geodetic_to_enu, make_origin
from .jobs import Job, emit
from .mesh import agreeing_points, clean_cloud, mesh_from_points, write_ply
from .spatial import write_spatial
from .scene import write_scene_manifest
from .spatial_model.inference import run_optional_stage
from .metric import evaluate_segment
from .photogrammetry.run import mesh_json_file, run_photogrammetry
from .reconstruct import ReconPoint, triangulate_pair, voxel_downsample
from .splats.train import splat_status, train_splats
from .telemetry import TelemetrySample, TelemetryTrack, parse_telemetry_csv

log = logging.getLogger(__name__)


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
        "source": p.source,
        "provenance": list(p.provenance),
        "uncertainty_m": p.uncertainty_m,
        "synthetic": p.synthetic,
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
    scene = write_scene_manifest(
        job.artifact_dir,
        origin={"lat": o.lat, "lon": o.lon, "alt": ORIGIN_ALT},
        point_count=len(fused),
        mesh=mesh_info,
        spatial=None,
        cameras=len(telem),
    )
    spatial_model = await asyncio.to_thread(run_optional_stage, job.artifact_dir, scene, None)
    await emit(job, {"type": "stage", "stage": "spatial_model", "label": "Generative spatial intelligence", "status": spatial_model["status"], "progress": 98, "stats": {"reason": spatial_model.get("reason"), "model": spatial_model.get("model")}})
    job.result = {
        "points": len(fused),
        "mesh": mesh_info,
        "metric": ref,
        "adapters": adapters,
        "origin": {"lat": o.lat, "lon": o.lon, "alt": ORIGIN_ALT},
        "scene": scene,
        "spatial_model": {k: v for k, v in spatial_model.items() if k != "model"},
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
    mode = "photogrammetry" + ("+vggt" if use_vggt else "") + ("+colmap" if colmap_available() else "")

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

    # More frames => smaller baselines and far more dense MVS points (a 30 s clip
    # at 3 fps gives ~90 candidates). 60 is a good quality/time balance;
    # override with ONEPASS_MAX_FRAMES.
    max_frames = max(6, int(os.environ.get("ONEPASS_MAX_FRAMES", "60")))
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
        log.exception("photogrammetry failed")
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        await emit(job, {"type": "error", "message": job.error})
        return
    if pg.get("status") != "ok":
        job.status = "error"
        job.error = pg.get("reason", "photogrammetry failed")
        await emit(job, {"type": "error", "message": job.error})
        return
    fused: list[ReconPoint] = list(pg["fused"])

    colmap_points = 0
    colmap_info: dict = {"status": "skipped", "reason": "colmap not on PATH"}
    if colmap_available():
        image_dir = job.artifact_dir / "colmap_input"
        work_dir = job.artifact_dir / "colmap"
        image_dir.mkdir(parents=True, exist_ok=True)
        poses_by_name: dict[str, TelemetrySample] = {}
        for i, (_vidx, _t, img) in enumerate(frames):
            name = f"frame_{i:04d}.jpg"
            if cv2.imwrite(str(image_dir / name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                poses_by_name[name] = samples[i]
        await emit(job, {
            "type": "stage",
            "stage": "colmap",
            "label": "COLMAP SfM + fusion",
            "status": "running",
            "progress": 89,
            "stats": {"images": len(poses_by_name)},
        })
        try:
            colmap_pts, colmap_info = await asyncio.to_thread(
                reconstruct_aligned, image_dir, work_dir, poses_by_name, o, track.reliability
            )
        except Exception as exc:
            log.exception("colmap stage failed")
            colmap_pts, colmap_info = [], {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        if colmap_pts:
            # COLMAP happily triangulates far outliers that the built-in trim
            # (applied only to the built-in SfM result) never sees.
            gps_enu = np.array([geodetic_to_enu(s.lat, s.lon, s.alt, o) for s in samples])
            pe = np.array([p.e for p in colmap_pts])
            pn = np.array([p.n for p in colmap_pts])
            near = np.hypot(pe[:, None] - gps_enu[None, :, 0], pn[:, None] - gps_enu[None, :, 1]).min(axis=1)
            flight = float(np.ptp(gps_enu[:, :2], axis=0).max())
            keep = near <= max(flight * 1.5, 150.0)
            if int(keep.sum()) >= 50:
                colmap_pts = [p for p, k in zip(colmap_pts, keep.tolist()) if k]
            colmap_points = len(colmap_pts)
            if colmap_points >= 3000:
                # COLMAP's sparse model is far cleaner than the built-in dense
                # MVS, but the two are aligned to GPS independently and can sit
                # metres apart; stacking both draws layered "walls".
                builtin = await asyncio.to_thread(agreeing_points, colmap_pts, fused)
                colmap_info["builtin_kept"] = len(builtin)
                fused = list(colmap_pts) + builtin
            else:
                fused.extend(colmap_pts)
            fused = voxel_downsample(fused, voxel=0.12)
            await emit(job, {
                "type": "dense",
                "source": "colmap",
                "progress": 90,
                "points": [_point_payload(p, o, 0.0) for p in colmap_pts[:2500]],
                "count": colmap_points,
            })
        await emit(job, {
            "type": "stage",
            "stage": "colmap",
            "label": "COLMAP SfM + fusion",
            "status": "done" if colmap_points else "skipped",
            "progress": 90,
            "stats": {k: v for k, v in colmap_info.items() if k in {"points", "scale", "aligned_cameras", "builtin_kept", "reason"}},
        })
    else:
        await emit(job, {
            "type": "stage",
            "stage": "colmap",
            "label": "COLMAP SfM + fusion",
            "status": "skipped",
            "progress": 90,
            "stats": {"reason": "colmap not on PATH"},
        })

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

    eyes = np.array([[c["e"], c["n"], c["u"]] for c in pg.get("cameras_payload") or []], dtype=np.float64)
    fused = await asyncio.to_thread(clean_cloud, fused, eyes if len(eyes) else None)
    await emit(job, {"type": "stage", "stage": "mesh", "label": "Surface mesh", "status": "running", "progress": 96, "stats": {"points": len(fused)}})
    mesh_info = await asyncio.to_thread(mesh_from_points, fused, job.artifact_dir / "mesh.ply")
    mesh_json = mesh_json_file(job.artifact_dir / "mesh.ply", job.artifact_dir / "mesh.json", fused)
    if mesh_info.get("status") != "ok":
        mesh_info, mesh_json = pg["mesh"], pg["mesh_json"]
    await emit(job, {
        "type": "stage",
        "stage": "mesh",
        "label": "Surface mesh",
        "status": "done" if mesh_info.get("status") == "ok" else "failed",
        "progress": 97,
        "stats": {k: v for k, v in mesh_info.items() if k != "path"},
    })
    spatial_summary = None
    spatial_cams = pg.get("cameras_payload") or []
    if spatial_cams and mesh_info.get("status") == "ok" and (job.artifact_dir / "mesh.ply").exists():
        try:
            spatial_summary = await asyncio.to_thread(
                write_spatial, job.artifact_dir / "mesh.ply", spatial_cams, job.artifact_dir / "spatial.json"
            )
        except Exception:
            log.exception("spatial map failed")
    scene = write_scene_manifest(
        job.artifact_dir,
        origin={"lat": o.lat, "lon": o.lon, "alt": origin_alt},
        point_count=len(fused),
        mesh=mesh_info,
        spatial=spatial_summary,
        cameras=len(spatial_cams),
    )
    spatial_payload = None
    spatial_path = job.artifact_dir / "spatial.json"
    if spatial_path.exists():
        try:
            spatial_payload = json.loads(spatial_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            spatial_payload = None
    spatial_model = await asyncio.to_thread(run_optional_stage, job.artifact_dir, scene, spatial_payload)
    await emit(job, {"type": "stage", "stage": "spatial_model", "label": "Generative spatial intelligence", "status": spatial_model["status"], "progress": 98, "stats": {"reason": spatial_model.get("reason"), "model": spatial_model.get("model")}})
    write_ply(job.artifact_dir / "cloud.ply", fused)
    shown = fused
    if len(fused) > 50000:
        pick = np.random.default_rng(0).choice(len(fused), 50000, replace=False)
        shown = [fused[i] for i in np.sort(pick).tolist()]
    (job.artifact_dir / "cloud.json").write_text(json.dumps({"points": [_point_payload(p, o, 0.0) for p in shown]}), encoding="utf-8")
    try:
        splat_info = await _run_splats(job, pg.get("gs_dir"), fused)
    except Exception as exc:
        log.exception("gaussian stage failed")
        splat_info = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        await emit(job, _splat_event("failed", 99, reason=splat_info["reason"]))
    traj = [{"lat": s.lat, "lon": s.lon, "alt": s.alt} for s in track.samples]
    (job.artifact_dir / "trajectory.geojson").write_text(json.dumps({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"], p["alt"]] for p in traj]}}), encoding="utf-8")
    challenges = _challenge_state(blur_dropped=blur_dropped, sparse=pg["dense_points"] < 2000, gps_downweighted=gps_down, illumination=illum, dynamic=True, occlusions=True, near_rt=True, metric=True)
    job.result = {
        "points": len(fused),
        "sparse_points": pg["sparse_points"],
        "dense_points": pg["dense_points"],
        "vggt_points": vggt_points,
        "colmap_points": colmap_points,
        "colmap": {k: v for k, v in colmap_info.items() if k not in {"sparse_dir", "txt_dir"}},
        "cameras": pg["cameras"],
        "mesh": mesh_info,
        "mesh_json": mesh_json,
        "spatial": spatial_summary,
        "bundle_adjustment": pg["ba"],
        "gps_scale": pg["gps_scale"],
        "gps_rmse_m": pg["gps_rmse_m"],
        "focal_px": pg["focal_px"],
        "adapters": {"vggt": vggt_status(), "colmap": colmap_available()},
        "mode": mode,
        "telemetry_reliability": track.reliability,
        "frames_kept": len(frames),
        "origin": {"lat": o.lat, "lon": o.lon, "alt": origin_alt},
        "scene": scene,
        "spatial_model": {k: v for k, v in spatial_model.items() if k != "model"},
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
    stats.pop("status", None)
    stats.pop("path", None)
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
    stats = {k: v for k, v in info.items() if k not in {"status", "path"}}
    await emit(job, _splat_event(final, 99, **stats))
    await emit(job, {"type": "splat", "status": info.get("status"), "reason": info.get("reason"), "gaussians": info.get("gaussians")})
    return info
