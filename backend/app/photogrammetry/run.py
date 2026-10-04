"""Run SIFT → KNN/RANSAC → SfM/BA → MVS → Poisson and stream stage events."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from ..confidence import classify, point_confidence
from ..geo import Origin, enu_to_geodetic, geodetic_to_enu
from ..jobs import Job, emit
from ..mesh import clean_cloud, mesh_from_points, write_ply
from ..reconstruct import ReconPoint, voxel_downsample
from ..telemetry import TelemetrySample
from ..splats.export import export_gaussian_dataset
from .features import build_tracks, candidate_pairs, extract_features, match_pair
from .mvs import dense_reconstruct
from .sfm import align_to_gps, initial_intrinsics, run_sfm

STAGES = [
    ("features", "Feature extraction (SIFT)"),
    ("matching", "KNN matching + RANSAC"),
    ("sfm", "Sparse SfM triangulation"),
    ("ba", "Bundle adjustment"),
    ("mvs", "Dense multi-view stereo"),
    ("mesh", "Surface mesh"),
]


def _stage(key: str, status: str, progress: int, **stats) -> dict:
    return {"type": "stage", "stage": key, "label": dict(STAGES)[key], "status": status, "progress": progress, "stats": _safe_stats(stats)}


def _safe_stats(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if k not in {"status", "progress", "type", "stage", "label"}}


def _payload(pts, rgb, conf, origin: Origin, limit: int = 4000) -> list[dict]:
    if len(pts) > limit:
        sel = np.random.default_rng(0).choice(len(pts), limit, replace=False)
        pts, rgb, conf = pts[sel], rgb[sel], conf[sel]
    out = []
    for p, c, cf in zip(pts, rgb, conf):
        lat, lon, alt = enu_to_geodetic(float(p[0]), float(p[1]), float(p[2]), origin)
        out.append({
            "lat": lat, "lon": lon, "height": alt,
            "e": float(p[0]), "n": float(p[1]), "u": float(p[2]),
            "r": int(c[0]), "g": int(c[1]), "b": int(c[2]),
            "conf": float(cf), "band": classify(float(cf)),
            "source": "photogrammetry",
            "provenance": ["mvs"],
            "uncertainty_m": max(0.05, min(20.0, 1.0 / max(float(cf), 0.05))),
            "synthetic": False,
        })
    return out


def _cameras(result, size, origin, samples) -> list[dict]:
    rows = []
    for i, cam in sorted(result.cameras.items()):
        c = cam.center
        lat, lon, alt = enu_to_geodetic(float(c[0]), float(c[1]), float(c[2]), origin)
        rows.append({
            "index": i,
            "e": float(c[0]), "n": float(c[1]), "u": float(c[2]),
            "lat": lat, "lon": lon, "alt": alt,
            "rotation": [float(v) for v in cam.rotation.T.ravel()],
            "fx": float(result.k[0, 0]),
            "width": size[0], "height": size[1],
            "gps": None if i >= len(samples) else {"lat": samples[i].lat, "lon": samples[i].lon, "alt": samples[i].alt},
        })
    return rows


def _trim_to_flight(result):
    """Drop triangulations that sit far beside the flight. Tiny parallax still passes the reprojection test and stretches the cloud for kilometres."""
    centers = [c.center for c in result.cameras.values()]
    if len(result.points) == 0 or not centers:
        return result, 0
    cams = np.asarray(centers, dtype=np.float64)
    if not np.isfinite(cams).all():
        return result, 0
    pts = result.points
    horiz = np.hypot(pts[:, None, 0] - cams[None, :, 0], pts[:, None, 1] - cams[None, :, 1]).min(1)
    flight = float(np.ptp(cams[:, :2], axis=0).max())
    alt = float(np.median(cams[:, 2]) - np.median(pts[:, 2]))
    limit = max(flight * 1.5, abs(alt) * 3.0, 50.0)
    keep = horiz <= limit
    if int(keep.sum()) < 50 or float(keep.mean()) > 0.98:
        return result, 0
    idx = np.flatnonzero(keep)
    result.points = pts[idx]
    result.colors = result.colors[idx]
    result.errors = result.errors[idx]
    result.triangulation_angles = result.triangulation_angles[idx]
    result.observations = [result.observations[i] for i in idx.tolist()]
    return result, int((~keep).sum())


async def run_photogrammetry(job: Job, frames, samples: list[TelemetrySample], origin: Origin, gps_reliability: float, progress_base: int = 8) -> dict:
    n = len(frames)
    size = (frames[0][2].shape[1], frames[0][2].shape[0])
    await emit(job, _stage("features", "running", progress_base, images=n))
    features = []
    for i, (vidx, _, img) in enumerate(frames):
        features.append(await asyncio.to_thread(extract_features, img, i, f"frame_{vidx:06d}"))
    await emit(job, _stage("features", "done", progress_base + 8, images=n, keypoints=int(sum(len(f.keypoints) for f in features))))

    gps = np.array([geodetic_to_enu(s.lat, s.lon, s.alt, origin) for s in samples])
    pairs_idx = candidate_pairs(n, gps)
    await emit(job, _stage("matching", "running", progress_base + 9, pairs=len(pairs_idx)))
    matches = []
    knn_total = ratio_total = inlier_total = 0
    for i, j in pairs_idx:
        pm = await asyncio.to_thread(match_pair, features[i], features[j])
        knn_total += pm.knn_matches
        ratio_total += pm.ratio_matches
        inlier_total += pm.n_inliers
        if pm.n_inliers >= 15:
            matches.append(pm)
    tracks = build_tracks(matches)
    await emit(job, _stage("matching", "done", progress_base + 22, pairs=len(pairs_idx), good_pairs=len(matches), knn=knn_total, ratio_kept=ratio_total, ransac_inliers=inlier_total, outliers_removed=ratio_total - inlier_total, tracks=len(tracks)))

    await emit(job, _stage("sfm", "running", progress_base + 23))
    loop = asyncio.get_running_loop()
    last = {"t": 0.0}

    def on_progress(reg, total, npts):
        now = time.time()
        if now - last["t"] > 0.5:
            last["t"] = now
            asyncio.run_coroutine_threadsafe(emit(job, _stage("sfm", "running", progress_base + 23 + int(16 * reg / max(total, 1)), cameras=reg, points=npts)), loop)

    sfm = await asyncio.to_thread(run_sfm, features, matches, tracks, initial_intrinsics(*size), on_progress)
    if sfm is None:
        await emit(job, _stage("sfm", "failed", progress_base + 40, reason="no pair with enough parallax"))
        return {"status": "failed", "reason": "sfm initialisation failed"}
    result = align_to_gps(sfm.result(), {i: gps[i] for i in range(n)})
    result, far_removed = _trim_to_flight(result)
    ba = result.ba_history[-1] if result.ba_history else {}
    await emit(job, _stage("sfm", "done", progress_base + 40, cameras=len(result.cameras), points=int(len(result.points)), far_points_removed=far_removed, mean_reproj_px=round(float(result.errors.mean()), 3) if len(result.errors) else None, gps_scale=round(result.gps_scale, 4), gps_rmse_m=None if result.gps_rmse_m is None else round(result.gps_rmse_m, 2)))
    await emit(job, _stage("ba", "done", progress_base + 42, **_safe_stats(ba)))

    sparse_conf = np.array([
        point_confidence(len(o), min(a / 12.0, 1.0), 0.8, gps_reliability, e)
        for o, a, e in zip(result.observations, result.triangulation_angles, result.errors)
    ]) if len(result.points) else np.zeros(0)
    cameras = _cameras(result, size, origin, samples)
    await emit(job, {"type": "sparse", "progress": progress_base + 44, "cameras": cameras, "points": _payload(result.points, result.colors, sparse_conf, origin, 6000), "count": int(len(result.points))})

    await emit(job, _stage("mvs", "running", progress_base + 46, views=len(result.cameras)))

    def mvs_progress(phase, done, total):
        asyncio.run_coroutine_threadsafe(emit(job, _stage("mvs", "running", progress_base + (46 if phase == "depth" else 60) + int(12 * done / max(total, 1)), phase=phase, done=done, views=total)), loop)

    dense = await asyncio.to_thread(dense_reconstruct, result, {i: frames[i][2] for i in result.cameras}, 640, 8000, 0.03, mvs_progress)
    dense_conf = np.array([
        point_confidence(float(v), 0.6, float(s), gps_reliability, 1.0 + (1.0 - float(s)) * 3.0)
        for v, s in zip(dense.views, dense.ncc)
    ]) if len(dense.xyz) else np.zeros(0)
    await emit(job, _stage("mvs", "done", progress_base + 74, views=len(result.cameras), dense_points=int(len(dense.xyz)), mean_consistent_views=round(float(dense.views.mean()), 2) if len(dense.views) else 0))
    if len(dense.xyz):
        step = max(1, len(dense.xyz) // 16000)
        for s in range(0, len(dense.xyz), max(step * 4000, 4000)):
            sl = slice(s, min(s + max(step * 4000, 4000), len(dense.xyz)))
            await emit(job, {"type": "dense", "progress": progress_base + 74, "points": _payload(dense.xyz[sl][::step], dense.rgb[sl][::step], dense_conf[sl][::step], origin, 4000), "count": int(len(dense.xyz))})

    await emit(job, _stage("mesh", "running", progress_base + 82))
    fused: list[ReconPoint] = []
    for p, c, cf in zip(result.points, result.colors, sparse_conf):
        fused.append(ReconPoint(
            float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2]), float(cf), 2,
            source="photogrammetry", provenance=("sfm",), uncertainty_m=max(0.05, min(20.0, 1.0 / max(float(cf), 0.05))),
        ))
    for p, c, cf, v in zip(dense.xyz, dense.rgb, dense_conf, dense.views):
        fused.append(ReconPoint(
            float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2]), float(cf), int(v),
            source="photogrammetry", provenance=("mvs",), uncertainty_m=max(0.05, min(20.0, 1.0 / max(float(cf), 0.05))),
        ))
    fused = voxel_downsample(fused, voxel=0.15 if len(fused) > 40000 else 0.08)
    write_ply(job.artifact_dir / "cloud.ply", fused)
    eyes = np.array([[c["e"], c["n"], c["u"]] for c in cameras], dtype=np.float64)
    fused = await asyncio.to_thread(clean_cloud, fused, eyes if len(eyes) else None)
    mesh_info = await asyncio.to_thread(mesh_from_points, fused, job.artifact_dir / "mesh.ply")
    mesh_json = mesh_json_file(job.artifact_dir / "mesh.ply", job.artifact_dir / "mesh.json", fused)
    (job.artifact_dir / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")
    try:
        gs_export = export_gaussian_dataset(job.artifact_dir / "gs", frames, result)
    except Exception as exc:
        gs_export = {"status": "skipped", "reason": str(exc), "frames": 0}
    await emit(job, _stage("mesh", "done" if mesh_info.get("status") == "ok" else "failed", progress_base + 90, **_safe_stats({k: v for k, v in mesh_info.items() if k != "path"})))
    return {
        "status": "ok",
        "cameras": len(result.cameras),
        "sparse_points": int(len(result.points)),
        "dense_points": int(len(dense.xyz)),
        "mesh": mesh_info,
        "mesh_json": mesh_json,
        "ba": ba,
        "gps_scale": result.gps_scale,
        "gps_rmse_m": result.gps_rmse_m,
        "focal_px": float(result.k[0, 0]),
        "fused": fused,
        "cameras_payload": cameras,
        "gs_dir": gs_export.get("dir"),
        "gs_frames": gs_export.get("frames", 0),
    }


def mesh_json_file(ply: Path, out: Path, points: list[ReconPoint] | None = None) -> dict | None:
    if not ply.exists():
        return None
    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(ply))
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.triangles, dtype=np.int32)
        c = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else np.full((len(v), 3), 210, np.uint8)
    except Exception:
        v, f, c = _ascii_ply(ply)
    if len(v) == 0 or len(f) == 0:
        return None
    if len(f) > 250000:
        f = f[np.random.default_rng(0).choice(len(f), 250000, replace=False)]
    vertex_confidence = np.ones(len(v), dtype=np.float32)
    vertex_source = ["photogrammetry"] * len(v)
    vertex_state = ["observed"] * len(v)
    if points:
        from scipy.spatial import cKDTree

        xyz = np.asarray([(p.e, p.n, p.u) for p in points], dtype=np.float64)
        if len(xyz):
            nearest = cKDTree(xyz).query(np.asarray(v, dtype=np.float64), k=1)[1]
            vertex_confidence = np.asarray([points[int(i)].conf for i in nearest], dtype=np.float32)
            vertex_source = [points[int(i)].source for i in nearest]
            vertex_state = ["generated" if points[int(i)].synthetic else "observed" for i in nearest]
    face_confidence = vertex_confidence[f].mean(axis=1) if len(f) else np.zeros(0, dtype=np.float32)
    face_source = [vertex_source[int(i)] for i in f[:, 0]] if len(f) else []
    face_state = ["generated" if any(vertex_state[int(i)] == "generated" for i in face) else "observed" for face in f]
    payload = {
        "positions": np.round(v, 3).ravel().tolist(),
        "colors": c.ravel().tolist(),
        "indices": f.ravel().tolist(),
        "vertex_confidence": np.round(vertex_confidence, 4).tolist(),
        "vertex_source": vertex_source,
        "vertex_state": vertex_state,
        "face_confidence": np.round(face_confidence, 4).tolist(),
        "face_source": face_source,
        "face_state": face_state,
        "measurement_safe": [False for _ in face_state],
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return {"vertices": int(len(v)), "triangles": int(len(f))}


def _ascii_ply(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    nv = nf = i = 0
    while i < len(lines) and lines[i].strip() != "end_header":
        if lines[i].startswith("element vertex"):
            nv = int(lines[i].split()[-1])
        if lines[i].startswith("element face"):
            nf = int(lines[i].split()[-1])
        i += 1
    i += 1
    verts, cols, faces = [], [], []
    for row in lines[i : i + nv]:
        p = row.split()
        verts.append([float(p[0]), float(p[1]), float(p[2])])
        cols.append([int(float(p[3])), int(float(p[4])), int(float(p[5]))] if len(p) >= 6 else [210, 210, 210])
    for row in lines[i + nv : i + nv + nf]:
        p = row.split()
        if len(p) >= 4:
            faces.append([int(p[1]), int(p[2]), int(p[3])])
    return np.array(verts, np.float32), np.array(faces, np.int32).reshape(-1, 3) if faces else np.zeros((0, 3), np.int32), np.array(cols, np.uint8)
