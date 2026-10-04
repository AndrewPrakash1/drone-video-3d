"""VGGT feed-forward reconstruction adapter.

Runs only when CUDA + the official `vggt` package are available. Visual
geometry is similarity-aligned onto GPS ENU so the dashboard stays metric.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..align import apply_similarity, camera_center_from_w2c, umeyama
from ..confidence import point_confidence
from ..geo import Origin, geodetic_to_enu
from ..reconstruct import ReconPoint, voxel_downsample
from ..telemetry import TelemetrySample

log = logging.getLogger("onepass.vggt")

_MODEL = None
_DEVICE = None


def vggt_available() -> bool:
    if os.environ.get("ONEPASS_DISABLE_VGGT", "").strip() in {"1", "true", "yes"}:
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
    except Exception:
        return False
    try:
        from vggt.models.vggt import VGGT  # noqa: F401
    except Exception:
        return False
    return True


def vggt_status() -> dict:
    info: dict = {"available": False, "cuda": False, "package": False, "device": None, "reason": None}
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["reason"] = f"torch: {exc}"
        return info
    try:
        from vggt.models.vggt import VGGT  # noqa: F401

        info["package"] = True
    except Exception as exc:
        info["reason"] = f"vggt package missing: {exc}"
        return info
    if os.environ.get("ONEPASS_DISABLE_VGGT", "").strip() in {"1", "true", "yes"}:
        info["reason"] = "disabled by ONEPASS_DISABLE_VGGT"
        return info
    info["available"] = info["cuda"] and info["package"]
    if not info["available"] and not info["reason"]:
        info["reason"] = "CUDA required for VGGT"
    return info


def _device_dtype():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    return device, dtype


def _load_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE
    import torch
    from vggt.models.vggt import VGGT

    device, _ = _device_dtype()
    weights = os.environ.get("VGGT_WEIGHTS", "").strip()
    repo_id = os.environ.get("VGGT_HF_ID", "facebook/VGGT-1B").strip()
    if weights and Path(weights).is_file():
        model = VGGT()
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        log.info("loaded VGGT weights from %s", weights)
    else:
        model = VGGT.from_pretrained(repo_id)
        log.info("loaded VGGT from Hugging Face %s", repo_id)
    model.eval()
    model = model.to(device)
    _MODEL = model
    _DEVICE = device
    return model, device


def reconstruct_chunk(
    frames_bgr: list[np.ndarray],
    poses: list[TelemetrySample],
    origin: Origin,
    gps_reliability: float,
) -> list[ReconPoint] | None:
    if not frames_bgr or not vggt_available():
        return None
    try:
        return _reconstruct(frames_bgr, poses, origin, gps_reliability)
    except Exception:
        log.exception("VGGT chunk failed; CPU path will be used")
        return None


def _write_temp_jpegs(frames_bgr: list[np.ndarray], folder: Path) -> list[str]:
    paths = []
    for i, img in enumerate(frames_bgr):
        path = folder / f"{i:03d}.jpg"
        h, w = img.shape[:2]
        max_edge = int(os.environ.get("VGGT_MAX_EDGE", "1024"))
        if max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        paths.append(str(path))
    return paths


def _extract_world_points(predictions: dict, images_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (points Nx3, conf N, rgb Nx3 or None) in VGGT world frame."""
    pts = None
    conf = None
    for key in ("world_points", "point_map", "pts3d"):
        if key in predictions and predictions[key] is not None:
            pts = predictions[key]
            break
    for key in ("world_points_conf", "point_conf", "conf"):
        if key in predictions and predictions[key] is not None:
            conf = predictions[key]
            break
    ext = predictions.get("extrinsic")
    depth = predictions.get("depth") or predictions.get("depth_map")
    intra = predictions.get("intrinsic")
    if pts is None and depth is not None and ext is not None:
        try:
            from vggt.utils.geometry import unproject_depth_map_to_point_map

            d = depth.squeeze(0) if hasattr(depth, "squeeze") else depth
            e = ext.squeeze(0) if hasattr(ext, "squeeze") else ext
            k = intra.squeeze(0) if intra is not None and hasattr(intra, "squeeze") else intra
            pts = unproject_depth_map_to_point_map(d, e, k)
        except Exception:
            pts = None
    if pts is None:
        raise RuntimeError("VGGT predictions missing world points / depth")

    import torch

    if hasattr(pts, "detach"):
        pts = pts.detach().float().cpu().numpy()
    else:
        pts = np.asarray(pts)
    while pts.ndim > 4:
        pts = pts[0]
    # expected (S, H, W, 3) or (B, S, H, W, 3)
    if pts.ndim == 5:
        pts = pts[0]
    if conf is not None and hasattr(conf, "detach"):
        conf = conf.detach().float().cpu().numpy()
    if conf is not None:
        while conf.ndim > 4:
            conf = conf[0]
        if conf.ndim == 5:
            conf = conf[0]
        conf = conf.reshape(-1)
    pts = pts.reshape(-1, 3)
    colors = None
    if images_np is not None:
        # images typically (S, 3, H, W) in 0-1
        img = images_np
        if img.ndim == 5:
            img = img[0]
        if img.shape[1] == 3:
            img = np.transpose(img, (0, 2, 3, 1))
        colors = (np.clip(img.reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
        n = min(len(pts), len(colors))
        pts, colors = pts[:n], colors[:n]
        if conf is not None:
            conf = conf[:n]
    return pts, (conf if conf is not None else np.ones(len(pts))), colors


def _vggt_camera_centers(predictions: dict) -> np.ndarray | None:
    ext = predictions.get("extrinsic")
    if ext is None:
        return None
    import torch

    if hasattr(ext, "detach"):
        ext = ext.detach().float().cpu().numpy()
    ext = np.asarray(ext)
    while ext.ndim > 3:
        ext = ext[0]
    if ext.ndim == 2:
        ext = ext[None, ...]
    centers = []
    for e in ext:
        try:
            centers.append(camera_center_from_w2c(e))
        except Exception:
            return None
    return np.stack(centers, axis=0)


def _reconstruct(
    frames_bgr: list[np.ndarray],
    poses: list[TelemetrySample],
    origin: Origin,
    gps_reliability: float,
) -> list[ReconPoint]:
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images

    model, device = _load_model()
    _, dtype = _device_dtype()
    with tempfile.TemporaryDirectory(prefix="onepass-vggt-") as tmp:
        paths = _write_temp_jpegs(frames_bgr, Path(tmp))
        images = load_and_preprocess_images(paths).to(device)
        with torch.no_grad():
            autocast = torch.cuda.amp.autocast if device == "cuda" else torch.cpu.amp.autocast
            with autocast(dtype=dtype):
                predictions = model(images)
        images_np = images.detach().float().cpu().numpy()

    pts, conf, colors = _extract_world_points(predictions, images_np)
    gps = np.stack([geodetic_to_enu(p.lat, p.lon, p.alt, origin) for p in poses], axis=0)
    vggt_cam = _vggt_camera_centers(predictions)
    aligned = None
    if vggt_cam is not None and len(vggt_cam) == len(gps):
        aligned = umeyama(vggt_cam, gps)
    if aligned is None and len(pts) > 100:
        # last-resort: scale VGGT median camera altitude vs GPS AGL
        agl = max(float(np.median([p.alt for p in poses]) - origin.alt), 8.0)
        zs = np.abs(pts[:, 2])
        zs = zs[np.isfinite(zs)]
        med = float(np.median(zs)) if len(zs) else 1.0
        scale = agl / max(med, 1e-3)
        aligned = (scale, np.eye(3), gps[0] - scale * np.zeros(3))
    if aligned is None:
        raise RuntimeError("could not align VGGT to GPS")
    scale, rot, trans = aligned
    finite = np.isfinite(pts).all(axis=1)
    pts, conf = pts[finite], conf[finite]
    if colors is not None:
        colors = colors[finite]
    world = apply_similarity(pts, scale, rot, trans)

    # Keep in-scene points near the GPS track
    span = float(np.linalg.norm(gps[-1] - gps[0])) + 40.0
    mid = gps.mean(axis=0)
    dist = np.linalg.norm(world - mid, axis=1)
    keep = dist < max(span * 1.8, 80.0)
    world, conf = world[keep], conf[keep]
    if colors is not None:
        colors = colors[keep]

    # Subsample for Cesium / fusion
    max_pts = int(os.environ.get("VGGT_MAX_POINTS", "8000"))
    if len(world) > max_pts:
        # prefer higher confidence
        order = np.argsort(-conf.reshape(-1))
        sel = np.sort(order[:max_pts])
        world, conf = world[sel], conf[sel]
        if colors is not None:
            colors = colors[sel]

    conf = np.clip(conf.reshape(-1), 0.0, 1.0)
    points: list[ReconPoint] = []
    n_views = float(len(frames_bgr))
    for i, xyz in enumerate(world):
        c = float(conf[i]) if i < len(conf) else 0.5
        rgb = colors[i] if colors is not None else np.array([180, 180, 180])
        score = point_confidence(
            observations=n_views,
            viewpoint_div=min(n_views / 6.0, 1.0),
            sharpness=0.75,
            gps_reliability=gps_reliability,
            residual=max(0.2, 2.0 * (1.0 - c)),
        )
        score = float(np.clip(0.55 * score + 0.45 * c, 0.08, 0.98))
        points.append(
            ReconPoint(
                e=float(xyz[0]),
                n=float(xyz[1]),
                u=float(xyz[2]),
                r=int(rgb[0]),
                g=int(rgb[1]),
                b=int(rgb[2]),
                conf=score,
                observations=max(2, int(n_views)),
                source="vggt",
                provenance=("vggt",),
                uncertainty_m=max(0.1, min(20.0, 2.0 * (1.0 - c) + 0.25)),
            )
        )
    return voxel_downsample(points, voxel=float(os.environ.get("VGGT_VOXEL", "0.45")))
