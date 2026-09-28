"""Short 3D Gaussian optimisation on the metric cloud.

Training is optional: without CUDA and gsplat the upload job still finishes
and the dashboard keeps the point cloud and Poisson mesh. When distortion
coefficients are present, frames are undistorted first so the pinhole
rasterizer sees the full image (the practical 3DGUT step). A scale-and-center
penalty keeps the optimised cloud in the same metres as the GPS model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from ..reconstruct import ReconPoint, voxel_downsample
from .ply import write_gaussian_ply


def splat_status() -> dict:
    try:
        import torch
    except Exception:
        return {"available": False, "cuda": False, "package": False, "device": None, "reason": "torch is not installed"}
    cuda = bool(torch.cuda.is_available())
    device = None
    if cuda:
        try:
            device = torch.cuda.get_device_name(0)
        except Exception:
            device = "cuda"
    try:
        import gsplat  # noqa: F401
        package = True
    except Exception:
        package = False
    if cuda and package:
        reason = None
    elif not package:
        reason = "gsplat is not installed"
    else:
        reason = "needs an NVIDIA CUDA GPU"
    return {"available": bool(cuda and package), "cuda": cuda, "package": package, "device": device, "reason": reason}


def load_views(dataset: Path) -> dict:
    meta_path = dataset / "cameras.json"
    if not meta_path.exists():
        raise FileNotFoundError("cameras.json is missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    width = int(meta["width"])
    height = int(meta["height"])
    k = _source_k(meta)
    dist = meta.get("distortion")
    dist_coeff = None
    new_k = k
    if dist:
        dist_coeff = np.asarray(dist, dtype=np.float64).reshape(-1)
        if not np.any(np.abs(dist_coeff) > 1e-8):
            dist_coeff = None
    if dist_coeff is not None:
        new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist_coeff, (width, height), 0)

    edge = max(8, int(os.environ.get("ONEPASS_SPLAT_EDGE", "640")))
    scale = 1.0
    long_edge = max(width, height)
    if long_edge > edge:
        scale = edge / float(long_edge)
    out_w = max(8, int(round(width * scale)))
    out_h = max(8, int(round(height * scale)))
    k = new_k.copy()
    k[0, 0] *= scale
    k[1, 1] *= scale
    k[0, 2] *= scale
    k[1, 2] *= scale

    images = []
    viewmats = []
    for frame in meta.get("frames") or []:
        path = dataset / "images" / frame["file"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        if dist_coeff is not None:
            bgr = cv2.undistort(bgr, _source_k(meta), dist_coeff, None, new_k)
        if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(rgb.astype(np.float32) / 255.0)
        rotation = np.asarray(frame["R"], dtype=np.float32).reshape(3, 3)
        translation = np.asarray(frame["t"], dtype=np.float32).reshape(3)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation
        view[:3, 3] = translation
        viewmats.append(view)

    if len(images) < 2:
        raise ValueError("need at least two posed frames")
    return {
        "images": np.stack(images, axis=0),
        "viewmats": np.stack(viewmats, axis=0),
        "K": k.astype(np.float32),
        "width": out_w,
        "height": out_h,
        "undistorted": dist_coeff is not None,
    }


def train_splats(dataset: Path, points: list[ReconPoint], out_ply: Path, progress=None) -> dict:
    status = splat_status()
    if not status["available"]:
        return {
            "status": "skipped",
            "reason": status["reason"] or "CUDA gsplat unavailable",
            "cuda": status["cuda"],
            "package": status["package"],
            "device": status["device"],
        }
    if not (dataset / "cameras.json").exists():
        return {"status": "skipped", "reason": "no posed frames"}
    if len(points) < 32:
        return {"status": "skipped", "reason": "metric cloud is too small to initialise Gaussians"}
    try:
        return _train(dataset, points, out_ply, progress, status.get("device"))
    except Exception as exc:
        if out_ply.exists():
            out_ply.unlink()
        return {"status": "failed", "reason": str(exc)}


def _source_k(meta: dict) -> np.ndarray:
    return np.array(
        [[meta["fx"], 0.0, meta["cx"]], [0.0, meta["fy"], meta["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _cap_points(points: list[ReconPoint], cap: int) -> list[ReconPoint]:
    if len(points) <= cap:
        return points
    span = 1.0
    for axis in ("e", "n", "u"):
        vals = [getattr(p, axis) for p in points]
        span = max(span, max(vals) - min(vals))
    voxel = max(span / (cap ** (1.0 / 3.0)), 0.05)
    chosen = points
    for _ in range(8):
        chosen = voxel_downsample(points, voxel)
        if len(chosen) <= cap:
            break
        voxel *= 1.35
    if len(chosen) > cap:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(chosen), cap, replace=False)
        chosen = [chosen[int(i)] for i in keep]
    return chosen


def _primitive_scales(xyz: np.ndarray) -> np.ndarray:
    n = len(xyz)
    if n < 2:
        return np.full((n, 3), 0.05, np.float32)
    rng = np.random.default_rng(0)
    ref = xyz[rng.choice(n, size=min(n, 1024), replace=False)]
    nearest = np.empty(n, np.float32)
    for start in range(0, n, 2048):
        chunk = xyz[start : start + 2048]
        dist = np.linalg.norm(chunk[:, None, :] - ref[None, :, :], axis=2)
        dist.sort(axis=1)
        col = dist[:, 0]
        if dist.shape[1] > 1:
            col = np.where(col < 1e-4, dist[:, 1], col)
        nearest[start : start + len(chunk)] = col
    nearest = np.clip(nearest, 0.02, 1.5).astype(np.float32)
    return np.repeat((nearest * 0.6)[:, None], 3, axis=1)


def _ssim(pred, target):
    import torch
    import torch.nn.functional as F

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    return score.mean()


def _train(dataset: Path, points: list[ReconPoint], out_ply: Path, progress, device_name: str | None) -> dict:
    import torch
    from gsplat import rasterization

    views = load_views(dataset)
    cap = max(256, int(os.environ.get("ONEPASS_SPLAT_MAX_POINTS", "80000")))
    steps = max(1, int(os.environ.get("ONEPASS_SPLAT_STEPS", "7000")))
    metric_weight = float(os.environ.get("ONEPASS_SPLAT_METRIC_WEIGHT", "0.05"))
    cloud = _cap_points(points, cap)
    xyz = np.array([[p.e, p.n, p.u] for p in cloud], dtype=np.float32)
    rgb = np.array([[p.r, p.g, p.b] for p in cloud], dtype=np.float32) / 255.0
    scales0 = _primitive_scales(xyz)

    device = torch.device("cuda")
    means = torch.nn.Parameter(torch.tensor(xyz, device=device))
    log_scales = torch.nn.Parameter(torch.log(torch.tensor(scales0, device=device)))
    quats = torch.nn.Parameter(torch.zeros((len(cloud), 4), device=device))
    with torch.no_grad():
        quats[:, 0] = 1.0
    logit_opac = torch.nn.Parameter(torch.full((len(cloud),), torch.logit(torch.tensor(0.1)), device=device))
    raw_rgb = torch.nn.Parameter(torch.logit(torch.tensor(np.clip(rgb, 0.02, 0.98), device=device)))
    init_mean = means.detach().mean(0)
    init_std = means.detach().std(0).clamp(min=1e-3)

    images = torch.tensor(views["images"], device=device)
    viewmats = torch.tensor(views["viewmats"], device=device)
    k_mat = torch.tensor(views["K"], device=device)
    height, width = int(views["height"]), int(views["width"])
    opt = torch.optim.Adam(
        [
            {"params": means, "lr": 1.6e-4},
            {"params": log_scales, "lr": 5e-3},
            {"params": quats, "lr": 1e-3},
            {"params": logit_opac, "lr": 5e-2},
            {"params": raw_rgb, "lr": 1e-2},
        ]
    )
    rng = np.random.default_rng(0)
    last_loss = 0.0
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        cam = int(rng.integers(0, images.shape[0]))
        renders, _alphas, _meta = rasterization(
            means,
            torch.nn.functional.normalize(quats, dim=-1),
            torch.exp(log_scales).clamp(1e-4, 8.0),
            torch.sigmoid(logit_opac),
            torch.sigmoid(raw_rgb),
            viewmats[cam : cam + 1],
            k_mat[None],
            width,
            height,
            near_plane=0.05,
            far_plane=5000.0,
            sh_degree=None,
            render_mode="RGB",
        )
        pred = renders[0, ..., :3]
        target = images[cam]
        photo = (pred - target).abs().mean() + 0.2 * (1.0 - _ssim(pred, target))
        center = (means.mean(0) - init_mean).pow(2).sum()
        spread = (means.std(0) - init_std).pow(2).sum()
        loss = photo + metric_weight * (center + spread)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
        if progress and (step == 1 or step == steps or step % 250 == 0):
            progress(step, steps, last_loss)

    with torch.no_grad():
        kept = torch.sigmoid(logit_opac) > 0.02
        if int(kept.sum()) < 16:
            kept = torch.ones_like(kept, dtype=torch.bool)
        count = write_gaussian_ply(
            out_ply,
            means[kept].detach().cpu().numpy(),
            torch.exp(log_scales[kept]).clamp(1e-4, 8.0).detach().cpu().numpy(),
            torch.nn.functional.normalize(quats[kept], dim=-1).detach().cpu().numpy(),
            torch.sigmoid(logit_opac[kept]).detach().cpu().numpy(),
            torch.sigmoid(raw_rgb[kept]).detach().cpu().numpy(),
        )
    return {
        "status": "ok",
        "gaussians": count,
        "steps": steps,
        "loss": round(last_loss, 4),
        "frames": int(images.shape[0]),
        "undistorted": bool(views["undistorted"]),
        "device": device_name,
        "path": str(out_ply),
    }
