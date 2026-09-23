"""Similarity alignment of visual reconstructions onto GPS ENU."""

from __future__ import annotations

import numpy as np


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Return (scale, R, t) such that target ≈ scale * R @ source + t.

    source/target are Nx3. Needs at least 3 non-collinear points; with 2 points
    we estimate scale + translation only (identity rotation).
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        return None
    n = src.shape[0]
    if n < 2:
        return None
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float((src_c**2).sum() / n)
    if var_s < 1e-12:
        return None
    if n == 2:
        scale = float(np.linalg.norm(dst[1] - dst[0]) / max(np.linalg.norm(src[1] - src[0]), 1e-9))
        t = mu_d - scale * mu_s
        return scale, np.eye(3), t
    cov = (dst_c.T @ src_c) / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vt
    scale = float((s * d).sum() / var_s)
    if not np.isfinite(scale) or scale <= 1e-8:
        return None
    t = mu_d - scale * (r @ mu_s)
    return float(scale), r, t


def apply_similarity(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (scale * (rot @ points.T)).T + trans


def camera_center_from_w2c(extrinsic: np.ndarray) -> np.ndarray:
    """OpenCV world-to-camera 3x4 or 3x3+t → camera center in world coords."""
    ext = np.asarray(extrinsic, dtype=np.float64)
    if ext.shape == (3, 4):
        r, t = ext[:, :3], ext[:, 3]
    elif ext.shape == (4, 4):
        r, t = ext[:3, :3], ext[:3, 3]
    else:
        raise ValueError(f"unexpected extrinsic shape {ext.shape}")
    return -r.T @ t
