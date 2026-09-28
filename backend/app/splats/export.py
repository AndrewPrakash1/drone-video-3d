"""Write posed frames and pinhole intrinsics for Gaussian training.

Cameras are the GPS-aligned SfM poses (ENU metres, OpenCV world-to-camera).
Distortion coefficients are stored when a model has them so training can
undistort before the pinhole rasterizer, which is the practical stand-in for
3DGUT's nonlinear projection.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def export_gaussian_dataset(dest: Path, frames: list, result, distortion: np.ndarray | None = None) -> dict:
    cameras = getattr(result, "cameras", None) or {}
    if len(cameras) < 2:
        return {"status": "skipped", "reason": "fewer than two registered cameras", "frames": len(cameras)}

    dest.mkdir(parents=True, exist_ok=True)
    image_dir = dest / "images"
    image_dir.mkdir(exist_ok=True)
    k = np.asarray(result.k, dtype=np.float64)
    height, width = frames[0][2].shape[:2]
    dist = None if distortion is None else [float(v) for v in np.ravel(distortion)]
    if dist is not None and not any(abs(v) > 1e-8 for v in dist):
        dist = None

    written = []
    for idx, cam in sorted(cameras.items()):
        if idx < 0 or idx >= len(frames):
            continue
        image = frames[idx][2]
        name = f"frame_{idx:04d}.jpg"
        if not cv2.imwrite(str(image_dir / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            continue
        rotation = np.asarray(cam.rotation, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(cam.tvec, dtype=np.float64).reshape(3)
        center = np.asarray(cam.center, dtype=np.float64).reshape(3)
        written.append(
            {
                "file": name,
                "index": int(idx),
                "R": rotation.reshape(-1).tolist(),
                "t": translation.tolist(),
                "center": center.tolist(),
            }
        )

    if len(written) < 2:
        return {"status": "skipped", "reason": "could not write two posed frames", "frames": len(written)}

    meta = {
        "width": int(width),
        "height": int(height),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "distortion": dist,
        "frames": written,
    }
    (dest / "cameras.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"status": "ok", "dir": str(dest), "frames": len(written), "width": int(width), "height": int(height)}
