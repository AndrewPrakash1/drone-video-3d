"""Intelligent frame selection for single-pass reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ScoredFrame:
    index: int
    t: float
    sharpness: float
    features: int
    texture: float
    exposure: float
    redundancy: float
    viewpoint: float
    keep: bool
    score: float
    reasons: list[str]


def score_frame(gray: np.ndarray) -> tuple[float, int, float, float]:
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    orb = cv2.ORB_create(nfeatures=600)
    kps = orb.detect(gray, None)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    mean = float(gray.mean())
    exposure = 1.0 - min(abs(mean - 118.0) / 118.0, 1.0)
    return sharp, len(kps or []), entropy, exposure


def select_frames(
    video_path: str,
    duration_s: float | None = None,
    target_fps: float = 3.0,
    max_frames: int = 90,
) -> list[ScoredFrame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(int(round(fps / target_fps)), 1)
    hashed_prev: np.ndarray | None = None
    results: list[ScoredFrame] = []
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if duration_s is not None and t > duration_s:
            break
        if idx % step != 0:
            idx += 1
            continue
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp, nfeat, texture, exposure = score_frame(gray)
        ah = cv2.resize(gray, (16, 16)).astype(np.float32)
        red = 0.0
        if hashed_prev is not None:
            red = float(np.corrcoef(ah.ravel(), hashed_prev.ravel())[0, 1])
            if np.isnan(red):
                red = 1.0
        hashed_prev = ah
        blur = sharp < 18.0
        washed = exposure < 0.18
        duplicate = red > 0.992
        reasons: list[str] = []
        if blur:
            reasons.append("motion_blur")
        if washed:
            reasons.append("illumination")
        if duplicate:
            reasons.append("redundant")
        score = (
            min(sharp / 120.0, 1.0) * 0.35
            + min(nfeat / 250.0, 1.0) * 0.3
            + min(texture / 5.0, 1.0) * 0.15
            + exposure * 0.1
            + (1.0 - red) * 0.1
        )
        keep = not (blur or washed or duplicate) and kept < max_frames
        if keep:
            kept += 1
        results.append(
            ScoredFrame(
                index=idx,
                t=t,
                sharpness=sharp,
                features=nfeat,
                texture=texture,
                exposure=exposure,
                redundancy=red,
                viewpoint=1.0 - red,
                keep=keep,
                score=float(score),
                reasons=reasons,
            )
        )
        idx += 1
    cap.release()
    if not any(f.keep for f in results) and results:
        best = max(results, key=lambda f: f.score)
        best.keep = True
        best.reasons = [r for r in best.reasons if r != "redundant"]
    return results
