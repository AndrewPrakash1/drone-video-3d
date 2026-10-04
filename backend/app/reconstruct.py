"""CPU geometric reconstruction for a single UAV pass."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .confidence import point_confidence
from .geo import Origin, geodetic_to_enu, heading_to_enu
from .telemetry import TelemetrySample


@dataclass
class ReconPoint:
    e: float
    n: float
    u: float
    r: int
    g: int
    b: int
    conf: float
    observations: int
    source: str = "photogrammetry"
    provenance: tuple[str, ...] = ()
    uncertainty_m: float | None = None
    synthetic: bool = False


def camera_matrix(width: int, height: int, hfov_deg: float = 68.0) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    fy = fx
    return np.array([[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _cam_center(sample: TelemetrySample, origin: Origin) -> np.ndarray:
    return geodetic_to_enu(sample.lat, sample.lon, sample.alt, origin)


def _rotation_world_from_cam(sample: TelemetrySample) -> np.ndarray:
    """Camera looks mostly nadir with aircraft heading around +Z_down / -U."""
    forward = heading_to_enu(sample.heading)
    down = np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    # Camera axes: x right, y down in image ~ -forward-ish for nadir, z optical ~ down
    z_cam = down
    x_cam = right
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / max(np.linalg.norm(y_cam), 1e-9)
    x_cam = np.cross(y_cam, z_cam)
    return np.stack([x_cam, y_cam, z_cam], axis=1)


def triangulate_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pose_a: TelemetrySample,
    pose_b: TelemetrySample,
    origin: Origin,
    gps_reliability: float,
) -> list[ReconPoint]:
    h, w = img_a.shape[:2]
    k = camera_matrix(w, h)
    orb = cv2.ORB_create(nfeatures=1800)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return _nadir_unproject(img_a, pose_a, origin, gps_reliability)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 10:
        return _nadir_unproject(img_a, pose_a, origin, gps_reliability)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    ca = _cam_center(pose_a, origin)
    cb = _cam_center(pose_b, origin)
    ra = _rotation_world_from_cam(pose_a)
    rb = _rotation_world_from_cam(pose_b)
    # World-to-camera
    rwa = ra.T
    rwb = rb.T
    ta = -rwa @ ca
    tb = -rwb @ cb
    p1 = k @ np.hstack([rwa, ta.reshape(3, 1)])
    p2 = k @ np.hstack([rwb, tb.reshape(3, 1)])
    homog = cv2.triangulatePoints(p1, p2, pts1.T, pts2.T)
    xyz = (homog[:3] / np.clip(homog[3], 1e-8, None)).T

    gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_n = min(sharp / 120.0, 1.0)
    baseline = float(np.linalg.norm(cb - ca))
    view_div = min(baseline / 12.0, 1.0)

    points: list[ReconPoint] = []
    for i, x in enumerate(xyz):
        if not np.isfinite(x).all():
            continue
        if abs(x[2]) > 80 or abs(x[0]) > 250 or abs(x[1]) > 250:
            continue
        # Cheirality / near-ground filter relative to camera altitude
        if x[2] > max(ca[2], 2.0) + 8:
            continue
        u, v = pts1[i]
        b, g, r = img_a[min(int(v), h - 1), min(int(u), w - 1)]
        residual = abs(float(x[2]))  # nadir scenes hug u≈0 ground after ENU
        conf = point_confidence(
            observations=2.0,
            viewpoint_div=view_div,
            sharpness=sharp_n,
            gps_reliability=gps_reliability,
            residual=min(residual, 8.0),
        )
        points.append(
            ReconPoint(
                e=float(x[0]),
                n=float(x[1]),
                u=float(x[2]),
                r=int(r),
                g=int(g),
                b=int(b),
                conf=conf,
                observations=2,
                source="photogrammetry",
                provenance=("sfm",),
                uncertainty_m=max(0.05, min(20.0, 1.0 / max(conf, 0.05))),
            )
        )
    if len(points) < 20:
        points.extend(_nadir_unproject(img_a, pose_a, origin, gps_reliability, subsample=4))
    return points


def _nadir_unproject(
    img: np.ndarray,
    pose: TelemetrySample,
    origin: Origin,
    gps_reliability: float,
    subsample: int = 6,
) -> list[ReconPoint]:
    """Altitude-as-depth fallback when stereo is too sparse (typical single pass)."""
    h, w = img.shape[:2]
    k = camera_matrix(w, h)
    k_inv = np.linalg.inv(k)
    center = _cam_center(pose, origin)
    rot = _rotation_world_from_cam(pose)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp_n = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 120.0, 1.0)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01, minDistance=8)
    uv: list[tuple[int, int]]
    if corners is None:
        uv = [(x, y) for y in range(8, h - 8, subsample * 8) for x in range(8, w - 8, subsample * 8)]
    else:
        uv = [(int(p[0][0]), int(p[0][1])) for p in corners]
        uv.extend((x, y) for y in range(16, h - 16, 28) for x in range(16, w - 16, 28))

    points: list[ReconPoint] = []
    for x, y in uv[:: max(subsample, 1)]:
        pix = np.array([x, y, 1.0])
        ray_cam = k_inv @ pix
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        ray_world = rot @ ray_cam  # camera z is down, ray mostly -U
        if abs(ray_world[2]) < 1e-4:
            continue
        # Intersect with local ground u = 0, with a small building bump from brightness
        t = -center[2] / ray_world[2]
        if t <= 1.0 or t > pose.alt * 4:
            continue
        p = center + t * ray_world
        bgr = img[min(y, h - 1), min(x, w - 1)]
        lum = float(np.mean(bgr))
        if lum > 70:
            p = p.copy()
            p[2] += (lum - 70.0) / 255.0 * 9.0  # crude height prior, marked medium/low
            extra_low = 0.22
        else:
            extra_low = 0.0
        conf = point_confidence(
            observations=1.0,
            viewpoint_div=0.25,
            sharpness=sharp_n,
            gps_reliability=gps_reliability,
            residual=2.5,
        )
        conf = max(0.08, conf - extra_low)
        points.append(
            ReconPoint(
                e=float(p[0]),
                n=float(p[1]),
                u=float(p[2]),
                r=int(bgr[2]),
                g=int(bgr[1]),
                b=int(bgr[0]),
                conf=conf,
                observations=1,
                source="photogrammetry",
                provenance=("nadir_fallback",),
                uncertainty_m=max(0.1, min(20.0, 2.0 / max(conf, 0.05))),
            )
        )
    return points


def voxel_downsample(points: list[ReconPoint], voxel: float = 0.8) -> list[ReconPoint]:
    buckets: dict[tuple[int, int, int], list[ReconPoint]] = {}
    for p in points:
        key = (int(p.e / voxel), int(p.n / voxel), int(p.u / voxel))
        buckets.setdefault(key, []).append(p)
    fused: list[ReconPoint] = []
    for group in buckets.values():
        n = len(group)
        labels = tuple(sorted({label for p in group for label in p.provenance}))
        source_counts: dict[str, int] = {}
        for p in group:
            source_counts[p.source] = source_counts.get(p.source, 0) + 1
        source = max(source_counts, key=source_counts.get)
        uncertainties = [p.uncertainty_m for p in group if p.uncertainty_m is not None]
        fused.append(
            ReconPoint(
                e=sum(p.e for p in group) / n,
                n=sum(p.n for p in group) / n,
                u=sum(p.u for p in group) / n,
                r=int(sum(p.r for p in group) / n),
                g=int(sum(p.g for p in group) / n),
                b=int(sum(p.b for p in group) / n),
                conf=min(0.98, max(p.conf for p in group) + min(0.12, 0.02 * n)),
                observations=sum(p.observations for p in group),
                source=source,
                provenance=labels,
                uncertainty_m=min(uncertainties) if uncertainties else None,
                synthetic=all(p.synthetic for p in group),
            )
        )
    return fused
