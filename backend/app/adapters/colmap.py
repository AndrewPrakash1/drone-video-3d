"""Optional COLMAP SfM adapter. Fuses sparse points into the live ENU cloud."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..align import umeyama
from ..confidence import point_confidence
from ..geo import Origin, geodetic_to_enu
from ..reconstruct import ReconPoint
from ..telemetry import TelemetrySample

log = logging.getLogger("onepass.colmap")


def colmap_available() -> bool:
    return shutil.which("colmap") is not None


def run_sfm(image_dir: Path, work_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("colmap")
    if not binary:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    sparse = work_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    try:
        gpu = os.environ.get("COLMAP_USE_GPU", "1").strip() not in {"0", "false", "no"}
        extract = [
            binary,
            "feature_extractor",
            "--database_path",
            str(db),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            "1" if gpu else "0",
        ]
        match = [binary, "exhaustive_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", "1" if gpu else "0"]
        try:
            _run(extract)
            _run(match)
        except subprocess.CalledProcessError:
            if not gpu:
                raise
            log.warning("COLMAP GPU SIFT failed; retrying on CPU")
            if db.exists():
                db.unlink()
            extract[-1] = "0"
            match[-1] = "0"
            _run(extract)
            _run(match)
        _run(
            [
                binary,
                "mapper",
                "--database_path",
                str(db),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(sparse),
            ]
        )
        model_dir = _first_model(sparse)
        if model_dir is None:
            return {"status": "skipped", "reason": "mapper produced no model"}
        txt_dir = work_dir / "sparse_txt"
        txt_dir.mkdir(exist_ok=True)
        _run(
            [
                binary,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(txt_dir),
                "--output_type",
                "TXT",
            ]
        )
        return {"sparse_dir": str(model_dir), "txt_dir": str(txt_dir), "status": "ok"}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("COLMAP failed: %s", exc)
        return {"status": "skipped", "reason": str(exc)}


def reconstruct_aligned(
    image_dir: Path,
    work_dir: Path,
    poses_by_name: dict[str, TelemetrySample],
    origin: Origin,
    gps_reliability: float,
) -> tuple[list[ReconPoint], dict[str, Any]]:
    info = run_sfm(image_dir, work_dir) or {"status": "skipped", "reason": "colmap missing"}
    if info.get("status") != "ok":
        return [], info
    txt = Path(info["txt_dir"])
    images = _parse_images_txt(txt / "images.txt")
    points = _parse_points3d_txt(txt / "points3D.txt")
    if not images or not points:
        info["reason"] = "empty reconstruction"
        info["status"] = "skipped"
        return [], info

    src, dst = [], []
    for name, (_qid, center) in images.items():
        pose = poses_by_name.get(name) or poses_by_name.get(Path(name).name)
        if pose is None:
            continue
        src.append(center)
        dst.append(geodetic_to_enu(pose.lat, pose.lon, pose.alt, origin))
    if len(src) < 2:
        info["reason"] = "could not match COLMAP images to GPS"
        info["status"] = "skipped"
        return [], info
    aligned = umeyama(np.stack(src), np.stack(dst))
    if aligned is None:
        info["reason"] = "Umeyama alignment failed"
        info["status"] = "skipped"
        return [], info
    scale, rot, trans = aligned
    out: list[ReconPoint] = []
    for xyz, rgb, err in points:
        world = scale * (rot @ xyz) + trans
        residual = float(min(err, 8.0))
        conf = point_confidence(
            observations=4.0,
            viewpoint_div=0.7,
            sharpness=0.8,
            gps_reliability=gps_reliability,
            residual=residual,
        )
        out.append(
            ReconPoint(
                e=float(world[0]),
                n=float(world[1]),
                u=float(world[2]),
                r=int(rgb[0]),
                g=int(rgb[1]),
                b=int(rgb[2]),
                conf=float(min(0.97, conf + 0.08)),
                observations=6,
            )
        )
    info["points"] = len(out)
    info["scale"] = scale
    info["aligned_cameras"] = len(src)
    return out, info


def _run(cmd: list[str], timeout: int = 420) -> None:
    subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)


def _first_model(sparse: Path) -> Path | None:
    numbered = sorted(p for p in sparse.iterdir() if p.is_dir())
    for cand in numbered:
        if (cand / "points3D.bin").exists() or (cand / "points3D.txt").exists():
            return cand
    if (sparse / "points3D.bin").exists():
        return sparse
    return None


def _parse_images_txt(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (quaternion_wxyz, camera_center)."""
    if not path.exists():
        return {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        r = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        center = -r.T @ t
        out[name] = (np.array([qw, qx, qy, qz]), center)
        # next line is 2D points — skip
        if i < len(lines) and not lines[i].strip().startswith("#"):
            i += 1
    return out


def _parse_points3d_txt(path: Path) -> list[tuple[np.ndarray, tuple[int, int, int], float]]:
    if not path.exists():
        return []
    pts = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
        rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
        err = float(parts[7])
        pts.append((xyz, rgb, err))
    return pts


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = max((qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5, 1e-12)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
