from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .contracts import SpatialTrainingSample
from .manifest import DatasetManifest


def _scene_image(size: int, scene_id: int, view_id: int) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = (35 + scene_id * 7, 55 + view_id * 5, 80 + scene_id * 3)
    rng = np.random.default_rng(scene_id * 100 + view_id)
    for _ in range(14):
        x, y = rng.integers(0, size - 20, 2)
        w, h = rng.integers(8, 42, 2)
        color = tuple(int(value) for value in rng.integers(70, 230, 3))
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
    cv2.putText(image, f"scene {scene_id} view {view_id}", (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
    return image


def _camera_features(view_id: int, views: int, size: int) -> list[float]:
    angle = 2.0 * np.pi * view_id / max(views, 1)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(np.cos(angle) * 12.0)
    pose[1, 3] = float(np.sin(angle) * 12.0)
    pose[2, 3] = 18.0
    k = np.array([[0.8 * size, 0.0, size / 2.0], [0.0, 0.8 * size, size / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.concatenate([pose.reshape(-1), k.reshape(-1)]).tolist()


def generate_dataset(root: Path, scenes: int = 12, views: int = 6, size: int = 128, seed: int = 7) -> DatasetManifest:
    rng = np.random.default_rng(seed)
    samples: list[SpatialTrainingSample] = []
    splits = {"train": [], "val": [], "test": []}
    for scene_id in range(scenes):
        scene_dir = root / f"scene_{scene_id:04d}"
        image_dir = scene_dir / "images"
        target_dir = scene_dir / "targets"
        image_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        target_paths = []
        poses = []
        intrinsics = []
        for view_id in range(views):
            clean = _scene_image(size, scene_id, view_id)
            target_path = target_dir / f"view_{view_id:03d}.png"
            cv2.imwrite(str(target_path), clean)
            noisy = clean.copy()
            if view_id % 2 == 0:
                noisy = cv2.GaussianBlur(noisy, (0, 0), sigmaX=1.4)
            noise = rng.normal(0.0, 10.0, noisy.shape)
            noisy = np.clip(noisy.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            frame_path = image_dir / f"view_{view_id:03d}.jpg"
            cv2.imwrite(str(frame_path), noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            frame_paths.append(str(frame_path.relative_to(root)))
            target_paths.append(str(target_path.relative_to(root)))
            poses.append(_camera_features(view_id, views, size)[:16])
            intrinsics.append(_camera_features(view_id, views, size)[16:])
        visible = np.array([float(scene_id) / max(scenes, 1), 0.0, 0.0, 1.0], dtype=np.float32)
        patch_targets = np.array([[0.0, 0.0, 4.0, 0.5, 0.0]], dtype=np.float32)
        view_targets = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        np.save(scene_dir / "patch_targets.npy", patch_targets)
        np.save(scene_dir / "view_targets.npy", view_targets)
        (scene_dir / "visibility.json").write_text(json.dumps({"seen": [1, 1, 0, 0], "patches": [{"id": 0, "state": "unseen"}]}), encoding="utf-8")
        (scene_dir / "visible_geometry.json").write_text(json.dumps({"coordinate_frame": "enu_m", "points": []}), encoding="utf-8")
        sample = SpatialTrainingSample(
            sample_id=f"scene_{scene_id:04d}",
            scene_id=f"scene_{scene_id:04d}",
            frame_paths=frame_paths,
            timestamps_s=[float(i) for i in range(views)],
            camera_poses_w2c=poses,
            intrinsics=intrinsics,
            origin={"lat": 0.0, "lon": 0.0, "alt": 0.0},
            visible_geometry_path=str((scene_dir / "visible_geometry.json").relative_to(root)),
            visibility_path=str((scene_dir / "visibility.json").relative_to(root)),
            target_view_paths=target_paths,
            patch_targets_path=str((scene_dir / "patch_targets.npy").relative_to(root)),
            view_targets_path=str((scene_dir / "view_targets.npy").relative_to(root)),
            license_name="OnePass synthetic generated",
            metadata={"synthetic": True, "noise": ["blur", "jpeg", "gaussian"]},
        )
        samples.append(sample)
        split = "test" if scene_id % 10 == 0 else "val" if scene_id % 10 == 1 else "train"
        splits[split].append(sample.sample_id)
    manifest = DatasetManifest(samples=samples, splits=splits, metadata={"generator": "onepass.synthetic.v1", "seed": seed})
    manifest.write(root / "manifest.json")
    return manifest
