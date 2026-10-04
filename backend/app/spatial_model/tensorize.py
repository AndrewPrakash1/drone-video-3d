from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .contracts import SpatialTrainingSample

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


class SpatialTensorDataset(Dataset):
    def __init__(self, samples: list[SpatialTrainingSample], root: Path, views: int = 6, image_size: int = 128) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for tensorization")
        if not samples:
            raise ValueError("dataset has no samples")
        self.samples = samples
        self.root = root
        self.views = views
        self.image_size = image_size
        for sample in samples:
            sample.validate()
            if len(sample.frame_paths) < views:
                raise ValueError(f"sample {sample.sample_id} has fewer than {views} frames")
            if not sample.patch_targets_path or not sample.view_targets_path:
                raise ValueError(f"sample {sample.sample_id} has no tensor targets")

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: str) -> np.ndarray:
        image = cv2.imread(str(self.root / path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.root / path)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        frame_paths = sample.frame_paths[: self.views]
        frames = np.stack([self._image(path) for path in frame_paths], axis=0)
        frames = np.transpose(frames, (0, 3, 1, 2))
        camera_features = np.asarray([
            sample.camera_poses_w2c[i][:16] + sample.intrinsics[i][:9]
            for i in range(self.views)
        ], dtype=np.float32)
        patch_targets = np.load(self.root / sample.patch_targets_path).astype(np.float32)
        view_targets = np.load(self.root / sample.view_targets_path).astype(np.float32)
        patch_query = np.asarray([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        view_query = np.asarray([[0.0, 0.0, 18.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        return {
            "frames": torch.from_numpy(frames),
            "camera_features": torch.from_numpy(camera_features),
            "patch_queries": torch.from_numpy(patch_query),
            "view_queries": torch.from_numpy(view_query),
            "targets": {
                "completion": torch.from_numpy(patch_targets),
                "novel_view": torch.from_numpy(view_targets),
            },
            "sample_id": sample.sample_id,
        }
