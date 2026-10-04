from __future__ import annotations

import json
from pathlib import Path

from .contracts import SpatialTrainingSample


def write_sample(path: Path, sample: SpatialTrainingSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample.to_dict(), indent=2), encoding="utf-8")


def read_sample(path: Path) -> SpatialTrainingSample:
    return SpatialTrainingSample.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_samples(root: Path) -> list[SpatialTrainingSample]:
    samples = []
    for path in sorted(root.glob("*.json")):
        samples.append(read_sample(path))
    return samples
