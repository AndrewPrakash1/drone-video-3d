from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .contracts import SpatialTrainingSample


@dataclass
class DatasetManifest:
    schema: str = "onepass.spatial-dataset.v1"
    dataset_id: str = "onepass-spatial"
    license_policy: str = "verify-each-source"
    samples: list[SpatialTrainingSample] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        ids = [sample.sample_id for sample in self.samples]
        scenes = {sample.scene_id for sample in self.samples}
        if len(ids) != len(set(ids)):
            raise ValueError("sample IDs must be unique")
        for sample in self.samples:
            sample.validate()
            if sample.license_name == "unknown":
                raise ValueError(f"sample {sample.sample_id} has no recorded license")
        assigned: dict[str, str] = {}
        for split, sample_ids in self.splits.items():
            if split not in {"train", "val", "test"}:
                raise ValueError(f"invalid split: {split}")
            for sample_id in sample_ids:
                if sample_id not in ids:
                    raise ValueError(f"split references unknown sample: {sample_id}")
                if sample_id in assigned:
                    raise ValueError(f"sample occurs in multiple splits: {sample_id}")
                assigned[sample_id] = split
        if set(assigned) != set(ids):
            raise ValueError("every sample must belong to exactly one split")
        split_scenes = {split: {next(sample.scene_id for sample in self.samples if sample.sample_id == sample_id) for sample_id in sample_ids} for split, sample_ids in self.splits.items()}
        if split_scenes.get("train", set()) & split_scenes.get("val", set()) or split_scenes.get("train", set()) & split_scenes.get("test", set()) or split_scenes.get("val", set()) & split_scenes.get("test", set()):
            raise ValueError("scene IDs must not cross train/val/test splits")
        if not scenes:
            raise ValueError("manifest must contain samples")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "license_policy": self.license_policy,
            "samples": [asdict(sample) for sample in self.samples],
            "splits": self.splits,
            "metadata": self.metadata,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "DatasetManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(
            schema=str(value.get("schema") or ""),
            dataset_id=str(value.get("dataset_id") or ""),
            license_policy=str(value.get("license_policy") or ""),
            samples=[SpatialTrainingSample.from_dict(item) for item in value.get("samples") or []],
            splits={str(key): [str(sample_id) for sample_id in sample_ids] for key, sample_ids in (value.get("splits") or {}).items()},
            metadata=dict(value.get("metadata") or {}),
        )
        manifest.validate()
        return manifest
