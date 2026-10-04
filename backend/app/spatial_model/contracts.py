from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SpatialTrainingSample:
    sample_id: str
    scene_id: str
    frame_paths: list[str]
    timestamps_s: list[float]
    camera_poses_w2c: list[list[float]]
    intrinsics: list[list[float]]
    origin: dict[str, float]
    visible_geometry_path: str
    visibility_path: str
    target_view_paths: list[str] = field(default_factory=list)
    target_geometry_path: str | None = None
    patch_targets_path: str | None = None
    view_targets_path: str | None = None
    license_name: str = "unknown"
    license_url: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id or not self.scene_id:
            raise ValueError("sample_id and scene_id are required")
        if not self.frame_paths:
            raise ValueError("at least one frame is required")
        lengths = {len(self.frame_paths), len(self.timestamps_s), len(self.camera_poses_w2c), len(self.intrinsics)}
        if len(lengths) != 1:
            raise ValueError("frame paths, timestamps, poses, and intrinsics must have equal lengths")
        if set(self.origin) != {"lat", "lon", "alt"}:
            raise ValueError("origin must contain lat, lon, and alt")
        if not self.visible_geometry_path or not self.visibility_path:
            raise ValueError("visible geometry and visibility paths are required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpatialTrainingSample":
        sample = cls(
            sample_id=str(value.get("sample_id") or ""),
            scene_id=str(value.get("scene_id") or ""),
            frame_paths=[str(path) for path in value.get("frame_paths") or []],
            timestamps_s=[float(timestamp) for timestamp in value.get("timestamps_s") or []],
            camera_poses_w2c=[[float(component) for component in pose] for pose in value.get("camera_poses_w2c") or []],
            intrinsics=[[float(component) for component in matrix] for matrix in value.get("intrinsics") or []],
            origin={key: float(value["origin"][key]) for key in ("lat", "lon", "alt")},
            visible_geometry_path=str(value.get("visible_geometry_path") or ""),
            visibility_path=str(value.get("visibility_path") or ""),
            target_view_paths=[str(path) for path in value.get("target_view_paths") or []],
            target_geometry_path=str(value["target_geometry_path"]) if value.get("target_geometry_path") else None,
            patch_targets_path=str(value["patch_targets_path"]) if value.get("patch_targets_path") else None,
            view_targets_path=str(value["view_targets_path"]) if value.get("view_targets_path") else None,
            license_name=str(value.get("license_name") or "unknown"),
            license_url=str(value["license_url"]) if value.get("license_url") else None,
            source_url=str(value["source_url"]) if value.get("source_url") else None,
            metadata=dict(value.get("metadata") or {}),
        )
        sample.validate()
        return sample


@dataclass
class CompletionProposal:
    proposal_id: str
    scene_revision: str
    patch_id: int
    state: str = "proposed"
    measurement_safe: bool = False
    uncertainty: float = 1.0
    generated_geometry_path: str | None = None
    novel_view_paths: list[str] = field(default_factory=list)
    conditioning_views: list[int] = field(default_factory=list)
    model_version: str = "unavailable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.proposal_id or not self.scene_revision:
            raise ValueError("proposal_id and scene_revision are required")
        if self.patch_id < 0:
            raise ValueError("patch_id must be non-negative")
        if self.state not in {"proposed", "accepted", "rejected"}:
            raise ValueError("invalid proposal state")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be between 0 and 1")
        if self.measurement_safe:
            raise ValueError("generated proposals cannot be measurement safe")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
