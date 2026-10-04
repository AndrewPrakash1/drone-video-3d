"""Versioned metric-scene contracts shared by reconstruction and spatial models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCENE_SCHEMA = "onepass.scene.v1"


def write_scene_manifest(
    artifact_dir: Path,
    *,
    origin: dict[str, float] | None,
    point_count: int,
    mesh: dict[str, Any] | None,
    spatial: dict[str, Any] | None,
    cameras: int,
    reconstruction_revision: str = "metric-1",
) -> dict[str, Any]:
    """Persist the authoritative scene boundary consumed by future model stages."""
    payload: dict[str, Any] = {
        "schema": SCENE_SCHEMA,
        "coordinate_frame": "enu_m",
        "origin": origin,
        "reconstruction": {
            "revision": reconstruction_revision,
            "measurement_authoritative": True,
            "points": point_count,
            "cameras": cameras,
        },
        "mesh": {
            "revision": "mesh-1",
            "status": (mesh or {}).get("status", "skipped"),
            "method": (mesh or {}).get("method"),
            "measurement_authoritative": False,
        },
        "spatial": {
            "revision": "spatial-1" if spatial else None,
            "available": spatial is not None,
            "generated_geometry_allowed": False,
        },
        "generated_layers": [],
        "measurement": {
            "source": "metric_cloud",
            "generated_geometry_included": False,
        },
    }
    (artifact_dir / "scene.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def read_scene_manifest(artifact_dir: Path) -> dict[str, Any] | None:
    path = artifact_dir / "scene.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
