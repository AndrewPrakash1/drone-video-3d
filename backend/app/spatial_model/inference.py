from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapter import ScaffoldSpatialModelAdapter


def run_optional_stage(artifact_dir: Path, scene: dict[str, Any] | None, spatial: dict[str, Any] | None) -> dict[str, Any]:
    status = ScaffoldSpatialModelAdapter().status()
    result = {
        "status": "skipped",
        "reason": status["reason"],
        "model": status,
        "proposals": [],
        "measurement_safe": False,
    }
    if scene is None or spatial is None:
        result["reason"] = "scene or spatial evidence is unavailable"
    (artifact_dir / "spatial_model.json").write_text(json.dumps(result), encoding="utf-8")
    return result
