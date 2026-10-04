from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from .contracts import CompletionProposal
from .model import model_status


class SpatialModelAdapter(Protocol):
    def status(self) -> dict[str, Any]: ...

    def propose_completion(self, scene: dict[str, Any], patch: dict[str, Any]) -> CompletionProposal: ...


class ScaffoldSpatialModelAdapter:
    def status(self) -> dict[str, Any]:
        checkpoint = os.environ.get("ONEPASS_SPATIAL_MODEL_CHECKPOINT")
        details = model_status(checkpoint)
        return {
            "available": False,
            "status": "scaffold",
            "checkpoint": checkpoint,
            "device": None,
            "version": details["version"],
            "architecture": details["architecture"],
            "architecture_ready": details["architecture_ready"],
            "training_ready": details["training_ready"],
            "inference_ready": details["inference_ready"],
            "reason": "model tensorization, checkpoint loading, and inference are not implemented yet",
        }

    def propose_completion(self, scene: dict[str, Any], patch: dict[str, Any]) -> CompletionProposal:
        raise RuntimeError("spatial model inference is not implemented")


def spatial_model_status() -> dict[str, Any]:
    return ScaffoldSpatialModelAdapter().status()


def checkpoint_path() -> Path | None:
    value = os.environ.get("ONEPASS_SPATIAL_MODEL_CHECKPOINT")
    return Path(value) if value else None
