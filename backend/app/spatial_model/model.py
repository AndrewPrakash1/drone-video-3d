from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on the CPU-only baseline
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


MODEL_VERSION = "onepass-spatial-0.1"


if TORCH_AVAILABLE:

    class MetricSpatialModel(nn.Module):
        """Small metric-conditioned backbone for the first OnePass model.

        The model predicts held-out camera views and completion attributes from
        noisy frame tokens, camera metadata, and a patch query in ENU metres.
        It is deliberately compact so it can be trained on a single GPU before
        scaling the same contracts into a foundation model.
        """

        def __init__(self, image_size: int = 128, frame_dim: int = 192, model_dim: int = 256, layers: int = 4, heads: int = 8) -> None:
            super().__init__()
            self.image_size = image_size
            self.frame_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.frame_projection = nn.Linear(96, frame_dim)
            self.camera_projection = nn.Sequential(nn.Linear(25, frame_dim), nn.GELU(), nn.Linear(frame_dim, frame_dim))
            self.token_projection = nn.Linear(frame_dim * 2, model_dim)
            encoder_layer = nn.TransformerEncoderLayer(model_dim, heads, model_dim * 4, batch_first=True, norm_first=True)
            self.scene_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.patch_query = nn.Sequential(nn.Linear(6, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
            self.view_query = nn.Sequential(nn.Linear(9, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
            self.completion_head = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Linear(model_dim, 5))
            self.view_head = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Linear(model_dim, 4))

        def encode_scene(self, frames: Tensor, camera_features: Tensor) -> Tensor:
            batch, views, channels, height, width = frames.shape
            image_tokens = self.frame_encoder(frames.reshape(batch * views, channels, height, width)).flatten(1)
            image_tokens = self.frame_projection(image_tokens).reshape(batch, views, -1)
            camera_tokens = self.camera_projection(camera_features)
            return self.scene_encoder(self.token_projection(torch.cat([image_tokens, camera_tokens], dim=-1)))

        def forward(
            self,
            frames: Tensor,
            camera_features: Tensor,
            patch_queries: Tensor,
            view_queries: Tensor,
        ) -> dict[str, Tensor]:
            scene_tokens = self.encode_scene(frames, camera_features)
            scene_context = scene_tokens.mean(dim=1)
            patch_context = scene_context[:, None, :] + self.patch_query(patch_queries)
            view_context = scene_context[:, None, :] + self.view_query(view_queries)
            completion = self.completion_head(torch.cat([patch_context, scene_context[:, None, :].expand_as(patch_context)], dim=-1))
            view = self.view_head(torch.cat([view_context, scene_context[:, None, :].expand_as(view_context)], dim=-1))
            return {
                "completion": completion,
                "novel_view": view,
                "scene_tokens": scene_tokens,
            }

        @staticmethod
        def loss(outputs: dict[str, Tensor], targets: dict[str, Tensor], completion_weight: float = 1.0, view_weight: float = 1.0) -> Tensor:
            completion_target = targets["completion"]
            view_target = targets["novel_view"]
            completion_mask = targets.get("completion_mask", torch.ones_like(completion_target[..., :1]))
            view_mask = targets.get("view_mask", torch.ones_like(view_target[..., :1]))
            completion_error = ((outputs["completion"] - completion_target).abs() * completion_mask).mean()
            view_error = ((outputs["novel_view"] - view_target).abs() * view_mask).mean()
            uncertainty_penalty = outputs["completion"][..., 4:5].sigmoid().mean() * 0.01
            return completion_weight * completion_error + view_weight * view_error + uncertainty_penalty

else:

    class MetricSpatialModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("MetricSpatialModel requires PyTorch; install the optional spatial-model dependencies")


def model_status(checkpoint: str | None = None) -> dict[str, Any]:
    from pathlib import Path

    path = Path(checkpoint) if checkpoint else None
    exists = bool(path and path.exists())
    return {
        "version": MODEL_VERSION,
        "torch": TORCH_AVAILABLE,
        "architecture": "metric-conditioned-spatial-transformer",
        "architecture_ready": TORCH_AVAILABLE,
        "training_ready": TORCH_AVAILABLE,
        "checkpoint": str(path) if path else None,
        "checkpoint_exists": exists,
        "inference_ready": False,
    }
