from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import SpatialTrainingSample
from .manifest import DatasetManifest
from .model import MODEL_VERSION, MetricSpatialModel, TORCH_AVAILABLE


def train_model(
    samples: list[SpatialTrainingSample],
    output: Path,
    *,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    device: str | None = None,
    data_root: Path | None = None,
    val_samples: list[SpatialTrainingSample] | None = None,
    image_size: int = 128,
    views: int = 6,
) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "PyTorch is not installed"}
    if not samples:
        return {"status": "skipped", "reason": "no training samples"}
    if data_root is None:
        raise ValueError("data_root is required for tensorization")
    for sample in samples:
        sample.validate()
    import torch
    from torch.utils.data import DataLoader

    from .tensorize import SpatialTensorDataset

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader = DataLoader(SpatialTensorDataset(samples, data_root, views=views, image_size=image_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SpatialTensorDataset(val_samples or samples[:1], data_root, views=views, image_size=image_size), batch_size=batch_size)
    model = MetricSpatialModel(image_size=image_size).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            frames = batch["frames"].to(target_device)
            camera_features = batch["camera_features"].to(target_device)
            patch_queries = batch["patch_queries"].to(target_device)
            view_queries = batch["view_queries"].to(target_device)
            targets = {key: value.to(target_device) for key, value in batch["targets"].items()}
            loss = model.loss(model(frames, camera_features, patch_queries, view_queries), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(target_device)
                camera_features = batch["camera_features"].to(target_device)
                patch_queries = batch["patch_queries"].to(target_device)
                view_queries = batch["view_queries"].to(target_device)
                targets = {key: value.to(target_device) for key, value in batch["targets"].items()}
                val_loss += float(model.loss(model(frames, camera_features, patch_queries, view_queries), targets).cpu())
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        row = {"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        if val_loss <= best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "model_version": MODEL_VERSION, "config": {"image_size": image_size, "views": views}, "epoch": epoch + 1, "val_loss": val_loss}, output)

    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps({"model_version": MODEL_VERSION, "device": str(target_device), "history": history}), encoding="utf-8")
    return {"status": "ok", "model_version": MODEL_VERSION, "device": str(target_device), "checkpoint": str(output), "metrics": str(metrics_path), "history": history}


def train_manifest(manifest_path: Path, output: Path, **kwargs: Any) -> dict[str, Any]:
    manifest = DatasetManifest.read(manifest_path)
    samples_by_id = {sample.sample_id: sample for sample in manifest.samples}
    train = [samples_by_id[sample_id] for sample_id in manifest.splits["train"]]
    val = [samples_by_id[sample_id] for sample_id in manifest.splits.get("val", [])]
    return train_model(train, output, data_root=manifest_path.parent, val_samples=val, **kwargs)


def build_model(**kwargs: Any) -> MetricSpatialModel:
    return MetricSpatialModel(**kwargs)
