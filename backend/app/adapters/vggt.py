"""Optional VGGT feed-forward reconstruction adapter."""

from __future__ import annotations

import os
from typing import Any


def vggt_available() -> bool:
    weights = os.environ.get("VGGT_WEIGHTS", "").strip()
    if not weights or not os.path.isfile(weights):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reconstruct_chunk(*_args: Any, **_kwargs: Any) -> None:
    if not vggt_available():
        return None
    raise RuntimeError("VGGT weights found but the inference wrapper is not bundled in this slice")
