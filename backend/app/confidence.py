"""Per-point reconstruction confidence."""

from __future__ import annotations

import numpy as np


def classify(value: float) -> str:
    if value >= 0.72:
        return "high"
    if value >= 0.42:
        return "medium"
    return "low"


def point_confidence(
    observations: float,
    viewpoint_div: float,
    sharpness: float,
    gps_reliability: float,
    residual: float,
) -> float:
    obs = min(observations / 6.0, 1.0)
    view = min(viewpoint_div, 1.0)
    sharp = min(sharpness, 1.0)
    res = float(np.clip(1.0 - residual / 4.0, 0.0, 1.0))
    return float(np.clip(0.28 * obs + 0.22 * view + 0.18 * sharp + 0.18 * gps_reliability + 0.14 * res, 0.05, 0.98))
