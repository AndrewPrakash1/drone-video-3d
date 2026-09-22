"""Metric evaluation against a known reference length."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceSegment:
    name: str
    true_length_m: float
    a_enu: tuple[float, float, float]
    b_enu: tuple[float, float, float]
    tolerance_m: float | None = None
    tolerance_pct: float = 5.0


def default_tolerance(true_length_m: float, tolerance_m: float | None, tolerance_pct: float) -> float:
    pct = abs(true_length_m) * (tolerance_pct / 100.0)
    floor = 1.0 if tolerance_m is None else tolerance_m
    return max(pct, floor)


def measure_length(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(b) - np.array(a)))


def evaluate(
    measured_m: float,
    true_length_m: float,
    tolerance_m: float | None = None,
    tolerance_pct: float = 5.0,
) -> dict:
    err = abs(measured_m - true_length_m)
    pct = 0.0 if true_length_m == 0 else 100.0 * err / abs(true_length_m)
    tol = default_tolerance(true_length_m, tolerance_m, tolerance_pct)
    return {
        "measured_m": round(measured_m, 3),
        "true_length_m": round(true_length_m, 3),
        "abs_error_m": round(err, 3),
        "pct_error": round(pct, 2),
        "tolerance_m": round(tol, 3),
        "pass": err <= tol,
    }


def evaluate_segment(seg: ReferenceSegment) -> dict:
    measured = measure_length(seg.a_enu, seg.b_enu)
    result = evaluate(measured, seg.true_length_m, seg.tolerance_m, seg.tolerance_pct)
    result["name"] = seg.name
    result["a_enu"] = list(seg.a_enu)
    result["b_enu"] = list(seg.b_enu)
    return result
