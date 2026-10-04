"""Proxy single-pass mission with a 20 m rooftop of known length."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .confidence import point_confidence
from .geo import Origin, enu_to_geodetic, make_origin
from .metric import ReferenceSegment, evaluate_segment
from .reconstruct import ReconPoint

# South Delhi proxy range — not operational data, a geometrically known synthetic site.
ORIGIN_LAT = 28.54480
ORIGIN_LON = 77.19240
ORIGIN_ALT = 216.0

ROOFTOP_LEN = 20.0
ROOFTOP_WIDTH = 12.0
ROOFTOP_HEIGHT = 9.0


def origin() -> Origin:
    return make_origin(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)


def reference_segment() -> ReferenceSegment:
    # Rooftop ridge along east, 20.0 m true length.
    return ReferenceSegment(
        name="north_rooftop_eave",
        true_length_m=ROOFTOP_LEN,
        a_enu=(-10.0, 8.0, ROOFTOP_HEIGHT),
        b_enu=(10.0, 8.0, ROOFTOP_HEIGHT),
        tolerance_m=1.0,
        tolerance_pct=5.0,
    )


def building_corners_geodetic() -> dict:
    o = origin()
    a = enu_to_geodetic(-10.0, 8.0, ROOFTOP_HEIGHT, o)
    b = enu_to_geodetic(10.0, 8.0, ROOFTOP_HEIGHT, o)
    return {
        "name": "north_rooftop_eave",
        "true_length_m": ROOFTOP_LEN,
        "a": {"lat": a[0], "lon": a[1], "height": float(a[2])},
        "b": {"lat": b[0], "lon": b[1], "height": float(b[2])},
        "a_enu": [-10.0, 8.0, ROOFTOP_HEIGHT],
        "b_enu": [10.0, 8.0, ROOFTOP_HEIGHT],
        "tolerance_m": 1.0,
        "tolerance_pct": 5.0,
        "note": "Synthetic proxy building. Event data will replace this reference.",
    }


def generate_telemetry(n: int = 64) -> list[dict]:
    """Single pass west→east, slight crab, 80 m AGL."""
    rows = []
    for i in range(n):
        t = i * 0.45
        e = -55.0 + 110.0 * (i / (n - 1))
        n_off = -4.0 + 1.2 * math.sin(i / 7.0)
        u = 78.0 + 0.8 * math.sin(i / 5.0)
        hdop = 0.9 + 0.15 * abs(math.sin(i / 9.0))
        if 22 <= i <= 26:
            hdop = 3.4  # brief GPS degradation
        lat, lon, alt = enu_to_geodetic(e, n_off, u, origin())
        rows.append(
            {
                "timestamp": round(t, 3),
                "lat": lat,
                "lon": lon,
                "alt": ORIGIN_ALT + u,
                "heading": 88.0 + 2.0 * math.sin(i / 11.0),
                "speed": 12.4,
                "hdop": round(hdop, 2),
                "rtk": 0,
            }
        )
    return rows


def _box_points(rng: np.random.Generator) -> list[ReconPoint]:
    pts: list[ReconPoint] = []
    # Rooftop (high confidence)
    for _ in range(420):
        e = rng.uniform(-10.0, 10.0)
        n = rng.uniform(2.0, 14.0)
        u = ROOFTOP_HEIGHT + rng.normal(0, 0.08)
        conf = point_confidence(8, 0.85, 0.9, 0.82, 0.3)
        pts.append(ReconPoint(e, n, u, 186, 168, 148, conf, 8, source="proxy", provenance=("proxy_scene",), synthetic=True))
    # Facades (medium — limited viewing angles on a single pass)
    for _ in range(260):
        side = rng.integers(0, 4)
        if side == 0:
            e, n = rng.uniform(-10, 10), 2.0 + rng.normal(0, 0.05)
        elif side == 1:
            e, n = rng.uniform(-10, 10), 14.0 + rng.normal(0, 0.05)
        elif side == 2:
            e, n = -10.0 + rng.normal(0, 0.05), rng.uniform(2, 14)
        else:
            e, n = 10.0 + rng.normal(0, 0.05), rng.uniform(2, 14)
        u = rng.uniform(0.2, ROOFTOP_HEIGHT)
        conf = point_confidence(3, 0.35, 0.7, 0.82, 1.4)
        pts.append(ReconPoint(e, n, u, 164, 142, 122, conf, 3, source="proxy", provenance=("proxy_scene",), synthetic=True))
    return pts


def _terrain_points(rng: np.random.Generator) -> list[ReconPoint]:
    pts: list[ReconPoint] = []
    for _ in range(900):
        e = rng.uniform(-60, 60)
        n = rng.uniform(-35, 40)
        if -11 < e < 11 and 1.5 < n < 14.5:
            continue
        u = 0.15 * math.sin(e / 9.0) + rng.normal(0, 0.12)
        # Road strip
        on_road = abs(n + 6) < 3.2
        r, g, b = (62, 64, 68) if on_road else (78, 112, 74)
        far = math.hypot(e, n) > 42
        conf = point_confidence(5 if on_road else 4, 0.55, 0.75, 0.78, 0.8 if far else 0.4)
        if far:
            conf *= 0.55
        pts.append(ReconPoint(e, n, u, r, g, b, conf, 4, source="proxy", provenance=("proxy_scene",), synthetic=True))
    # Vegetation blobs (medium/low)
    for cx, cy in [(-28, 18), (32, 22), (18, -18)]:
        for _ in range(80):
            e = cx + rng.normal(0, 3.2)
            n = cy + rng.normal(0, 3.2)
            u = abs(rng.normal(3.5, 1.1))
            conf = point_confidence(2, 0.4, 0.5, 0.78, 2.0)
            pts.append(ReconPoint(e, n, u, 46, 92, 48, conf, 2, source="proxy", provenance=("proxy_scene",), synthetic=True))
    # Occluded courtyard (low)
    for _ in range(90):
        e = rng.uniform(-6, 6)
        n = rng.uniform(5, 11)
        u = rng.uniform(0.0, 1.6)
        conf = point_confidence(1, 0.15, 0.4, 0.78, 3.5)
        pts.append(ReconPoint(e, n, u, 90, 88, 70, conf, 1, source="proxy", provenance=("proxy_scene",), synthetic=True))
    return pts


def generate_cloud(seed: int = 7) -> list[ReconPoint]:
    rng = np.random.default_rng(seed)
    pts = _box_points(rng) + _terrain_points(rng)
    return pts


def chunk_cloud(points: list[ReconPoint], chunks: int = 10) -> list[list[ReconPoint]]:
    # Stream west to east so the model grows along the flight.
    ordered = sorted(points, key=lambda p: p.e)
    n = len(ordered)
    size = math.ceil(n / chunks)
    return [ordered[i : i + size] for i in range(0, n, size)]


def write_demo_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    telem = generate_telemetry()
    with (root / "telemetry.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(telem[0].keys()))
        writer.writeheader()
        writer.writerows(telem)
    ref = building_corners_geodetic()
    proof = evaluate_segment(reference_segment())
    ref["self_check"] = proof
    (root / "reference.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
