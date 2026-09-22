"""Flight telemetry parsing, interpolation, and reliability scoring."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class TelemetrySample:
    t: float
    lat: float
    lon: float
    alt: float
    heading: float
    speed: float
    hdop: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    roll: float | None = None
    pitch: float | None = None
    rtk: bool = False


@dataclass
class TelemetryTrack:
    samples: list[TelemetrySample]
    reliability: float
    notes: list[str] = field(default_factory=list)

    def interpolate(self, t: float) -> TelemetrySample:
        if not self.samples:
            raise ValueError("empty telemetry")
        if t <= self.samples[0].t:
            return self.samples[0]
        if t >= self.samples[-1].t:
            return self.samples[-1]
        for i in range(1, len(self.samples)):
            a, b = self.samples[i - 1], self.samples[i]
            if a.t <= t <= b.t:
                w = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
                return TelemetrySample(
                    t=t,
                    lat=a.lat + w * (b.lat - a.lat),
                    lon=a.lon + w * (b.lon - a.lon),
                    alt=a.alt + w * (b.alt - a.alt),
                    heading=_lerp_heading(a.heading, b.heading, w),
                    speed=a.speed + w * (b.speed - a.speed),
                    hdop=a.hdop + w * (b.hdop - a.hdop),
                    rtk=a.rtk and b.rtk,
                )
        return self.samples[-1]


def _lerp_heading(a: float, b: float, w: float) -> float:
    da = ((b - a + 180.0) % 360.0) - 180.0
    return (a + w * da) % 360.0


def parse_telemetry_csv(text: str) -> TelemetryTrack:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("telemetry CSV has no header")
    fields = {name.strip().lower(): name for name in reader.fieldnames}

    def col(*aliases: str) -> str | None:
        for alias in aliases:
            if alias in fields:
                return fields[alias]
        return None

    t_key = col("t", "time", "timestamp", "time_s")
    lat_key = col("lat", "latitude")
    lon_key = col("lon", "lng", "longitude")
    alt_key = col("alt", "altitude", "height", "rel_alt")
    hdg_key = col("heading", "yaw", "course")
    spd_key = col("speed", "vel", "velocity")
    hdop_key = col("hdop", "gps_hdop", "accuracy", "eph")
    rtk_key = col("rtk", "rtk_fix")

    if not (t_key and lat_key and lon_key):
        raise ValueError("telemetry CSV must include timestamp, lat, lon")

    samples: list[TelemetrySample] = []
    for row in reader:
        try:
            samples.append(
                TelemetrySample(
                    t=float(row[t_key]),
                    lat=float(row[lat_key]),
                    lon=float(row[lon_key]),
                    alt=float(row[alt_key]) if alt_key and row.get(alt_key) else 80.0,
                    heading=float(row[hdg_key]) if hdg_key and row.get(hdg_key) else 0.0,
                    speed=float(row[spd_key]) if spd_key and row.get(spd_key) else 0.0,
                    hdop=float(row[hdop_key]) if hdop_key and row.get(hdop_key) else 1.5,
                    rtk=_truthy(row.get(rtk_key, "")) if rtk_key else False,
                )
            )
        except (TypeError, ValueError):
            continue

    if len(samples) < 2:
        raise ValueError("need at least two valid telemetry samples")
    samples.sort(key=lambda s: s.t)
    return TelemetryTrack(samples=samples, reliability=_score_reliability(samples), notes=[])


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "rtk", "fix"}


def _score_reliability(samples: Iterable[TelemetrySample]) -> float:
    samples = list(samples)
    hdops = np.array([s.hdop for s in samples], dtype=np.float64)
    hdop_score = float(np.clip(1.0 - (np.median(hdops) - 0.8) / 4.0, 0.15, 1.0))

    jumps = 0
    for a, b in zip(samples, samples[1:]):
        dt = max(b.t - a.t, 1e-3)
        dlat = (b.lat - a.lat) * 111_320.0
        dlon = (b.lon - a.lon) * 111_320.0 * math.cos(math.radians(a.lat))
        speed = math.hypot(dlat, dlon) / dt
        expected = max(b.speed, a.speed, 1.0)
        if speed > expected * 6.0 and speed > 40.0:
            jumps += 1
    jump_penalty = min(0.6, jumps / max(len(samples), 1) * 8.0)
    rtk_bonus = 0.12 if any(s.rtk for s in samples) else 0.0
    return float(np.clip(hdop_score - jump_penalty + rtk_bonus, 0.1, 1.0))
