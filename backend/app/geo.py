"""WGS84 geodetic <-> local ENU conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


@dataclass(frozen=True)
class Origin:
    lat: float
    lon: float
    alt: float
    ecef: np.ndarray
    enu_rot: np.ndarray


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    return math.degrees(lat), math.degrees(lon), alt


def make_origin(lat: float, lon: float, alt: float = 0.0) -> Origin:
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
    sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)
    rot = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return Origin(lat=lat, lon=lon, alt=alt, ecef=geodetic_to_ecef(lat, lon, alt), enu_rot=rot)


def geodetic_to_enu(lat: float, lon: float, alt: float, origin: Origin) -> np.ndarray:
    delta = geodetic_to_ecef(lat, lon, alt) - origin.ecef
    return origin.enu_rot @ delta


def enu_to_geodetic(e: float, n: float, u: float, origin: Origin) -> tuple[float, float, float]:
    ecef = origin.ecef + origin.enu_rot.T @ np.array([e, n, u], dtype=np.float64)
    return ecef_to_geodetic(float(ecef[0]), float(ecef[1]), float(ecef[2]))


def heading_to_enu(heading_deg: float) -> np.ndarray:
    """Aircraft heading (deg clockwise from north) as ENU unit vector."""
    h = math.radians(heading_deg)
    return np.array([math.sin(h), math.cos(h), 0.0], dtype=np.float64)
