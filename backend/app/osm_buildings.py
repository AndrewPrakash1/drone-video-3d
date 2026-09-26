"""OpenStreetMap building footprints extruded for the Cesium city view."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

_CACHE: dict[tuple[float, float], list[dict]] = {}
_BUNDLED = Path(__file__).resolve().parents[2] / "data" / "demo" / "osm_buildings.json"
_DEMO_LAT = 28.5448
_DEMO_LON = 77.1924


def _height_m(tags: dict) -> float:
    raw = tags.get("height")
    if raw:
        try:
            return max(4.0, min(float(str(raw).split()[0]), 140.0))
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return max(4.0, min(float(str(levels).split(";")[0]) * 3.1, 140.0))
        except ValueError:
            pass
    kind = tags.get("building", "")
    if kind in {"apartments", "residential", "commercial", "office", "hotel", "retail"}:
        return 16.0
    return 9.0


def _bundled(lat: float, lon: float) -> list[dict] | None:
    if not _BUNDLED.exists():
        return None
    dist = math.hypot((lat - _DEMO_LAT) * 111_320, (lon - _DEMO_LON) * 111_320 * math.cos(math.radians(lat)))
    if dist > 2500:
        return None
    data = json.loads(_BUNDLED.read_text(encoding="utf-8"))
    return data.get("buildings") or []


def fetch_buildings(lat: float, lon: float, radius_m: float = 1100.0) -> list[dict]:
    key = (round(lat, 3), round(lon, 3))
    if key in _CACHE:
        return _CACHE[key]

    local = _bundled(lat, lon)
    if local is not None:
        _CACHE[key] = local
        return local

    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    south, north = lat - dlat, lat + dlat
    west, east = lon - dlon, lon + dlon
    query = (
        f"[out:json][timeout:25];"
        f'way["building"]({south:.5f},{west:.5f},{north:.5f},{east:.5f});'
        f"out geom;"
    )
    url = "https://overpass-api.de/api/interpreter?" + urllib.parse.urlencode({"data": query})
    req = urllib.request.Request(url, headers={"User-Agent": "OnePass/1.0 (SIH26158 3D city view)"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    buildings: list[dict] = []
    for el in payload.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 4:
            continue
        ring = [[float(p["lon"]), float(p["lat"])] for p in geom]
        tags = el.get("tags") or {}
        buildings.append({"id": el.get("id"), "height": _height_m(tags), "ring": ring})
        if len(buildings) >= 700:
            break
    _CACHE[key] = buildings
    return buildings
