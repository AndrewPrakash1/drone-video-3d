"""Occlusion map, ground-agent routes, and metric Chisel blocks.

Everything is in the same ENU metres as the GPS-aligned mesh. Nothing here
invents geometry, so a length measured on the metric cloud does not move.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np

TAGS = ("road", "facade", "ground", "fill")
_MAX_FAULTLINES = 2000


def analyze_mesh(verts: np.ndarray, faces: np.ndarray, cameras: list[dict]) -> dict:
    """Per-face visibility, occlusion-boundary faultlines, and a short summary."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    empty = {
        "seen": [],
        "face_state": [],
        "patches": [],
        "summary": {"faces": 0, "seen_faces": 0, "seen_fraction": 0.0, "largest_unseen_patch": 0, "faultlines": 0},
        "faultlines": [],
    }
    if len(faces) == 0 or len(verts) == 0:
        return empty
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    centroids = (v0 + v1 + v2) / 3.0
    seen = _seen_counts(centroids, normals, cameras)
    faultlines = _faultlines(verts, faces, seen)
    unseen = seen == 0
    patches = _unseen_patches(verts, faces, unseen)
    return {
        "seen": seen.astype(int).tolist(),
        "face_state": ["unseen" if flag else "observed" for flag in unseen.tolist()],
        "patches": patches,
        "summary": {
            "faces": int(len(faces)),
            "seen_faces": int((~unseen).sum()),
            "seen_fraction": round(float((~unseen).mean()), 4),
            "largest_unseen_patch": int(max((len(p["face_ids"]) for p in patches), default=0)),
            "faultlines": int(len(faultlines)),
        },
        "faultlines": faultlines,
    }


def write_spatial(mesh_ply: Path, cameras: list[dict], out: Path) -> dict:
    verts, faces = load_mesh(mesh_ply)
    payload = analyze_mesh(verts, faces, cameras)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return payload["summary"]


def normalize_blocks(raw: list) -> list[dict]:
    blocks: list[dict] = []
    for i, item in enumerate(raw or []):
        if not isinstance(item, dict):
            raise ValueError(f"block {i} is not an object")
        tag = str(item.get("tag") or "").lower()
        if tag not in TAGS:
            raise ValueError(f"block {i} tag must be one of {', '.join(TAGS)}")
        try:
            lo = [float(v) for v in item["min"]]
            hi = [float(v) for v in item["max"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"block {i} needs min and max ENU") from exc
        if len(lo) != 3 or len(hi) != 3 or not np.isfinite(lo + hi).all():
            raise ValueError(f"block {i} min/max must be three finite numbers")
        lo_a = [min(a, b) for a, b in zip(lo, hi)]
        hi_a = [max(a, b) for a, b in zip(lo, hi)]
        if hi_a[0] - lo_a[0] < 1e-3 or hi_a[1] - lo_a[1] < 1e-3 or hi_a[2] - lo_a[2] < 1e-3:
            raise ValueError(f"block {i} has no volume")
        ident = str(item.get("id") or f"b{i}")
        blocks.append({"id": ident, "tag": tag, "min": [round(v, 3) for v in lo_a], "max": [round(v, 3) for v in hi_a]})
    return blocks


def plan_route(
    verts: np.ndarray,
    faces: np.ndarray,
    seen: np.ndarray | None,
    start: np.ndarray,
    goal: np.ndarray,
    blocks: list[dict],
    faultlines: list | None = None,
) -> dict:
    """A* on the height-map grid. Facade boxes and steep steps are blocked.

    Unseen cells cost more, so the route prefers ground the cameras actually saw.
    Fill boxes waive that penalty: the user has marked the blind spot on purpose.
    """
    grid = _height_grid(np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32), seen)
    if not grid["cells"]:
        return {"path": [], "length_m": 0.0, "flags": [], "reason": "mesh has no walkable cells"}
    start_k = _snap(grid, np.asarray(start, dtype=np.float64))
    goal_k = _snap(grid, np.asarray(goal, dtype=np.float64))
    if start_k is None or goal_k is None:
        return {"path": [], "length_m": 0.0, "flags": [], "reason": "start or goal is off the mesh"}
    came = _astar(grid, start_k, goal_k, blocks)
    if goal_k not in came and start_k != goal_k:
        return {"path": [], "length_m": 0.0, "flags": [], "reason": "no route (facade or steep ground blocks the way)"}
    keys = []
    cur: tuple[int, int] | None = goal_k
    while cur is not None:
        keys.append(cur)
        cur = came.get(cur)
    keys.reverse()
    cell = grid["cell"]
    path = []
    for ix, iy in keys:
        info = grid["cells"][(ix, iy)]
        path.append([round(grid["origin"][0] + (ix + 0.5) * cell, 3), round(grid["origin"][1] + (iy + 0.5) * cell, 3), round(info["h"] + 0.15, 3)])
    length = 0.0
    for a, b in zip(path, path[1:]):
        length += float(np.linalg.norm(np.array(b) - np.array(a)))
    return {
        "path": path,
        "length_m": round(length, 3),
        "flags": _route_flags(grid, keys, faultlines or []),
    }


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(path))
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        if len(verts) and len(faces):
            return verts, faces
    except Exception:
        pass
    return _ascii_mesh(path)


def _seen_counts(centroids: np.ndarray, normals: np.ndarray, cameras: list[dict]) -> np.ndarray:
    seen = np.zeros(len(centroids), dtype=np.int32)
    nlen = np.linalg.norm(normals, axis=1)
    usable = nlen > 1e-8
    for cam in cameras:
        rot = cam.get("rotation") or []
        if len(rot) != 9:
            continue
        rotation = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        center = np.array([float(cam["e"]), float(cam["n"]), float(cam["u"])], dtype=np.float64)
        fx = float(cam.get("fx") or 800.0)
        fy = float(cam.get("fy") or fx)
        width = int(cam.get("width") or 1280)
        height = int(cam.get("height") or 720)
        cx = float(cam["cx"]) if cam.get("cx") is not None else width / 2.0
        cy = float(cam["cy"]) if cam.get("cy") is not None else height / 2.0
        cam_pts = (centroids - center) @ rotation
        depth = cam_pts[:, 2]
        view = center - centroids
        view_n = np.linalg.norm(view, axis=1)
        facing = usable & ((normals * view).sum(1) / (nlen * view_n + 1e-9) > 0.25)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fx * cam_pts[:, 0] / depth + cx
            v = fy * cam_pts[:, 1] / depth + cy
        inside = (depth > 0.5) & facing & (u >= 0) & (u < width) & (v >= 0) & (v < height) & np.isfinite(u) & np.isfinite(v)
        if int(inside.sum()) == 0:
            continue
        bw = max(8, width // 8)
        bh = max(8, height // 8)
        ui = np.clip((u[inside] * bw / width).astype(np.int32), 0, bw - 1)
        vi = np.clip((v[inside] * bh / height).astype(np.int32), 0, bh - 1)
        flat = vi * bw + ui
        zz = depth[inside]
        zbuf = np.full(bw * bh, np.inf)
        np.minimum.at(zbuf, flat, zz)
        nearest = zz <= zbuf[flat] * 1.03 + 0.4
        seen[np.flatnonzero(inside)[nearest]] += 1
    return seen


def _edge_table(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = faces.astype(np.int64)
    pairs = np.stack(
        [
            np.sort(tri[:, [0, 1]], axis=1),
            np.sort(tri[:, [1, 2]], axis=1),
            np.sort(tri[:, [2, 0]], axis=1),
        ],
        axis=1,
    ).reshape(-1, 2)
    face_of = np.repeat(np.arange(len(faces), dtype=np.int64), 3)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order], face_of[order], order


def _faultlines(verts: np.ndarray, faces: np.ndarray, seen: np.ndarray) -> list[list[float]]:
    pairs, face_of, _ = _edge_table(faces)
    if len(pairs) == 0:
        return []
    change = np.flatnonzero(np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1), True])
    occlusion: list[list[float]] = []
    boundary: list[list[float]] = []
    for a, b in zip(change[:-1].tolist(), change[1:].tolist()):
        group = face_of[a:b]
        if len(group) == 1:
            bucket = boundary
        elif len(group) == 2 and (seen[group[0]] == 0) != (seen[group[1]] == 0):
            bucket = occlusion
        else:
            continue
        i, j = int(pairs[a, 0]), int(pairs[a, 1])
        p, q = verts[i], verts[j]
        bucket.append([round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), round(float(q[0]), 2), round(float(q[1]), 2), round(float(q[2]), 2)])
    lines = occlusion + boundary
    if len(lines) > _MAX_FAULTLINES:
        lines = lines[:_MAX_FAULTLINES]
    return lines


def _unseen_patches(verts: np.ndarray, faces: np.ndarray, unseen: np.ndarray) -> list[dict]:
    if int(unseen.sum()) == 0:
        return []
    pairs, face_of, _ = _edge_table(faces)
    parent = np.arange(len(faces), dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    change = np.flatnonzero(np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1), True])
    for a, b in zip(change[:-1].tolist(), change[1:].tolist()):
        group = face_of[a:b]
        if len(group) != 2 or not unseen[group[0]] or not unseen[group[1]]:
            continue
        ra, rb = find(int(group[0])), find(int(group[1]))
        if ra != rb:
            parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for face_id in np.flatnonzero(unseen).tolist():
        groups.setdefault(find(int(face_id)), []).append(int(face_id))
    patches = []
    for patch_id, face_ids in enumerate(sorted(groups.values(), key=lambda ids: (-len(ids), min(ids)))):
        xyz = verts[faces[np.asarray(face_ids, dtype=np.int32)]].reshape(-1, 3)
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        patches.append({
            "id": patch_id,
            "face_ids": face_ids,
            "bounds": {"min": [round(float(v), 3) for v in lo], "max": [round(float(v), 3) for v in hi]},
            "area_hint_m2": round(float(sum(np.linalg.norm(np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])) for f in faces[np.asarray(face_ids, dtype=np.int32)]) * 0.5), 3),
            "state": "unseen",
            "measurement_safe": False,
        })
    return patches


def _largest_unseen_patch(faces: np.ndarray, unseen: np.ndarray) -> int:
    if int(unseen.sum()) == 0:
        return 0
    pairs, face_of, _ = _edge_table(faces)
    parent = np.arange(len(faces), dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    change = np.flatnonzero(np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1), True])
    for a, b in zip(change[:-1].tolist(), change[1:].tolist()):
        group = face_of[a:b]
        if len(group) != 2 or not unseen[group[0]] or not unseen[group[1]]:
            continue
        ra, rb = find(int(group[0])), find(int(group[1]))
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in np.flatnonzero(unseen).tolist()], dtype=np.int32)
    if len(roots) == 0:
        return 0
    _, counts = np.unique(roots, return_counts=True)
    return int(counts.max())


def _height_grid(verts: np.ndarray, faces: np.ndarray, seen: np.ndarray | None) -> dict:
    if len(verts) == 0:
        return {"cells": {}, "cell": 1.0, "origin": np.zeros(2)}
    xs = np.unique(np.round(verts[:, 0], 2))
    ys = np.unique(np.round(verts[:, 1], 2))
    dx = np.diff(xs)
    dy = np.diff(ys)
    steps = np.concatenate([dx[dx > 1e-3], dy[dy > 1e-3]]) if len(dx) or len(dy) else np.array([])
    cell = float(np.median(steps)) if len(steps) else 1.0
    cell = max(cell, 0.05)
    span = float(max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1]), cell))
    if span / cell > 700:
        cell = span / 700.0
    origin = verts[:, :2].min(axis=0)
    ix = np.clip(np.floor((verts[:, 0] - origin[0]) / cell).astype(np.int32), 0, 699)
    iy = np.clip(np.floor((verts[:, 1] - origin[1]) / cell).astype(np.int32), 0, 699)
    acc: dict[tuple[int, int], list[float]] = {}
    for key, h in zip(zip(ix.tolist(), iy.tolist()), verts[:, 2].tolist()):
        acc.setdefault(key, []).append(h)
    cells = {key: {"h": float(np.median(hs)), "seen": 1.0} for key, hs in acc.items()}
    if seen is not None and len(seen) == len(faces) and len(faces):
        centroids = verts[faces].mean(axis=1)
        fx = np.clip(np.floor((centroids[:, 0] - origin[0]) / cell).astype(np.int32), 0, 699)
        fy = np.clip(np.floor((centroids[:, 1] - origin[1]) / cell).astype(np.int32), 0, 699)
        tally: dict[tuple[int, int], list[int]] = {}
        for key, flag in zip(zip(fx.tolist(), fy.tolist()), (seen > 0).astype(int).tolist()):
            tally.setdefault(key, []).append(flag)
        for key, flags in tally.items():
            if key in cells:
                cells[key]["seen"] = float(np.mean(flags))
    return {"cells": cells, "cell": cell, "origin": origin}


def _snap(grid: dict, point: np.ndarray) -> tuple[int, int] | None:
    cell = grid["cell"]
    origin = grid["origin"]
    cells = grid["cells"]
    ix = int(np.floor((point[0] - origin[0]) / cell))
    iy = int(np.floor((point[1] - origin[1]) / cell))
    best = None
    best_d = 1e18
    for radius in range(0, 6):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                key = (ix + dx, iy + dy)
                if key not in cells:
                    continue
                center = origin + np.array([key[0] + 0.5, key[1] + 0.5]) * cell
                dist = float(np.hypot(*(center - point[:2])))
                if dist < best_d:
                    best, best_d = key, dist
        if best is not None:
            return best
    return None


def _blocks_at(blocks: list[dict], e: float, n: float, u: float) -> set[str]:
    tags = set()
    for block in blocks:
        lo, hi = block["min"], block["max"]
        if lo[0] <= e <= hi[0] and lo[1] <= n <= hi[1] and lo[2] - 0.5 <= u <= hi[2] + 0.5:
            tags.add(block["tag"])
    return tags


def _astar(grid: dict, start: tuple[int, int], goal: tuple[int, int], blocks: list[dict]) -> dict[tuple[int, int], tuple[int, int] | None]:
    cell = grid["cell"]
    origin = grid["origin"]
    cells = grid["cells"]

    def center(key: tuple[int, int]) -> np.ndarray:
        return np.array([origin[0] + (key[0] + 0.5) * cell, origin[1] + (key[1] + 0.5) * cell, cells[key]["h"]])

    def tags(key: tuple[int, int]) -> set[str]:
        c = center(key)
        return _blocks_at(blocks, float(c[0]), float(c[1]), float(c[2]))

    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    if start == goal:
        return came
    gscore = {start: 0.0}
    goal_c = center(goal)
    heap = [(float(np.hypot(center(start)[0] - goal_c[0], center(start)[1] - goal_c[1])), start)]
    closed: set[tuple[int, int]] = set()
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            return came
        cx, cy = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (cx + dx, cy + dy)
            if nxt not in cells:
                continue
            step = cell * (1.41421356237 if dx and dy else 1.0)
            rise = abs(cells[nxt]["h"] - cells[cur]["h"])
            if rise / step > 0.8:
                continue
            mark = tags(nxt)
            if "facade" in mark:
                continue
            cost = step
            if "road" in mark:
                cost *= 0.5
            if cells[nxt]["seen"] < 0.5 and "fill" not in mark:
                cost *= 4.0
            cand = gscore[cur] + cost
            if cand >= gscore.get(nxt, 1e18):
                continue
            gscore[nxt] = cand
            came[nxt] = cur
            heur = float(np.hypot(center(nxt)[0] - goal_c[0], center(nxt)[1] - goal_c[1]))
            heapq.heappush(heap, (cand + heur, nxt))
    return came


def _route_flags(grid: dict, keys: list[tuple[int, int]], faultlines: list) -> list[dict]:
    cell = grid["cell"]
    origin = grid["origin"]
    cells = grid["cells"]
    flags: list[dict] = []
    run: list[tuple[int, int]] = []

    def flush() -> None:
        if not run:
            return
        mid = run[len(run) // 2]
        info = cells[mid]
        flags.append({
            "e": round(origin[0] + (mid[0] + 0.5) * cell, 3),
            "n": round(origin[1] + (mid[1] + 0.5) * cell, 3),
            "u": round(info["h"], 3),
            "kind": "occlusion",
        })
        run.clear()

    for key in keys:
        if cells[key]["seen"] < 0.5:
            run.append(key)
        else:
            flush()
    flush()
    if not faultlines:
        return flags
    mids = np.array([[(seg[0] + seg[3]) / 2, (seg[1] + seg[4]) / 2] for seg in faultlines if len(seg) >= 6], dtype=np.float64)
    if len(mids) == 0:
        return flags
    used = np.zeros(len(mids), dtype=bool)
    reach = max(cell * 1.5, 1.0)
    for key in keys:
        e = origin[0] + (key[0] + 0.5) * cell
        n = origin[1] + (key[1] + 0.5) * cell
        dist = np.hypot(mids[:, 0] - e, mids[:, 1] - n)
        hit = np.flatnonzero((dist <= reach) & ~used)
        for idx in hit[:1].tolist():
            used[idx] = True
            seg = faultlines[idx]
            flags.append({"e": round(float(mids[idx, 0]), 3), "n": round(float(mids[idx, 1]), 3), "u": round((seg[2] + seg[5]) / 2, 3), "kind": "faultline"})
            if sum(1 for f in flags if f["kind"] == "faultline") >= 40:
                return flags
    return flags


def _ascii_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    nv = nf = i = 0
    while i < len(lines) and lines[i].strip() != "end_header":
        if lines[i].startswith("element vertex"):
            nv = int(lines[i].split()[-1])
        if lines[i].startswith("element face"):
            nf = int(lines[i].split()[-1])
        i += 1
    i += 1
    verts = np.zeros((nv, 3), dtype=np.float64)
    for row in range(nv):
        parts = lines[i + row].split()
        verts[row] = [float(parts[0]), float(parts[1]), float(parts[2])]
    i += nv
    faces = np.zeros((nf, 3), dtype=np.int32)
    for row in range(nf):
        parts = lines[i + row].split()
        faces[row] = [int(parts[1]), int(parts[2]), int(parts[3])]
    return verts, faces
