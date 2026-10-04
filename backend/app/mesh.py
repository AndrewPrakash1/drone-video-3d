"""Headless 2.5D meshing (Open3D Poisson when EGL is available)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .reconstruct import ReconPoint


def write_ply(path: Path, points: list[ReconPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float confidence\nend_header\n")
        for p in points:
            f.write(f"{p.e:.4f} {p.n:.4f} {p.u:.4f} {p.r} {p.g} {p.b} {p.conf:.3f}\n")


def _grid_mesh(points: list[ReconPoint], cell: float = 1.4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets: dict[tuple[int, int], list[ReconPoint]] = {}
    for p in points:
        key = (int(np.floor(p.e / cell)), int(np.floor(p.n / cell)))
        buckets.setdefault(key, []).append(p)
    cells = {}
    for key, group in buckets.items():
        n = len(group)
        cells[key] = (
            sum(p.e for p in group) / n,
            sum(p.n for p in group) / n,
            sum(p.u for p in group) / n,
            int(sum(p.r for p in group) / n),
            int(sum(p.g for p in group) / n),
            int(sum(p.b for p in group) / n),
        )
    keys = list(cells)
    index = {k: i for i, k in enumerate(keys)}
    verts = np.array([[cells[k][0], cells[k][1], cells[k][2]] for k in keys], dtype=np.float64)
    colors = np.array([[cells[k][3], cells[k][4], cells[k][5]] for k in keys], dtype=np.int32)
    faces: list[list[int]] = []
    for i, j in keys:
        a = (i, j)
        b = (i + 1, j)
        c = (i, j + 1)
        d = (i + 1, j + 1)
        if b in index and c in index:
            faces.append([index[a], index[b], index[c]])
        if b in index and c in index and d in index:
            faces.append([index[b], index[d], index[c]])
    return verts, np.array(faces, dtype=np.int32), colors


def _write_mesh_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v, c in zip(verts, colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def clean_cloud(
    points: list[ReconPoint],
    cameras: np.ndarray | None = None,
    k: int = 12,
    std_ratio: float = 2.0,
) -> list[ReconPoint]:
    """Range gate, statistical outlier removal and a local height-residual test.

    Far-field drone triangulations leave isolated specks and vertical streaks;
    both read as noise in the viewer and wreck any surface fit. Depth error
    grows with range, so points far beyond the typical viewing distance go too.
    """
    if len(points) < 200:
        return points
    from scipy.spatial import cKDTree

    xyz = np.array([(p.e, p.n, p.u) for p in points], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    xyz_f = xyz[finite]
    keep = np.ones(len(xyz_f), dtype=bool)
    if cameras is not None and len(cameras):
        rng = cKDTree(np.asarray(cameras, dtype=np.float64)).query(xyz_f)[0]
        keep &= rng <= 1.6 * float(np.median(rng))
    tree = cKDTree(xyz_f)
    kk = min(max(k, 24), len(xyz_f) - 1)
    dist, idx = tree.query(xyz_f, k=kk + 1)
    mean_d = dist[:, 1 : k + 1].mean(axis=1)
    keep &= mean_d <= mean_d.mean() + std_ratio * mean_d.std()
    resid = np.abs(xyz_f[:, 2] - np.median(xyz_f[idx[:, 1:], 2], axis=1))
    keep &= resid <= max(3.0, 6.0 * float(np.median(resid)))
    kept_idx = np.flatnonzero(finite)[keep]
    if len(kept_idx) < 0.5 * len(points):
        return points
    return [points[i] for i in kept_idx.tolist()]


def agreeing_points(base: list[ReconPoint], extra: list[ReconPoint], tol: float | None = None) -> list[ReconPoint]:
    """Points of `extra` lying on the `base` surface, so two independently aligned reconstructions don't stack into layers."""
    if not base or not extra:
        return extra
    from scipy.spatial import cKDTree

    b = np.array([(p.e, p.n, p.u) for p in base], dtype=np.float64)
    x = np.array([(p.e, p.n, p.u) for p in extra], dtype=np.float64)
    tree = cKDTree(b)
    if tol is None:
        spacing = float(np.median(tree.query(b, k=2)[0][:, 1]))
        tol = max(1.5, 3.0 * spacing)
    d = tree.query(x, distance_upper_bound=tol)[0]
    return [p for p, ok in zip(extra, np.isfinite(d).tolist()) if ok]


def _dsm_mesh(points: list[ReconPoint], max_cells: int = 110_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """2.5D digital surface model: robust median height per grid cell, small holes filled, lightly smoothed."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    xyz = np.array([(p.e, p.n, p.u) for p in points], dtype=np.float64)
    rgb = np.array([(p.r, p.g, p.b) for p in points], dtype=np.float64)
    ok = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[ok], rgb[ok]
    if len(xyz) < 50:
        return None
    lo = np.percentile(xyz[:, :2], 0.5, axis=0)
    hi = np.percentile(xyz[:, :2], 99.5, axis=0)
    inside = ((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi)).all(axis=1)
    xyz, rgb = xyz[inside], rgb[inside]
    spacing = float(np.median(cKDTree(xyz[:, :2]).query(xyz[:, :2], k=5)[0][:, 1:].mean(axis=1)))
    cell = float(np.clip(spacing * 1.6, 0.2, 6.0))
    extent = np.maximum(hi - lo, cell)
    while (extent[0] / cell + 1) * (extent[1] / cell + 1) > max_cells:
        cell *= 1.25
    nx = int(extent[0] / cell) + 1
    ny = int(extent[1] / cell) + 1
    ix = np.clip(((xyz[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((xyz[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    ends = np.r_[starts[1:], len(flat_s)]
    height = np.full(nx * ny, np.nan)
    color = np.zeros((nx * ny, 3))
    count = np.zeros(nx * ny, dtype=np.int32)
    for s, e in zip(starts.tolist(), ends.tolist()):
        members = order[s:e]
        c = flat_s[s]
        height[c] = np.median(xyz[members, 2])
        color[c] = np.median(rgb[members], axis=0)
        count[c] = e - s
    height = height.reshape(ny, nx)
    color = color.reshape(ny, nx, 3)
    count = count.reshape(ny, nx)
    valid = np.isfinite(height)
    if (count >= 2).sum() >= 0.6 * valid.sum():
        valid &= count >= 2
    if valid.sum() < 16:
        return None

    for size in (5, 3):
        _, (ry, rx) = ndimage.distance_transform_edt(~valid, return_indices=True)
        med = ndimage.median_filter(height[ry, rx], size=size)
        valid &= np.abs(height - med) <= max(2.0 * cell, 1.5)

    mask = ndimage.binary_closing(valid, structure=np.ones((3, 3)), iterations=3) | valid
    mask = ndimage.binary_fill_holes(mask) & ndimage.binary_dilation(valid, iterations=4)
    _, (ry, rx) = ndimage.distance_transform_edt(~valid, return_indices=True)
    height = np.where(valid, height, height[ry, rx])
    color = np.where(valid[..., None], color, color[ry, rx])
    height = ndimage.median_filter(height, size=3)

    w = mask.astype(float)
    num = ndimage.gaussian_filter(np.where(mask, height, 0.0), sigma=1.0)
    den = ndimage.gaussian_filter(w, sigma=1.0)
    height = np.where(den > 1e-6, num / np.maximum(den, 1e-6), height)

    vid = -np.ones((ny, nx), dtype=np.int64)
    vid[mask] = np.arange(int(mask.sum()))
    gy, gx = np.nonzero(mask)
    verts = np.column_stack([lo[0] + (gx + 0.5) * cell, lo[1] + (gy + 0.5) * cell, height[gy, gx]])
    colors = np.clip(color[gy, gx], 0, 255).astype(np.int32)
    a = vid[:-1, :-1]
    b = vid[:-1, 1:]
    c = vid[1:, :-1]
    d = vid[1:, 1:]
    quad = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    faces = np.concatenate([
        np.column_stack([a[quad], b[quad], d[quad]]),
        np.column_stack([a[quad], d[quad], c[quad]]),
    ]).astype(np.int32)
    z = verts[faces, 2]
    faces = faces[np.ptp(z, axis=1) <= max(5.0 * cell, 3.0)]
    if len(faces) < 8:
        return None
    return verts, faces, colors, cell


def mesh_from_points(points: list[ReconPoint], out_ply: Path) -> dict:
    if len(points) < 30:
        return {"status": "skipped", "reason": "too few points"}
    method = os.environ.get("ONEPASS_MESH", "dsm").lower()
    dsm_reason = None
    if method != "poisson":
        try:
            dsm = _dsm_mesh(points)
        except Exception as exc:  # pragma: no cover - numeric guard
            dsm, dsm_reason = None, f"{type(exc).__name__}: {exc}"
        if dsm is not None:
            verts, faces, colors, cell = dsm
            _write_mesh_ply(out_ply, verts, faces, colors)
            return {
                "status": "ok",
                "method": "dsm25d",
                "cell_m": round(cell, 3),
                "vertices": int(len(verts)),
                "triangles": int(len(faces)),
                "path": str(out_ply),
            }
        dsm_reason = dsm_reason or "surface grid too sparse"
    poisson = _try_open3d_poisson(points, out_ply)
    if poisson.get("status") == "ok":
        if dsm_reason:
            poisson["fallback_from"] = dsm_reason
        return poisson
    verts, faces, colors = _grid_mesh(points)
    if len(faces) < 8:
        return {"status": "skipped", "reason": poisson.get("reason", "mesh too thin")}
    _write_mesh_ply(out_ply, verts, faces, colors)
    return {
        "status": "ok",
        "method": "grid25d",
        "fallback_from": poisson.get("reason"),
        "vertices": int(len(verts)),
        "triangles": int(len(faces)),
        "path": str(out_ply),
    }


def _try_open3d_poisson(points: list[ReconPoint], out_ply: Path) -> dict:
    """Poisson-mesh in a subprocess.

    Open3D's bundled Poisson can call std::terminate on degenerate input (e.g.
    duplicated points from two fused reconstructions), which kills the whole
    server. Running it in a subprocess contains that crash so we can fall back
    to the grid mesh instead of taking the API down.
    """
    pts = _clean_for_poisson(points)
    if len(pts) < 30:
        return {"status": "skipped", "reason": "too few points after cleaning"}
    in_ply = out_ply.with_name(out_ply.stem + "_poisson_input.ply")
    write_ply(in_ply, pts)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _POISSON_SCRIPT, str(in_ply), str(out_ply)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"status": "skipped", "reason": "poisson timeout"}
    finally:
        in_ply.unlink(missing_ok=True)
    if proc.returncode != 0:
        return {"status": "skipped", "reason": (proc.stderr or "poisson failed").strip()[:160]}
    try:
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(str(out_ply))
        nv = int(np.asarray(mesh.vertices).shape[0])
        nt = int(np.asarray(mesh.triangles).shape[0])
    except Exception as exc:  # pragma: no cover - read-back guard
        return {"status": "skipped", "reason": str(exc)}
    if nv == 0 or nt == 0:
        return {"status": "skipped", "reason": "poisson produced an empty mesh"}
    return {"status": "ok", "method": "poisson", "vertices": nv, "triangles": nt, "path": str(out_ply)}


def _clean_for_poisson(points: list[ReconPoint], max_points: int = 40000) -> list[ReconPoint]:
    pts = [p for p in points if np.isfinite(p.e) and np.isfinite(p.n) and np.isfinite(p.u)]
    if len(pts) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts), max_points, replace=False)
        pts = [pts[i] for i in idx]
    return pts


_POISSON_SCRIPT = r"""
import sys
import numpy as np
import open3d as o3d

inp, outp = sys.argv[1], sys.argv[2]
pcd = o3d.io.read_point_cloud(inp)
if len(pcd.points) == 0:
    raise SystemExit(2)
pcd = pcd.remove_duplicated_points()
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
if len(pcd.points) < 30:
    raise SystemExit(3)
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
dens = np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.08))
mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
mesh.compute_vertex_normals()
tree = o3d.geometry.KDTreeFlann(pcd)
cols = np.asarray(pcd.colors)
mc = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
for i, v in enumerate(np.asarray(mesh.vertices)):
    _, idx, _ = tree.search_knn_vector_3d(v, 1)
    mc[i] = cols[idx[0]]
mesh.vertex_colors = o3d.utility.Vector3dVector(mc)
o3d.io.write_triangle_mesh(outp, mesh)
"""
