"""Headless 2.5D meshing (Open3D Poisson when EGL is available)."""

from __future__ import annotations

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


def mesh_from_points(points: list[ReconPoint], out_ply: Path) -> dict:
    if len(points) < 30:
        return {"status": "skipped", "reason": "too few points"}
    poisson = _try_open3d_poisson(points, out_ply)
    if poisson.get("status") == "ok":
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
    try:
        import open3d as o3d
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}
    xyz = np.array([[p.e, p.n, p.u] for p in points], dtype=np.float64)
    rgb = np.array([[p.r, p.g, p.b] for p in points], dtype=np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    try:
        pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
        mesh.compute_vertex_normals()
        tree = o3d.geometry.KDTreeFlann(pcd)
        colors = []
        for v in np.asarray(mesh.vertices):
            _, idx, _ = tree.search_knn_vector_3d(v, 1)
            colors.append(rgb[idx[0]])
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out_ply), mesh)
        return {
            "status": "ok",
            "method": "poisson",
            "vertices": int(np.asarray(mesh.vertices).shape[0]),
            "triangles": int(np.asarray(mesh.triangles).shape[0]),
            "path": str(out_ply),
        }
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}
