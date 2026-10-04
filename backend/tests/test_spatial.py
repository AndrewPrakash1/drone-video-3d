"""Occlusion flags and a ground agent that respects metric Chisel blocks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.app.mesh import _write_mesh_ply
from backend.app.spatial import analyze_mesh, load_mesh, plan_route, write_spatial


def _grid(e0: float, e1: float, n0: float, n1: float, step: float, height) -> tuple[np.ndarray, np.ndarray]:
    es = np.arange(e0, e1 + 1e-6, step)
    ns = np.arange(n0, n1 + 1e-6, step)
    ee, nn = np.meshgrid(es, ns, indexing="xy")
    verts = np.column_stack([ee.ravel(), nn.ravel(), np.asarray(height(ee, nn), dtype=np.float64).ravel()])
    nx, ny = len(es), len(ns)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b = j * nx + i, j * nx + i + 1
            c, d = a + nx, b + nx
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int32)


def _camera(eye: np.ndarray, target: np.ndarray, fx: float = 700.0, width: int = 1280, height: int = 720) -> dict:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0]) if abs(float(forward[2])) > 0.95 else np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward]).reshape(-1)
    return {
        "e": float(eye[0]), "n": float(eye[1]), "u": float(eye[2]),
        "rotation": rotation.tolist(),
        "fx": fx, "width": width, "height": height,
    }


class SpatialTests(unittest.TestCase):
    def test_far_side_of_hill_is_unseen(self) -> None:
        verts, faces = _grid(0, 24, 0, 16, 1.0, lambda e, n: np.maximum(0.0, 8.0 - 0.9 * np.abs(e - 10.0)))
        cams = [
            _camera(np.array([-8.0, 4.0, 7.0]), np.array([10.0, 4.0, 3.0])),
            _camera(np.array([-8.0, 12.0, 7.0]), np.array([10.0, 12.0, 3.0])),
        ]
        report = analyze_mesh(verts, faces, cams)
        seen = np.asarray(report["seen"])
        centroids = verts[faces].mean(axis=1)
        west = centroids[:, 0] < 7.0
        east = centroids[:, 0] > 14.0
        self.assertGreater(float(seen[west].mean() > 0), 0.6)
        self.assertLess(float((seen[east] > 0).mean()), 0.05)
        self.assertGreater(report["summary"]["largest_unseen_patch"], 10)
        self.assertGreater(report["summary"]["faultlines"], 0)
        self.assertTrue(report["patches"])
        self.assertEqual(report["face_state"][0] in {"observed", "unseen"}, True)
        self.assertFalse(report["patches"][0]["measurement_safe"])

    def test_clear_path_stays_on_the_line(self) -> None:
        verts, faces = _grid(0, 40, 10, 30, 1.0, lambda e, n: np.zeros_like(e))
        seen = np.ones(len(faces), dtype=np.int32)
        route = plan_route(verts, faces, seen, np.array([1.0, 20.0, 0.0]), np.array([39.0, 20.0, 0.0]), [])
        path = np.asarray(route["path"], dtype=np.float64)
        self.assertGreater(len(path), 2)
        self.assertLess(float(np.max(np.abs(path[:, 1] - 20.0))), 1.0)

    def test_facade_block_forces_a_detour(self) -> None:
        verts, faces = _grid(0, 40, 10, 30, 1.0, lambda e, n: np.zeros_like(e))
        seen = np.ones(len(faces), dtype=np.int32)
        facade = [{"id": "wall", "tag": "facade", "min": [18.0, 12.0, -1.0], "max": [22.0, 28.0, 6.0]}]
        route = plan_route(verts, faces, seen, np.array([1.0, 20.0, 0.0]), np.array([39.0, 20.0, 0.0]), facade)
        path = np.asarray(route["path"], dtype=np.float64)
        self.assertGreater(len(path), 2, route.get("reason"))
        self.assertGreater(float(np.max(np.abs(path[:, 1] - 20.0))), 4.0)

    def test_write_spatial_roundtrip(self) -> None:
        verts, faces = _grid(0, 4, 0, 4, 1.0, lambda e, n: np.zeros_like(e))
        colors = np.full((len(verts), 3), 180, dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "mesh.ply"
            _write_mesh_ply(ply, verts, faces, colors)
            cam = _camera(np.array([2.0, 2.0, 8.0]), np.array([2.0, 2.0, 0.0]), fx=400, width=320, height=240)
            summary = write_spatial(ply, [cam], Path(tmp) / "spatial.json")
            loaded_v, loaded_f = load_mesh(ply)
        self.assertEqual(len(loaded_v), len(verts))
        self.assertEqual(len(loaded_f), len(faces))
        self.assertGreater(summary["seen_faces"], 0)


if __name__ == "__main__":
    unittest.main()
