"""Surface mesh and cloud cleaning on a synthetic aerial scene."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.app.mesh import agreeing_points, clean_cloud, mesh_from_points
from backend.app.reconstruct import ReconPoint


def _hill(n: int = 6000, seed: int = 0) -> list[ReconPoint]:
    rng = np.random.default_rng(seed)
    e = rng.uniform(-60, 60, n)
    nn = rng.uniform(-60, 60, n)
    u = 0.08 * e + 4.0 * np.sin(nn / 15.0) + rng.normal(0, 0.15, n)
    return [ReconPoint(float(a), float(b), float(c), 120, 140, 90, 0.8, 3) for a, b, c in zip(e, nn, u)]


class MeshTests(unittest.TestCase):
    def test_clean_drops_spikes_and_far_points(self) -> None:
        pts = _hill()
        spikes = [ReconPoint(float(x), float(x), 40.0, 255, 0, 0, 0.5, 2) for x in np.linspace(-50, 50, 40)]
        far = [ReconPoint(900.0 + i, 900.0, 0.0, 0, 0, 255, 0.5, 2) for i in range(40)]
        eyes = np.array([[0.0, y, 80.0] for y in np.linspace(-40, 40, 9)])
        kept = clean_cloud(pts + spikes + far, eyes)
        self.assertFalse(any(p.u > 20 for p in kept))
        self.assertFalse(any(p.e > 500 for p in kept))
        self.assertGreater(len(kept), 0.9 * len(pts))

    def test_dsm_mesh_follows_surface(self) -> None:
        pts = _hill()
        with tempfile.TemporaryDirectory() as tmp:
            info = mesh_from_points(pts, Path(tmp) / "mesh.ply")
        self.assertEqual(info["status"], "ok")
        self.assertEqual(info["method"], "dsm25d")
        self.assertGreater(info["triangles"], 1000)

    def test_agreeing_points_rejects_offset_layer(self) -> None:
        base = _hill(seed=1)
        on = _hill(n=500, seed=2)
        layer = [ReconPoint(p.e, p.n, p.u + 15.0, p.r, p.g, p.b, p.conf, p.observations) for p in _hill(n=500, seed=3)]
        kept = agreeing_points(base, on + layer)
        self.assertGreater(len(kept), 450)
        self.assertLess(len(kept), 520)


if __name__ == "__main__":
    unittest.main()
