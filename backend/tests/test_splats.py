"""Export and PLY checks that do not need a GPU."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.app.photogrammetry.sfm import Camera
from backend.app.reconstruct import ReconPoint
from backend.app.splats.export import export_gaussian_dataset
from backend.app.splats.ply import SH_C0, write_gaussian_ply
from backend.app.splats.train import load_views, splat_status, train_splats


class _Result:
    def __init__(self, k, cameras):
        self.k = k
        self.cameras = cameras


def _camera(index: int, center: np.ndarray) -> Camera:
    rotation = np.eye(3)
    rvec = cv2.Rodrigues(rotation)[0].ravel()
    tvec = -rotation @ center
    return Camera(index, rvec, tvec)


class SplatExportTest(unittest.TestCase):
    def test_export_writes_enu_cameras(self) -> None:
        frames = []
        cameras = {}
        for i, center in enumerate((np.array([0.0, 0.0, 10.0]), np.array([5.0, 1.0, 12.0]))):
            image = np.full((16, 24, 3), 20 + i, np.uint8)
            frames.append((i, float(i), image))
            cameras[i] = _camera(i, center)
        k = np.array([[18.0, 0.0, 12.0], [0.0, 18.0, 8.0], [0.0, 0.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            info = export_gaussian_dataset(Path(tmp), frames, _Result(k, cameras), distortion=np.zeros(4))
            self.assertEqual(info["status"], "ok")
            meta = json.loads((Path(tmp) / "cameras.json").read_text())
            self.assertIsNone(meta["distortion"])
            self.assertEqual(len(meta["frames"]), 2)
            self.assertEqual(meta["frames"][1]["center"][0], 5.0)
            self.assertTrue((Path(tmp) / "images" / "frame_0001.jpg").exists())

    def test_load_views_undistorts_when_coefficients_exist(self) -> None:
        frames = []
        cameras = {}
        for i in range(2):
            image = np.zeros((32, 40, 3), np.uint8)
            image[:, :, 1] = np.linspace(0, 255, 40, dtype=np.uint8)[None, :]
            frames.append((i, 0.0, image))
            cameras[i] = _camera(i, np.array([float(i), 0.0, 8.0]))
        k = np.array([[30.0, 0.0, 20.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            export_gaussian_dataset(dest, frames, _Result(k, cameras), distortion=np.array([0.2, 0.0, 0.0, 0.0]))
            bent = load_views(dest)
            meta = json.loads((dest / "cameras.json").read_text())
            meta["distortion"] = None
            (dest / "cameras.json").write_text(json.dumps(meta))
            straight = load_views(dest)
            self.assertTrue(bent["undistorted"])
            self.assertFalse(straight["undistorted"])
            self.assertFalse(np.allclose(bent["images"], straight["images"]))

    def test_ply_roundtrip_uses_viewer_quaternion(self) -> None:
        means = np.array([[1.0, 2.0, 3.0]], np.float32)
        scales = np.array([[0.2, 0.3, 0.4]], np.float32)
        quats = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
        colors = np.array([[1.0, 0.0, 0.0]], np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splat.ply"
            self.assertEqual(write_gaussian_ply(path, means, scales, quats, np.array([0.5]), colors), 1)
            blob = path.read_bytes()
            header, raw = blob.split(b"end_header\n", 1)
            self.assertIn(b"property float f_dc_0", header)
            self.assertNotIn(b"f_rest", header)
            vals = struct.unpack("<17f", raw)
            self.assertEqual(vals[0:3], (1.0, 2.0, 3.0))
            self.assertAlmostEqual(vals[6], (1.0 - 0.5) / SH_C0, places=5)
            # identity wxyz is stored as wxyz: rot_0 = 1, rot_1..3 = 0
            self.assertAlmostEqual(vals[13], 1.0, places=5)
            self.assertEqual(vals[14:17], (0.0, 0.0, 0.0))

    def test_train_skips_without_cuda_gsplat(self) -> None:
        info = splat_status()
        if info["available"]:
            self.skipTest("CUDA gsplat is installed; skip the CPU path")
        with tempfile.TemporaryDirectory() as tmp:
            out = train_splats(Path(tmp), [ReconPoint(0, 0, 0, 1, 2, 3, 1.0, 1)], Path(tmp) / "splat.ply")
        self.assertEqual(out["status"], "skipped")
        self.assertTrue(out["reason"])


if __name__ == "__main__":
    unittest.main()
