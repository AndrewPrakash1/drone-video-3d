from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.spatial_model.manifest import DatasetManifest
from backend.app.spatial_model.synthetic import generate_dataset


class SpatialModelDataTests(unittest.TestCase):
    def test_synthetic_manifest_is_scene_disjoint_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = generate_dataset(root, scenes=12, views=3, size=32)
            loaded = DatasetManifest.read(root / "manifest.json")
        self.assertEqual(len(loaded.samples), 12)
        self.assertTrue(loaded.splits["train"])
        self.assertTrue(loaded.splits["val"])
        self.assertTrue(loaded.splits["test"])
        self.assertEqual(manifest.metadata["generator"], "onepass.synthetic.v1")

    def test_manifest_rejects_unknown_license(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_dataset(root, scenes=3, views=2, size=24)
            value = (root / "manifest.json").read_text(encoding="utf-8").replace("OnePass synthetic generated", "unknown")
            (root / "manifest.json").write_text(value, encoding="utf-8")
            with self.assertRaises(ValueError):
                DatasetManifest.read(root / "manifest.json")

    def test_tensor_dataset_shapes_when_torch_is_available(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        from backend.app.spatial_model.manifest import DatasetManifest
        from backend.app.spatial_model.tensorize import SpatialTensorDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_dataset(root, scenes=2, views=3, size=32)
            manifest = DatasetManifest.read(root / "manifest.json")
            item = SpatialTensorDataset(manifest.samples, root, views=3, image_size=32)[0]
        self.assertEqual(tuple(item["frames"].shape), (3, 3, 32, 32))
        self.assertEqual(tuple(item["camera_features"].shape), (3, 25))
        self.assertEqual(tuple(item["targets"]["completion"].shape), (1, 5))


if __name__ == "__main__":
    unittest.main()
