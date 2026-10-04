from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.reconstruct import ReconPoint, voxel_downsample
from backend.app.scene import read_scene_manifest, write_scene_manifest
from backend.app.spatial_model.contracts import CompletionProposal, SpatialTrainingSample
from backend.app.spatial_model.inference import run_optional_stage


class SceneContractTests(unittest.TestCase):
    def test_voxel_fusion_preserves_sources_and_uncertainty(self) -> None:
        points = [
            ReconPoint(0.01, 0.01, 0.0, 1, 2, 3, 0.7, 2, source="photogrammetry", provenance=("mvs",), uncertainty_m=1.2),
            ReconPoint(0.02, 0.02, 0.0, 2, 3, 4, 0.8, 3, source="vggt", provenance=("vggt",), uncertainty_m=0.5),
        ]
        fused = voxel_downsample(points, voxel=1.0)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].source, "photogrammetry")
        self.assertEqual(fused[0].provenance, ("mvs", "vggt"))
        self.assertEqual(fused[0].uncertainty_m, 0.5)

    def test_scene_manifest_keeps_generated_geometry_non_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_scene_manifest(
                root,
                origin={"lat": 1.0, "lon": 2.0, "alt": 3.0},
                point_count=12,
                mesh={"status": "ok", "method": "dsm25d"},
                spatial={"summary": {"faces": 4}},
                cameras=2,
            )
            self.assertEqual(read_scene_manifest(root), manifest)
            self.assertTrue(manifest["reconstruction"]["measurement_authoritative"])
            self.assertFalse(manifest["measurement"]["generated_geometry_included"])
            self.assertEqual(manifest["generated_layers"], [])

    def test_training_sample_roundtrip_validates_parallel_inputs(self) -> None:
        sample = SpatialTrainingSample(
            sample_id="sample-1",
            scene_id="scene-1",
            frame_paths=["a.jpg", "b.jpg"],
            timestamps_s=[0.0, 1.0],
            camera_poses_w2c=[[1.0] * 16, [1.0] * 16],
            intrinsics=[[1.0] * 9, [1.0] * 9],
            origin={"lat": 1.0, "lon": 2.0, "alt": 3.0},
            visible_geometry_path="cloud.ply",
            visibility_path="spatial.json",
        )
        self.assertEqual(SpatialTrainingSample.from_dict(sample.to_dict()).scene_id, "scene-1")
        sample.timestamps_s = [0.0]
        with self.assertRaises(ValueError):
            sample.validate()

    def test_completion_proposal_is_never_measurement_safe(self) -> None:
        proposal = CompletionProposal("p-1", "metric-1", 0, uncertainty=0.8)
        self.assertFalse(proposal.to_dict()["measurement_safe"])
        with self.assertRaises(ValueError):
            CompletionProposal("p-2", "metric-1", 0, measurement_safe=True).validate()

    def test_optional_model_stage_is_skipped_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_optional_stage(Path(tmp), {"schema": "onepass.scene.v1"}, {"patches": []})
            self.assertEqual(result["status"], "skipped")
            self.assertFalse(result["measurement_safe"])
            self.assertTrue((Path(tmp) / "spatial_model.json").exists())


if __name__ == "__main__":
    unittest.main()
