from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from backend.app.spatial_model.contracts import SpatialTrainingSample
from backend.app.spatial_model.dataset import write_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a OnePass job as a spatial-model sample")
    parser.add_argument("--job", type=Path, required=True, help="Completed job artifact directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--license-name", required=True)
    parser.add_argument("--license-url")
    parser.add_argument("--source-url")
    args = parser.parse_args()
    cameras = json.loads((args.job / "cameras.json").read_text(encoding="utf-8"))
    frames = cameras.get("frames") or []
    if not frames:
        raise SystemExit("job has no gs/cameras.json frames")
    image_dir = args.job / "gs" / "images"
    if not image_dir.exists():
        raise SystemExit("job has no exported gs/images directory")
    scene_dir = args.output / args.scene_id
    frame_paths = []
    target_paths = []
    poses = []
    intrinsics = []
    for frame in frames:
        source = image_dir / str(frame["file"])
        if not source.exists():
            continue
        dest = scene_dir / "images" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            continue
        cv2.imwrite(str(dest), image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_paths.append(str(dest.relative_to(args.output)))
        target_paths.append(str(dest.relative_to(args.output)))
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.asarray(frame["R"], dtype=np.float32).reshape(3, 3)
        pose[:3, 3] = np.asarray(frame["t"], dtype=np.float32)
        poses.append(pose.reshape(-1).tolist())
        intrinsics.append([cameras["fx"], cameras["fy"], cameras["cx"], cameras["cy"], cameras["width"], cameras["height"], 0.0, 0.0, 1.0])
    sample = SpatialTrainingSample(
        sample_id=args.scene_id,
        scene_id=args.scene_id,
        frame_paths=frame_paths,
        timestamps_s=[float(i) for i in range(len(frame_paths))],
        camera_poses_w2c=poses,
        intrinsics=intrinsics,
        origin={"lat": 0.0, "lon": 0.0, "alt": 0.0},
        visible_geometry_path=str((args.job / "cloud.ply").resolve()),
        visibility_path=str((args.job / "spatial.json").resolve()),
        target_view_paths=target_paths,
        target_geometry_path=str((args.job / "cloud.ply").resolve()),
        license_name=args.license_name,
        license_url=args.license_url,
        source_url=args.source_url,
        metadata={"requires_review": True, "target_views_are_input_reuse": True},
    )
    write_sample(scene_dir / "sample.json", sample)
    print(f"wrote {scene_dir / 'sample.json'}")
    print("Replace target_view_paths/target_geometry_path with independent ground truth before training.")


if __name__ == "__main__":
    main()
