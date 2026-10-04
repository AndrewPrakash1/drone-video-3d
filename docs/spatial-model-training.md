# OnePass spatial-model training

This repository now contains a first trainable, metric-conditioned spatial model. It is a narrow research model, not a finished foundation model. It predicts compact completion attributes and novel-view attributes from noisy frames plus camera features. The next research steps are richer image/depth decoders, true 3D tokens, calibrated uncertainty, and large-scale pretraining.

## 1. Install the training environment

Use Python 3.11 and a PyTorch build matching the machine. On an NVIDIA machine, install the CUDA-enabled wheel from the official PyTorch selector first, then install the backend and GPU extras:

```bash
python3.11 -m venv .venv-spatial
source .venv-spatial/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r backend/requirements-gpu.txt
```

The model can run on CPU for a smoke test, but serious training requires CUDA. The current trainer selects `cuda` automatically when available; override it with `--device cuda` or `--device cpu`.

## 2. Generate the first dataset immediately

This produces a small, fully synthetic dataset. It is useful for testing tensorization, augmentation contracts, losses, checkpointing, and the end-to-end pipeline. It is not sufficient to teach real architecture or drone geometry.

```bash
python scripts/spatial_model_generate.py \
  --output data/spatial_synth \
  --scenes 120 \
  --views 6 \
  --size 128
```

Each scene contains degraded input frames, clean target views, camera metadata, visibility metadata, compact patch/view targets, and `license_name: OnePass synthetic generated`. The manifest is scene-disjoint: train, validation, and test scenes do not overlap.

## 3. Run a smoke training job

```bash
python scripts/spatial_model_train.py \
  --manifest data/spatial_synth/manifest.json \
  --output checkpoints/spatial-0.1.pt \
  --epochs 20 \
  --batch-size 8 \
  --device cuda
```

Outputs:

- `checkpoints/spatial-0.1.pt` — best validation checkpoint.
- `checkpoints/spatial-0.1.json` — loss/device history.

The current checkpoint is not yet consumed by runtime inference. Checkpoint loading and proposal generation are the next integration task.

## 4. Public data mixture

Do not put downloaded datasets in Git. Normalize each scene into the `SpatialTrainingSample` format and record the exact source/license in the manifest.

Recommended sequence:

1. **BlendedMVS / BlendedMVG** for supervised multi-view warm-up. It provides images, camera files, and rendered depth; the project states that it is CC BY 4.0. Use it to teach multi-view/depth structure, not drone-specific priors.
2. **Tanks and Temples training data** for real video/image sequences with ground-truth geometry. Its official page provides training ground truth and states non-commercial terms; verify the current license before any redistribution or commercial use.
3. **UrbanScene3D** for aerial/urban reconstruction and drone path patterns. It includes high-resolution images, LiDAR, synthetic scenes, and simulator data, but the official page states non-commercial-only access and prohibits distributing the dataset or derivatives. Use it only where its terms fit the project.
4. **MegaDepth-X** for sparse/noisy in-the-wild reconstruction research. It contains scene archives with images, dense depth, and COLMAP sparse models; the released reconstruction assets are CC BY 4.0, while original images retain their own licenses. Read `license/license.parquet` and filter images before training or redistribution.
5. **Your own drone missions** for the final domain adaptation. Each scene should have video, telemetry, camera calibration, independent target geometry or held-out views, and a signed data/license record. A reconstruction produced from the same input video is not independent ground truth.

For each external sample, store:

```json
{
  "license_name": "CC BY 4.0",
  "license_url": "https://creativecommons.org/licenses/by/4.0/",
  "source_url": "...",
  "metadata": {
    "attribution": "...",
    "commercial_use_allowed": false,
    "redistribution_allowed": false
  }
}
```

## 5. Training curriculum

### Stage A — synthetic smoke test

Train on generated scenes until tensorization and both losses decrease. Test held-out scene IDs, not held-out frames.

### Stage B — geometry warm-up

Train on BlendedMVS/BlendedMVG and Tanks and Temples samples with clean depth/pose targets. Add blur, JPEG, exposure, rolling-shutter-like row offsets, frame dropping, and sparse-view sampling online.

### Stage C — aerial adaptation

Fine-tune on UrbanScene3D-compatible samples and your own flights. Make the input sampling resemble OnePass: mostly nadir/oblique views, limited overlap, noisy GPS-conditioned poses, and large unseen patches.

### Stage D — sparse generative completion

For each dense scene, hide entire view sectors or connected mesh patches. The model sees only the remaining views and predicts the held-out target views/geometry. Never use a random per-frame split; that leaks the same building into train and validation.

### Stage E — product evaluation

Report separately for observed and generated areas:

- held-out RGB/novel-view error;
- depth/geometry error;
- surface completeness;
- uncertainty calibration;
- seam consistency at observed/generated boundaries;
- known-length error on the metric cloud;
- hallucination rate on deliberately unsupported details.

## 6. Data rule for OnePass

The GPS-aligned cloud and measurement API remain authoritative. A generated proposal must contain its parent scene revision, patch ID, conditioning views, model version, uncertainty, and `measurement_safe: false`. Generated completion can be displayed, edited, or used for exploratory planning, but it must not overwrite `cloud.ply`, `cloud.json`, or certified measurements.

## 7. Preparing a real job

The helper copies the posed frames and records the existing artifacts:

```bash
python scripts/spatial_model_prepare_job.py \
  --job /path/to/onepass-job-artifacts \
  --output data/spatial_real \
  --scene-id brighton-001 \
  --license-name "Owned project data" \
  --source-url "internal"
```

It intentionally warns that reused input frames are not independent supervision. Replace `target_view_paths` and `target_geometry_path` with held-out views and independently surveyed/LiDAR geometry before using the sample for validation or serious training.
