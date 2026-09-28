# RTX 5070 Ti laptop — VGGT + COLMAP

Your NVIDIA GeForce RTX 5070 Ti is the GPU we want. The AMD Radeon 610M is the weak iGPU — **do not** run PyTorch/COLMAP on it.

OnePass now:

1. Runs **VGGT** in overlapping frame chunks when CUDA + the `vggt` package are present.
2. Similarity-aligns that cloud onto GPS ENU (Umeyama) so it stays metric.
3. After the live stream, runs **COLMAP SfM**, aligns it the same way, and **fuses those points into the same Cesium model** (not a sidecar folder).
4. Falls back to CPU geometry if VGGT/COLMAP are missing or fail.

---

## 0. Confirm the NVIDIA GPU is actually used

### Windows

```powershell
nvidia-smi
```

You must see `RTX 5070 Ti`. In Windows Graphics Settings, set Python / Cursor / your terminal to **High performance NVIDIA**.

### WSL2 (recommended for COLMAP + VGGT together)

Install NVIDIA’s [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) driver on Windows, then in Ubuntu:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Blackwell (50-series) needs a **recent** PyTorch CUDA build (CUDA 12.8+). If `torch.cuda.is_available()` is false, update the NVIDIA driver, then install a matching torch wheel from https://pytorch.org.

---

## 1. Install VGGT (neural reconstruction)

From the OnePass repo:

```bash
# 1) PyTorch with CUDA — pick the command pytorch.org shows for your CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Official VGGT
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
pip install -e .
cd ..

# 3) OnePass backend
pip install -r backend/requirements.txt
```

First VGGT run downloads **facebook/VGGT-1B** (~1–2 GB) from Hugging Face. Optional local weights:

```bash
export VGGT_WEIGHTS=/path/to/model.pt
# or
export VGGT_HF_ID=facebook/VGGT-1B
```

Tunables for a 12 GB laptop GPU:

```bash
export VGGT_CHUNK_FRAMES=4      # overlapping views per forward pass
export VGGT_MAX_EDGE=1024       # downscale long 4K edges
export VGGT_MAX_POINTS=8000     # points kept per chunk
```

If VRAM blows up, drop `VGGT_CHUNK_FRAMES` to `2`.

Check:

```bash
curl -s http://127.0.0.1:8765/health
# "vggt": { "available": true, "cuda": true, "package": true, "device": "NVIDIA GeForce RTX 5070 Ti ..." }
```

The dashboard ingest card should read **VGGT ready on NVIDIA GeForce RTX 5070 Ti**. Then **upload** a clip (the proxy demo is synthetic and does not call VGGT).

---

## 2. Install COLMAP and fuse it into the live pipeline

You do **not** need extra OnePass code after install. If `colmap` is on `PATH`, upload jobs automatically:

- dump selected keyframes
- run feature_extractor → exhaustive_matcher → mapper
- convert to TXT, align cameras to GPS, **emit a live Cesium chunk** of sparse points
- voxel-fuse with VGGT/CPU points

### Windows (simplest)

1. Download the prebuilt installer: https://github.com/colmap/colmap/releases  
2. Install, then add `C:\Program Files\COLMAP\bin` (or similar) to PATH.  
3. New terminal:

```powershell
colmap -h
```

CUDA COLMAP builds will use the 5070 Ti for SIFT. If GPU SIFT fails, OnePass retries on CPU automatically.

### Ubuntu / WSL2

```bash
sudo apt update
sudo apt install colmap
colmap -h
```

If the distro package is too old, follow https://colmap.github.io/install.html (build with CUDA so SIFT hits the 5070 Ti).

### Confirm fusion

Restart the API, upload video + telemetry. Logs / job result should include:

```json
"mode": "hybrid-vggt-colmap",
"colmap": { "status": "ok", "points": 1234, "aligned_cameras": 18 }
```

Cesium should get a late chunk with `"source": "colmap"`. The ingest line should say **COLMAP on PATH — live SfM fusion**.

If COLMAP is missing, the job still completes on VGGT/CPU — it will not crash.

---

## 3. Run the stack on the GPU laptop

Terminal 1:

```bash
cd backend
PYTHONPATH=. python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8765
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:43123 → upload a single-pass clip + CSV → watch VGGT chunks, then COLMAP fusion.

---

## 4. Optional Gaussian render

The metric model is still the GPS-aligned cloud. After an upload, OnePass can train a short 3D Gaussian appearance layer on those same metres and serve `splat.ply` to the viewport's Neural mode.

This needs the CUDA torch wheel from section 1, then:

```bash
pip install -r backend/requirements-splat.txt
```

CPU-only runs skip the stage and keep the point cloud and Poisson mesh. The proxy mission has no photographs, so it never trains Gaussians. Length checks stay on the Cesium / GPS cloud.

---

## 5. License note

`facebook/VGGT-1B` is research / non-commercial. For a commercial product use `facebook/VGGT-1B-Commercial` (`VGGT_HF_ID`). SIH student use of the research checkpoint is the usual path.
