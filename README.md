# OnePass

Single-pass drone video → georeferenced, **metrically checkable** 3D (SIH26158 / NTRO).

The dashboard reconstructs incrementally in Cesium, colors the cloud by confidence, and lets you measure a known length on the model. That measurement — not a pretty mesh — is what this problem is judged on.

## What runs here

- **Proxy mission** (no upload): South Delhi synthetic flyby of a building whose north eave is **20.0 m**. It has no photographs, so it does not train a Gaussian splat.
- **Upload**: 1080p/4K clip + telemetry CSV. CPU reconstruction always runs. **VGGT** runs on CUDA when the official package is installed; **COLMAP** sparse points are fused when `colmap` is on PATH. **Gaussian splatting** then trains on that GPS-aligned cloud when CUDA and `gsplat` are installed.
- Viewport: **Points**, **Mesh**, and **Neural**. Neural loads `/jobs/{id}/splat.ply` after training. Measure still uses the metric cloud.
- GPU laptop (RTX 5070 Ti): [docs/gpu-colmap.md](docs/gpu-colmap.md)
- Export: georeferenced cloud (`/jobs/{id}/cloud.ply`) and, after training, the neural splat (`/jobs/{id}/splat.ply`).

## Run locally

Need Python 3.11+, Node 20+, FFmpeg. COLMAP and CUDA+VGGT are optional — see [docs/gpu-colmap.md](docs/gpu-colmap.md).

```bash
python3 -m pip install -r backend/requirements.txt
python3 -c "from backend.app.demo_scene import write_demo_files; from pathlib import Path; write_demo_files(Path('data/demo'))"

# terminal 1
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8765 --app-dir /workspace
# if you launched from repo root:
cd /workspace && PYTHONPATH=. python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8765

# terminal 2
cd frontend
npm install
cp -a node_modules/cesium/Build/Cesium/. public/cesium/
npm run dev -- --port 43123 --hostname 0.0.0.0
```

Open http://127.0.0.1:43123 — **Run proxy mission**, then **Measure** the two violet eave markers. You should see ~20 m and PASS within 1 m / 5%.

### Neural render (NVIDIA GPU)

Gaussian training is a CUDA stage. Install the PyTorch CUDA wheel from [docs/gpu-colmap.md](docs/gpu-colmap.md), then:

```bash
pip install -r backend/requirements-splat.txt
```

Upload a real clip. The photogrammetry card shows **Gaussian splatting** for about 7k steps. When it finishes, switch the viewport to **Neural**. Each Gaussian starts on the metric cloud in east-north-up metres, and a scale penalty keeps the building from resizing, so the 20 m check and the splat stay in the same frame. Without CUDA or `gsplat`, that stage is skipped and Points / Mesh stay available.

### Telemetry CSV

```
timestamp,lat,lon,alt,heading,speed,hdop,rtk
0.0,28.5448,77.1924,294.0,88.0,12.4,0.9,0
```

`timestamp` must match video time in seconds.

## Architecture

Hybrid reconstruction: intelligent frame selection → GPS-weighted poses → VGGT chunks (if CUDA) or CPU geometry → COLMAP sparse fusion (if installed) → Open3D mesh → 3D Gaussian training in the same ENU frame (if CUDA and gsplat) → Three.js viewport.

See [docs/sih26158.md](docs/sih26158.md) for the official eight challenges and adapter notes.

## Hardware

A field box with an RTX-class GPU (e.g. 5070 Ti laptop) should follow [docs/gpu-colmap.md](docs/gpu-colmap.md). Gaussian training uses that NVIDIA GPU through CUDA; an Apple GPU or the AMD Radeon 610M cannot run this stage. Reconstruction still finishes on CPU, and Neural turns on once `splat.ply` exists.
