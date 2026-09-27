# OnePass

Single-pass drone video → georeferenced, **metrically checkable** 3D (SIH26158 / NTRO).

The dashboard reconstructs incrementally in Cesium, colors the cloud by confidence, and lets you measure a known length on the model. That measurement — not a pretty mesh — is what this problem is judged on.

## What runs here

- **Proxy mission** (no upload):synthetic flyby of a building whose north eave is **20.0 m**.
- **Upload**: 1080p/4K clip + telemetry CSV. CPU reconstruction always runs. **VGGT** runs on CUDA when the official package is installed; **COLMAP** sparse points are fused into the live globe when `colmap` is on PATH.
- GPU laptop (RTX 5070 Ti): [docs/gpu-colmap.md](docs/gpu-colmap.md)
- Export: georeferenced PLY (`/jobs/{id}/cloud.ply`).

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

### Telemetry CSV

```
timestamp,lat,lon,alt,heading,speed,hdop,rtk
0.0,28.5448,77.1924,294.0,88.0,12.4,0.9,0
```

`timestamp` must match video time in seconds.

## Architecture

Hybrid reconstruction: intelligent frame selection → GPS-weighted poses → VGGT chunks (if CUDA) or CPU geometry → COLMAP sparse fusion (if installed) → Open3D mesh → confidence + metric report → Cesium.

See [docs/sih26158.md](docs/sih26158.md) for the official eight challenges and adapter notes.

## Hardware

A field box with an RTX-class GPU (e.g. 5070 Ti laptop) should follow [docs/gpu-colmap.md](docs/gpu-colmap.md). The AMD Radeon 610M iGPU is ignored. This prototype still runs on CPU without those extras.
