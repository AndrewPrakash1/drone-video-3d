# OnePass

Single-pass drone video → georeferenced, **metrically checkable** 3D (SIH26158 / NTRO).

The dashboard builds a metric model you can measure. A known length on that model — not a pretty picture — is what this problem is judged on. The viewport can also show a neural render of the same scene, but that render is appearance only. It lives in the same metres as the GPS cloud and does not replace the length check.

## What runs here

- **Proxy mission** (no upload): South Delhi synthetic flyby of a building whose north eave is **20.0 m**. It has no photographs, so it does not train a Gaussian splat.
- **Upload**: 1080p/4K clip + telemetry CSV. Classical photogrammetry always runs on the CPU. **VGGT** runs on CUDA when the official package is installed. **COLMAP** sparse points are fused when `colmap` is on PATH. **Gaussian splatting** then trains on that GPS-aligned cloud when CUDA and `gsplat` are installed.
- Viewport: **Points**, **Mesh**, and **Neural**. Neural loads `/jobs/{id}/splat.ply` after training. Measure still uses the metric cloud.
- GPU laptop (RTX 5070 Ti): [docs/gpu-colmap.md](docs/gpu-colmap.md)
- Export: georeferenced cloud (`/jobs/{id}/cloud.ply`) and, after training, the neural splat (`/jobs/{id}/splat.ply`).

## Run locally

Need Python 3.11+, Node 20+, FFmpeg. COLMAP, CUDA+VGGT, and CUDA+gsplat are extra — see [docs/gpu-colmap.md](docs/gpu-colmap.md).

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

Upload a real clip. The photogrammetry card shows **Gaussian splatting** for about 7,000 steps. When it finishes, switch the viewport to **Neural**. Without an NVIDIA GPU or without `gsplat`, that stage is skipped and Points / Mesh stay available. An Apple GPU and the AMD Radeon 610M cannot run the CUDA rasterizer.

### Telemetry CSV

```
timestamp,lat,lon,alt,heading,speed,hdop,rtk
0.0,28.5448,77.1924,294.0,88.0,12.4,0.9,0
```

`timestamp` must match video time in seconds.

## How the pieces fit

```
video + GPS
  → keep sharp, well-exposed frames
  → SIFT, matching, sparse SfM, bundle adjustment
  → scale and rotate the model onto GPS (east, north, up)
  → dense multi-view stereo and a Poisson mesh
  → optional VGGT / COLMAP points fused into the same cloud
  → optional 3D Gaussian training on those metres
  → Three.js: Points, Mesh, or Neural
```

Photogrammetry answers “where is the surface, in metres?” Neural rendering answers “what does that surface look like from a new camera?” OnePass runs the first on every upload and the second only when CUDA is present. See [docs/sih26158.md](docs/sih26158.md) for the eight official challenges.

## Photogrammetry and 3D reconstruction

Photogrammetry recovers 3D geometry from overlapping photographs whose camera positions are known only roughly. A single drone pass is a hard version of that problem: the camera mostly looks down, overlap is limited, GPS is noisy, and there are no surveyed ground control points. The pipeline below is what `backend/app/photogrammetry/` actually runs.

### Frames and telemetry

The video is decoded and each candidate frame is scored for blur, exposure, and redundancy. Blurred, blown-out, and near-duplicate frames are dropped so later geometry is not built on bad pixels. Telemetry (`lat`, `lon`, `alt`, heading, HDOP) is interpolated to the timestamp of each kept frame.

Those GPS samples become a local **ENU** frame: east, north, and up in metres, with the origin at the first fix. ENU is the coordinate system used for the point cloud, the mesh, the cameras, and the Gaussians. Geodetic latitude and longitude are only for the globe and for the measurement API.

### Feature extraction (SIFT)

**SIFT** finds interest points that survive small changes in scale and viewpoint, and describes each one with a vector. Two photos of the same roof corner should produce similar vectors even if the drone has moved.

### Matching (KNN, ratio test, RANSAC)

Descriptors are paired with a **k-nearest-neighbour** search. Lowe’s ratio test keeps a match only when the best neighbour is much closer than the second best, which rejects repetitive texture such as roof tiles and road markings. **RANSAC** then fits a geometric model and throws out matches that disagree with it. What remains is a set of tentative correspondences: “this pixel in frame A is the same physical point as that pixel in frame B.”

### Sparse Structure-from-Motion

**Structure-from-Motion (SfM)** estimates camera poses and a sparse 3D point cloud together.

1. An initial pair with enough parallax yields an essential matrix and two relative poses.
2. Matched pixels are **triangulated**: the 3D point is the intersection of the two camera rays.
3. Further frames are registered by **PnP** (perspective-n-point), which solves for a camera given known 3D points and their pixels.
4. New points are triangulated from the growing set of cameras.

The internal camera is a pinhole. Focal length starts from an assumed horizontal field of view (`initial_intrinsics` in `backend/app/photogrammetry/sfm.py`). That model is good enough to triangulate, and it is the model the Gaussian trainer uses unless a distortion vector is supplied.

### Bundle adjustment

Triangulation and PnP accumulate small errors. **Bundle adjustment** moves the cameras and the 3D points together to reduce reprojection error: the distance, in pixels, between a predicted projection and the observed feature. OnePass runs this repeatedly while the model grows, then once more at the end.

### GPS alignment

SfM’s own coordinate system has an arbitrary origin, rotation, and scale. **Umeyama alignment** finds the similarity transform (scale, rotation, translation) that maps reconstructed camera centres onto the GPS centres in ENU. After that, a 20 m eave is about 20 m in the model. The residual of that fit is reported as a GPS RMSE. This is metric control without ground control points. It is only as good as the GPS, so the UI reports the measured error instead of claiming centimetre accuracy.

### Dense multi-view stereo

The sparse cloud is only the feature points. **Multi-view stereo (MVS)** estimates a depth for many more pixels by matching patches across the registered cameras, then keeps depths that agree from several views. Those dense points carry the colour of the source pixel. The fused cloud is voxel-downsampled so nearby duplicates collapse into one point.

### Poisson mesh

A point cloud has no surface. **Poisson reconstruction** (Open3D) fits a smooth watertight mesh through the points and crops it to the cloud’s bounds. The viewport’s Mesh mode draws that surface with vertex colours. If the cloud is too thin, meshing is skipped and the points remain.

### Confidence and the length check

Each point gets a confidence from how many views saw it, how large the triangulation angle was, how sharp the frame was, and how trustworthy the GPS was. High, medium, and low bands are what the colour mode can show. **Measure** picks two positions and reports the straight-line distance in the ENU frame against a known length when one exists (the proxy eave is 20.0 m, pass within 1 m or 5%).

### Optional neural geometry (VGGT) and COLMAP

**VGGT** (Visual Geometry Grounded Transformer) is a feed-forward network that predicts cameras and points from images in one pass. When CUDA and the package are installed, OnePass runs it on short overlapping chunks and similarity-aligns those points onto the same ENU cloud. It is a neural reconstructor, not the renderer.

**COLMAP**, if it is on `PATH`, runs its own feature extraction, matching, and mapper. Its sparse points are aligned the same way and fused in. COLMAP may estimate lens distortion. The Gaussian exporter can store those coefficients so training undistorts the frames first.

## Neural rendering

Classical rendering draws a mesh with a hand-written lighting model. The Poisson mesh in this project is a Lambertian surface with vertex colours, so it looks flat. **Neural rendering** instead stores a scene representation that was optimised to reproduce the photographs, then draws new viewpoints from that representation.

OnePass uses **3D Gaussian Splatting** for that representation. It does not replace the metric cloud, and it does not run a NeRF viewer.

### 3D Gaussian Splatting

Kerbl et al. (SIGGRAPH 2023) represent a scene as millions of fuzzy elliptical blobs. Each Gaussian has:

- a centre (where it sits in 3D),
- a covariance (how stretched and rotated it is), stored as a scale and a quaternion,
- an opacity,
- a colour (here, a view-independent RGB colour).

To draw a pixel, the Gaussians that overlap that pixel are projected to 2D and composited from front to back, the same alpha blending used for transparent particles. Because the projection is a rasterizer rather than a ray march through a neural network, a trained scene can be drawn at interactive rates. Training is still an optimisation: render the photographs from the known cameras, compare to the real pixels, and nudge every Gaussian’s parameters with gradients.

The paper is [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). NVIDIA’s later work, below, is what we used to decide how a drone camera should feed that optimiser.

### NVIDIA 3DGUT (the paper this training follows)

Drone video is a poor fit for the original Gaussian Splatting rasterizer. That rasterizer projects each 3D Gaussian with a linearised pinhole model (EWA splatting). Wide-angle lenses, residual distortion, and rolling shutter break that linearisation, so roofs bow and edges smear.

[**3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting**](https://arxiv.org/pdf/2412.12507) (Wu, Martinez Esturo, Mirzaei, Moenne-Loccoz, Gojcic, NVIDIA, arXiv:2412.12507) replaces that linearisation with an **Unscented Transform**. A Gaussian is summarised by a few sigma points, those points are pushed through the real camera model (distortion, rolling shutter, any other nonlinear projection), and a 2D Gaussian is refit from the projected points. The scene is still rasterized, so it stays fast. The same paper also lines the math up with ray tracing so reflections and refraction can be added later. Their code is [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut).

What OnePass takes from 3DGUT:

- Train on the real image, including pixels a naive pinhole crop would throw away.
- If the camera has distortion coefficients, undistort with OpenCV and update the intrinsics, then run ordinary Gaussian Splatting. That is the practical form of 3DGUT’s main quality point for a moderate drone lens.
- Keep rasterization. Do not switch the viewer to volumetric ray marching.

What OnePass does not take, on purpose:

- The full Unscented Transform CUDA stack in `3dgrut`. That build is a poor bet on a Blackwell 5070 Ti, and OpenCV undistort covers the lens model we actually have.
- Secondary rays for glass, wet roads, and refraction. Those are not what makes a rooftop flyby look bad.
- Rolling-shutter projection per image row. That remains a follow-up if roofs still bow after undistort.

The other papers on NVIDIA’s [neural-rendering list](https://research.nvidia.com/labs/rtr/tag/neural-rendering/) (8DNA, neural materials, UniRelight, DiffusionRenderer, real-time neural appearance models) replace shaders on rich physically based materials. This project has no such materials. They would not turn a sparse flyby into a photoreal building.

### Instant NeRF, and why we did not ship it

[Instant NeRF](https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/) is NVIDIA’s product form of **Instant-NGP** (Müller et al., SIGGRAPH 2022): a radiance field stored in a multiresolution hash grid plus a tiny network, trained in seconds to minutes. A **NeRF** answers “what colour and density is at this 3D point?” by querying that network along a camera ray and integrating. Quality can be high. Drawing a frame still costs a march through the volume, so even Instant-NGP is typically tens of milliseconds per frame and does not drop into a browser viewport.

3DGUT’s own related work makes the same comparison: hash-grid NeRFs struggle to stay interactive, while Gaussian rasterization does not. OnePass keeps Instant-NGP’s speed lesson only — a short per-scene optimisation (about 7,000 steps, the fast regime used when NVIDIA compared hash-grid NeRFs to Gaussians), an explicit scene, no giant MLP. The delivered viewer is a 3DGS `.ply` in Three.js, not the Instant-NGP desktop app.

### What training does in this repository

`backend/app/splats/train.py` runs only after an upload has posed frames and a metric cloud.

1. Registered frames and their OpenCV world-to-camera poses are written next to the job (`backend/app/splats/export.py`). Poses are already in ENU metres.
2. Images are undistorted when distortion coefficients are present, then scaled so the long edge is about 640 px.
3. One Gaussian is created per point of the fused metric cloud (capped near 80,000). Its centre and colour come from that point. Its initial size comes from the local spacing of the cloud, not from a random initialisation.
4. For each of ~7,000 steps, `gsplat` rasterizes the Gaussians into one training camera on the CUDA GPU. The loss is photometric (L1 plus a structural-similarity term) plus a **scale-and-centre penalty**. That penalty keeps the cloud’s mean and spread near the GPS-aligned original, so optimisation cannot quietly resize the building. The 20 m check still uses the GPS cloud. The splat is forced to occupy the same metres.
5. Gaussians with almost no opacity are dropped. The rest are written as `splat.ply` (positions in ENU). The browser viewer reads rotations in xyzw order, so the writer converts from gsplat’s wxyz quaternions on the way out.

Training is minutes on an RTX-class GPU, not a multi-hour NeRF. The proxy mission never enters this stage.

### What the viewport draws

`frontend/components/recon-viewport.tsx` is a Three.js scene:

- **Points** — the reconstructed cloud, constant-size markers.
- **Mesh** — the Poisson surface.
- **Neural** — `@mkkellogg/gaussian-splats-3d` loads `splat.ply` and sorts the Gaussians on the GPU as you orbit. ENU (east, north, up) is rotated into Three.js (east, up, −north), the same axis map as the point cloud. Camera frustums stay visible in every mode.

Neural mode is disabled until a job reports that `splat.ply` is ready. Cesium measurement, where it is used, is unchanged. Gaussians are not drawn inside the globe.

## Hardware

A field box with an RTX-class GPU (e.g. 5070 Ti laptop) should follow [docs/gpu-colmap.md](docs/gpu-colmap.md). Gaussian training uses that NVIDIA GPU through CUDA. Reconstruction still finishes on CPU, and Neural turns on once `splat.ply` exists.

## Further reading

- Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), SIGGRAPH 2023. The representation and the rasterizer.
- Wu et al., [3DGUT](https://arxiv.org/pdf/2412.12507), NVIDIA, arXiv:2412.12507. Distorted cameras and rolling shutter for Gaussian Splatting. This is the NVIDIA paper the training path follows.
- Müller et al., [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/), NVIDIA, SIGGRAPH 2022, and the [Instant NeRF](https://developer.nvidia.com/blog/getting-started-with-nvidia-instant-nerfs/) write-up. The speed lesson. Not the viewer we ship.
- [gsplat](https://github.com/nerfstudio-project/gsplat) — the CUDA rasterizer used for the 7,000-step fit.
