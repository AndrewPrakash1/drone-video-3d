"""Binary 3D Gaussian PLY for the Three.js viewer.

GaussianSplats3D's INRIA parser reads rot_0..rot_3 as (w, x, y, z), the same order
gsplat stores quaternions in, so this writer stores them unchanged.
Scales are log-encoded and opacity is a logit, matching that parser.
Spherical harmonics are degree 0 (f_dc only). The viewer does not render degree 3.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

_HEADER = """ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""


def write_gaussian_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats_wxyz: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
) -> int:
    means = np.asarray(means, dtype=np.float32).reshape(-1, 3)
    scales = np.clip(np.asarray(scales, dtype=np.float32).reshape(-1, 3), 1e-6, 1e3)
    quats = np.asarray(quats_wxyz, dtype=np.float32).reshape(-1, 4)
    opacities = np.clip(np.asarray(opacities, dtype=np.float32).reshape(-1), 1e-4, 1.0 - 1e-4)
    colors = np.clip(np.asarray(colors, dtype=np.float32).reshape(-1, 3), 0.0, 1.0)
    n = int(means.shape[0])
    if not (scales.shape[0] == quats.shape[0] == opacities.shape[0] == colors.shape[0] == n):
        raise ValueError("gaussian attributes must share the same vertex count")

    f_dc = ((colors - 0.5) / SH_C0).astype(np.float32)
    logit = np.log(opacities / (1.0 - opacities)).astype(np.float32)
    log_scale = np.log(scales).astype(np.float32)
    # Viewer reads rot_0..rot_3 as (w, x, y, z); gsplat quats are already wxyz.
    rot = quats

    path.parent.mkdir(parents=True, exist_ok=True)
    header = _HEADER.format(n=n).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for i in range(n):
            handle.write(
                struct.pack(
                    "<17f",
                    float(means[i, 0]),
                    float(means[i, 1]),
                    float(means[i, 2]),
                    0.0,
                    0.0,
                    0.0,
                    float(f_dc[i, 0]),
                    float(f_dc[i, 1]),
                    float(f_dc[i, 2]),
                    float(logit[i]),
                    float(log_scale[i, 0]),
                    float(log_scale[i, 1]),
                    float(log_scale[i, 2]),
                    float(rot[i, 0]),
                    float(rot[i, 1]),
                    float(rot[i, 2]),
                    float(rot[i, 3]),
                )
            )
    return n
