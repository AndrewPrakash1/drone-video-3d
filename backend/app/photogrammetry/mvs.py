"""Multi-view stereo: rectified depth maps fused when neighbouring views agree."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .sfm import Camera, SfMResult, _depth

log = logging.getLogger(__name__)


@dataclass
class DenseCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    views: np.ndarray
    ncc: np.ndarray


def _scale_k(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    ks = k.copy()
    ks[0, 0] *= sx
    ks[1, 1] *= sy
    ks[0, 2] *= sx
    ks[1, 2] *= sy
    return ks


def _neighbours(result: SfMResult, max_per_view: int = 2) -> dict[int, list[int]]:
    cams = result.cameras
    ids = sorted(cams)
    shared: dict[tuple[int, int], int] = {}
    for obs in result.observations:
        imgs = [i for i in obs if i in cams]
        for a in range(len(imgs)):
            for b in range(a + 1, len(imgs)):
                key = (min(imgs[a], imgs[b]), max(imgs[a], imgs[b]))
                shared[key] = shared.get(key, 0) + 1
    out: dict[int, list[int]] = {}
    for i in ids:
        cands = []
        for j in ids:
            if i == j:
                continue
            n = shared.get((min(i, j), max(i, j)), 0)
            if n < 15:
                continue
            pts = np.array([result.points[p] for p, o in enumerate(result.observations) if i in o and j in o]).reshape(-1, 3)
            if len(pts) == 0:
                continue
            va = cams[i].center[None] - pts
            vb = cams[j].center[None] - pts
            ang = np.degrees(np.arccos(np.clip((va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12), -1, 1)))
            mean_ang = float(np.median(ang))
            if 3.0 <= mean_ang <= 30.0:
                cands.append((abs(mean_ang - 10.0) - 0.002 * n, j))
        cands.sort()
        out[i] = [j for _, j in cands[:max_per_view]]
    return out


def _depth_map(img_a, img_b, k, cam_a: Camera, cam_b: Camera, depth_range):
    h, w = img_a.shape[:2]
    r_rel = np.ascontiguousarray(cam_b.rotation @ cam_a.rotation.T)
    t_rel = np.ascontiguousarray((cam_b.tvec - r_rel @ cam_a.tvec).reshape(3, 1))
    baseline = float(np.linalg.norm(t_rel))
    if baseline < 1e-6:
        return None
    dist = np.zeros(5)
    # alpha=0 crops to the overlapping region and can multiply the focal length
    # several times over, which pushes the disparity window off the image and
    # the depth map comes back empty. -1 keeps the original intrinsics.
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(k, dist, k, dist, (w, h), r_rel, t_rel, alpha=-1)
    map1x, map1y = cv2.initUndistortRectifyMap(k, dist, r1, p1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(k, dist, r2, p2, (w, h), cv2.CV_32FC1)
    ga = cv2.cvtColor(cv2.remap(img_a, map1x, map1y, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.remap(img_b, map2x, map2y, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    f_rect = float(p1[0, 0])
    cx_r, cy_r = float(p1[0, 2]), float(p1[1, 2])
    dmin, dmax = depth_range
    disp_max = f_rect * baseline / max(dmin, 1e-3)
    disp_min = f_rect * baseline / max(dmax, 1e-3)
    num_disp = int(np.ceil(np.clip(disp_max - disp_min, 32, 512) / 16.0) * 16)
    min_disp = int(max(np.floor(disp_min) - 8, 0))
    # OpenCV throws when minDisparity + numDisparities leaves no room in the
    # image. The depth range comes from sparse points and can be unreliable, so
    # shrink the window (or give up) rather than letting SGBM raise.
    limit = w - 4
    if min_disp >= limit - 32:
        return None
    if min_disp + num_disp > limit:
        num_disp = int(np.floor((limit - min_disp) / 16.0) * 16)
    if num_disp < 32:
        return None
    vertical = abs(p2[1, 3]) > abs(p2[0, 3])
    if vertical:
        ga = cv2.rotate(ga, cv2.ROTATE_90_CLOCKWISE)
        gb = cv2.rotate(gb, cv2.ROTATE_90_CLOCKWISE)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=5,
        P1=8 * 3 * 25, P2=32 * 3 * 25, disp12MaxDiff=1, uniquenessRatio=8,
        speckleWindowSize=120, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    d1 = sgbm.compute(ga, gb).astype(np.float32) / 16.0
    d2 = np.fliplr(sgbm.compute(np.ascontiguousarray(np.fliplr(ga)), np.ascontiguousarray(np.fliplr(gb))).astype(np.float32) / 16.0)
    disp = d1 if (d1 > min_disp + 0.5).sum() >= (d2 > min_disp + 0.5).sum() else d2
    if vertical:
        disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    valid = disp > (min_disp + 0.5)
    if int(valid.sum()) < 200:
        return None
    z_rect = np.full((h, w), np.nan, np.float32)
    z_rect[valid] = f_rect * baseline / disp[valid]
    vv, uu = np.mgrid[0:h, 0:w]
    pts_rect = np.stack([(uu - cx_r) * z_rect / f_rect, (vv - cy_r) * z_rect / f_rect, z_rect], axis=-1)
    xyz = pts_rect.reshape(-1, 3) @ r1
    depth = xyz[:, 2].reshape(h, w)
    valid &= np.isfinite(depth) & (depth > dmin * 0.5) & (depth < dmax * 1.5)
    depth_unrect = np.full((h, w), np.nan, np.float32)
    cam_pts = xyz[valid.ravel()]
    proj = (k @ cam_pts.T).T
    ui = np.round(proj[:, 0] / proj[:, 2]).astype(int)
    vi = np.round(proj[:, 1] / proj[:, 2]).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    depth_unrect[vi[inside], ui[inside]] = cam_pts[inside, 2]
    mask = np.isfinite(depth_unrect)
    if int(mask.sum()) < 200:
        return None
    return depth_unrect, mask


def _views_to_check(i: int, nbrs: dict[int, list[int]], order: list[int], depth_maps: dict, limit: int = 3) -> list[int]:
    """Neighbour views that actually have a depth map. Missing maps used to raise KeyError."""
    seen: list[int] = []
    for j in list(nbrs.get(i, [])) + [o for o in order if o in depth_maps and o != i]:
        if j == i or j not in depth_maps or j in seen:
            continue
        seen.append(j)
        if len(seen) >= limit:
            break
    return seen


def dense_reconstruct(result: SfMResult, images: dict[int, np.ndarray], work_width: int = 640, max_points_per_view: int = 8000, consistency_tol: float = 0.03, progress=None) -> DenseCloud:
    cams = result.cameras
    empty = DenseCloud(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, int), np.zeros(0))
    if not cams:
        return empty
    any_img = next(iter(images.values()))
    h0, w0 = any_img.shape[:2]
    size = (work_width, int(round(h0 * work_width / float(w0))))
    small = {i: cv2.resize(im, size, interpolation=cv2.INTER_AREA) for i, im in images.items() if i in cams}
    ks = _scale_k(result.k, size[0] / w0, size[1] / h0)
    nbrs = _neighbours(result)
    ranges = {}
    for i, cam in cams.items():
        pts = np.array([result.points[p] for p, o in enumerate(result.observations) if i in o]).reshape(-1, 3)
        if len(pts) < 5:
            continue
        d = _depth(cam, pts)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) < 5:
            continue
        # A handful of badly triangulated near points can collapse the range,
        # which pushes the SGBM disparity window off the image. Keep the band
        # within a sane multiple of the median depth instead.
        med = float(np.median(d))
        lo = max(float(np.percentile(d, 5)), med * 0.25)
        hi = min(float(np.percentile(d, 95)), med * 6.0)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        ranges[i] = (lo * 0.9, hi * 1.1)
    depth_maps = {}
    order = sorted(cams)
    for n, i in enumerate(order):
        if i in ranges and nbrs.get(i):
            try:
                dm = _depth_map(small[i], small[nbrs[i][0]], ks, cams[i], cams[nbrs[i][0]], ranges[i])
            except cv2.error as exc:
                log.warning("depth map failed for camera %s: %s", i, exc)
                dm = None
            if dm is None:
                log.debug("no depth map for camera %s", i)
            if dm is not None:
                depth_maps[i] = dm
        if progress:
            progress("depth", n + 1, len(order))
    hs, ws = size[1], size[0]
    all_xyz, all_rgb, all_views = [], [], []
    for n, i in enumerate(order):
        if i not in depth_maps:
            continue
        depth, mask = depth_maps[i]
        cam = cams[i]
        vs, us = np.where(mask)
        if len(us) > max_points_per_view:
            sel = np.random.default_rng(i).choice(len(us), max_points_per_view, replace=False)
            us, vs = us[sel], vs[sel]
        z = depth[vs, us]
        pix = np.stack([us, vs, np.ones_like(us)], 1).astype(np.float64)
        cam_pts = (np.linalg.inv(ks) @ pix.T).T * z[:, None]
        world = (cam.rotation.T @ (cam_pts - cam.tvec[None]).T).T
        agree = np.ones(len(world), dtype=int)
        for j in _views_to_check(i, nbrs, order, depth_maps):
            dj, _mj = depth_maps[j]
            cj = cams[j]
            cp = (cj.rotation @ world.T + cj.tvec[:, None]).T
            zj = cp[:, 2]
            ok = zj > 1e-6
            uj = np.zeros(len(world), int)
            vj = np.zeros(len(world), int)
            uj[ok] = np.round(ks[0, 0] * cp[ok, 0] / zj[ok] + ks[0, 2]).astype(int)
            vj[ok] = np.round(ks[1, 1] * cp[ok, 1] / zj[ok] + ks[1, 2]).astype(int)
            inside = ok & (uj >= 0) & (uj < ws) & (vj >= 0) & (vj < hs)
            sampled = np.full(len(world), np.nan, np.float32)
            sampled[inside] = dj[vj[inside], uj[inside]]
            agree += (inside & np.isfinite(sampled) & (np.abs(sampled - zj) < consistency_tol * np.abs(zj) + 0.05)).astype(int)
        keep = agree >= 2
        if keep.sum() == 0:
            continue
        all_xyz.append(world[keep])
        all_rgb.append(small[i][vs[keep], us[keep]][:, ::-1])
        all_views.append(agree[keep])
        if progress:
            progress("fuse", n + 1, len(order))
    if not all_xyz:
        return empty
    views = np.concatenate(all_views)
    return DenseCloud(np.concatenate(all_xyz), np.concatenate(all_rgb).astype(np.uint8), views, np.clip(views / 4.0, 0, 1).astype(np.float32))
