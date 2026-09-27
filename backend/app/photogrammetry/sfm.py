"""Incremental Structure-from-Motion: triangulation, PnP, bundle adjustment, GPS scale."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ..align import umeyama
from .features import ImageFeatures, PairMatch


@dataclass
class Camera:
    index: int
    rvec: np.ndarray
    tvec: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.tvec

    def projection(self, k: np.ndarray) -> np.ndarray:
        return k @ np.hstack([self.rotation, self.tvec.reshape(3, 1)])


@dataclass
class SfMResult:
    k: np.ndarray
    cameras: dict[int, Camera]
    points: np.ndarray
    colors: np.ndarray
    observations: list[dict[int, int]]
    errors: np.ndarray
    triangulation_angles: np.ndarray
    ba_history: list[dict] = field(default_factory=list)
    gps_scale: float = 1.0
    gps_rmse_m: float | None = None


def initial_intrinsics(width: int, height: int, hfov_deg: float = 73.0) -> np.ndarray:
    f = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _project(k: np.ndarray, cam: Camera, pts: np.ndarray) -> np.ndarray:
    proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), cam.rvec, cam.tvec, k, None)
    return proj.reshape(-1, 2)


def _triangulate(k: np.ndarray, ca: Camera, cb: Camera, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    h = cv2.triangulatePoints(ca.projection(k), cb.projection(k), pa.T.astype(np.float64), pb.T.astype(np.float64))
    w = np.where(np.abs(h[3]) < 1e-12, 1e-12, h[3])
    return (h[:3] / w).T


def _angle_deg(ca: np.ndarray, cb: np.ndarray, x: np.ndarray) -> np.ndarray:
    va = ca[None, :] - x
    vb = cb[None, :] - x
    cosang = (va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def _depth(cam: Camera, x: np.ndarray) -> np.ndarray:
    return (cam.rotation @ x.T + cam.tvec.reshape(3, 1))[2]


def _pick_initial_pair(features: list[ImageFeatures], pairs: list[PairMatch]) -> PairMatch | None:
    best, best_score = None, -1.0
    for pm in pairs:
        if pm.n_inliers < 60:
            continue
        a = features[pm.i].keypoints[pm.inliers[:, 0]]
        b = features[pm.j].keypoints[pm.inliers[:, 1]]
        _, hmask = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
        hom_ratio = float(hmask.sum() / len(a)) if hmask is not None else 1.0
        score = pm.n_inliers * (1.15 - hom_ratio)
        if score > best_score:
            best, best_score = pm, score
    return best


class IncrementalSfM:
    def __init__(self, features, pairs, tracks, k, min_angle_deg=1.5, max_reproj_px=4.0) -> None:
        self.features = features
        self.pairs = {(p.i, p.j): p for p in pairs}
        self.tracks = tracks
        self.k = k.copy()
        self.min_angle = min_angle_deg
        self.max_reproj = max_reproj_px
        self.cameras: dict[int, Camera] = {}
        self.points: list[np.ndarray] = []
        self.obs: list[dict[int, int]] = []
        self.track_to_point: dict[int, int] = {}
        self.ba_history: list[dict] = []
        self._track_lookup: dict[tuple[int, int], int] = {}
        for ti, tr in enumerate(tracks):
            for img, kp in tr.items():
                self._track_lookup[(img, kp)] = ti

    def initialize(self) -> PairMatch | None:
        pm = _pick_initial_pair(self.features, list(self.pairs.values()))
        if pm is None:
            return None
        fa, fb = self.features[pm.i], self.features[pm.j]
        a = fa.keypoints[pm.inliers[:, 0]].astype(np.float64)
        b = fb.keypoints[pm.inliers[:, 1]].astype(np.float64)
        e, mask = cv2.findEssentialMat(a, b, self.k, cv2.RANSAC, 0.999, 1.0)
        if e is None:
            return None
        if e.shape[0] > 3:
            e = e[:3]
        _, r, t, pose_mask = cv2.recoverPose(e, a, b, self.k, mask=mask)
        self.cameras[pm.i] = Camera(pm.i, np.zeros(3), np.zeros(3))
        self.cameras[pm.j] = Camera(pm.j, cv2.Rodrigues(r)[0].ravel(), t.ravel())
        self._add_points_from_pair(pm, pose_mask.ravel().astype(bool))
        return pm

    def _add_points_from_pair(self, pm: PairMatch, keep: np.ndarray | None = None) -> int:
        fa, fb = self.features[pm.i], self.features[pm.j]
        ca, cb = self.cameras[pm.i], self.cameras[pm.j]
        inl = pm.inliers if keep is None else pm.inliers[keep]
        if len(inl) == 0:
            return 0
        a = fa.keypoints[inl[:, 0]]
        b = fb.keypoints[inl[:, 1]]
        xyz = _triangulate(self.k, ca, cb, a, b)
        da, db = _depth(ca, xyz), _depth(cb, xyz)
        ang = _angle_deg(ca.center, cb.center, xyz)
        ea = np.linalg.norm(_project(self.k, ca, xyz) - a, axis=1)
        eb = np.linalg.norm(_project(self.k, cb, xyz) - b, axis=1)
        good = (da > 0) & (db > 0) & (ang > self.min_angle) & (ea < self.max_reproj) & (eb < self.max_reproj)
        added = 0
        for idx in np.where(good)[0]:
            ka, kb = int(inl[idx, 0]), int(inl[idx, 1])
            ti = self._track_lookup.get((pm.i, ka))
            if ti is None or ti in self.track_to_point:
                continue
            self.track_to_point[ti] = len(self.points)
            self.points.append(xyz[idx])
            self.obs.append({pm.i: ka, pm.j: kb})
            added += 1
        return added

    def _valid(self, idx: int) -> bool:
        c = self.cameras.get(idx)
        return c is not None and np.isfinite(c.rvec).all()

    def _next_image(self):
        best = None
        for img_idx, feat in enumerate(self.features):
            if img_idx in self.cameras:
                continue
            pts3, pts2 = [], []
            for kp_idx in range(len(feat.keypoints)):
                ti = self._track_lookup.get((img_idx, kp_idx))
                if ti is None:
                    continue
                pi = self.track_to_point.get(ti)
                if pi is None:
                    continue
                pts3.append(self.points[pi])
                pts2.append(feat.keypoints[kp_idx])
            if len(pts3) >= 12 and (best is None or len(pts3) > len(best[1])):
                best = (img_idx, np.array(pts3, dtype=np.float64), np.array(pts2, dtype=np.float64))
        return best

    def register_next(self) -> int | None:
        nxt = self._next_image()
        if nxt is None:
            return None
        img_idx, pts3, pts2 = nxt
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3, pts2, self.k, None, iterationsCount=2000, reprojectionError=self.max_reproj, confidence=0.999, flags=cv2.SOLVEPNP_EPNP
        )
        if not ok or inliers is None or len(inliers) < 10:
            self.cameras[img_idx] = Camera(img_idx, np.full(3, np.nan), np.full(3, np.nan))
            return img_idx
        inl = inliers.ravel()
        ok2, rvec, tvec = cv2.solvePnP(pts3[inl], pts2[inl], self.k, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        cam = Camera(img_idx, rvec.ravel(), tvec.ravel())
        self.cameras[img_idx] = cam
        feat = self.features[img_idx]
        for kp_idx in range(len(feat.keypoints)):
            ti = self._track_lookup.get((img_idx, kp_idx))
            if ti is None:
                continue
            pi = self.track_to_point.get(ti)
            if pi is None:
                continue
            err = np.linalg.norm(_project(self.k, cam, self.points[pi][None]) - feat.keypoints[kp_idx])
            if err < self.max_reproj:
                self.obs[pi][img_idx] = kp_idx
        for (i, j), pm in self.pairs.items():
            if img_idx not in (i, j):
                continue
            other = j if i == img_idx else i
            if self._valid(other):
                self._add_points_from_pair(pm)
        return img_idx

    def bundle_adjust(self, max_nfev: int = 40, refine_focal: bool = True) -> dict:
        cam_ids = [i for i in sorted(self.cameras) if self._valid(i)]
        cam_pos = {c: n for n, c in enumerate(cam_ids)}
        pt_ids = [p for p in range(len(self.points)) if len(self.obs[p]) >= 2]
        pt_pos = {p: n for n, p in enumerate(pt_ids)}
        cam_i, pt_i, uv = [], [], []
        for p in pt_ids:
            for img, kp in self.obs[p].items():
                if img in cam_pos:
                    cam_i.append(cam_pos[img])
                    pt_i.append(pt_pos[p])
                    uv.append(self.features[img].keypoints[kp])
        if len(uv) < 20:
            return {"status": "skipped", "observations": len(uv)}
        cam_i = np.array(cam_i)
        pt_i = np.array(pt_i)
        uv = np.array(uv, dtype=np.float64)
        n_c, n_p = len(cam_ids), len(pt_ids)
        x0 = np.concatenate(
            [
                [self.k[0, 0]],
                np.concatenate([np.concatenate([self.cameras[c].rvec, self.cameras[c].tvec]) for c in cam_ids]),
                np.concatenate([self.points[p] for p in pt_ids]),
            ]
        )
        cx, cy = self.k[0, 2], self.k[1, 2]

        def residuals(x: np.ndarray) -> np.ndarray:
            f = x[0]
            cams = x[1 : 1 + 6 * n_c].reshape(n_c, 6)
            pts = x[1 + 6 * n_c :].reshape(n_p, 3)
            rv = cams[cam_i, :3]
            tv = cams[cam_i, 3:]
            p = pts[pt_i]
            theta = np.linalg.norm(rv, axis=1, keepdims=True)
            axis = np.where(theta > 1e-12, rv / np.maximum(theta, 1e-12), 0.0)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dot = (p * axis).sum(1, keepdims=True)
            rot = p * cos_t + np.cross(axis, p) * sin_t + axis * dot * (1 - cos_t)
            cam_pt = rot + tv
            z = np.clip(cam_pt[:, 2:3], 1e-6, None)
            proj = cam_pt[:, :2] / z * f + np.array([cx, cy])
            return (proj - uv).ravel()

        m = len(uv) * 2
        sparsity = lil_matrix((m, len(x0)), dtype=int)
        rows = np.arange(len(uv))
        for s in range(2):
            sparsity[2 * rows + s, 0] = 1
            for d in range(6):
                sparsity[2 * rows + s, 1 + cam_i * 6 + d] = 1
            for d in range(3):
                sparsity[2 * rows + s, 1 + 6 * n_c + pt_i * 3 + d] = 1
        if not refine_focal:
            sparsity[:, 0] = 0
        before = float(np.sqrt(np.mean(residuals(x0) ** 2)))
        res = least_squares(residuals, x0, jac_sparsity=sparsity, method="trf", loss="huber", f_scale=2.0, max_nfev=max_nfev, x_scale="jac", ftol=1e-5, xtol=1e-6)
        x = res.x
        if refine_focal and 0.4 * self.k[0, 0] < x[0] < 2.5 * self.k[0, 0]:
            self.k[0, 0] = self.k[1, 1] = x[0]
        cams = x[1 : 1 + 6 * n_c].reshape(n_c, 6)
        for c, n in cam_pos.items():
            self.cameras[c].rvec = cams[n, :3].copy()
            self.cameras[c].tvec = cams[n, 3:].copy()
        pts = x[1 + 6 * n_c :].reshape(n_p, 3)
        for p, n in pt_pos.items():
            self.points[p] = pts[n].copy()
        after = float(np.sqrt(np.mean(residuals(x) ** 2)))
        info = {
            "status": "ok",
            "cameras": n_c,
            "points": n_p,
            "observations": int(len(uv)),
            "rmse_before_px": round(before, 3),
            "rmse_after_px": round(after, 3),
            "focal_px": round(float(self.k[0, 0]), 1),
            "iterations": int(res.nfev),
        }
        self.ba_history.append(info)
        return info

    def prune(self) -> int:
        removed = 0
        for p in range(len(self.points)):
            if not self.obs[p]:
                continue
            x = self.points[p][None]
            for img in list(self.obs[p]):
                cam = self.cameras.get(img)
                if cam is None or not self._valid(img):
                    del self.obs[p][img]
                    continue
                kp = self.features[img].keypoints[self.obs[p][img]]
                err = np.linalg.norm(_project(self.k, cam, x) - kp)
                if err > self.max_reproj or _depth(cam, x)[0] <= 0:
                    del self.obs[p][img]
                    removed += 1
            if len(self.obs[p]) < 2:
                self.obs[p] = {}
        return removed

    def result(self) -> SfMResult:
        keep = [p for p in range(len(self.points)) if len(self.obs[p]) >= 2]
        pts = np.array([self.points[p] for p in keep], dtype=np.float64).reshape(-1, 3)
        cols, errs, angs, obs = [], [], [], []
        for p in keep:
            o = self.obs[p]
            imgs = list(o)
            cols.append(np.mean([self.features[i].colors[o[i]] for i in imgs], axis=0))
            errs.append(float(np.mean([
                np.linalg.norm(_project(self.k, self.cameras[i], self.points[p][None]) - self.features[i].keypoints[o[i]])
                for i in imgs
            ])))
            centers = np.array([self.cameras[i].center for i in imgs])
            best = 0.0
            for a in range(len(centers)):
                for b in range(a + 1, len(centers)):
                    best = max(best, float(_angle_deg(centers[a], centers[b], self.points[p][None])[0]))
            angs.append(best)
            obs.append(dict(o))
        return SfMResult(
            k=self.k.copy(),
            cameras={i: c for i, c in self.cameras.items() if self._valid(i)},
            points=pts,
            colors=np.array(cols, dtype=np.uint8).reshape(-1, 3),
            observations=obs,
            errors=np.array(errs, dtype=np.float64),
            triangulation_angles=np.array(angs, dtype=np.float64),
            ba_history=list(self.ba_history),
        )


def align_to_gps(result: SfMResult, gps_centers: dict[int, np.ndarray]) -> SfMResult:
    ids = [i for i in result.cameras if i in gps_centers]
    if len(ids) < 2:
        return result
    sim = umeyama(np.array([result.cameras[i].center for i in ids]), np.array([gps_centers[i] for i in ids]))
    if sim is None:
        return result
    s, r, t = sim
    if len(result.points):
        result.points = (s * (r @ result.points.T)).T + t
    for cam in result.cameras.values():
        c_new = s * (r @ cam.center) + t
        r_w2c = cam.rotation @ r.T
        cam.rvec = cv2.Rodrigues(r_w2c)[0].ravel()
        cam.tvec = -r_w2c @ c_new
    aligned = np.array([result.cameras[i].center for i in ids])
    dst = np.array([gps_centers[i] for i in ids])
    result.gps_scale = float(s)
    result.gps_rmse_m = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return result


def run_sfm(features, pairs, tracks, k, progress=None):
    sfm = IncrementalSfM(features, pairs, tracks, k)
    if sfm.initialize() is None:
        return None
    sfm.bundle_adjust(max_nfev=30, refine_focal=False)
    registered, since = 2, 0
    while True:
        idx = sfm.register_next()
        if idx is None:
            break
        if sfm._valid(idx):
            registered += 1
            since += 1
        if since >= 3:
            sfm.prune()
            sfm.bundle_adjust(max_nfev=25)
            since = 0
        if progress:
            progress(registered, len(features), len(sfm.points))
    sfm.prune()
    sfm.bundle_adjust(max_nfev=60)
    sfm.prune()
    return sfm
