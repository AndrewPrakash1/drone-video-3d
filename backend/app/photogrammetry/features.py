"""SIFT features, KNN ratio matching, and RANSAC outlier removal."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ImageFeatures:
    index: int
    name: str
    width: int
    height: int
    keypoints: np.ndarray
    descriptors: np.ndarray
    colors: np.ndarray


@dataclass
class PairMatch:
    i: int
    j: int
    knn_matches: int
    ratio_matches: int
    inliers: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    fundamental: np.ndarray | None = None

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.shape[0])


def extract_features(img_bgr: np.ndarray, index: int, name: str, max_features: int = 4000) -> ImageFeatures:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    kps, des = sift.detectAndCompute(gray, None)
    if des is None or len(kps) == 0:
        return ImageFeatures(index, name, img_bgr.shape[1], img_bgr.shape[0], np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), np.zeros((0, 3), np.uint8))
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    h, w = gray.shape
    xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
    colors = img_bgr[ys, xs][:, ::-1].astype(np.uint8)
    return ImageFeatures(index, name, w, h, pts, des.astype(np.float32), colors)


def knn_ratio_match(des_a: np.ndarray, des_b: np.ndarray, ratio: float = 0.8) -> tuple[np.ndarray, int]:
    """k=2 FLANN match, Lowe ratio test, then cross-check."""
    if len(des_a) < 8 or len(des_b) < 8:
        return np.zeros((0, 2), dtype=np.int32), 0
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    knn_ab = flann.knnMatch(des_a, des_b, k=2)
    fwd: dict[int, int] = {}
    total = 0
    for pair in knn_ab:
        if len(pair) < 2:
            continue
        total += 1
        m, n = pair
        if m.distance < ratio * n.distance:
            fwd[m.queryIdx] = m.trainIdx
    kept: list[tuple[int, int]] = []
    for pair in flann.knnMatch(des_b, des_a, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and fwd.get(m.trainIdx) == m.queryIdx:
            kept.append((m.trainIdx, m.queryIdx))
    return np.array(kept, dtype=np.int32).reshape(-1, 2), total


def ransac_filter(pts_a: np.ndarray, pts_b: np.ndarray, matches: np.ndarray, threshold_px: float = 1.5) -> tuple[np.ndarray, np.ndarray | None]:
    if matches.shape[0] < 8:
        return np.zeros((0, 2), dtype=np.int32), None
    a = pts_a[matches[:, 0]].astype(np.float64)
    b = pts_b[matches[:, 1]].astype(np.float64)
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    f, mask = cv2.findFundamentalMat(a, b, method, threshold_px, 0.999)
    if f is None or mask is None:
        return np.zeros((0, 2), dtype=np.int32), None
    if f.shape[0] > 3:
        f = f[:3]
    return matches[mask.ravel().astype(bool)], f


def candidate_pairs(n: int, camera_centers: np.ndarray | None, window: int = 4, max_neighbors: int = 6) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, min(n, i + window + 1)):
            pairs.add((i, j))
    if camera_centers is not None and len(camera_centers) == n and n > 2:
        d = np.linalg.norm(camera_centers[:, None, :] - camera_centers[None, :, :], axis=2)
        for i in range(n):
            for j in np.argsort(d[i])[1 : max_neighbors + 1]:
                a, b = (i, int(j)) if i < j else (int(j), i)
                if a != b:
                    pairs.add((a, b))
    return sorted(pairs)


def match_pair(fa: ImageFeatures, fb: ImageFeatures) -> PairMatch:
    raw, total = knn_ratio_match(fa.descriptors, fb.descriptors)
    inl, f = ransac_filter(fa.keypoints, fb.keypoints, raw)
    return PairMatch(fa.index, fb.index, total, int(raw.shape[0]), inl, f)


def build_tracks(pairs: list[PairMatch]) -> list[dict[int, int]]:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pm in pairs:
        for ka, kb in pm.inliers:
            union((pm.i, int(ka)), (pm.j, int(kb)))
    groups: dict[tuple[int, int], dict[int, int]] = {}
    for node in list(parent):
        g = groups.setdefault(find(node), {})
        img, kp = node
        if img in g and g[img] != kp:
            g[-1] = -1
        g[img] = kp
    return [g for g in groups.values() if -1 not in g and len(g) >= 2]
