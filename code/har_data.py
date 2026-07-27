#!/usr/bin/env python3
"""Dataset over the npz cache. One item = one clip, per-modality tensors."""
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
CACHE = os.path.join(ROOT, "cache")
FOLDS = os.path.join(ROOT, "research", "artifacts", "cv_folds.json")
DEVICES = ("WTC", "WTLA", "WTRA", "WTLL", "WTRL")


def load_meta(split):
    rows = {}
    with open(os.path.join(CACHE, f"meta_{split}.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["sample_id"]] = r
    return rows


def fold_users(fold):
    with open(FOLDS) as f:
        p = json.load(f)
    val = set(p["folds"][fold]["val_users"])
    return val


def uniform_idx(n, t, jitter=False, rng=None):
    """t indices spanning [0, n)."""
    if n <= 0:
        return np.zeros(t, np.int64)
    grid = np.linspace(0, n - 1e-6, t + 1)
    if jitter and rng is not None:
        pos = grid[:-1] + rng.random(t) * np.maximum(np.diff(grid), 1e-6)
    else:
        pos = (grid[:-1] + grid[1:]) / 2
    return np.clip(pos.astype(np.int64), 0, n - 1)


class HARDataset(Dataset):
    H36M_PARENT = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)

    def __init__(self, split, modality, fold=None, part="train", t_skel=32, t_imu=64,
                 n_frames=8, aug=False, seed=0, feat="jv"):
        # aug: False/""=none, True=all components, or comma spec from
        # {rot,scale,jit,jdrop,tjit} (skel) / {chan}(imu/visual keep bool behavior)
        if aug is True:
            aug = "rot,scale,jit,jdrop,tjit"
        self.aug_set = set(aug.split(",")) if aug else set()
        self.split, self.mod, self.aug = split, modality, bool(self.aug_set)
        self.feat = feat  # "jv" | "jvb" (+bones); "+tn" suffix = torso-scale norm
        self.t_skel, self.t_imu, self.n_frames = t_skel, t_imu, n_frames
        meta = load_meta(split)
        if split == "train":
            assert fold is not None
            if fold == "rand":  # random (non-subject) split — DG-gap diagnostic only
                import hashlib
                def is_val(s):
                    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 4 == 0
                keep = is_val if part == "val" else (lambda s: not is_val(s))
                self.ids = sorted(s for s in meta if keep(s))
            else:
                val = fold_users(fold)
                keep = (lambda u: u in val) if part == "val" else (lambda u: u not in val)
                self.ids = sorted(s for s, m in meta.items() if keep(m["user"]))
            self.labels = {s: int(meta[s]["class_id"]) for s in self.ids}
        else:
            self.ids = sorted(meta.keys())
            self.labels = None
        self.meta = meta
        self.epoch_seed = seed

    def __len__(self):
        return len(self.ids)

    def skel_dim(self):
        base = 17 * 3 * 2  # joints + velocity
        return base + 17 * 3 if self.feat.startswith("jvb") else base

    def _skel(self, z, rng):
        if "skel_pos" not in z.files:
            return torch.zeros(self.t_skel, self.skel_dim())
        pos = z["skel_pos"][:, 0]  # (T,17,3) person-0 policy (ablate later)
        T = len(pos)
        idx = uniform_idx(T, self.t_skel, jitter="tjit" in self.aug_set, rng=rng)
        x = pos[idx].astype(np.float32)  # (t,17,3)
        if "+tn" in self.feat:  # torso-scale normalization (pelvis->thorax)
            spine = np.linalg.norm(x[:, 8] - x[:, 0], axis=-1)
            s = np.median(spine[spine > 1e-4])
            if s > 1e-4:
                x = x / s * 0.5
        if rng is not None:
            if "rot" in self.aug_set:
                ang = (rng.random() - 0.5) * (np.pi / 3)  # yaw ±30°
                c, s = np.cos(ang), np.sin(ang)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
                x = x @ R.T
            if "scale" in self.aug_set:
                x *= 1.0 + (rng.random() - 0.5) * 0.3  # ±15%
            if "jit" in self.aug_set:
                x += rng.normal(0, 0.01, x.shape).astype(np.float32)
            if "jdrop" in self.aug_set and rng.random() < 0.2:
                x[:, rng.integers(0, 17, 2)] = 0
        v = np.diff(x, axis=0, prepend=x[:1])
        parts = [x, v]
        if self.feat.startswith("jvb"):
            parts.append(x - x[:, list(self.H36M_PARENT)])  # bone vectors
        return torch.from_numpy(np.concatenate(parts, -1).reshape(self.t_skel, -1))

    def _imu(self, z, rng):
        span = [np.inf, -np.inf]
        for d in DEVICES:
            k = f"imu_{d}_t"
            if k in z.files and len(z[k]):
                span[0] = min(span[0], z[k][0])
                span[1] = max(span[1], z[k][-1])
        out = np.zeros((5, 17, self.t_imu), np.float32)
        if span[0] < span[1]:
            grid = np.linspace(span[0], span[1], self.t_imu)
            for i, d in enumerate(DEVICES):
                tk, xk = f"imu_{d}_t", f"imu_{d}_x"
                if tk in z.files and len(z[tk]) >= 2:
                    t, x = z[tk], z[xk]
                    for c in range(16):
                        out[i, c] = np.interp(grid, t, x[:, c])
                    out[i, 16] = 1.0  # presence mask
        # normalize: acc in g (ok), gyro /500, angle /180, mag /100, quat ok
        out[:, 3:6] /= 500.0
        out[:, 6:9] /= 180.0
        out[:, 9:12] /= 100.0
        if self.aug and rng is not None:
            out[:, :16] *= 1.0 + rng.normal(0, 0.1, (5, 1, 1))
            out[:, :16] += rng.normal(0, 0.02, out[:, :16].shape)
            if rng.random() < 0.2:
                out[rng.integers(0, 5)] = 0
        return torch.from_numpy(out.reshape(85, self.t_imu))

    def _frames(self, z, key, rng):
        if key not in z.files:
            return torch.zeros(self.n_frames, 120, 160)
        v = z[key]
        idx = uniform_idx(len(v), self.n_frames, jitter=self.aug, rng=rng)
        x = v[idx].astype(np.float32) / 255.0
        if self.aug and rng is not None:
            if rng.random() < 0.5:  # random resized crop (same for all frames)
                sc = 0.8 + rng.random() * 0.2
                h, w = int(120 * sc), int(160 * sc)
                y0, x0 = rng.integers(0, 121 - h), rng.integers(0, 161 - w)
                import cv2
                x = np.stack([cv2.resize(f[y0:y0 + h, x0:x0 + w], (160, 120)) for f in x])
            if rng.random() < 0.3:  # random erase
                y0, x0 = rng.integers(0, 90), rng.integers(0, 120)
                x[:, y0:y0 + 30, x0:x0 + 40] = 0
        return torch.from_numpy(x)

    def __getitem__(self, i):
        sid = self.ids[i]
        rng = np.random.default_rng((hash(sid) + self.epoch_seed * 1000003) % (1 << 31)) if self.aug else None
        with np.load(os.path.join(CACHE, self.split, sid + ".npz")) as z:
            if self.mod == "skel":
                x = self._skel(z, rng)
            elif self.mod == "imu":
                x = self._imu(z, rng)
            elif self.mod in ("depth", "ir"):
                x = self._frames(z, self.mod, rng)
            else:
                raise ValueError(self.mod)
        y = self.labels[sid] if self.labels else -1
        return x, y, sid
