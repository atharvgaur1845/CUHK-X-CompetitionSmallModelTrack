#!/usr/bin/env python3
"""M-01: aligned multimodal dataset for the fusion transformer (RESET-001 design).

One item = one clip, all streams on the shared 10 Hz timeline, pad+mask (NO stretch):
  skel  (T,102|153) f32 + mask (T,)    droi (T,112,112) u8->f32 + mask (T,)
  imu   (T,85) f32 resampled to the same 10 Hz grid + mask (T,)
  meta: log-duration scalar, per-stream presence flags
T = fixed budget (default 48 @ 10 Hz = 4.8 s); longer clips take a random (train) or
center (eval) contiguous window — duration signal is carried by the scalar + masks.
Aug: trunc (random truncation), skel jitter (mild), visual crop/erase (shared params).
"""
import os
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

import har_data as hd

T_GRID = 48


class MMDataset(Dataset):
    def __init__(self, split, fold=None, part="train", t=T_GRID, feat="jvb",
                 aug=False, seed=0, roi="droi"):
        base = hd.HARDataset(split, "skel", fold, part, feat=feat)
        self.ids, self.labels, self.meta = base.ids, base.labels, base.meta
        self.skel_feat = feat
        self.base = base
        self.split, self.t, self.aug, self.seed, self.roi = split, t, aug, seed, roi
        self.roi_dir = os.path.join(hd.CACHE, split + ("_roi" if roi == "droi" else "_roi"))

    def __len__(self):
        return len(self.ids)

    def _window(self, t0_all, t1_all, rng):
        """pick the clip's grid start time and length in steps"""
        dur = max(0.1, t1_all - t0_all)
        n = min(self.t, max(2, int(dur * 10) + 1))
        if self.aug and rng is not None and rng.random() < 0.5 and n > 6:
            n = max(4, int(n * (0.4 + 0.6 * rng.random())))  # trunc-aug
        span = dur - (n - 1) / 10.0
        off = (rng.random() * max(0, span)) if (self.aug and rng is not None) else max(0, span) / 2
        return t0_all + off, n, dur

    def __getitem__(self, i):
        sid = self.ids[i]
        rng = np.random.default_rng((zlib.crc32(sid.encode()) + self.seed * 1000003) % (1 << 31)) if self.aug else None
        out = {}
        with np.load(os.path.join(hd.CACHE, self.split, sid + ".npz")) as z:
            starts = [z[k][0] for k in ("skel_t", "depth_t", "ir_t") if k in z.files]
            ends = [z[k][-1] for k in ("skel_t", "depth_t", "ir_t") if k in z.files]
            for d in hd.DEVICES:
                k = f"imu_{d}_t"
                if k in z.files and len(z[k]):
                    starts.append(z[k][0]); ends.append(z[k][-1])
            if not starts:
                starts, ends = [0.0], [0.1]
            g0, n, dur = self._window(min(starts), max(ends), rng)
            grid = g0 + np.arange(self.t) / 10.0
            valid_n = np.zeros(self.t, bool); valid_n[:n] = True
            # skeleton: nearest-frame match on the grid
            skel = np.zeros((self.t, self.base.skel_dim()), np.float32)
            skel_m = np.zeros(self.t, np.float32)
            if "skel_pos" in z.files and len(z["skel_t"]):
                st, sp = z["skel_t"], z["skel_pos"][:, 0]
                idx = np.clip(np.searchsorted(st, grid), 0, len(st) - 1)
                near = np.abs(st[idx] - grid) < 0.15
                x = sp[idx].astype(np.float32)
                if rng is not None:  # mild jitter only (EXP-007: strong geometric aug hurts)
                    x = x + rng.normal(0, 0.008, x.shape).astype(np.float32)
                v = np.diff(x, axis=0, prepend=x[:1]) * near[:, None, None]
                parts = [x, v]
                if self.skel_feat.startswith("jvb"):
                    parts.append(x - x[:, list(hd.HARDataset.H36M_PARENT)])
                skel = np.concatenate(parts, -1).reshape(self.t, -1) * (near & valid_n)[:, None]
                skel_m = (near & valid_n).astype(np.float32)
            out["skel"], out["skel_m"] = torch.from_numpy(skel), torch.from_numpy(skel_m)
            # imu: interp per channel onto grid
            imu = np.zeros((self.t, 85), np.float32)
            imu_m = np.zeros(self.t, np.float32)
            any_dev = False
            for di, d in enumerate(hd.DEVICES):
                tk, xk = f"imu_{d}_t", f"imu_{d}_x"
                if tk in z.files and len(z[tk]) >= 2:
                    tt, xx = z[tk], z[xk].copy()
                    xx[:, 3:6] /= 500.0; xx[:, 6:9] /= 180.0; xx[:, 9:12] /= 100.0
                    cover = (grid >= tt[0] - 0.2) & (grid <= tt[-1] + 0.2) & valid_n
                    for c in range(16):
                        imu[:, di * 17 + c] = np.interp(grid, tt, xx[:, c]) * cover
                    imu[:, di * 17 + 16] = cover
                    any_dev = True
            if any_dev:
                imu_m = valid_n.astype(np.float32)
            out["imu"], out["imu_m"] = torch.from_numpy(imu), torch.from_numpy(imu_m)
        # visual ROI stream (separate npz)
        vis = np.zeros((self.t, 112, 112), np.float32)
        vis_m = np.zeros(self.t, np.float32)
        rp = os.path.join(self.roi_dir, sid + ".npz")
        key = "depth_roi" if self.roi == "droi" else "ir_roi"
        if os.path.exists(rp):
            with np.load(rp) as zr:
                if key in zr.files and len(zr[key]):
                    vt = zr[key + "_t"] if key + "_t" in zr.files else None
                    fr = zr[key]
                    if vt is not None and len(vt) == len(fr):
                        idx = np.clip(np.searchsorted(vt, grid), 0, len(fr) - 1)
                        near = np.abs(vt[idx] - grid) < 0.15
                    else:
                        idx = hd.uniform_idx(len(fr), self.t)
                        near = np.ones(self.t, bool)
                    x = fr[idx].astype(np.float32) / 255.0
                    if self.aug and rng is not None:
                        if rng.random() < 0.5:
                            import cv2
                            sc = 0.8 + rng.random() * 0.2
                            h = int(112 * sc)
                            y0, x0 = rng.integers(0, 113 - h), rng.integers(0, 113 - h)
                            x = np.stack([cv2.resize(f[y0:y0 + h, x0:x0 + h], (112, 112)) for f in x])
                        if rng.random() < 0.3:
                            y0, x0 = rng.integers(0, 84), rng.integers(0, 84)
                            x[:, y0:y0 + 28, x0:x0 + 28] = 0
                    m = near & valid_n
                    vis = x * m[:, None, None]
                    vis_m = m.astype(np.float32)
        out["vis"], out["vis_m"] = torch.from_numpy(vis), torch.from_numpy(vis_m)
        out["dur"] = torch.tensor([np.log1p(dur)], dtype=torch.float32)
        y = self.labels[sid] if self.labels else -1
        return out, y, sid
