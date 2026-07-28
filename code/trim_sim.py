#!/usr/bin/env python3
"""E1 (DA-001 Q5): trim-shift simulation. Re-score skel_noaug fold checkpoints on val
clips truncated to the TEST duration distribution (quantile-mapped via n_ir frame counts).
Delta = full-OOF minus trimmed-OOF ≈ the part of the CV→LB offset explained by trimming.
CPU-only (skeleton model is tiny).
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import har_data
import har_models

ROOT = "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
CKPT = os.path.join(ROOT, "checkpoints")

# target frame-count distribution from test meta (n_ir ~ 10fps frames)
test_frames = []
with open(os.path.join(ROOT, "cache", "meta_test.csv")) as f:
    for r in csv.DictReader(f):
        n = int(r["n_ir"] or 0)
        if n > 0:
            test_frames.append(n)
test_frames = np.sort(np.array(test_frames))


class TrimmedSkel(har_data.HARDataset):
    """Truncate each clip's skeleton stream to a quantile-mapped test length before sampling."""

    def set_quantiles(self):
        lens = []
        for sid in self.ids:
            with np.load(os.path.join(har_data.CACHE, self.split, sid + ".npz")) as z:
                lens.append(len(z["skel_t"]) if "skel_t" in z.files else 0)
        lens = np.array(lens)
        ranks = lens.argsort().argsort() / max(1, len(lens) - 1)
        tgt = test_frames[np.clip((ranks * (len(test_frames) - 1)).astype(int), 0, len(test_frames) - 1)]
        self.trim_to = {sid: int(min(l, t)) for sid, l, t in zip(self.ids, lens, tgt)}

    def _skel(self, z, rng):
        if "skel_pos" not in z.files:
            return torch.zeros(self.t_skel, self.skel_dim())
        pos = z["skel_pos"][:, 0]
        k = self.trim_to.get(self._sid, len(pos))
        pos = pos[:max(2, k)]
        idx = har_data.uniform_idx(len(pos), self.t_skel)
        x = pos[idx].astype(np.float32)
        v = np.diff(x, axis=0, prepend=x[:1])
        return torch.from_numpy(np.concatenate([x, v], -1).reshape(self.t_skel, -1))

    def __getitem__(self, i):
        self._sid = self.ids[i]
        return super().__getitem__(i)


def evaluate(trimmed):
    hits = n = 0
    torch.set_num_threads(8)
    for fold in range(4):
        ds = (TrimmedSkel if trimmed else har_data.HARDataset)("train", "skel", fold, "val")
        if trimmed:
            ds.set_quantiles()
        model = har_models.build("skel")
        model.load_state_dict(torch.load(os.path.join(CKPT, f"skel_noaug_f{fold}.pt"), map_location="cpu"))
        model.eval()
        with torch.no_grad():
            for x, y, _ in DataLoader(ds, 128, num_workers=4):
                hits += (model(x).argmax(1) == y).sum().item()
                n += len(y)
    return hits / n


full = evaluate(False)
trim = evaluate(True)
print(f"skel_noaug OOF full: {full:.4f}  trimmed-to-test-dist: {trim:.4f}  Δ_trim = {full - trim:+.4f}")
print(f"(test frame counts: med {np.median(test_frames):.0f}, p90 {np.percentile(test_frames, 90):.0f})")
