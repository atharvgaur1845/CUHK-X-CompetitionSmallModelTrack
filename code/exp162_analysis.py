#!/usr/bin/env python3
"""EXP-162 -- analysis-only experiments for the report (no shipped change).

(4) Missing-modality ablation on the 2,700 pooled OOF clips: each member's posterior is
    replaced, for ALL clips, by what the frozen pipeline produces for a clip with that
    modality absent (a zero-input forward pass gives a constant posterior; here we use the
    member's own constant = its mean posterior over clips, the closest label-free proxy
    available without re-running the models on zero tensors) -- and, separately, the member is
    removed from the fusion (weight 0). Reports fused accuracy pre-decoder for each case.
(6) Fusion-weight sensitivity surface: fused OOF accuracy when each coefficient is scaled by
    0.5 .. 1.5 with the others fixed at the shipped values.
Both use the exact fusion of the package (code/exp160_gate.py harness: 2109/2700 baseline).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT / "research" / "artifacts"
W = json.loads((ART / "weights_p384.json").read_text()); W = {"person384": W["person"], "wrist": W["wrist"], "thermal": W["thermal"], "skel": W["skel"], "imu": W["imu"], "prior": 0.25}
EPS, SM = 1e-12, 0.001
L = lambda a: np.log(np.maximum(a, EPS))


def load(tag):
    z = np.load(ART / f"oof_{tag}.npz"); return dict(zip(z["sids"], z["probs"]))


def stack(tags, sids):
    d = {}
    for t in tags: d.update(load(t))
    return np.stack([d[s] for s in sids])


def main() -> int:
    ref = np.load(ART / "oof_p384w_fused.npz"); sids, y = list(ref["sids"]), ref["labels"]
    rows = list(csv.DictReader(open(ROOT / "cache" / "meta_train.csv")))
    cls = np.array([int(r["class_id"]) for r in rows]); prior = np.bincount(cls, minlength=40) / len(cls)
    P = {"person384": stack([f"k224_mvit384_f{f}" for f in range(4)], sids),
         "wrist": stack([f"k224_mvitwrist_f{f}" for f in range(4)], sids),
         "thermal": stack([f"k224_mvit_th_f{f}" for f in range(4)], sids),
         "skel": stack(["astgcn_world25"], sids),
         "imu": (stack(["imu_stats_t200_d12"], sids) + SM) / (1 + 40 * SM)}
    def fuse(P, W):
        z = sum(W[r] * (L(P[r] / prior) if r == "skel" else L(P[r])) for r in P) + W["prior"] * L(prior)[None, :]
        return z.argmax(1)
    base = int((fuse(P, W) == y).sum())
    print(f"baseline fused OOF (pre-decoder): {base}/2700 = {base/2700:.4f}")
    print("\n(4) missing-modality ablation, pre-decoder, 2,700 pooled OOF clips")
    print(f"{'member absent':12s} {'constant posterior':>20s} {'removed (w=0)':>16s}   member alone")
    for r in P:
        Pc = dict(P); Pc[r] = np.repeat(P[r].mean(0, keepdims=True), len(sids), 0)
        Wr = dict(W); Wr[r] = 0.0
        alone = int((P[r].argmax(1) == y).sum())
        c = int((fuse(Pc, W) == y).sum()); d = int((fuse(P, Wr) == y).sum())
        print(f"{r:12s} {c:5d} ({c-base:+4d})        {d:5d} ({d-base:+4d})        {alone:5d}")
    print("\n(6) fusion-weight sensitivity, pre-decoder (each coefficient scaled, others fixed)")
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    print(f"{'coef':10s} " + " ".join(f"{s:>7.2f}" for s in scales))
    for r in W:
        vals = []
        for s in scales:
            Ws = dict(W); Ws[r] = W[r] * s; vals.append(int((fuse(P, Ws) == y).sum()))
        print(f"{r:10s} " + " ".join(f"{v:7d}" for v in vals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
