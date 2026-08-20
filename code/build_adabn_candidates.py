#!/usr/bin/env python3
"""Rebuild the champion fusion from AdaBN member probabilities (EXP-099).

Emits three candidates so each carries exactly one change against sub_h8all (162):
  n1  = h8all recipe, AdaBN members            <- single change: AdaBN
  n2  = h8all recipe, AdaBN members, sw=0.5    <- AdaBN + start-weight (bundled)
Falls back to the non-AdaBN file for any member whose AdaBN run has not landed,
and says so, so a partial run can never masquerade as a complete one.
"""
import numpy as np, collections, csv, os, sys
ART = "research/artifacts"; EPS = 1e-12
L = lambda a: np.log(np.maximum(a, EPS))
tr = collections.Counter(int(r["class_id"]) for r in csv.DictReader(open("cache/meta_train.csv")))
n = sum(tr.values()); prior = np.array([tr[k] / n for k in range(40)])

def load(tag, adabn=False):
    path = f"{ART}/testprobs_{tag}{'_adabn' if adabn else ''}.npz"
    if adabn and not os.path.exists(path):
        print(f"  !! {tag}: no AdaBN file, falling back to as-trained")
        path = f"{ART}/testprobs_{tag}.npz"
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["probs"], np.float64), [str(s) for s in d["sids"]]

VID = ["vid_r2p1d_f0", "vid_r2p1d_f1", "vid_r2p1d_f2", "vid_r2p1d_f3",
       "vid_ig65m_f0", "vid_ig65m_f1", "vid_ig65m_f2", "vid_ig65m_f3",
       "vid_f32_f2", "vid_res160_f2", "vid_upper_f2"]
missing = [t for t in VID if not os.path.exists(f"{ART}/testprobs_{t}_adabn.npz")]
if missing:
    print(f"WARNING: {len(missing)}/{len(VID)} members lack AdaBN probs: {missing}")

B, sb = load("astgcn_world25_int8"); I, _ = load("imu_stats"); MB, _ = load("skel_mb_f2")
vids = [load(t, adabn=True)[0] for t in VID]
lv = np.mean([L(v) for v in vids], 0)
lg = 0.35 * L(B / prior) + 0.2925 * lv + 0.3575 * L(I)
lg = 0.9 * lg + 0.10 * L(MB)
lg = lg + 0.25 * L(prior)[None, :]
lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
out = f"{ART}/testprobs_n1.npz"
np.savez_compressed(out, probs=p.astype(np.float32), sids=np.array(sb, dtype="<U15"))
print(f"wrote {out}  ({len(VID)} video members, {len(VID)-len(missing)} of them AdaBN)")
