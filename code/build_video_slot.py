#!/usr/bin/env python3
"""Rebuild the champion fusion with an arbitrary video slot (EXP-105).

The video slot is a weighted log-mean over *views*; each view is itself an equal
log-mean over its member folds. This separates the two questions that n7 confounded:
how much weight a view deserves, and how many folds back it.

  --view mvit=k224_mvit_f0,k224_mvit_f1,k224_mvit_f2,k224_mvit_f3:0.5 \
  --view wrist=k224_mvitwrist_f2:0.5

Everything outside the video slot is the verified 166 champion recipe and is not a
parameter here — one change per submission.
"""
import argparse, collections, csv, os
import numpy as np

ART = "research/artifacts"; EPS = 1e-12
L = lambda a: np.log(np.maximum(a, EPS))

ap = argparse.ArgumentParser()
ap.add_argument("--view", action="append", required=True,
                help="name=tag1,tag2,...[:weight]  (weights renormalised to 1)")
ap.add_argument("--tag", required=True, help="output tag, writes testprobs_<tag>.npz")
ap.add_argument("--skel", default="astgcn_world25_int8", help="skeleton-branch tag (for packaging prunes)")
ap.add_argument("--imu", default="imu_stats", help="IMU tree-member tag (for packaging prunes)")
ap.add_argument("--no-motionbert", action="store_true",
                help="drop the 0.10 MotionBERT term — 241 MB fp32 for 9 of 405 rows")
a = ap.parse_args()

tr = collections.Counter(int(r["class_id"]) for r in csv.DictReader(open("cache/meta_train.csv")))
n = sum(tr.values()); prior = np.array([tr[k] / n for k in range(40)])

def load(tag):
    d = np.load(f"{ART}/testprobs_{tag}.npz", allow_pickle=True)
    return np.asarray(d["probs"], np.float64), [str(s) for s in d["sids"]]

ref_sids = None
views = []
for spec in a.view:
    name, rest = spec.split("=", 1)
    tags, _, w = rest.partition(":")
    w = float(w) if w else 1.0
    tags = tags.split(",")
    logs = []
    for t in tags:
        p, s = load(t)
        if ref_sids is None: ref_sids = s
        assert s == ref_sids, f"{t}: sid order differs from reference"
        logs.append(L(p))
    views.append((name, w, np.mean(logs, 0), tags))

tot = sum(w for _, w, _, _ in views)
lv = sum((w / tot) * lg for _, w, lg, _ in views)
for name, w, _, tags in views:
    print(f"  view {name:8s} weight {w/tot:.3f}  members {len(tags)}: {','.join(tags)}")

B, sb = load(a.skel); I, _ = load(a.imu)
assert sb == ref_sids
lg = 0.35 * L(B / prior) + 0.2925 * lv + 0.3575 * L(I)
if not a.no_motionbert:
    MB, _ = load("skel_mb_f2")
    lg = 0.9 * lg + 0.10 * L(MB)
lg = lg + 0.25 * L(prior)[None, :]
lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
out = f"{ART}/testprobs_{a.tag}.npz"
np.savez_compressed(out, probs=p.astype(np.float32), sids=np.array(ref_sids, dtype="<U15"))
print(f"wrote {out}")
