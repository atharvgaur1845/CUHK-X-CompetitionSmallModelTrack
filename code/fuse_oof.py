#!/usr/bin/env python3
"""Fuse OOF prob files (research/artifacts/oof_<tag>.npz), grid-search weights,
report CV accuracy with/without recording-group Hungarian.
Usage: python3 fuse_oof.py skel_noaug imu_aug [depth_aug ...]
"""
import itertools
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hungarian_sim import radar_groups

# Portable root: env CUHKX_ROOT wins, else the repo dir two levels up from this
# file. Was a hardcoded absolute path (with a space in it) in 11 files, which was
# the #1 blocker for running anywhere but the original laptop.
ROOT = os.environ.get("CUHKX_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "research", "artifacts")


def hungarian_acc(P, Y, sids, gkey):
    pred = P.argmax(1).copy()
    groups = defaultdict(list)
    for i, s in enumerate(sids):
        groups[gkey.get(s, f"solo_{s}")].append(i)
    for g, idx in groups.items():
        if len(idx) < 2:
            continue
        Pn = P[idx] / P[idx].sum(1, keepdims=True)
        r, c = linear_sum_assignment(-np.log(Pn + 1e-9))
        for k, cls in zip(r, c):
            pred[idx[k]] = cls
    return (pred == Y).mean()


def main(tags):
    data = {}
    ref_sids = None
    for t in tags:
        z = np.load(os.path.join(ART, f"oof_{t}.npz"), allow_pickle=True)
        sids = list(z["sids"])
        order = {s: i for i, s in enumerate(sids)}
        if ref_sids is None:
            ref_sids = sids
        idx = [order[s] for s in ref_sids]
        data[t] = (z["probs"][idx], z["labels"][idx])
    Y = data[tags[0]][1]
    gkey = radar_groups()
    for t in tags:
        P = data[t][0]
        print(f"{t}: argmax {(P.argmax(1) == Y).mean():.4f}")
    best = (0, None)
    grid = np.arange(0, 1.05, 0.1)
    for ws in itertools.product(grid, repeat=len(tags)):
        if abs(sum(ws) - 1) > 1e-6 or ws[0] == 0:
            continue
        P = sum(w * data[t][0] for w, t in zip(ws, tags))
        a = (P.argmax(1) == Y).mean()
        if a > best[0]:
            best = (a, ws)
    a, ws = best
    P = sum(w * data[t][0] for w, t in zip(ws, tags))
    print(f"best fusion {dict(zip(tags, ws))}: argmax {a:.4f}, hungarian {hungarian_acc(P, Y, ref_sids, gkey):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
