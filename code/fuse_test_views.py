#!/usr/bin/env python3
"""EXP-140 test-side fusion with an explicit, named member set.

Rebuilds the champion fusion from its members as a CONTROL (it must reproduce
sub_r2_dist.csv at rowdiff 0 before any variant it produces is trusted), then emits
arbitrary view combinations so the person view can be dropped in favour of thermal.
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
L = lambda a: np.log(np.maximum(a, 1e-12))


def bag(tags):
    """Geometric mean over fold members, asserting identical sid order."""
    ref, acc = None, None
    for t in tags:
        d = np.load(os.path.join(ART, f"testprobs_{t}.npz"), allow_pickle=True)
        s = [str(x) for x in d["sids"]]
        if ref is None:
            ref = s
        assert s == ref, f"sid order differs for {t}"
        lg = L(np.asarray(d["probs"], np.float64))
        acc = lg if acc is None else acc + lg
    return ref, acc / len(tags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--person", type=float, default=0.0)
    ap.add_argument("--wrist", type=float, default=0.0)
    ap.add_argument("--thermal", type=float, default=0.0)
    ap.add_argument("--skel", type=float, default=0.35)
    ap.add_argument("--imu", type=float, default=0.3575)
    ap.add_argument("--prior", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    counts = np.zeros(40)
    for r in csv.DictReader(open(os.path.join(ROOT, "cache", "meta_train.csv"))):
        counts[int(r["class_id"])] += 1
    prior = counts / counts.sum()

    SRC = {
        "person":  (["k224_mvit_all"], "uniform"),
        "wrist":   (["k224_mvitwrist_all"], "uniform"),
        "thermal": (["th224_bag"], "uniform"),
        "skel":    (["astgcn_world25"], "train"),
        "imu":     (["imu_stats_t200_d12"], "uniform"),
    }
    sids, z = None, None
    for name, (tags, space) in SRC.items():
        w = getattr(args, name)
        if not w:
            continue
        s, lg = bag(tags)
        if sids is None:
            sids, z = s, np.zeros_like(lg)
        assert s == sids, f"sid order differs for {name}"
        if space == "train":
            lg = lg - L(prior)[None, :]
        z = z + w * lg
        print(f"  + {name:8s} w={w:.4f}  ({len(tags)} member(s))")
    z = z + args.prior * L(prior)[None, :]
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert (p > 0).all(), "zero cell in fused posterior"
    out = os.path.join(ART, f"testprobs_{args.out}.npz")
    np.savez_compressed(out, probs=p.astype(np.float32),
                        sids=np.array(sids, dtype="<U15"))
    print(f"wrote {out}  rows={len(sids)}  distinct argmax={len(set(p.argmax(1).tolist()))}/40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
