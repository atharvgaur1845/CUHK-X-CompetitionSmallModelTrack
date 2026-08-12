#!/usr/bin/env python3
"""Reproduce the priorB champion fusion with a swappable visual member.

The champion `sub_priorB_trans05.csv` (0.62686 = 126/201) was built ad hoc and
never committed.  Recovered and verified here at 404/405 argmax parity against
`testprobs_w25vg035_priorB.npz`; the single differing clip is a 1.6e-4 top-2 tie,
i.e. float32 storage precision, not a recipe difference.

    fused = geometric_mean( base / train_prior , visual ; w )

Why the prior division (this is a mis-specification fix, not a tuned knob).  The
visual member trains under balanced softmax, so its logits target a *uniform*
prior; the skeleton base trains under plain CE, so its logits carry the 28x-skewed
*train* prior.  Fusing them geometrically without reconciling the two prior spaces
double-counts the train prior.  Dividing the base by it moves both members into
uniform space before they are combined.  Public: priorB 0.62686 vs priorA
(the opposite direction, visual -> train space) 0.58706, so the direction is
measured, not assumed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
EPS = 1e-12


def load(path: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(path if os.path.isabs(path) else os.path.join(ART, path),
                   allow_pickle=True)
    return np.asarray(data["probs"], dtype=np.float64), [str(s) for s in data["sids"]]


def train_prior() -> np.ndarray:
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as handle:
        counts = collections.Counter(
            int(row["class_id"]) for row in csv.DictReader(handle)
        )
    prior = np.array([counts.get(i, 0) for i in range(40)], dtype=np.float64)
    return prior / prior.sum()


def geometric_fuse(base: np.ndarray, visual: np.ndarray, weight: float) -> np.ndarray:
    logit = (1.0 - weight) * np.log(np.maximum(base, EPS)) + weight * np.log(
        np.maximum(visual, EPS)
    )
    logit -= logit.max(axis=1, keepdims=True)
    fused = np.exp(logit)
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), EPS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="testprobs_astgcn_world25_int8.npz")
    parser.add_argument("--visual", nargs="+", required=True,
                        help="one or more visual members; several are combined "
                             "by geometric mean before fusion")
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", help="existing fused NPZ to compare against")
    args = parser.parse_args()

    base, sids = load(args.base)
    members = []
    for name in args.visual:
        probs, member_sids = load(name)
        if member_sids != sids:
            raise SystemExit(f"{name}: sid order differs from the base")
        members.append(np.log(np.maximum(probs, EPS)))
    stacked = np.exp(np.mean(members, axis=0))
    visual = stacked / np.maximum(stacked.sum(axis=1, keepdims=True), EPS)

    fused = geometric_fuse(base / train_prior(), visual, args.weight)
    print(f"base    {os.path.basename(args.base)}  {base.shape}")
    print(f"visual  {', '.join(os.path.basename(v) for v in args.visual)}")
    print(f"w={args.weight}  argmax changes vs base: "
          f"{int((fused.argmax(1) != base.argmax(1)).sum())}/{len(sids)}")

    if args.verify:
        reference, _ = load(args.verify)
        same = int((fused.argmax(1) == reference.argmax(1)).sum())
        print(f"verify vs {os.path.basename(args.verify)}: "
              f"argmax {same}/{len(sids)}  maxdelta "
              f"{np.abs(fused - reference).max():.3e}")

    out = args.output if os.path.isabs(args.output) else os.path.join(ART, args.output)
    np.savez_compressed(out, probs=fused.astype(np.float32),
                        sids=np.array(sids, dtype="<U15"))
    with open(out, "rb") as handle:
        print(f"wrote {out}\n  sha256={hashlib.sha256(handle.read()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
