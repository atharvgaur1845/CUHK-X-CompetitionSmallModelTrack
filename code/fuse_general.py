#!/usr/bin/env python3
"""Generalized inference-time pooling over the champion's members.

The champion is the special case

    fused = softmax( (1-w) * log(base / train_prior) + w * mean_m log(visual_m) )

with w=0.45 and two visual members pooled by an *unweighted* geometric mean.
This script exposes the three pooling degrees of freedom the champion fixes by
convention rather than by measurement:

* ``--visual-weights``  the split *within* the visual block.  The champion's
  equal split is an unstated uniform prior over member quality; the two members
  are not equally good on public (solo 78 vs 63 clips).
* ``--power``           the pooling exponent.  p=0 is the geometric mean the
  champion uses, p=1 is the arithmetic mean already measured as strictly worse.
  p<0 extrapolates along that measured gradient toward an agreement/veto pool.
* ``--motion-weight``   a fixed, rule-based visual weight applied only to the
  9 motion class ids (28-36), where the skeleton stack is strong and the visual
  members are weak.  Class membership is a fixed rule, not a fitted partition.

``--temp-base`` / ``--temp-visual`` exist only to *demonstrate* that per-member
temperature is degenerate with (weight, transition-weight) for a two-term log
pool; see ``code/degeneracy_check.py``.  They are not a new degree of freedom.
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
MOTION_CLASSES = tuple(range(28, 37))  # 9 gait/motion classes


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


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(x.sum(axis=1, keepdims=True), EPS)


def log_pool(logs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted mean in log space; weights broadcast over classes."""
    total = np.zeros_like(logs[0])
    for lg, wt in zip(logs, weights):
        total = total + wt * lg
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="testprobs_astgcn_world25_int8.npz")
    parser.add_argument("--visual", nargs="+", required=True)
    parser.add_argument("--weight", type=float, required=True,
                        help="total mass on the visual block")
    parser.add_argument("--visual-weights", nargs="*", type=float, default=None,
                        help="relative split inside the visual block; "
                             "renormalized to sum to 1 (default: equal)")
    parser.add_argument("--motion-weight", type=float, default=None,
                        help="visual block mass used on class ids 28-36 only")
    parser.add_argument("--power", type=float, default=0.0,
                        help="pooling exponent; 0 = geometric (champion)")
    parser.add_argument("--temp-base", type=float, default=1.0)
    parser.add_argument("--temp-visual", type=float, default=1.0)
    parser.add_argument("--no-prior", action="store_true",
                        help="skip the base/train_prior correction")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", help="existing fused NPZ to compare against")
    args = parser.parse_args()

    base, sids = load(args.base)
    if not args.no_prior:
        base = normalize(base / train_prior())
    else:
        base = normalize(base)

    members = []
    for name in args.visual:
        probs, member_sids = load(name)
        if member_sids != sids:
            raise SystemExit(f"{name}: sid order differs from the base")
        members.append(np.log(np.maximum(probs, EPS)))

    if args.visual_weights is None:
        vw = np.full(len(members), 1.0 / len(members))
    else:
        if len(args.visual_weights) != len(members):
            raise SystemExit("--visual-weights must match --visual count")
        vw = np.asarray(args.visual_weights, dtype=np.float64)
        vw = vw / vw.sum()

    visual = normalize(np.exp(log_pool(members, vw) - log_pool(members, vw).max(1, keepdims=True)))

    log_b = np.log(np.maximum(base, EPS)) / args.temp_base
    log_v = np.log(np.maximum(visual, EPS)) / args.temp_visual

    # Per-class visual mass: fixed rule, motion ids get their own scalar.
    w_vec = np.full(40, args.weight, dtype=np.float64)
    if args.motion_weight is not None:
        w_vec[list(MOTION_CLASSES)] = args.motion_weight

    if args.power == 0.0:
        logit = (1.0 - w_vec)[None, :] * log_b + w_vec[None, :] * log_v
        logit -= logit.max(axis=1, keepdims=True)
        fused = normalize(np.exp(logit))
    else:
        p = args.power
        b = np.exp(log_b)
        v = np.exp(log_v)
        b = b / np.maximum(b.max(1, keepdims=True), EPS)
        v = v / np.maximum(v.max(1, keepdims=True), EPS)
        floor = 1e-8
        b = np.maximum(b, floor)
        v = np.maximum(v, floor)
        mixed = (1.0 - w_vec)[None, :] * b ** p + w_vec[None, :] * v ** p
        fused = normalize(np.maximum(mixed, EPS) ** (1.0 / p))

    print(f"base    {os.path.basename(args.base)}  {base.shape}  prior={'off' if args.no_prior else 'on'}")
    print(f"visual  {', '.join(os.path.basename(v) for v in args.visual)}")
    print(f"        split={np.round(vw, 4).tolist()}")
    print(f"w={args.weight} motion_w={args.motion_weight} power={args.power} "
          f"Tb={args.temp_base} Tv={args.temp_visual}")
    print(f"argmax changes vs base: {int((fused.argmax(1) != base.argmax(1)).sum())}/{len(sids)}")

    if args.verify:
        reference, _ = load(args.verify)
        same = int((fused.argmax(1) == reference.argmax(1)).sum())
        print(f"verify vs {os.path.basename(args.verify)}: argmax {same}/{len(sids)} "
              f"(differs {len(sids)-same})  maxdelta {np.abs(fused-reference).max():.3e}")

    out = args.output if os.path.isabs(args.output) else os.path.join(ART, args.output)
    np.savez_compressed(out, probs=fused.astype(np.float32),
                        sids=np.array(sids, dtype="<U15"))
    with open(out, "rb") as handle:
        print(f"wrote {out}\n  sha256={hashlib.sha256(handle.read()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
