#!/usr/bin/env python3
"""Constrain predicted class marginals to the protocol prior (Sinkhorn / transport).

Why this is legitimate here
---------------------------
The recording protocol, not the subject, sets how often each activity occurs:
class 36 is 11.2% of clips in 17 of the 18 train users, and 448 of 528 non-zero
per-user-class counts are exact multiples of 3 (each activity is performed in
three rounds).  So the class marginal is a property of the *protocol* and should
carry to unseen subjects, unlike anything fitted on train-user identity -- which
is the distinction that killed the previous four levers (GBDT stacker, structure
decoder, cohort weights, learned gate: all +OOF, all <=0 public).

Measured on the 126/201 champion, the predicted test histogram is off the protocol
prior by 83.3 clips of 405 (20.6%): class 26 predicted 24x against 5.5 expected,
class 25 15x against 1.7, while class 6 is 11.6 short.  Rare classes are massively
over-predicted -- the signature of pushing logits toward a uniform prior and never
putting the protocol prior back.

Method
------
Sinkhorn scaling in log space: alternately renormalise rows to sum to 1 and columns
to the target counts.  `--tau` sharpens (<1) or softens (>1) the probabilities first;
`--iters` 0 disables, recovering the input exactly.  `--hard` instead solves the exact
transportation problem, forcing integer per-class counts.

Validation is on OOF with the *OOF's own* prior as target, which tests the method
rather than the transfer.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
EPS = 1e-12


def train_prior() -> np.ndarray:
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as handle:
        counts = collections.Counter(
            int(row["class_id"]) for row in csv.DictReader(handle)
        )
    prior = np.array([counts.get(i, 0) for i in range(40)], dtype=np.float64)
    return prior / prior.sum()


def sinkhorn(probs: np.ndarray, target: np.ndarray, iters: int, tau: float) -> np.ndarray:
    """Scale probs so column masses approach `target` (counts) and rows stay simplices."""
    logp = np.log(np.maximum(probs, EPS)) / tau
    logp -= logp.max(axis=1, keepdims=True)
    log_target = np.log(np.maximum(target, EPS))
    for _ in range(iters):
        logp -= np.logaddexp.reduce(logp, axis=1, keepdims=True)
        col = np.logaddexp.reduce(logp, axis=0, keepdims=True)
        logp += log_target[None, :] - col
    logp -= np.logaddexp.reduce(logp, axis=1, keepdims=True)
    return np.exp(logp)


def hard_assign(probs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Exact min-cost assignment of clips to classes under integer per-class capacities."""
    from scipy.optimize import linear_sum_assignment

    slots = np.repeat(np.arange(len(counts)), counts.astype(int))
    cost = -np.log(np.maximum(probs, EPS))[:, slots]
    rows, cols = linear_sum_assignment(cost)
    out = np.zeros(len(probs), dtype=np.int64)
    out[rows] = slots[cols]
    return out


def largest_remainder(prior: np.ndarray, total: int) -> np.ndarray:
    raw = prior * total
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: total - base.sum()]:
        base[i] += 1
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probs", required=True)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", action="store_true",
                        help="exact transport with integer per-class counts")
    parser.add_argument("--evaluate", action="store_true",
                        help="input carries labels; report accuracy before/after")
    parser.add_argument("--target", choices=("train", "empirical"), default="train",
                        help="'empirical' uses the input's own label marginal (OOF only)")
    parser.add_argument("--output")
    args = parser.parse_args()

    path = args.probs if os.path.isabs(args.probs) else os.path.join(ART, args.probs)
    data = np.load(path, allow_pickle=True)
    probs = np.asarray(data["probs"], dtype=np.float64)
    sids = [str(s) for s in data["sids"]]
    labels = np.asarray(data["labels"]) if "labels" in data.files else None

    if args.target == "empirical":
        if labels is None:
            raise SystemExit("--target empirical needs labels in the NPZ")
        counts = np.bincount(labels, minlength=40).astype(float)
        prior = counts / counts.sum()
    else:
        prior = train_prior()
    target = largest_remainder(prior, len(probs))

    if args.hard:
        prediction = hard_assign(probs, target)
        adjusted = np.eye(40)[prediction]
    else:
        adjusted = sinkhorn(probs, target.astype(float), args.iters, args.tau)
        prediction = adjusted.argmax(1)

    before = probs.argmax(1)
    print(f"{os.path.basename(path)}  n={len(probs)}  "
          f"{'hard transport' if args.hard else f'sinkhorn iters={args.iters} tau={args.tau}'}")
    print(f"  argmax changed: {int((prediction != before).sum())}/{len(probs)}")
    miss = np.abs(np.bincount(before, minlength=40) - target).sum() / 2
    miss_after = np.abs(np.bincount(prediction, minlength=40) - target).sum() / 2
    print(f"  marginal misallocation: {miss:.0f} -> {miss_after:.0f} clips")
    if args.evaluate and labels is not None:
        print(f"  accuracy: {(before == labels).mean():.4f} -> "
              f"{(prediction == labels).mean():.4f}  "
              f"({int((prediction == labels).sum() - (before == labels).sum()):+d} clips)")

    if args.output:
        out = args.output if os.path.isabs(args.output) else os.path.join(ART, args.output)
        np.savez_compressed(out, probs=adjusted.astype(np.float32),
                            sids=np.array(sids, dtype="<U15"))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
