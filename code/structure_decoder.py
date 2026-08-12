#!/usr/bin/env python3
"""Apply the recovered recording structure to clip probabilities.

Two constraints, both validated in EXP-070 on train without using any label:

  consensus     clips at the same position in the three repetitions of a block
                share one class (94.55% correct, covers 75.7% of clips)
  distinctness  clips inside one repetition have pairwise-distinct classes
                (781/781 runs = 100% correct)

Consensus runs first because it sharpens the distributions that distinctness then
assigns over.  Both are pure inference-time transduction on the split's own
timing metadata; no labels and no manual inspection are involved.

Usage
-----
    python3 code/structure_decoder.py --probs research/artifacts/testprobs_x.npz \
        --split test --output research/artifacts/testprobs_x_struct.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import recording_structure as rs

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


def apply_structure(
    probs: np.ndarray,
    sids: list[str],
    runs: list[list[str]],
    triples: list[list[str]],
    consensus: bool = True,
    distinct: bool = True,
    consensus_weight: float = 1.0,
) -> np.ndarray:
    """Return updated log-probabilities of shape (n, classes)."""
    index = {s: i for i, s in enumerate(sids)}
    log = np.log(np.asarray(probs, dtype=np.float64) + EPS)

    if consensus:
        for triple in triples:
            rows = [index[s] for s in triple if s in index]
            if len(rows) < 2:
                continue
            pooled = log[rows].sum(0)
            # weight 1.0 replaces each member with the pooled evidence
            log[rows] = (1.0 - consensus_weight) * log[rows] + consensus_weight * pooled

    if distinct:
        for run in runs:
            rows = [index[s] for s in run if s in index]
            if len(rows) < 2:
                continue
            cost = -log[rows]
            r, c = linear_sum_assignment(cost)
            boost = np.full_like(log[rows], -np.inf)
            for ri, ci in zip(r, c):
                boost[ri, ci] = 0.0
            # hard assignment, but keep the original ordering as the tie-break
            picked = np.full(len(rows), -1, dtype=int)
            for ri, ci in zip(r, c):
                picked[ri] = ci
            for k, row in enumerate(rows):
                if picked[k] >= 0:
                    log[row, picked[k]] += 50.0
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-consensus", action="store_true")
    parser.add_argument("--no-distinct", action="store_true")
    args = parser.parse_args()

    payload = np.load(args.probs, allow_pickle=True)
    probs = payload["probs"]
    sids = [str(x) for x in payload["sids"]]
    cache_dir = args.cache_dir or (ROOT / "cache" / args.split)
    found = rs.structure(cache_dir)

    log = apply_structure(
        probs, sids, found["runs"], found["triples"],
        consensus=not args.no_consensus, distinct=not args.no_distinct,
    )
    updated = np.exp(log - log.max(1, keepdims=True))
    updated /= updated.sum(1, keepdims=True)

    changed = int((updated.argmax(1) != probs.argmax(1)).sum())
    print(f"{args.probs.name}: {len(sids)} clips, {changed} predictions changed "
          f"({changed / len(sids):.3f})")
    if "labels" in payload.files:
        y = payload["labels"].astype(int)
        print(f"  accuracy {(probs.argmax(1) == y).mean():.4f} -> "
              f"{(updated.argmax(1) == y).mean():.4f}")
    if args.output:
        np.savez(args.output, probs=updated.astype(np.float32), sids=np.array(sids))
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
