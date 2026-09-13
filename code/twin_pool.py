#!/usr/bin/env python3
"""EXP-157 -- TWIN POOLING: pool evidence across repeated trials of the same scene.

The structure (measured on train, code/twin_pool.py --audit): every recording group is
all-distinct (743/743), and each scene is performed 2-3 times by the same subject in the
SAME order (227 of 241 repeats identical). Repeats are recognisable from time and length
alone -- consecutive groups of equal size within 120 s are the same scene 97.0% of the time
and in the same order 94.3% -- and 81% of clips sit in such a run. No user id is needed,
which is what makes it usable on test.

So clip i of trial 2 is, with ~94% confidence, the SAME ACTIVITY as clip i of trial 1 and
trial 3. Pooling their probabilities is a 2-3x evidence gain per clip, from a prior the
per-clip models and the Markov decoder cannot see: the decoder chains WITHIN a group; this
chains ACROSS groups.

Implementation: geometric mean of the member probabilities at each position across the
run, written back to every member, then the standard decoder runs unchanged. The decoder
still applies transitions and distinctness on top.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import ordered_transition_decoder as D  # noqa: E402


def runs_of_twins(seqs, gap_s: float):
    """seqs: list of groups, each a list of Clip sorted by t0. Returns list of runs
    (lists of groups) of consecutive equal-length groups within gap_s."""
    seqs = sorted((s for s in seqs if len(s) >= 2), key=lambda s: s[0].t0)
    runs, i = [], 0
    while i < len(seqs):
        j = i
        while (j + 1 < len(seqs) and len(seqs[j + 1]) == len(seqs[i])
               and seqs[j + 1][0].t0 - seqs[j][-1].t0 < gap_s):
            j += 1
        runs.append(seqs[i:j + 1]); i = j + 1
    return runs


def pool(probs: np.ndarray, sids: list[str], split: str, gap_s: float, power: float = 1.0):
    clips = D.build_train_clips("trial") if split == "train" else D.build_test_clips("trial")
    have = set(sids)
    seqs = [sorted(s, key=lambda c: c.t0)
            for s in D.group_sequences(c for c in clips.values() if c.sid in have)]
    idx = {s: i for i, s in enumerate(sids)}
    out = probs.copy()
    L = np.log(np.maximum(probs, 1e-12))
    n_pooled = 0; n_runs = 0
    for run in runs_of_twins(seqs, gap_s):
        if len(run) < 2:
            continue
        n_runs += 1
        n = len(run[0])
        for pos in range(n):
            rows = [idx[g[pos].sid] for g in run]
            m = L[rows].mean(0)
            # power < 1 softens the twins' pull (a hedge against the ~6% misaligned pairs)
            pooled = L[rows] * (1 - power) + m[None, :] * power
            p = np.exp(pooled - pooled.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            out[rows] = p
            n_pooled += len(rows)
    return out, n_runs, n_pooled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--gap", type=float, default=120.0)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.probs, allow_pickle=True)
    sids = [str(s) for s in d["sids"]]; P = np.asarray(d["probs"], np.float64)
    out, nr, npooled = pool(P, sids, a.split, a.gap, a.power)
    extra = {"labels": d["labels"]} if "labels" in d.files else {}
    np.savez_compressed(a.out, probs=out.astype(np.float32), sids=np.array(sids, dtype="<U15"), **extra)
    print(f"twin-pooled {npooled}/{len(sids)} clips in {nr} runs (gap<{a.gap:.0f}s, power={a.power}) -> {a.out}")
    if "labels" in d.files:
        y = np.asarray(d["labels"])
        print(f"  pre-decoder top-1: {(P.argmax(1)==y).mean():.5f} -> {(out.argmax(1)==y).mean():.5f}"
              f"  ({int((out.argmax(1)==y).sum())-int((P.argmax(1)==y).sum()):+d} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
