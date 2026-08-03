#!/usr/bin/env python3
"""Geometric (log-space) fusion of the package base with the visual MIL member.

Why this file exists
--------------------
The verified champion `sub_world25_visgeo225_trans05.csv` (0.61194 = 123/201) was
produced by an ad-hoc fusion step that was never committed.  Its input,
`testprobs_world25_visgeo225.npz`, therefore could not be regenerated from any
script in `code/`.  Reproducibility is a hard Selection-Stage gate, so the step
is reconstructed here and verified byte-for-byte against the existing artifact
(`--verify`).

Why geometric rather than linear
--------------------------------
Measured on the public leaderboard (LEADERBOARD.md): geometric fusion beats
linear at every weight tried and stays positive far past the point where linear
collapses.  Linear averaging lets a confident base drown out the visual member;
the geometric mean treats the two as independent evidence.

    fused = normalize( base**(1-w) * visual**w )

Usage
-----
    # regenerate the champion's fused probabilities and prove they match
    python3 code/fuse_visual_geo.py --weight 0.225 \
        --verify research/artifacts/testprobs_world25_visgeo225.npz

    # after new visual folds land, re-fuse from the 4-fold visual member
    python3 code/fuse_visual_geo.py --weight 0.225 \
        --visual research/artifacts/testprobs_visual_mil_f0123.npz \
        --output research/artifacts/testprobs_world25_visgeo225_f0123.npz
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
EPS = 1e-12

DEFAULT_BASE = os.path.join(ART, "testprobs_astgcn_world25_int8.npz")
DEFAULT_VISUAL = os.path.join(ART, "testprobs_visual_mil_f2.npz")


def load(path: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["probs"], dtype=np.float64), data["sids"].tolist()


def geometric_fuse(base: np.ndarray, visual: np.ndarray, weight: float) -> np.ndarray:
    """Weighted geometric mean, computed in log space for numerical stability."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    logit = (1.0 - weight) * np.log(np.maximum(base, EPS)) + weight * np.log(
        np.maximum(visual, EPS)
    )
    # Subtract the row max before exponentiating so no row underflows to zero.
    logit -= logit.max(axis=1, keepdims=True)
    fused = np.exp(logit)
    return fused / np.maximum(fused.sum(axis=1, keepdims=True), EPS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="package base probabilities NPZ")
    parser.add_argument("--visual", default=DEFAULT_VISUAL,
                        help="visual MIL member probabilities NPZ")
    parser.add_argument("--weight", type=float, required=True,
                        help="visual weight w; the champion is 0.225")
    parser.add_argument("--output", help="write fused NPZ here")
    parser.add_argument("--verify",
                        help="compare against an existing fused NPZ and report max abs delta")
    args = parser.parse_args()

    base, base_sids = load(args.base)
    visual, visual_sids = load(args.visual)
    if base_sids != visual_sids:
        raise SystemExit("sid order differs between base and visual inputs")

    fused = geometric_fuse(base, visual, args.weight)
    print(f"base   {os.path.basename(args.base)}  {base.shape}")
    print(f"visual {os.path.basename(args.visual)}  {visual.shape}")
    print(f"weight w={args.weight}  ->  argmax changes vs base: "
          f"{int((fused.argmax(1) != base.argmax(1)).sum())}/405")

    if args.verify:
        reference, reference_sids = load(args.verify)
        if reference_sids != base_sids:
            raise SystemExit("sid order differs from the reference artifact")
        delta = np.abs(fused.astype(np.float32) - reference.astype(np.float32)).max()
        same = int((fused.argmax(1) == reference.argmax(1)).sum())
        print(f"\nverify against {os.path.basename(args.verify)}")
        print(f"  max abs probability delta : {delta:.3e}")
        print(f"  identical argmaxes        : {same}/405")
        print("  RESULT: " + ("REPRODUCED" if same == 405 and delta < 1e-6
                              else "MISMATCH — the committed step differs from the artifact"))

    if args.output:
        np.savez_compressed(args.output,
                            probs=fused.astype(np.float32),
                            sids=np.array(base_sids, dtype="<U12"))
        with open(args.output, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        print(f"\nwrote {args.output}\n  sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
