#!/usr/bin/env python3
"""Recover the recording block/repetition structure of a split from timestamps.

EXP-069 established the generative structure of this dataset: a *block* is an
ordered list of K pairwise-distinct classes, performed three times.  Train exposes
it in the trial name (`station-block-repetition`); test has only timestamps, so it
must be inferred.

EXP-070 found the inference is trivial once measured.  Inter-clip gaps separate
perfectly on train (n=2713):

    within a repetition   median   1.3 s   (p90   4.2 s)
    repetition boundary   median  81.9 s   (p25  65.8 s)
    block boundary        median 185.1 s   (p25 126.4 s)

so a 20 s cut recovers 100% of repetition boundaries with a 0.1% false-positive
rate inside repetitions.  A *run* (maximal sequence of clips separated by <20 s)
is therefore exactly one repetition, and its clips carry pairwise-distinct classes
(true for 267/267 train groups).  Blocks are then consecutive runs of equal length,
and clips at the same position in the three runs of a block share one class
(94.5% correct on train).

This module is inference-only: it reads timestamps, never labels.  Test-time
transductive processing is permitted by the organizer ruling recorded in
OBJECTIVE.md, and no test sample is manually labelled.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP_GAP_SECONDS = 20.0
SESSION_GAP_SECONDS = 1200.0


def clip_times(cache_dir: Path) -> dict[str, tuple[float, float]]:
    """sid -> (start, end) wall-clock seconds, from the depth frame timestamps."""
    out: dict[str, tuple[float, float]] = {}
    for path in sorted(cache_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=True) as handle:
                stamps = handle["depth_t"]
        except Exception:
            continue
        if stamps.size == 0:
            continue
        out[path.stem] = (float(np.min(stamps)), float(np.max(stamps)))
    return out


def runs_from_times(times: dict[str, tuple[float, float]],
                    rep_gap: float = REP_GAP_SECONDS) -> list[list[str]]:
    """Split the whole split into runs; one run == one repetition."""
    items = sorted(times.items(), key=lambda kv: kv[1][0])
    runs: list[list[str]] = []
    current: list[str] = []
    previous_end: float | None = None
    for sid, (start, end) in items:
        if current and previous_end is not None and start - previous_end > rep_gap:
            runs.append(current)
            current = []
        current.append(sid)
        previous_end = end
    if current:
        runs.append(current)
    return runs


def blocks_from_runs(runs: list[list[str]],
                     times: dict[str, tuple[float, float]]) -> list[list[list[str]]]:
    """Group consecutive runs into blocks of three repetitions.

    Requires three consecutive runs of EQUAL length.  A gap-only rule (cut at the
    150 s block boundary) recovers the right number of blocks -- 265 against 264
    true groups -- but admits ragged blocks whose positional triples are only
    81.8% pure, against 94.6% here.  Cross-repetition matching by prediction
    similarity was tried to lift coverage on test and rejected: monotone alignment
    reached 0.579 clip-weighted purity and a positional prior made it worse
    (0.522), because the model's own predictions are too noisy to align with.

    Runs that do not form a clean triple become their own single-run block, so no
    clip is dropped -- it simply gains distinctness without consensus.
    """
    blocks: list[list[list[str]]] = []
    i = 0
    while i < len(runs):
        if i + 2 < len(runs) and len(runs[i]) == len(runs[i + 1]) == len(runs[i + 2]):
            if times[runs[i + 2][0]][0] - times[runs[i][-1]][1] < SESSION_GAP_SECONDS:
                blocks.append([runs[i], runs[i + 1], runs[i + 2]])
                i += 3
                continue
        blocks.append([runs[i]])
        i += 1
    return blocks


def positional_triples(blocks: list[list[list[str]]]) -> list[list[str]]:
    """Same position across equal-length runs of a block. Exact when lengths match."""
    out = []
    for block in blocks:
        if len(block) < 2:
            continue
        if len({len(r) for r in block}) != 1:
            continue
        for position in range(len(block[0])):
            out.append([run[position] for run in block])
    return out


def structure(cache_dir: Path) -> dict:
    times = clip_times(cache_dir)
    runs = runs_from_times(times)
    blocks = blocks_from_runs(runs, times)
    return {
        "times": times,
        "runs": runs,
        "blocks": blocks,
        "triples": positional_triples(blocks),
    }


def _truth(sid: str):
    parts = sid.split("_")
    station, block, repetition = parts[1 + 1].split("-")
    return (parts[1], station, block), int(repetition), int(parts[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache_dir = args.cache_dir or (ROOT / "cache" / args.split)
    found = structure(cache_dir)
    runs, blocks, tri = found["runs"], found["blocks"], found["triples"]
    covered = sum(len(t) for t in tri)
    print(f"{args.split}: {len(found['times'])} clips")
    print(f"  runs (repetitions): {len(runs)}  median size "
          f"{int(np.median([len(r) for r in runs]))}  sizes "
          f"{Counter(len(r) for r in runs).most_common(6)}")
    print(f"  blocks: {len(blocks)}  complete (3 runs): "
          f"{sum(1 for b in blocks if len(b) == 3)}")
    print(f"  triples: {len(tri)} covering {covered} clips "
          f"({covered / max(len(found['times']), 1):.3f} of the split)")

    if args.split == "train":
        pure_runs = sum(1 for r in runs if len({_truth(s)[0] for s in r}) == 1
                        and len({_truth(s)[1] for s in r}) == 1)
        print(f"  VALIDATION runs matching exactly one true (group, repetition): "
              f"{pure_runs}/{len(runs)} = {pure_runs / len(runs):.4f}")
        distinct = sum(1 for r in runs if len({_truth(s)[2] for s in r}) == len(r))
        print(f"  VALIDATION runs with pairwise-distinct classes: "
              f"{distinct}/{len(runs)} = {distinct / len(runs):.4f}")
        good = sum(1 for t in tri if len({_truth(s)[2] for s in t}) == 1)
        print(f"  VALIDATION triples that are truly one class: "
              f"{good}/{len(tri)} = {good / max(len(tri), 1):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "split": args.split,
            "rep_gap_seconds": REP_GAP_SECONDS,
            "runs": runs,
            "triples": tri,
        }, indent=1))
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
