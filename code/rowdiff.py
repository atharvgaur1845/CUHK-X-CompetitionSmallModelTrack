#!/usr/bin/env python3
"""Row-level diff between submission CSVs, plus per-class-group breakdown.

Usage: python3 code/rowdiff.py REFERENCE.csv CANDIDATE.csv [CANDIDATE2.csv ...]
"""
from __future__ import annotations

import csv
import sys

MOTION = set(range(28, 37))


def read(path: str) -> dict[str, int]:
    with open(path, newline="") as handle:
        return {r["path"]: int(r["prediction"]) for r in csv.DictReader(handle)}


def main() -> int:
    ref = read(sys.argv[1])
    print("reference: %s" % sys.argv[1])
    print("%-52s %7s %7s %7s %9s" % ("candidate", "rowdiff", "->motion", "->sed", "predmotion"))
    print("-" * 88)
    base_motion = sum(1 for v in ref.values() if v in MOTION)
    print("%-52s %7s %7s %7s %9d" % ("(reference)", "-", "-", "-", base_motion))
    for path in sys.argv[2:]:
        cand = read(path)
        keys = [k for k in ref if k in cand]
        diff = [k for k in keys if ref[k] != cand[k]]
        to_motion = sum(1 for k in diff if cand[k] in MOTION and ref[k] not in MOTION)
        to_sed = sum(1 for k in diff if cand[k] not in MOTION and ref[k] in MOTION)
        pm = sum(1 for v in cand.values() if v in MOTION)
        name = path.split("/")[-1]
        print("%-52s %7d %7d %7d %9d" % (name, len(diff), to_motion, to_sed, pm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
