#!/usr/bin/env python3
"""EXP-141 -- the 18-subject leave-one-subject-out table, and the selection statistic.

Why this exists. Private (8 subjects) and the on-site test (8 subjects, 30% of the grade)
are 8-subject draws from the same subject population. EXP-124 measured between-subject sd
at 5.26 points against a seed sd of 1.16, so WHICH subjects are drawn dominates every
other source of variance -- and the public 4 are one draw of that same lottery. The mean
is the wrong summary: an ensemble that is excellent on easy subjects and collapses on hard
ones has a worse expected 8-subject draw than a flatter one with the same mean.

The statistic this file produces is the LOWER QUARTILE -- the mean of the 4 worst subjects
-- computed under true LOSO (17 training subjects, 1 held out) rather than the 4-fold
grouping, which trains on only 13-14 subjects and holds out 4.
"""
from __future__ import annotations
import csv, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")


def main() -> int:
    meta = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "cache", "meta_train.csv"))):
        meta[r["sample_id"]] = (int(r["class_id"]), r["user"])

    rows = []
    for f in sorted(os.listdir(ART)):
        if not (f.startswith("oof_loso_user") and f.endswith(".npz")):
            continue
        user = f[len("oof_loso_"):-len(".npz")]
        d = np.load(os.path.join(ART, f), allow_pickle=True)
        sids = [str(s) for s in d["sids"]]
        p = np.asarray(d["probs"], np.float64)
        y = np.array([meta[s][0] for s in sids])
        who = {meta[s][1] for s in sids}
        assert who == {user}, f"{f} holds {who}, expected {user}"
        rows.append((user, len(y), float((p.argmax(1) == y).mean())))

    rows.sort(key=lambda r: r[2])
    n = len(rows)
    acc = np.array([r[2] for r in rows])
    print(f"LOSO over {n} subjects -- person view (k224_mvit), 17 train subjects each\n")
    print(f"{'subject':>9s} {'clips':>6s} {'acc':>8s}")
    for u, k, a in rows:
        bar = "#" * int(round(a * 40))
        print(f"{u:>9s} {k:6d} {a:8.5f}  {bar}")

    q = int(np.ceil(n / 4))
    print(f"\nmean            {acc.mean():.5f}")
    print(f"sd (n-1)        {acc.std(ddof=1):.5f}   <- between-subject, {n-1} df")
    print(f"SE of an 8-subject draw   {acc.std(ddof=1)/np.sqrt(8):.5f}"
          f"  ({100*acc.std(ddof=1)/np.sqrt(8):.2f} points)")
    print(f"SE of a  4-subject draw   {acc.std(ddof=1)/np.sqrt(4):.5f}"
          f"  ({100*acc.std(ddof=1)/np.sqrt(4):.2f} points)  <- the public leaderboard")
    print(f"\nLOWER QUARTILE (worst {q}) {acc[:q].mean():.5f}   <- SELECTION STATISTIC")
    print(f"upper quartile  (best {q}) {acc[-q:].mean():.5f}")
    print(f"spread worst->best        {acc[-1]-acc[0]:.5f} "
          f"({100*(acc[-1]-acc[0]):.1f} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
