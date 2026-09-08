#!/usr/bin/env python3
"""Does accuracy still RISE with the number of training subjects at n=18?

This is the question that decides whether pseudo-labelling the test split is worth GPU
time. The test split holds 12 subjects we have never seen (R-4 permits self-training on
it). If the subject learning curve has plateaued by 18, those 12 subjects buy nothing and
the route is dead before it costs anything. If it is still climbing, subject count is the
binding constraint -- which is what EXP-124's between-subject sd of 5.26 points (4.5x the
seed sigma) already suggests.

Protocol: features from the fold-2 MViT for every clip; probe trained on k of the 14
non-fold-2 users; evaluated on the 4 held-out fold-2 users, whose features that model
never trained on. The probe is a proxy for fine-tuning, so read the SHAPE, not the level.
Each k is averaged over several random user draws to keep subject sampling from
dominating -- between-subject sd is the largest variance in this dataset.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "research" / "artifacts" / "emb"
FOLD2 = ("user22", "user24", "user5", "user6")


def main():
    d = np.load(EMB / "emb_person_f2model_all.npz", allow_pickle=True)
    X = d["feats"].astype(np.float32); sids = [str(s) for s in d["sids"]]
    idx = json.loads((ROOT / "cache" / "crop_224" / "train_index.json").read_text())
    y = np.array([int(idx["labels"][s]) for s in sids])
    u = np.array([idx["users"][s] for s in sids])

    te = np.isin(u, FOLD2)
    pool = sorted(set(u[~te]))
    print(f"eval = {len(FOLD2)} held-out users, {te.sum()} clips; "
          f"train pool = {len(pool)} users, {(~te).sum()} clips")
    Xe, ye = X[te], y[te]

    rng = np.random.default_rng(0)
    print("\n  k users   mean acc   sd     (draws)   clips")
    prev = None
    for k in (2, 4, 6, 8, 10, 12, 14):
        accs, ns = [], []
        draws = 6 if k < 14 else 1
        for _ in range(draws):
            sel = rng.choice(pool, k, replace=False) if k < 14 else np.array(pool)
            m = np.isin(u, sel)
            sc = StandardScaler().fit(X[m])
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[m]), y[m])
            accs.append((clf.predict(sc.transform(Xe)) == ye).mean())
            ns.append(m.sum())
        a = np.array(accs)
        gain = f"  (+{100*(a.mean()-prev):.2f} vs k-2)" if prev is not None else ""
        print(f"  {k:2d}       {a.mean():.4f}   {a.std(ddof=1) if len(a)>1 else 0:.4f}"
              f"   ({draws})    {int(np.mean(ns)):4d}{gain}")
        prev = a.mean()
    print("\nRead: if the last steps still add clearly, subject count is still binding and "
          "the 12 unlabelled test subjects are worth pseudo-labelling (R-4). If flat, dead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
