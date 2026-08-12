#!/usr/bin/env python3
"""Fit the pairwise GBDT stacker on all train OOF and predict the test split.

EXP-073.  Member pairs (OOF artifact, test artifact) must describe the SAME model;
they are listed explicitly rather than inferred from names, because the visual MIL
member is stored as `oof_visual_mil_v1` on the train side and
`testprobs_visual_mil_f0123` on the test side, and silently mismatching two members
would be invisible in the output.

    python3 code/stack_submit.py --output submissions/sub_stack9.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import stack_members as sm

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "research" / "artifacts"

MEMBERS = [
    ("oof_astgcn_world25", "testprobs_astgcn_world25"),
    ("oof_visual_mil_v1", "testprobs_visual_mil_f0123"),
    ("oof_frame_app_mm", "testprobs_frame_app_mm"),
    ("oof_sed_spec", "testprobs_sed_spec"),
    ("oof_tjitter_v2_p3_astgcn_v1", "testprobs_tjitter_v2_p3_astgcn_v1"),
    ("oof_tjitter_v2_p3_skel_multitcn", "testprobs_tjitter_v2_p3_skel_multitcn"),
    ("oof_stgcn_w96", "testprobs_stgcn_w96"),
    ("oof_imu_world", "testprobs_imu_world"),
    ("oof_droi_big", "testprobs_droi_big"),
]


def load_split(names, key="probs"):
    tables, sid_sets = {}, []
    for name in names:
        d = np.load(ARTIFACTS / f"{name}.npz", allow_pickle=True)
        sids = [str(x) for x in d["sids"]]
        tables[name] = (sids, np.nan_to_num(d[key].astype(np.float64)),
                        d["labels"].astype(int) if "labels" in d.files else None)
        sid_sets.append(set(sids))
    common = sorted(set.intersection(*sid_sets))
    probs, labels = [], None
    for name in names:
        sids, p, y = tables[name]
        index = {s: i for i, s in enumerate(sids)}
        rows = [index[s] for s in common]
        probs.append(p[rows])
        if labels is None and y is not None:
            labels = y[rows]
    return common, np.stack(probs, 1), labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--class-id", action="store_true", default=True)
    ap.add_argument("--no-class-id", dest="class_id", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-submission", type=Path,
                    default=ROOT / "Small-Model-Track" / "Testing" / "sample_submission.csv")
    args = ap.parse_args()

    tr_names = [a for a, _ in MEMBERS]
    te_names = [b for _, b in MEMBERS]
    tr_sids, tr_probs, y = load_split(tr_names)
    te_sids, te_probs, _ = load_split(te_names)
    if tr_probs.shape[1] != te_probs.shape[1]:
        raise SystemExit("member count differs between splits")
    print(f"train {tr_probs.shape}  test {te_probs.shape}  members {len(MEMBERS)}")

    n_tr, m, C = tr_probs.shape
    Xtr = sm.pair_features(tr_probs, args.class_id)
    ytr = (np.tile(np.arange(C), n_tr) == np.repeat(y, C)).astype(int)
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=args.seed, early_stopping=False)
    model.fit(Xtr, ytr)

    n_te = te_probs.shape[0]
    Xte = sm.pair_features(te_probs, args.class_id)
    scores = model.predict_proba(Xte)[:, 1].reshape(n_te, C)
    pred = scores.argmax(1)
    # Save a normalised distribution so the ordered-transition decoder -- which is
    # part of the 125/201 champion pipeline -- can run on top of the stacker.
    dist = scores / np.maximum(scores.sum(1, keepdims=True), 1e-12)
    np.savez(ARTIFACTS / f"testprobs_{args.output.stem}.npz",
             probs=dist.astype(np.float32), sids=np.array(te_sids))

    geo = (0.65 * np.log(te_probs[:, 0] + sm.EPS)
           + 0.35 * np.log(te_probs[:, 1] + sm.EPS)).argmax(1)
    print(f"stacker differs from w=0.35 fusion on {(pred != geo).sum()}/{n_te} clips")

    lookup = {s: int(c) for s, c in zip(te_sids, pred)}
    rows = []
    with args.sample_submission.open() as fh:
        for row in csv.DictReader(fh):
            sid = row["path"].rstrip("/").split("/")[-1]
            rows.append((row["path"], lookup.get(sid, 34)))
    missing = sum(1 for _, c in rows if c == 34) - sum(1 for c in pred if c == 34)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "prediction"])
        w.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} rows, {max(missing,0)} sids unmatched)")


if __name__ == "__main__":
    main()
