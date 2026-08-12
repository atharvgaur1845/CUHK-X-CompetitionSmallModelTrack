#!/usr/bin/env python3
"""Non-linear stacking over member probabilities, fold-safe.

EXP-071 showed the appearance member holds +7.1 clips of information that weighted
fusion cannot reach, and I wrongly concluded the information was unreachable.
Linear failure is not non-linear failure, and the ledger shows no GBDT/stacking
attempt was ever made -- only a "learned logit gate" (-6.8 nested).

Design note that matters more than the model choice: this scores (clip, class)
PAIRS, not clips.  A 40-way classifier over concatenated member probabilities can
memorise which classes the training users perform, which is precisely the kind of
subject-specific structure that fails to transfer here (skeleton -7.55, transition
+10.4 OOF -> +3 public).  A pairwise scorer sees only "how do the members rate
THIS class for THIS clip", is invariant to class identity, and so has no per-class
prior to overfit.  Class identity is offered only as an optional feature so its
value can be measured rather than assumed.

Evaluation is fold-safe: for outer fold f the stacker trains on the clips of the
other three folds and predicts fold f, so no clip is scored by a model that saw
its user.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "research" / "artifacts"
EPS = 1e-12
SED = (0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,
       25, 26, 27, 37, 38, 39)
DEFAULT_MEMBERS = ("oof_astgcn_world25", "oof_visual_mil_v1", "oof_frame_app_mm")


def load_members(names, ) -> tuple[list[str], np.ndarray, np.ndarray]:
    tables, sid_sets = {}, []
    for name in names:
        d = np.load(ARTIFACTS / f"{name}.npz", allow_pickle=True)
        sids = [str(x) for x in d["sids"]]
        tables[name] = (sids, d["probs"].astype(np.float64),
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
    return common, np.stack(probs, 1), labels          # (n, n_members, 40)


def pair_features(probs: np.ndarray, use_class_id: bool) -> np.ndarray:
    """(n, m, C) member probabilities -> (n*C, F) features for (clip, class) pairs."""
    n, m, C = probs.shape
    logp = np.log(probs + EPS)
    # per (clip, class): each member's probability, log-probability and rank
    order = (-probs).argsort(2)
    ranks = np.empty_like(order)
    for i in range(m):
        ranks[:, i, :] = order[:, i, :].argsort(1)
    top1 = probs.max(2, keepdims=True)
    margin = probs - top1                                  # 0 for the member's argmax
    entropy = -(probs * logp).sum(2, keepdims=True)
    feats = [
        probs.transpose(0, 2, 1).reshape(n * C, m),
        logp.transpose(0, 2, 1).reshape(n * C, m),
        ranks.transpose(0, 2, 1).reshape(n * C, m).astype(np.float64),
        margin.transpose(0, 2, 1).reshape(n * C, m),
        np.repeat(entropy.transpose(0, 2, 1).reshape(n, m), C, 0),
        np.repeat(top1.transpose(0, 2, 1).reshape(n, m), C, 0),
    ]
    geo = logp.mean(1)                                     # current fusion, as a feature
    feats.append(geo.reshape(n * C, 1))
    feats.append(probs.mean(1).reshape(n * C, 1))
    # how many members put this class first -- a direct agreement signal
    feats.append((ranks == 0).sum(1).reshape(n * C, 1).astype(np.float64))
    if use_class_id:
        feats.append(np.tile(np.arange(C, dtype=np.float64), n).reshape(n * C, 1))
    return np.concatenate(feats, 1)


def run(members, folds_of, use_class_id, seed, max_iter):
    common, probs, y = load_members(members)
    n, m, C = probs.shape
    fold = np.array([folds_of[s] for s in common])
    X = pair_features(probs, use_class_id)
    pair_y = (np.tile(np.arange(C), n) == np.repeat(y, C)).astype(int)
    clip_of = np.repeat(np.arange(n), C)

    scores = np.zeros(n * C)
    for f in sorted(set(fold.tolist())):
        tr = np.isin(clip_of, np.where(fold != f)[0])
        te = np.isin(clip_of, np.where(fold == f)[0])
        model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed, early_stopping=False)
        model.fit(X[tr], pair_y[tr])
        scores[te] = model.predict_proba(X[te])[:, 1]
    pred = scores.reshape(n, C).argmax(1)
    return common, probs, y, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", nargs="*", default=list(DEFAULT_MEMBERS))
    ap.add_argument("--class-id", action="store_true",
                    help="offer class identity as a feature (measure, do not assume)")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    v = np.load(ARTIFACTS / "oof_visual_mil_v1.npz", allow_pickle=True)
    folds_of = {str(s): int(f) for s, f in zip(v["sids"], v["folds"])}

    common, probs, y, pred = run(args.members, folds_of, args.class_id,
                                 args.seed, args.max_iter)
    m = np.isin(y, SED)
    geo = (0.65 * np.log(probs[:, 0] + EPS) + 0.35 * np.log(probs[:, 1] + EPS)).argmax(1)
    base = (geo == y).mean()
    got = (pred == y)
    print(f"members: {args.members}")
    print(f"clips {len(y)} | class-id feature: {args.class_id}")
    print(f"  weighted fusion (w=0.35) : {base:.4f}  sedentary {(geo==y)[m].mean():.4f}")
    print(f"  GBDT pairwise stacker    : {got.mean():.4f}  sedentary {got[m].mean():.4f}"
          f"   ({(got.mean()-base)*201:+.1f} clips/201)")


if __name__ == "__main__":
    main()
