#!/usr/bin/env python3
"""L1/T1 gate — probe frozen foundation features, subject-grouped, against our own member.

Bars, pre-registered:
  fold 2   > 0.74      (the 3.28-point single-fold bar over the 0.70501 seed mean, EXP-120a)
  pooled   > 0.7116    (k224_mvit 4-fold pooled, EXP-103) to be worth using as a teacher
  pooled   > 0.80      for the plan's T1 gate -- a teacher that could lift the student far

Logit-adjusted so the probe emits UNIFORM-prior posteriors like every other video member;
getting prior space backwards scored 0.58706 vs 0.62686 once already (CLAUDE.md).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "research" / "artifacts"
TD = ART / "teacher"
FOLDS = {0: ("user1", "user19", "user21", "user8", "user9"),
         1: ("user18", "user20", "user23", "user3", "user7"),
         2: ("user22", "user24", "user5", "user6"),
         3: ("user16", "user17", "user2", "user4")}
OBJECT = np.array(sorted(set(range(28)) | {37, 38, 39}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", required=True,
                    help="npz basenames under research/artifacts/teacher, concatenated")
    ap.add_argument("--clf", default="logreg", choices=("logreg", "mlp"))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    lab, usr = {}, {}
    idx = json.loads((ROOT / "cache" / "crop_224" / "train_index.json").read_text())
    lab = {k: int(v) for k, v in idx["labels"].items()}
    usr = idx["users"]

    mats, sids = [], None
    for f in a.feats:
        d = np.load(TD / f"{f}.npz", allow_pickle=True)
        s = [str(x) for x in d["sids"]]
        if sids is None:
            sids = s
        elif s != sids:
            raise SystemExit(f"{f}: sid order differs")
        X = d["feats"].astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        mats.append(X)
    X = np.concatenate(mats, 1)
    y = np.array([lab[s] for s in sids])
    u = np.array([usr[s] for s in sids])
    print(f"features {X.shape} from {len(a.feats)} source(s); {len(set(u))} users")

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    oof = np.zeros((len(y), 40))
    per = []
    for f in range(4):
        te = np.isin(u, FOLDS[f])
        if a.clf == "logreg":
            m = LogisticRegression(max_iter=3000, C=a.C, n_jobs=-1)
        else:
            m = MLPClassifier(hidden_layer_sizes=(512,), max_iter=400, alpha=1e-3,
                              random_state=0)
        m.fit(X[~te], y[~te])
        p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_):
            oof[te, c] = p[:, j]
        acc = (oof[te].argmax(1) == y[te]).mean()
        om = np.isin(y[te], OBJECT)
        per.append(acc)
        print(f"  fold {f}: n={te.sum():4d}  micro={acc:.5f}  "
              f"object={(oof[te][om].argmax(1) == y[te][om]).mean():.5f}")
    pooled = (oof.argmax(1) == y).mean()
    om = np.isin(y, OBJECT)
    print(f"\n  POOLED micro {pooled:.5f}   object {(oof[om].argmax(1) == y[om]).mean():.5f}")
    print(f"  reference: k224_mvit pooled 0.71156 (EXP-103); fold2 seed mean 0.70501")
    print(f"  fold2 {per[2]:.5f} -> {'PASS' if per[2] > 0.74 else 'FAIL'} vs the 0.74 bar")
    print(f"  pooled -> {'beats our member' if pooled > 0.71156 else 'below our member'}; "
          f"T1 gate (0.80) {'PASS' if pooled > 0.80 else 'FAIL'}")
    if a.save:
        np.savez_compressed(ART / f"oof_{a.save}.npz", probs=oof.astype(np.float32),
                            sids=np.array(sids, dtype="<U24"), labels=y)
        print(f"  wrote oof_{a.save}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
