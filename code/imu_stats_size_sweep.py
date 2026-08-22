#!/usr/bin/env python3
"""EXP-105: how small can imu_stats get before it stops paying its 0.3575 weight?

Dropping imu_stats moves 43 of 405 rows, so it is load-bearing and must ship. An
unconstrained 1000-tree ExtraTrees serializes near 100 MB on its own, which the 100 MB
Stage-2 budget cannot absorb. This sweeps (n_estimators, max_depth) against the same
GroupKFold-by-user OOF the member already uses, and against serialized bytes.
"""
import csv, io, os, sys, time
import numpy as np, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imu_stats_member as M
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT, ART = M.ROOT, M.ART
SMOOTH = 1e-3
meta = [r for r in csv.DictReader(open(os.path.join(ROOT, "cache", "meta_train.csv")))
        if os.path.exists(os.path.join(ROOT, "cache", "train", f"{r['sample_id']}.npz"))]
sids = [r["sample_id"] for r in meta]
y = np.array([int(r["class_id"]) for r in meta])
groups = np.array([r["user"] for r in meta])

cache = "/tmp/claude-1000/-home-atharv-Desktop-projects-KAggle--CUHK-X-CompetitionSmallModelTrack/31c5debc-4002-4341-8b19-64336fb4bd53/scratchpad/imu_X.npy"
if os.path.exists(cache):
    X = np.load(cache)
else:
    X = M.build_matrix(sids, "train"); np.save(cache, X)
print(f"features {X.shape}", flush=True)
om = np.isin(y, list(M.OBJECT))
folds = list(GroupKFold(n_splits=4).split(X, y, groups))

print(f"\n{'config':26s} {'micro':>8s} {'object':>8s} {'motion':>8s} {'MB':>7s} {'fit_s':>6s}")
for trees, depth in [(1000, None), (300, None), (300, 16), (300, 12), (200, 12), (150, 10)]:
    t0 = time.time()
    oof = np.zeros((len(y), 40), np.float32)
    for tr, va in folds:
        m = ExtraTreesClassifier(n_estimators=trees, max_depth=depth,
                                 n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])
        for j, c in enumerate(m.classes_):
            oof[va, c] = p[:, j]
    oof = (oof + SMOOTH) / (1.0 + 40.0 * SMOOTH)
    pred = oof.argmax(1)
    full = ExtraTreesClassifier(n_estimators=trees, max_depth=depth,
                                n_jobs=-1, random_state=0).fit(X, y)
    buf = io.BytesIO(); joblib.dump(full, buf, compress=3)
    name = f"{trees} trees, depth={depth}"
    print(f"{name:26s} {(pred==y).mean():8.5f} {(pred[om]==y[om]).mean():8.5f} "
          f"{(pred[~om]==y[~om]).mean():8.5f} {buf.tell()/1e6:7.2f} {time.time()-t0:6.0f}",
          flush=True)
    np.savez_compressed(f"{ART}/oof_imu_stats_t{trees}_d{depth}.npz",
                        probs=oof, sids=np.array(sids, dtype="<U15"), labels=y)
