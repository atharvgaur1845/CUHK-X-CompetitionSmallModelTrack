#!/usr/bin/env python3
"""EXP-141 -- leave-one-subject-out for the IMU ExtraTrees member (CPU only).

The video views' LOSO runs on the cluster; this member is 18 sklearn fits and does not
need a GPU. Feature matrix is cached because building it re-reads 2,933 clip archives.
"""
from __future__ import annotations
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imu_stats_member import build_matrix, OBJECT   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
CACHE = os.path.join(ART, "imu_feats_train.npz")


def main() -> int:
    meta = [r for r in csv.DictReader(open(os.path.join(ROOT, "cache", "meta_train.csv")))
            if os.path.exists(os.path.join(ROOT, "cache", "train", f"{r['sample_id']}.npz"))]
    sids = [r["sample_id"] for r in meta]
    y = np.array([int(r["class_id"]) for r in meta])
    g = np.array([r["user"] for r in meta])

    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        assert [str(s) for s in d["sids"]] == sids, "cached features are stale"
        X = d["X"]
        print(f"loaded cached features {X.shape}")
    else:
        X = build_matrix(sids, "train")
        np.savez_compressed(CACHE, X=X, sids=np.array(sids, dtype="<U15"))
        print(f"built and cached features {X.shape}")

    from sklearn.ensemble import ExtraTreesClassifier
    oof = np.zeros((len(y), 40), dtype=np.float32)
    for u in sorted(set(g)):
        te = g == u
        m = ExtraTreesClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=0)
        m.fit(X[~te], y[~te])
        p = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_):
            oof[te, c] = p[:, j]
        print(f"  {u:8s} n={te.sum():4d} acc={(oof[te].argmax(1)==y[te]).mean():.5f}", flush=True)
    oof = (oof + 1e-3) / (1.0 + 40.0 * 1e-3)
    np.savez_compressed(os.path.join(ART, "oof_loso_imu_stats.npz"),
                        probs=oof, sids=np.array(sids, dtype="<U15"), labels=y)
    print(f"pooled LOSO micro={(oof.argmax(1)==y).mean():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
