#!/usr/bin/env python3
"""EXP-087: IMU member as engineered statistics + trees, not a sequence net.

Measured motivation (data-probe agent, subject-grouped CV on the identical 2700
clips our neural IMU models cover):

    oof_imu_lstm   0.2504     oof_imu_aug   0.2674
    oof_imu_inv4   0.2963     oof_imu_world 0.3207   <- our best neural IMU
    stats + ExtraTrees                      0.4067   <- this file

The IMU stream is ~10.8 Hz with a MEDIAN OF 23 SAMPLES PER DEVICE PER CLIP. That
is far too short for a recurrent/temporal-conv model to earn its parameters, and
it is exactly the regime where fixed descriptors + a variance-reducing tree
ensemble win. Our neural IMU members were the wrong model class for the data.

Two deliberate design choices, both measured:

* CHANNELS 0-5 ONLY (accelerometer XYZ, gyroscope XYZ). The stream also carries
  absolute Euler angle (6-8), magnetometer (9-11) and quaternion (12-15), and
  DROPPING them IMPROVES accuracy (0.3856 -> 0.4030, sedentary 0.2697 -> 0.2939).
  Absolute orientation and magnetic heading encode where the subject was standing
  and which way they faced -- session/room nuisance that inflates local CV and
  cannot survive new subjects in a new room. Magnetometer alone still scores
  0.2728, which is heading leakage, not activity.
* Device placement is measured, not assumed: WTC chest, WTLA/WTRA left/right
  wrist, WTLL/WTRL left/right leg. Arm/leg gyro energy ratio is 0.87-2.3 on the
  gross-motion classes and 7-24 on the hand-object classes, and the two wrists
  alone recover 96% of the 5-device sedentary score.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
DEVICES = ("WTC", "WTLA", "WTRA", "WTLL", "WTRL")
USE_CHANNELS = 6            # acc XYZ + gyro XYZ; see docstring
OBJECT = np.array(sorted(set(range(28)) | {37, 38, 39}))


def channel_stats(v: np.ndarray) -> list[float]:
    """13 descriptors for one channel of one device."""
    if v.size == 0:
        return [0.0] * 13
    d = np.diff(v) if v.size > 1 else np.zeros(1)
    centred = v - v.mean()
    zcr = float((np.diff(np.sign(centred)) != 0).mean()) if v.size > 2 else 0.0
    spec = np.abs(np.fft.rfft(centred)) if v.size > 3 else np.zeros(2)
    low = float(spec[: max(1, len(spec) // 4)].sum() / (spec.sum() + 1e-9))
    peak = float(np.argmax(spec[1:]) + 1) if len(spec) > 2 else 0.0
    return [float(v.mean()), float(v.std()), float(v.min()), float(v.max()),
            float(np.percentile(v, 25)), float(np.percentile(v, 75)),
            float(v[-1] - v[0]), float(np.abs(d).mean()), float(d.std()),
            float(np.abs(d).max()), zcr, low, peak]


def clip_features(data) -> np.ndarray:
    feats: list[float] = []
    norms: dict[str, np.ndarray] = {}
    for dev in DEVICES:
        key = f"imu_{dev}_x"
        x = np.asarray(data[key], dtype=np.float64) if key in data.files else np.zeros((0, 16))
        x = x[:, :USE_CHANNELS]
        for c in range(USE_CHANNELS):
            feats.extend(channel_stats(x[:, c] if x.size else np.zeros(0)))
        acc = np.linalg.norm(x[:, 0:3], axis=1) if x.size else np.zeros(0)
        gyr = np.linalg.norm(x[:, 3:6], axis=1) if x.size else np.zeros(0)
        norms[dev] = acc
        feats.extend(channel_stats(acc))
        feats.extend(channel_stats(gyr))
        feats.append(float(x.shape[0]))
    # Cross-device coupling on a fixed-length grid: which limbs move together is
    # what separates two-handed from one-handed object manipulation.
    grid = np.linspace(0, 1, 32)
    resampled = {}
    for dev, v in norms.items():
        if v.size > 1:
            resampled[dev] = np.interp(grid, np.linspace(0, 1, v.size), v)
        else:
            resampled[dev] = np.zeros(32)
    for i, a in enumerate(DEVICES):
        for b in DEVICES[i + 1:]:
            u, w = resampled[a], resampled[b]
            if u.std() > 1e-9 and w.std() > 1e-9:
                feats.append(float(np.corrcoef(u, w)[0, 1]))
            else:
                feats.append(0.0)
            feats.append(float(np.abs(u - w).mean()))
    return np.nan_to_num(np.array(feats, dtype=np.float32))


def build_matrix(sids, split):
    rows = []
    for i, sid in enumerate(sids):
        path = os.path.join(ROOT, "cache", split, f"{sid}.npz")
        with np.load(path) as data:
            rows.append(clip_features(data))
        if i % 500 == 0:
            print(f"  {split} {i}/{len(sids)}", flush=True)
    return np.stack(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="imu_stats")
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--smooth", type=float, default=1e-3,
                    help=("Laplace smoothing. ExtraTrees emits EXACT zeros "
                          "(measured: 2382 cells, 392/405 rows), and a zero in "
                          "a geometric-mean fusion vetoes that class for the "
                          "whole ensemble -- the defect that scored 0.29 once "
                          "already. Never publish an unsmoothed tree member."))
    args = ap.parse_args()

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import GroupKFold

    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        meta = list(csv.DictReader(h))
    meta = [r for r in meta
            if os.path.exists(os.path.join(ROOT, "cache", "train", f"{r['sample_id']}.npz"))]
    sids = [r["sample_id"] for r in meta]
    y = np.array([int(r["class_id"]) for r in meta])
    groups = np.array([r["user"] for r in meta])
    print(f"train clips={len(sids)} users={len(set(groups))}", flush=True)

    X = build_matrix(sids, "train")
    print(f"feature matrix {X.shape}", flush=True)

    oof = np.zeros((len(y), 40), dtype=np.float32)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=4).split(X, y, groups)):
        model = ExtraTreesClassifier(n_estimators=args.trees, n_jobs=-1, random_state=0)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[va])
        for j, c in enumerate(model.classes_):
            oof[va, c] = proba[:, j]
        acc = (oof[va].argmax(1) == y[va]).mean()
        print(f"  fold {fold}: n={len(va)} acc={acc:.4f}", flush=True)

    oof = (oof + args.smooth) / (1.0 + 40.0 * args.smooth)
    pred = oof.argmax(1)
    om = np.isin(y, OBJECT)
    print(f"{args.tag} OOF: micro={(pred==y).mean():.5f} "
          f"object={(pred[om]==y[om]).mean():.5f} "
          f"motion={(pred[~om]==y[~om]).mean():.5f}", flush=True)
    np.savez_compressed(os.path.join(ART, f"oof_{args.tag}.npz"),
                        probs=oof, sids=np.array(sids, dtype="<U15"), labels=y)

    sample = os.path.join(ROOT, "Small-Model-Track", "Testing", "sample_submission.csv")
    with open(sample) as h:
        body = list(csv.reader(h))[1:]
    test_sids = [r[0].strip("/").split("/")[-1] for r in body]
    Xt = build_matrix(test_sids, "test")
    final = ExtraTreesClassifier(n_estimators=args.trees, n_jobs=-1, random_state=0)
    final.fit(X, y)
    proba = final.predict_proba(Xt)
    test_probs = np.zeros((len(test_sids), 40), dtype=np.float32)
    for j, c in enumerate(final.classes_):
        test_probs[:, c] = proba[:, j]
    test_probs = (test_probs + args.smooth) / (1.0 + 40.0 * args.smooth)
    assert (test_probs > 0).all(), "smoothing failed to remove zero cells"
    np.savez_compressed(os.path.join(ART, f"testprobs_{args.tag}.npz"),
                        probs=test_probs.astype(np.float32),
                        sids=np.array(test_sids, dtype="<U15"))
    print(f"wrote testprobs_{args.tag}.npz  distinct argmax="
          f"{len(set(test_probs.argmax(1).tolist()))}/40", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
