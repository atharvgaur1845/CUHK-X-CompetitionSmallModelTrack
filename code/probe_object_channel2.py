#!/usr/bin/env python3
"""L3b probe, HONEST version. The first pass reported mean pair-AUC 1.000 and it was a bug.

Why the first pass was wrong, and it is worth recording:
  * it took max AUC over 30 object features on 12+12 clips -- selection over 30 features on
    24 points produces a near-perfect split from noise alone;
  * the winning features were semantically absurd ('tv' separating Read_documents from
    Turn_pages, 'cell phone' separating Sweep from Mop), which is the tell;
  * the top detections are FURNITURE -- chair .42, couch .39, sink .38, bed .28,
    refrigerator .23 -- i.e. the detector is naming the STATION, and each activity is
    performed at a fixed station. EXP-000c already measured station recovery as worth
    0.09 bits and it does not survive new subjects.

This version answers the question that actually matters: does the object vector predict the
pair label on HELD-OUT SUBJECTS? Same-subject evaluation cannot distinguish an object cue
from a station/user cue, and the whole competition is cross-subject.

Controls, all pre-registered:
  A  subject-grouped CV (train users disjoint from test users)
  B  label permutation within the pair -> must collapse to 0.5
  C  a NEGATIVE pair the object channel should not help on (28 Jog vs 30 Jumping_jacks)
  D  furniture-only vs handheld-only feature subsets -- if furniture carries it, it is
     station, not object
"""
from __future__ import annotations
import argparse, collections, csv, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IR_ROOT = os.path.join(ROOT, "Small-Model-Track", "Training", "data", "HAR", "data", "IR")
ART = os.path.join(ROOT, "research", "artifacts")

HANDHELD = {39: "bottle", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
            46: "banana", 47: "apple", 49: "orange", 63: "laptop", 64: "mouse",
            65: "remote", 66: "keyboard", 67: "cell phone", 73: "book", 76: "scissors",
            79: "toothbrush", 24: "backpack", 26: "handbag", 75: "vase"}
FURNITURE = {62: "tv", 56: "chair", 57: "couch", 59: "bed", 60: "dining table",
             68: "microwave", 69: "oven", 71: "sink", 72: "refrigerator", 74: "clock"}
KEEP = {**HANDHELD, **FURNITURE}


def sorted_frames(d):
    import re
    if not os.path.isdir(d):
        return []
    fr = re.compile(r"_(\d+)\.png$")
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")]
    return sorted(fs, key=lambda p: int(m.group(1)) if (m := fr.search(p)) else -1)


def auc_grouped(X, y, g):
    """Leave-one-user-group-out logistic AUC. Returns pooled AUC over held-out folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    users = np.array(sorted(set(g)))
    if len(users) < 4:
        return np.nan
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(users), 4)
    sc = np.zeros(len(y))
    for f in folds:
        te = np.isin(g, f)
        if te.all() or not te.any() or len(set(y[~te])) < 2:
            return np.nan
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X[~te], y[~te])
        sc[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, sc) if len(set(y)) == 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--max-per-class", type=int, default=60)
    a = ap.parse_args()

    z = np.load(os.path.join(ART, "exp132_champion_oof.npz"), allow_pickle=True)
    y_all, pred = z["y"], z["z"].argmax(1)
    conf = collections.Counter()
    for t, p in zip(y_all, pred):
        if t != p:
            conf[(min(t, p), max(t, p))] += 1
    pairs = [p for p, _ in conf.most_common(8)] + [(28, 30)]   # control C appended
    focus = sorted({c for p in pairs for c in p})

    rows = []
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        for r in csv.DictReader(h):
            if int(r["class_id"]) in focus:
                rows.append((r["sample_id"], int(r["class_id"]), r["user"], r["trial"],
                             r.get("station", "")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    rng = np.random.default_rng(0)
    sample = []
    for c in focus:
        rs = by[c]
        idx = rng.choice(len(rs), size=min(a.max_per_class, len(rs)), replace=False)
        sample += [rs[i] for i in idx]
    print(f"{len(sample)} clips, {len(focus)} classes, model={a.model}")

    cls_dirs = {int(d.split("_")[0]): d for d in os.listdir(IR_ROOT)}
    from ultralytics import YOLO
    model = YOLO(a.model)
    order = sorted(KEEP)
    names = [KEEP[k] for k in order]

    F, Y, U, S = [], [], [], []
    for n, (sid, c, user, trial, st) in enumerate(sample):
        fs = sorted_frames(os.path.join(IR_ROOT, cls_dirs[c], user, trial))
        if not fs:
            continue
        idx = np.linspace(0, len(fs) - 1, min(a.frames, len(fs))).round().astype(int)
        vec = np.zeros(len(order))
        for res in model.predict([fs[i] for i in sorted(set(idx))], conf=a.conf,
                                 verbose=False, device=0):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for k, s in zip(res.boxes.cls.cpu().numpy().astype(int),
                            res.boxes.conf.cpu().numpy()):
                if k in KEEP:
                    j = order.index(k)
                    vec[j] = max(vec[j], float(s))
        F.append(vec); Y.append(c); U.append(user); S.append(st)
        if (n + 1) % 100 == 0:
            print(f"   {n+1}/{len(sample)}", flush=True)
    X = np.stack(F); Y = np.array(Y); U = np.array(U); S = np.array(S)

    hand = np.array([i for i, k in enumerate(order) if k in HANDHELD])
    furn = np.array([i for i, k in enumerate(order) if k in FURNITURE])

    print("\n                          subject-grouped logistic AUC")
    print("pair        n     ALL    hand    furn    permuted   station-only-baseline")
    res = []
    for (i, j) in pairs:
        m = (Y == i) | (Y == j)
        if m.sum() < 20:
            continue
        yy = (Y[m] == j).astype(int)
        if len(set(yy)) < 2:
            continue
        xa, xh, xf = X[m], X[m][:, hand], X[m][:, furn]
        # station one-hot as the confound baseline
        st = S[m]
        xs = np.stack([(st == v).astype(float) for v in sorted(set(st))], 1)
        pm = yy.copy(); np.random.default_rng(0).shuffle(pm)
        a_all = auc_grouped(xa, yy, U[m]); a_h = auc_grouped(xh, yy, U[m])
        a_f = auc_grouped(xf, yy, U[m]); a_p = auc_grouped(xa, pm, U[m])
        a_s = auc_grouped(xs, yy, U[m])
        tag = "  <-- CONTROL" if (i, j) == (28, 30) else ""
        print(f"{i:2d} vs {j:2d}  {m.sum():4d}  {a_all:.3f}  {a_h:.3f}  {a_f:.3f}     "
              f"{a_p:.3f}       {a_s:.3f}{tag}")
        if (i, j) != (28, 30):
            res.append((a_all, a_h, a_f, a_s))
    r = np.array(res)
    print(f"\nMEAN over the 8 real pairs:  all {r[:,0].mean():.3f}   handheld {r[:,1].mean():.3f}"
          f"   furniture {r[:,2].mean():.3f}   station-only {r[:,3].mean():.3f}")
    print(f"\nKILL RULE: handheld-only AUC must be >= 0.60 AND must beat station-only.")
    verdict = "ALIVE" if (r[:,1].mean() >= 0.60 and r[:,1].mean() > r[:,3].mean()) else "DEAD"
    print(f"VERDICT: {verdict}")
    np.savez_compressed(os.path.join(ART, "objprobe_grouped.npz"),
                        X=X.astype(np.float32), y=Y, users=U, station=S, objects=np.array(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
