#!/usr/bin/env python3
"""L3b probe — can a COCO detector name the object in the hand, on IR frames?

EXP-068 established the visual branch is a MOTION model that learned essentially no
appearance ("the objects ARE in the pixels ... the network latched onto motion").
EXP-067 established that 75% of residual error is object-identity confusion inside an
identical posture: Drink_water vs Eat_food, Read_documents vs Turn_pages,
Play_games vs Use_a_mobile_phone.

A COCO-pretrained detector already names cup / bottle / book / laptop / cell phone /
spoon / fork / bowl. R-1 makes a small pretrained CNN legal, and YOLO11n is ALREADY in
the pipeline for the person crop. So the question is purely empirical and costs 20
minutes: does a detector fire on these objects in 640x480 IR, and do its firings
DISCRIMINATE the confused pair?

Kill rule (pre-registered): if the pair-discrimination AUC is < 0.60 on the pooled
confusable pairs, the branch is dead and no model gets built.
"""
from __future__ import annotations
import argparse, collections, csv, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IR_ROOT = os.path.join(ROOT, "Small-Model-Track", "Training", "data", "HAR", "data", "IR")
ART = os.path.join(ROOT, "research", "artifacts")

# COCO ids that could name a hand-held object in these 40 activities.
KEEP = {39: "bottle", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
        46: "banana", 47: "apple", 49: "orange", 62: "tv", 63: "laptop", 64: "mouse",
        65: "remote", 66: "keyboard", 67: "cell phone", 73: "book", 74: "clock",
        76: "scissors", 79: "toothbrush", 24: "backpack", 26: "handbag", 60: "dining table",
        56: "chair", 57: "couch", 59: "bed", 68: "microwave", 69: "oven", 71: "sink",
        72: "refrigerator", 75: "vase"}


def sorted_frames(d):
    import re
    if not os.path.isdir(d):
        return []
    fr = re.compile(r"_(\d+)\.png$")
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png")]
    return sorted(fs, key=lambda p: int(m.group(1)) if (m := fr.search(p)) else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--per-class", type=int, default=12, help="clips sampled per activity")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.10, help="low: we want recall, then rank")
    ap.add_argument("--out", default="objprobe")
    a = ap.parse_args()

    # The confusable pairs that actually cost us clips (EXP-132 champion OOF).
    z = np.load(os.path.join(ART, "exp132_champion_oof.npz"), allow_pickle=True)
    y, pred = z["y"], z["z"].argmax(1)
    conf = collections.Counter()
    for t, p in zip(y, pred):
        if t != p:
            conf[(min(t, p), max(t, p))] += 1
    pairs = [p for p, _ in conf.most_common(10)]
    focus = sorted({c for p in pairs for c in p})
    print("top-10 confusion pairs (champion OOF):")
    for (i, j), n in conf.most_common(10):
        print(f"   {i:2d} <-> {j:2d}   {n} errors")

    meta = []
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        for r in csv.DictReader(h):
            if int(r["class_id"]) in focus:
                meta.append((r["sample_id"], int(r["class_id"]), r["user"], r["trial"]))
    by_cls = collections.defaultdict(list)
    for m in meta:
        by_cls[m[1]].append(m)
    rng = np.random.default_rng(0)
    sample = []
    for c in focus:
        rows = by_cls[c]
        idx = rng.choice(len(rows), size=min(a.per_class, len(rows)), replace=False)
        sample += [rows[i] for i in idx]
    print(f"\nsampling {len(sample)} clips over {len(focus)} classes, "
          f"{a.frames} IR frames each, model={a.model}")

    cls_dirs = {int(d.split("_")[0]): d for d in os.listdir(IR_ROOT)}
    from ultralytics import YOLO
    model = YOLO(a.model)

    feats, labels, kept = [], [], []
    for n, (sid, c, user, trial) in enumerate(sample):
        d = os.path.join(IR_ROOT, cls_dirs[c], user, trial)
        fs = sorted_frames(d)
        if not fs:
            continue
        idx = np.linspace(0, len(fs) - 1, min(a.frames, len(fs))).round().astype(int)
        imgs = [fs[i] for i in sorted(set(idx))]
        vec = np.zeros(len(KEEP))
        order = sorted(KEEP)
        for res in model.predict(imgs, conf=a.conf, verbose=False, device=0):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            cid = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for k, s in zip(cid, cf):
                if k in KEEP:
                    j = order.index(k)
                    vec[j] = max(vec[j], float(s))
        feats.append(vec); labels.append(c); kept.append(sid)
        if (n + 1) % 25 == 0:
            print(f"   {n+1}/{len(sample)}", flush=True)

    X = np.stack(feats); Y = np.array(labels)
    order = sorted(KEEP)
    names = [KEEP[k] for k in order]
    print(f"\nfired at all on {(X.max(1) > 0).mean():.3f} of clips")
    print("per-object detection rate (fraction of clips with any hit):")
    rate = (X > 0).mean(0)
    for j in np.argsort(-rate)[:14]:
        print(f"   {names[j]:14s} {rate[j]:.3f}")

    # The decisive number: does the detection vector separate the CONFUSED pair?
    print("\npair discrimination (single best object feature, AUC):")
    aucs = []
    for (i, j) in pairs:
        mi, mj = Y == i, Y == j
        if mi.sum() < 4 or mj.sum() < 4:
            continue
        best, bestname = 0.5, "-"
        for k in range(X.shape[1]):
            u, v = X[mi, k], X[mj, k]
            if u.max() == 0 and v.max() == 0:
                continue
            allv = np.concatenate([u, v])
            r = np.argsort(np.argsort(allv)) + 1
            n1 = len(u)
            auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(v))
            auc = max(auc, 1 - auc)
            if auc > best:
                best, bestname = auc, names[k]
        aucs.append(best)
        print(f"   {i:2d} vs {j:2d}   AUC {best:.3f}   via '{bestname}'   "
              f"(n={mi.sum()}/{mj.sum()})")
    m = float(np.mean(aucs)) if aucs else 0.5
    print(f"\nMEAN PAIR AUC {m:.3f}   -> {'ALIVE' if m >= 0.60 else 'DEAD (< 0.60 kill rule)'}")

    np.savez_compressed(os.path.join(ART, f"{a.out}_{os.path.splitext(a.model)[0]}.npz"),
                        X=X.astype(np.float32), y=Y, sids=np.array(kept), objects=np.array(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
