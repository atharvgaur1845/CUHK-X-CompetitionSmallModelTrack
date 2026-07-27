#!/usr/bin/env python3
"""Test inference + fusion + recording-group Hungarian + submission writer.

Usage: python3 predict_test.py --tags skel_noaug:1.0,depth_aug:1.0 [--no-hungarian] [--out sub.csv]
Per tag: average softmax probs over the 4 fold checkpoints; fuse tags by weighted
average; optionally apply per-recording-group distinct-label assignment (B-011);
write Kaggle submission (path,prediction).
"""
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

import har_data
import har_models

ROOT = "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
TEST = os.path.join(ROOT, "Small-Model-Track/Testing/data/small_model_track_test")
CKPT = os.path.join(ROOT, "checkpoints")
SUBS = os.path.join(ROOT, "submissions")


def test_groups(sids):
    """sm_id -> group key: radar filename ts; fallback = camera anchor (t0 - idx/10) bucket."""
    key = {}
    for sid in sids:
        rdir = os.path.join(TEST, sid, "Radar")
        ts = None
        if os.path.isdir(rdir):
            for f in os.listdir(rdir):
                m = re.search(r"T(.+)\.csv", f)
                if m:
                    ts = "R" + m.group(1)
                    break
        key[sid] = ts
    # camera-anchor fallback for clips with no radar file
    need = [s for s, k in key.items() if k is None]
    if need:
        import datetime
        anchors = {}
        for sid in sids:
            d = os.path.join(TEST, sid, "IR")
            if not os.path.isdir(d):
                continue
            fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
            m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}\.\d+)_(\d{8})", fs[0])
            if m:
                t = datetime.datetime.strptime(
                    m.group(1) + " " + m.group(2).replace("-", ":"), "%Y-%m-%d %H:%M:%S.%f").timestamp()
                anchors[sid] = t - int(m.group(3)) / 10.0
        for sid in need:
            a = anchors.get(sid)
            if a is None:
                key[sid] = "solo_" + sid
                continue
            match = [s for s, k in key.items() if k and k.startswith("R") and s in anchors
                     and abs(anchors[s] - a) < 2.0]
            key[sid] = key[match[0]] if match else "A%.0f" % a
    return key


def predict_tag(tag, bs=64):
    modality = next(m for m in ("skel", "imu", "depth", "ir") if tag.startswith(m))
    ds = har_data.HARDataset("test", modality)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acc = np.zeros((len(ds), 40), np.float64)
    for fold in range(4):
        model = har_models.build(modality).to(dev)
        model.load_state_dict(torch.load(os.path.join(CKPT, f"{tag}_f{fold}.pt"), map_location=dev))
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _, _ in DataLoader(ds, bs, num_workers=4):
                probs.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
        acc += np.concatenate(probs)
    return acc / 4.0, ds.ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="tag:weight,tag:weight,...")
    ap.add_argument("--no-hungarian", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fused, ids = None, None
    parts = [t.split(":") for t in args.tags.split(",")]
    for tag, w in parts:
        p, ids = predict_tag(tag)
        np.savez_compressed(os.path.join(ROOT, "research", "artifacts", f"testprobs_{tag}.npz"),
                            probs=p, sids=np.array(ids))
        fused = p * float(w) if fused is None else fused + p * float(w)
        print(f"{tag} (w={w}) done")
    pred = fused.argmax(1)
    if not args.no_hungarian:
        gkey = test_groups(ids)
        groups = defaultdict(list)
        for i, s in enumerate(ids):
            groups[gkey[s]].append(i)
        changed = 0
        for g, idx in groups.items():
            if len(idx) < 2:
                continue
            r, c = linear_sum_assignment(-np.log(fused[idx] / fused[idx].sum(1, keepdims=True) + 1e-9))
            for k, cls in zip(r, c):
                if pred[idx[k]] != cls:
                    changed += 1
                pred[idx[k]] = cls
        print(f"hungarian: {len([g for g in groups.values() if len(g) > 1])} multi-groups, {changed} predictions changed")
    os.makedirs(SUBS, exist_ok=True)
    name = args.out or ("sub_" + "_".join(t for t, _ in parts) + ("_nohung" if args.no_hungarian else "_hung") + ".csv")
    out = os.path.join(SUBS, name)
    with open(out, "w") as f:
        f.write("path,prediction\n")
        for sid, p in zip(ids, pred):
            f.write(f"small_model_track_test/{sid}/,{p}\n")
    print(f"wrote {out} ({len(ids)} rows)")


if __name__ == "__main__":
    main()
