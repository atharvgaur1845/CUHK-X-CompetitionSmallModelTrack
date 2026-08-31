#!/usr/bin/env python3
"""Q-68: simulate recording-group Hungarian distinctness assignment on train OOF preds.

Groups = trials sharing a Radar filename timestamp (user-pure, class-distinct; EXP-000c).
Compares per-clip argmax accuracy vs per-group linear_sum_assignment accuracy.
Usage: python3 hungarian_sim.py <tag> (checkpoint prefix, e.g. skel_noaug)
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

import har_data
import har_models

# Portable root: env CUHKX_ROOT wins, else the repo dir two levels up from this
# file. Was a hardcoded absolute path (with a space in it) in 11 files, which was
# the #1 blocker for running anywhere but the original laptop.
ROOT = os.environ.get("CUHKX_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RADAR = os.path.join(ROOT, "Small-Model-Track/Training/data/HAR/data/Radar")
CKPT = os.path.join(ROOT, "checkpoints")

MOD = {"skel": "skel", "imu": "imu", "depth": "depth", "ir": "ir"}


def radar_groups():
    """sample_id -> group key (radar filename ts), for all train trials."""
    key = {}
    for cls in os.listdir(RADAR):
        cid = int(cls.split("_")[0])
        croot = os.path.join(RADAR, cls)
        for user in os.listdir(croot):
            for trial in os.listdir(os.path.join(croot, user)):
                for f in os.listdir(os.path.join(croot, user, trial)):
                    m = re.search(r"T(.+)\.csv", f)
                    if m:
                        key[f"{cid:02d}_{user}_{trial}"] = m.group(1)
    return key


def main(tag):
    modality = next(m for m in MOD if tag.startswith(m))
    gkey = radar_groups()
    probs, labels, sids = [], [], []
    for fold in range(4):
        ds = har_data.HARDataset("train", modality, fold, "val")
        model = har_models.build(modality)
        model.load_state_dict(torch.load(os.path.join(CKPT, f"{tag}_f{fold}.pt"), map_location="cpu"))
        model.eval()
        with torch.no_grad():
            for x, y, s in DataLoader(ds, 128, num_workers=4):
                probs.append(torch.softmax(model(x), 1).numpy())
                labels.append(y.numpy())
                sids.extend(s)
    P = np.concatenate(probs)
    Y = np.concatenate(labels)
    np.savez_compressed(os.path.join(ROOT, "research", "artifacts", f"oof_{tag}.npz"),
                        probs=P, labels=Y, sids=np.array(sids))
    base = (P.argmax(1) == Y).mean()
    # group and assign
    groups = defaultdict(list)
    for i, s in enumerate(sids):
        groups[gkey.get(s, f"solo_{s}")].append(i)
    hung_pred = P.argmax(1).copy()
    n_multi = n_collide = 0
    for g, idx in groups.items():
        if len(idx) < 2:
            continue
        n_multi += len(idx)
        sub = P[idx]
        if len(set(sub.argmax(1))) < len(idx):
            n_collide += 1
        r, c = linear_sum_assignment(-np.log(sub + 1e-9))
        for k, cls in zip(r, c):
            hung_pred[idx[k]] = cls
    hung = (hung_pred == Y).mean()
    multi_mask = np.zeros(len(Y), bool)
    for g, idx in groups.items():
        if len(idx) > 1:
            multi_mask[idx] = True
    print(f"tag={tag}  OOF n={len(Y)}")
    print(f"argmax acc:    {base:.4f}")
    print(f"hungarian acc: {hung:.4f}  (delta {hung - base:+.4f})")
    print(f"multi-group clips: {multi_mask.sum()} ({multi_mask.mean():.1%}); groups with argmax collision: {n_collide}")
    print(f"on multi-group clips only: argmax {(P.argmax(1) == Y)[multi_mask].mean():.4f} -> hungarian {(hung_pred == Y)[multi_mask].mean():.4f}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main(sys.argv[1])
