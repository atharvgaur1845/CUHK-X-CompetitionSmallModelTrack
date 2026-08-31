#!/usr/bin/env python3
"""Q-91 SIM: does per-user self-training help? Simulate on CV val users (CPU).
For each fold's val USERS separately: pseudo-label their (unlabeled) clips with the fold
checkpoint, fine-tune a copy on confident pseudo-labels, re-evaluate on that user.
Reports per-user delta. This measures the transduction multiplier without submissions.
"""
import copy
import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import har_data
import har_models

# Portable root: env CUHKX_ROOT wins, else the repo dir two levels up from this
# file. Was a hardcoded absolute path (with a space in it) in 11 files, which was
# the #1 blocker for running anywhere but the original laptop.
ROOT = os.environ.get("CUHKX_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
TAU = 0.60          # pseudo-label confidence threshold
FT_EPOCHS = 15
FT_LR = 2e-4
torch.set_num_threads(10)

user_of = {}
with open(os.path.join(ROOT, "cache", "meta_train.csv")) as f:
    for r in csv.DictReader(f):
        user_of[r["sample_id"]] = r["user"]

def predict(model, ds, idx=None):
    dl = DataLoader(ds if idx is None else Subset(ds, idx), 128, num_workers=4)
    P, Y = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in dl:
            P.append(torch.softmax(model(x), 1))
            Y.append(y)
    return torch.cat(P), torch.cat(Y)

rows = []
for fold in range(4):
    ds = har_data.HARDataset("train", "skel", fold, "val")
    base = har_models.build("skel")
    base.load_state_dict(torch.load(os.path.join(CKPT, f"skel_noaug_f{fold}.pt"), map_location="cpu"))
    by_user = defaultdict(list)
    for i, s in enumerate(ds.ids):
        by_user[user_of[s]].append(i)
    for u, idx in sorted(by_user.items()):
        P, Y = predict(base, ds, idx)
        acc0 = (P.argmax(1) == Y).float().mean().item()
        conf, plab = P.max(1)
        keep = (conf >= TAU).nonzero().flatten()
        model = copy.deepcopy(base)
        if len(keep) >= 10:
            opt = torch.optim.AdamW(model.parameters(), FT_LR, weight_decay=0.01)
            crit = nn.CrossEntropyLoss(label_smoothing=0.1)
            sub = Subset(ds, [idx[i] for i in keep.tolist()])
            lab = plab[keep]
            model.train()
            for ep in range(FT_EPOCHS):
                dl = DataLoader(list(zip(range(len(sub)), lab.tolist())), 32, shuffle=True)
                for bi, by in dl:
                    x = torch.stack([sub[i][0] for i in bi.tolist()])
                    loss = crit(model(x), by)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        P1, _ = predict(model, ds, idx)
        acc1 = (P1.argmax(1) == Y).float().mean().item()
        pl_acc = (plab[keep] == Y[keep]).float().mean().item() if len(keep) else float("nan")
        rows.append((fold, u, len(idx), acc0, acc1, acc1 - acc0, len(keep), pl_acc))
        print(f"f{fold} {u}: n={len(idx)} base {acc0:.3f} -> adapted {acc1:.3f} (Δ{acc1-acc0:+.3f}) "
              f"| pseudo n={len(keep)} acc {pl_acc:.3f}", flush=True)

d = np.array([r[5] for r in rows])
print(f"\nSUMMARY: mean Δ {d.mean():+.4f} · median {np.median(d):+.4f} · helped {sum(d>0)}/{len(d)} users "
      f"· clip-weighted Δ {np.average(d, weights=[r[2] for r in rows]):+.4f}")
