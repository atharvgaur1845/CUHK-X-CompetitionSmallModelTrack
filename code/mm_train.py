#!/usr/bin/env python3
"""M-02 trainer: fusion transformer on MMDataset. Gate: beat nested late fusion (56.15)."""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mm_data
import mm_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "research", "artifacts", "results.csv")
CKPT = os.path.join(ROOT, "checkpoints")


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def run_fold(args, fold):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    tr = mm_data.MMDataset("train", fold, "train", aug=True, seed=args.seed, roi=args.roi)
    va = mm_data.MMDataset("train", fold, "val", roi=args.roi)
    # Workers must restart each epoch so the updated dataset seed reaches them.
    lt = DataLoader(tr, args.bs, shuffle=True, num_workers=4, drop_last=True,
                    pin_memory=True)
    lv = DataLoader(va, args.bs, num_workers=3)
    model = mm_model.MMFusion(d=args.dim, layers=args.layers).to(dev)
    if args.init:
        sd = torch.load(os.path.join(CKPT, args.init + ".pt"), map_location=dev)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  init from {args.init}: {len(sd) - len(unexpected)} tensors loaded", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(lt), pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler()
    best, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        tr.seed = args.seed * 1000 + ep
        for b, y, _ in lt:
            b, y = to_dev(b, dev), y.to(dev)
            with torch.amp.autocast("cuda"):
                out = model(b)
                loss = crit(out["logits"], y) + 0.3 * sum(crit(out[m], y) for m in ("skel", "imu", "vis"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        hit = n = 0
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for b, y, _ in lv:
                p = model(to_dev(b, dev), train_mode=False)["logits"].float().argmax(1).cpu()
                hit += (p == y).sum().item()
                n += len(y)
        acc = hit / n
        if acc > best:
            best, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"  fold{fold} ep{ep + 1}: val {acc:.4f} (best {best:.4f})", flush=True)
    torch.save(best_state, os.path.join(CKPT, f"{args.tag}_f{fold}.pt"))
    return best, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mmfuse")
    ap.add_argument("--exp", default="EXP-020")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--roi", default="droi")
    ap.add_argument("--init", default="", help="checkpoint name to init trunk from (SSL)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="0,2")
    args = ap.parse_args()
    accs, t0 = [], time.time()
    for f in [int(x) for x in args.folds.split(",")]:
        acc, n_par = run_fold(args, f)
        accs.append(acc)
        print(f"fold {f}: {acc:.4f}", flush=True)
    mean = float(np.mean(accs))
    row = dict(exp=args.exp, tag=args.tag, modality="mm", aug=1, epochs=args.epochs,
               seed=args.seed, folds=args.folds, accs=json.dumps([round(a, 4) for a in accs]),
               mean=round(mean, 4), std=round(float(np.std(accs)), 4), params=n_par,
               mins=round((time.time() - t0) / 60, 1))
    with open(RESULTS, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writerow(row)
    print(f"RESULT {args.exp} {args.tag}: mean {mean:.4f} ({n_par / 1e6:.2f}M params)")


if __name__ == "__main__":
    main()
