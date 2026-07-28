#!/usr/bin/env python3
"""Per-modality fold training. Appends results to research/artifacts/results.csv."""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import har_data
import har_models

ROOT = "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
RESULTS = os.path.join(ROOT, "research", "artifacts", "results.csv")
CKPT = os.path.join(ROOT, "checkpoints")


def run_fold(args, fold):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    aug = args.aug_spec if args.aug_spec else args.aug
    kw2 = dict(feat=args.feat, n_frames=args.n_frames, person=args.person, imu_feat=args.imu_feat)
    tr = har_data.HARDataset("train", args.modality, fold, "train", aug=aug, seed=args.seed, **kw2)
    va = har_data.HARDataset("train", args.modality, fold, "val", aug=False, **kw2)
    if args.balanced:
        from collections import Counter
        cnt = Counter(tr.labels[s] for s in tr.ids)
        wts = torch.tensor([1.0 / cnt[tr.labels[s]] for s in tr.ids], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(wts, len(tr.ids), replacement=True)
        lt = DataLoader(tr, args.bs, sampler=sampler, num_workers=6, drop_last=True, persistent_workers=True)
    else:
        lt = DataLoader(tr, args.bs, shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    lv = DataLoader(va, args.bs, num_workers=4)
    in_ch = tr.skel_dim() if args.modality == "skel" else (tr.imu_dim() if args.modality == "imu" else None)
    kw = {"width": args.width} if args.width else {}
    if args.arch:
        kw["arch"] = args.arch
    model = har_models.build(args.modality, in_ch=in_ch, **kw).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, epochs=args.epochs, steps_per_epoch=len(lt), pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)
    best, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        tr.epoch_seed = args.seed * 1000 + ep
        for x, y, _ in lt:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(x), device=dev)
                out = model(lam * x + (1 - lam) * x[perm])
                loss = lam * crit(out, y) + (1 - lam) * crit(out, y[perm])
            else:
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        if len(va) == 0:  # refit-on-all mode: no val, keep last
            best, best_state = 0.0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ep % 20 == 19:
                print(f"  fold{fold} ep{ep + 1}: (refit, no val)", flush=True)
            continue
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _ in lv:
                ps.append(model(x.to(dev)).argmax(1).cpu())
                ys.append(y)
        yv, pv = torch.cat(ys), torch.cat(ps)
        acc = (yv == pv).float().mean().item()
        macro = float(np.mean([(pv[yv == c] == c).float().mean().item()
                               for c in yv.unique().tolist()]))
        if acc > best:  # select on MICRO (test prior ≈ train prior — SUB-003/004)
            best, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 9 or ep == args.epochs - 1:
            print(f"  fold{fold} ep{ep + 1}: micro {acc:.4f} macro {macro:.4f} (best {best:.4f})", flush=True)
    os.makedirs(CKPT, exist_ok=True)
    torch.save(best_state, os.path.join(CKPT, f"{args.tag}_f{fold}.pt"))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               os.path.join(CKPT, f"{args.tag}_f{fold}_last.pt"))
    return best, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--balanced", action="store_true", help="class-balanced sampling")
    ap.add_argument("--aug-spec", default="", help="comma list: rot,scale,jit,jdrop,tjit")
    ap.add_argument("--feat", default="jv", help="skel features: jv | jvb | jv+tn | jvb+tn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--width", type=int, default=0, help="model width override")
    ap.add_argument("--ls", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--person", default="first", help="skel person policy: first|motion")
    ap.add_argument("--imu-feat", default="raw", help="imu features: raw|inv")
    ap.add_argument("--arch", default="", help="alt architecture (imu: lstm)")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0=off)")
    args = ap.parse_args()
    accs, t0 = [], time.time()
    for f in [x if x in ("rand", "all") else int(x) for x in args.folds.split(",")]:
        acc, n_par = run_fold(args, f)
        accs.append(acc)
        print(f"fold {f}: {acc:.4f}", flush=True)
    mean, std = float(np.mean(accs)), float(np.std(accs))
    row = dict(exp=args.exp, tag=args.tag, modality=args.modality, aug=int(args.aug),
               epochs=args.epochs, seed=args.seed, folds=args.folds,
               accs=json.dumps([round(a, 4) for a in accs]), mean=round(mean, 4),
               std=round(std, 4), params=n_par, mins=round((time.time() - t0) / 60, 1))
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"RESULT {args.exp} {args.tag}: mean {mean:.4f} ± {std:.4f} ({n_par / 1e6:.2f}M params)")


if __name__ == "__main__":
    main()
