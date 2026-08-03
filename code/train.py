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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "research", "artifacts", "results.csv")
CKPT = os.path.join(ROOT, "checkpoints")


def run_fold(args, fold):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aug = args.aug_spec if args.aug_spec else args.aug
    kw2 = dict(feat=args.feat, n_frames=args.n_frames, person=args.person,
               imu_feat=args.imu_feat, t_skel=args.t_skel, skel_time=args.skel_time)
    tr = har_data.HARDataset("train", args.modality, fold, "train", aug=aug, seed=args.seed, **kw2)
    va = har_data.HARDataset("train", args.modality, fold, "val", aug=False, **kw2)
    if args.balanced:
        from collections import Counter
        cnt = Counter(tr.labels[s] for s in tr.ids)
        wts = torch.tensor([1.0 / cnt[tr.labels[s]] for s in tr.ids], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(wts, len(tr.ids), replacement=True)
        lt = DataLoader(tr, args.bs, sampler=sampler, num_workers=args.workers,
                        drop_last=True, pin_memory=dev == "cuda")
    else:
        lt = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=dev == "cuda")
    # Do not use persistent workers here. epoch_seed changes every epoch and worker
    # dataset copies must observe it; otherwise every clip gets one frozen augmented
    # view for the entire run.
    lv = DataLoader(va, args.bs, num_workers=max(0, args.workers // 2),
                    pin_memory=dev == "cuda")
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
    domain_crit = nn.CrossEntropyLoss()
    if args.dann > 0 and args.mixup > 0:
        raise ValueError("--dann and --mixup cannot be combined in this isolated ablation")
    best, best_state, best_ep = 0.0, None, -1
    total_steps = args.epochs * len(lt)
    global_step = 0
    for ep in range(args.epochs):
        model.train()
        tr.epoch_seed = args.seed * 1000 + ep
        for x, y, sids in lt:
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            if args.dann > 0:
                progress = global_step / max(1, total_steps - 1)
                strength = args.dann * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
                out, domain_out = model.forward_domain(x, float(strength))
                domain_y = torch.tensor(
                    [int(tr.meta[s]["user"].removeprefix("user")) for s in sids],
                    device=dev,
                )
                loss = crit(out, y) + domain_crit(domain_out, domain_y)
            elif args.mixup > 0:
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
            global_step += 1
        model.eval()
        if len(va) == 0:  # refit-on-all mode: no val, keep last
            best, best_ep = 0.0, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
            best, best_ep = acc, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 9 or ep == args.epochs - 1:
            print(f"  fold{fold} ep{ep + 1}: micro {acc:.4f} macro {macro:.4f} (best {best:.4f})", flush=True)
    os.makedirs(CKPT, exist_ok=True)
    torch.save(best_state, os.path.join(CKPT, f"{args.tag}_f{fold}.pt"))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               os.path.join(CKPT, f"{args.tag}_f{fold}_last.pt"))
    return best, n_par, best_ep + 1


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
    ap.add_argument(
        "--aug-spec",
        default="",
        help=("comma list; shared: trunc,tjit; skeleton: rot,scale,jit,jdrop; "
              "IMU: iscale,inoise,chan; visual: crop,erase"),
    )
    ap.add_argument("--feat", default="jv", help="skel features: jv | jvb | jv+tn | jvb+tn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--width", type=int, default=0, help="model width override")
    ap.add_argument("--ls", type=float, default=0.1, help="label smoothing")
    ap.add_argument("--person", default="first", help="skel person policy: first|motion")
    ap.add_argument("--imu-feat", default="raw", help="imu features: raw|inv")
    ap.add_argument("--arch", default="", help="alt architecture (imu: lstm)")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--t-skel", type=int, default=32)
    ap.add_argument("--skel-time", choices=("stretch", "pad"), default="stretch")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup alpha (0=off)")
    ap.add_argument("--dann", type=float, default=0.0,
                    help="maximum gradient-reversal strength (0=off)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    accs, best_eps, t0 = [], [], time.time()
    for f in [x if x in ("rand", "all") else int(x) for x in args.folds.split(",")]:
        acc, n_par, best_ep = run_fold(args, f)
        accs.append(acc)
        best_eps.append(best_ep)
        print(f"fold {f}: {acc:.4f} at epoch {best_ep}", flush=True)
    mean, std = float(np.mean(accs)), float(np.std(accs))
    row = dict(exp=args.exp, tag=args.tag, modality=args.modality,
               aug=int(bool(args.aug or args.aug_spec)),
               epochs=args.epochs, seed=args.seed, folds=args.folds,
               accs=json.dumps([round(a, 4) for a in accs]), mean=round(mean, 4),
               std=round(std, 4), params=n_par, mins=round((time.time() - t0) / 60, 1))
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    manifest = {
        "exp": args.exp,
        "tag": args.tag,
        "args": vars(args),
        # args["aug"] is the store_true flag, NOT what the dataset received: line
        # 26 is `aug = args.aug_spec if args.aug_spec else args.aug`, so a bare
        # --aug-spec turns augmentation on while args.aug stays False.  Rebuilding
        # a recipe from args["aug"] alone silently drops augmentation (EXP-057
        # lost a 4-seed run to exactly that).  Record what the dataset got.
        "effective_aug": args.aug_spec if args.aug_spec else args.aug,
        "best_epochs": best_eps,
        "fold_accuracies": accs,
        "params": n_par,
    }
    safe_folds = args.folds.replace(",", "-")
    with open(os.path.join(os.path.dirname(RESULTS),
                           f"run_{args.exp}_{args.tag}_{safe_folds}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"RESULT {args.exp} {args.tag}: mean {mean:.4f} ± {std:.4f} ({n_par / 1e6:.2f}M params)")


if __name__ == "__main__":
    main()
