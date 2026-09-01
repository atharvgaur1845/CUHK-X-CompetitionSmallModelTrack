#!/usr/bin/env python3
"""EXP-088: the EXP-086 crop+video recipe applied to THERMAL only.

Separate file rather than a --cache flag on train_video_crop.py on purpose: that
script's --resume fingerprint is `vars(args)`, so adding an argument to it while
folds 0/1/3 are training would invalidate their resume state and throw away every
completed epoch on the next watchdog restart.

Everything except the input tensor is held identical to EXP-086 so the comparison
is single-change: r2plus1d_18 Kinetics-400, lr 5e-5, EMA 0.99, 30 epochs, last
epoch kept, clip-level augmentation only, logit-adjusted CE.

Thermal is 3-channel (the ironbow JPG is not cleanly invertible to a scalar, so it
stays RGB), which means the stem needs NO adaptation at all -- the Kinetics weights
are used exactly as pretrained. The 4-channel IR/depth member had to average the RGB
kernels for its extra channel; this member does not, so if it underperforms, stem
surgery is not the explanation.

Prior evidence on this modality:
  * dataset paper ranks thermal FIRST of six sensors (92.57 > depth 90.5 > IR 90.2)
  * both public notebooks discard it entirely
  * our own 2D ImageNet thermal member scores 0.34783 solo and degrades the fusion
    monotonically from weight 0.05 -- but that is the same 2D family the crop+video
    recipe beat by +23.9 micro on IR/depth, so it does not speak to this member.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_video_crop import OBJECT, CropClips, build_model, predict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache", "thermal_v1")
ART = os.path.join(ROOT, "research", "artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
# Kinetics statistics, unmodified: thermal enters as a 3-channel colormapped image.
KIN_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KIN_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def main() -> int:
    global CACHE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="vidth_f2")
    p.add_argument("--cache", default="thermal_v1",
                   help="cache dir under cache/. thermal_v1 = YOLO person crop (EXP-088, "
                        "fold-2 micro 0.54448); thermal_full = uncropped whole frame "
                        "(EXP-118 hypothesis). Default keeps EXP-088 reproducible.")
    p.add_argument("--fold-oof", default="oof_visual_mil_v1_f2.npz")
    p.add_argument("--arch", default="r2plus1d_18")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    CACHE = os.path.join(ROOT, "cache", args.cache)
    print(f"  cache: {CACHE}", flush=True)

    torch.manual_seed(20260730)
    np.random.seed(20260730)
    device = torch.device("cuda")

    index = json.load(open(os.path.join(CACHE, "train_index.json")))
    sids = index["sids"]
    row_of = {s: i for i, s in enumerate(sids)}
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        lab = {r["sample_id"]: int(r["class_id"]) for r in csv.DictReader(h)}
    ref = np.load(os.path.join(ART, args.fold_oof), allow_pickle=True)
    val_sids = [s for s in (str(x) for x in ref["sids"]) if s in row_of]
    vset = set(val_sids)
    tr_sids = [s for s in sids if s in lab and s not in vset]

    memmap = np.load(os.path.join(CACHE, "train.npy"), mmap_mode="r")
    # 145 of 2933 clips have no thermal at all; training on all-zero tensors teaches
    # the model that "blank" is evidence for the majority class. Drop them from train,
    # but KEEP them in the outer set so the reported number is over the same subjects
    # as every other member and stays comparable.
    def nonempty(s):
        return bool(memmap[row_of[s]].any())
    dropped = [s for s in tr_sids if not nonempty(s)]
    tr_sids = [s for s in tr_sids if s not in set(dropped)]
    tr_rows = np.array([row_of[s] for s in tr_sids])
    va_rows = np.array([row_of[s] for s in val_sids])
    tr_lab = np.array([lab[s] for s in tr_sids])
    va_lab = np.array([lab[s] for s in val_sids])

    counts = np.bincount(tr_lab, minlength=40).astype(np.float64)
    log_prior = torch.tensor(np.log(np.maximum(counts / counts.sum(), 1e-9)),
                             dtype=torch.float32, device=device)

    tl = DataLoader(CropClips(memmap, tr_rows, tr_lab, True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=False)
    vl = DataLoader(CropClips(memmap, va_rows, va_lab, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=False)

    model = build_model(args.arch, in_channels=3).to(device)
    n = sum(q.numel() for q in model.parameters())
    print(f"{args.tag}: arch={args.arch} train={len(tr_rows)} (dropped {len(dropped)} "
          f"thermal-empty) outer={len(va_rows)} params={n:,} fp16={n*2/1e6:.1f}MB", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, len(tl) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps,
        pct_start=max(args.warmup / max(args.epochs, 1), 0.05))
    scaler = torch.amp.GradScaler("cuda")
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)

    resume_path = os.path.join(CKPT, f"{args.tag}_resume.pt")
    fingerprint = {k: v for k, v in vars(args).items() if k != "resume"}
    start_epoch = 1
    if args.resume and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("fingerprint") == fingerprint:
            model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
            start_epoch = int(state["epoch"]) + 1
            for _ in range(min((start_epoch - 1) * steps, args.epochs * steps - 1)):
                sched.step()
            print(f"  resumed from epoch {state['epoch']}", flush=True)
        else:
            print("  resume state recipe differs; starting fresh", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total, seen, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tl):
            x = ((x.to(device, non_blocking=True) - KIN_MEAN.to(device))
                 / KIN_STD.to(device))
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x) + log_prior, y,
                                       label_smoothing=args.label_smoothing)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < args.epochs * steps - 1:
                    sched.step()
                with torch.no_grad():
                    for e, m in zip(ema.parameters(), model.parameters()):
                        e.mul_(args.ema).add_(m.detach(), alpha=1 - args.ema)
                    for e, m in zip(ema.buffers(), model.buffers()):
                        e.copy_(m)
            total += float(loss.detach()) * y.numel(); seen += y.numel()
        tmp = resume_path + ".tmp"
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "epoch": epoch, "fingerprint": fingerprint}, tmp)
        os.replace(tmp, resume_path)
        if epoch % 3 == 0 or epoch in (1, args.epochs):
            print(f"  epoch {epoch:02d}/{args.epochs}: loss={total/seen:.5f} "
                  f"lr={sched.get_last_lr()[0]:.2e} [{time.time()-t0:.0f}s]", flush=True)

    probs, labels = predict(ema, vl, device, mean=KIN_MEAN, std=KIN_STD)
    pred = probs.argmax(1)
    om = np.isin(labels, OBJECT)
    micro = float((pred == labels).mean())
    obj = float((pred[om] == labels[om]).mean())
    mot = float((pred[~om] == labels[~om]).mean())
    print(f"{args.tag} OUTER-ONCE: micro={micro:.5f} object={obj:.5f} "
          f"({int((pred[om]==labels[om]).sum())}/{int(om.sum())}) "
          f"gross_motion={mot:.5f}", flush=True)

    torch.save({"state_dict": ema.state_dict(), "args": vars(args),
                "micro": micro, "object": obj}, os.path.join(CKPT, f"{args.tag}.pt"))
    np.savez_compressed(os.path.join(ART, f"oof_{args.tag}.npz"),
                        probs=probs.astype(np.float32),
                        sids=np.array(val_sids, dtype="<U15"), labels=labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
