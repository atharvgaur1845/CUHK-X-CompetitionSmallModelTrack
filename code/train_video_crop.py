#!/usr/bin/env python3
"""EXP-086: Kinetics-pretrained video backbone on the YOLO person-crop cache.

Ports the published LB 0.711 recipe (`phuongncn/lb-0-711-yolo-person-crop-r2plus1d-100mb`,
verified 143/201 vs our 131) with its hyperparameters taken as given, because it is
measured and we are not. Deviation from the notebook is limited to the backbone:
it uses R(2+1)D-34 pretrained on IG-65M then Kinetics-400 (63M params, MoabitCoin
weights); torchvision ships r2plus1d_18/mc3_18/r3d_18 with Kinetics-400 only, which
is the closest legally-available equivalent and is selectable via --arch.

Recipe held from the notebook, each with its stated reason:
  * lr 5e-5 -- "Anything near the usual 3e-4 destroys the pretrained features
    within a couple of epochs". This is 4x below what we used for ResNet.
  * EMA 0.99, and the LAST epoch is kept, never the best -- avoids selecting on a
    validation curve, which on this competition is worth ~9-10 public clips of noise.
  * hflip p=0.5 + random resized crop 0.75-1.0 applied IDENTICALLY to all 16 frames
    -- "per-frame jitter would inject fake motion", i.e. it would corrupt the signal
    the 3D kernels exist to read.
  * 4-channel stem: pretrained RGB kernels copied for Depth_Color, and the IR kernel
    initialized to the MEAN of the RGB kernels so it starts at the same scale.

Why this direction at all: EXP-083 measured spatial resolution as binding, and the
notebook's own rationale ("the background differs systematically between rooms.
Cropping around the person removes that background shortcut") matches our measured
day/session shift -- every visual model loses 3.7-10.9 points of calibrated
confidence on unseen recording days while the skeleton model loses none.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache", "crop_v1")
ART = os.path.join(ROOT, "research", "artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
OBJECT = np.array(sorted(set(range(28)) | {37, 38, 39}))
# Kinetics normalization; the IR channel takes the mean of the RGB statistics so
# the adapted stem sees all four channels on a comparable scale.
KIN_MEAN = torch.tensor([0.43216, 0.394666, 0.37645, 0.401092]).view(1, 4, 1, 1, 1)
KIN_STD = torch.tensor([0.22803, 0.22145, 0.216989, 0.222156]).view(1, 4, 1, 1, 1)


class CropClips(Dataset):
    def __init__(self, memmap, rows, labels, train):
        self.memmap, self.rows, self.labels, self.train = memmap, rows, labels, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.memmap[self.rows[i]])).float() / 255.0
        x = x.permute(1, 0, 2, 3)                       # (T,C,H,W) -> (C,T,H,W)
        if self.train:
            c, t, h, w = x.shape
            scale = float(np.random.uniform(0.75, 1.0))
            ch, cw = int(h * scale), int(w * scale)
            top = int(np.random.randint(0, h - ch + 1))
            left = int(np.random.randint(0, w - cw + 1))
            # One window for the whole clip: per-frame jitter would inject fake motion.
            x = x[:, :, top:top + ch, left:left + cw]
            x = F.interpolate(x.permute(1, 0, 2, 3), size=(h, w), mode="bilinear",
                              align_corners=False).permute(1, 0, 2, 3)
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[-1])
        y = -1 if self.labels is None else int(self.labels[i])
        return x, y


def build_model(arch: str, n_classes: int = 40, in_channels: int = 4) -> nn.Module:
    # in_channels defaults to 4 so every existing checkpoint and the in-flight folds
    # rebuild identically. The 3-channel case (EXP-088 thermal) skips the stem surgery
    # entirely and uses the Kinetics RGB kernels exactly as pretrained.
    import torchvision.models.video as tvv
    spec = {"r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights),
            "mc3_18": (tvv.mc3_18, tvv.MC3_18_Weights),
            "r3d_18": (tvv.r3d_18, tvv.R3D_18_Weights)}[arch]
    ctor, weights = spec
    model = ctor(weights=weights.KINETICS400_V1)
    if in_channels != 3:
        stem = model.stem[0]
        old = stem.weight.data                               # (out,3,kt,kh,kw)
        new = nn.Conv3d(in_channels, stem.out_channels, stem.kernel_size, stem.stride,
                        stem.padding, bias=stem.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = old
            for extra in range(3, in_channels):
                new.weight[:, extra:extra + 1] = old.mean(dim=1, keepdim=True)
        model.stem[0] = new
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


@torch.no_grad()
def predict(model, loader, device, flip_tta=True, mean=None, std=None):
    # mean/std are overridable so the 3-channel thermal member (EXP-088) can reuse
    # this exact evaluation path. Defaults reproduce the 4-channel behaviour byte for
    # byte, and these are plain kwargs, not argparse fields, so no --resume
    # fingerprint changes and in-flight folds are unaffected.
    mean = KIN_MEAN if mean is None else mean
    std = KIN_STD if std is None else std
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = ((x.to(device) - mean.to(device)) / std.to(device))
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(x).float(), -1)
            if flip_tta:
                p = 0.5 * (p + torch.softmax(model(torch.flip(x, dims=[-1])).float(), -1))
        probs.append(p.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="vid_f2")
    p.add_argument("--fold-oof", default="oof_visual_mil_v1_f2.npz")
    p.add_argument("--arch", default="r2plus1d_18",
                   choices=("r2plus1d_18", "mc3_18", "r3d_18"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0,
                   help=("1 by default: this job was OOM-killed at epoch ~24 of 30 "
                         "with workers=2 -- the 3GB memmap plus per-worker page "
                         "cache exhausts 15GB of system RAM."))
    p.add_argument("--resume", action="store_true",
                   help="continue from the per-epoch resume state if its recipe matches")
    p.add_argument("--all-train", action="store_true",
                   help=("train on all 18 users with NO held-out fold. The fold members "
                         "each see ~2,200 of 2,933 clips; this sees every one, which is "
                         "what the published 0.711 notebook does and what must ship. "
                         "Reports the fold's outer score anyway, but that number is "
                         "TRAIN-ON-TEST and is printed only as a sanity check -- it must "
                         "never be compared against a fold member's honest OOF."))
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--cache", default="crop_v1",
                   help="cache dir under cache/; crop_v160 is the 160x160 build")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    cache_dir = os.path.join(ROOT, "cache", args.cache)
    index = json.load(open(os.path.join(cache_dir, "train_index.json")))
    sids = index["sids"]
    row_of = {s: i for i, s in enumerate(sids)}
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        lab = {r["sample_id"]: int(r["class_id"]) for r in csv.DictReader(h)}
    ref = np.load(os.path.join(ART, args.fold_oof), allow_pickle=True)
    val_sids = [s for s in (str(x) for x in ref["sids"]) if s in row_of]
    vset = set(val_sids)
    # --all-train keeps the outer clips in training too: the fold split exists to
    # measure, and once the recipe is measured the shipped model should see all of it.
    tr_sids = [s for s in sids if s in lab and (args.all_train or s not in vset)]

    memmap = np.load(os.path.join(cache_dir, "train.npy"), mmap_mode="r")
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

    model = build_model(args.arch).to(device)
    n = sum(q.numel() for q in model.parameters())
    print(f"{args.tag}: arch={args.arch} train={len(tr_rows)} outer={len(va_rows)} "
          f"params={n:,} fp16={n*2/1e6:.1f}MB", flush=True)

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
            # Optimizer/scaler state is not persisted (it tripled the checkpoint to
            # 501MB and the process died on the write). Fast-forward the LR schedule
            # so the resumed run continues on the same cosine curve.
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

    # Last epoch, never the best: selecting on the val curve is worth ~9-10 clips
    # of noise on this competition.
    probs, labels = predict(ema, vl, device)
    pred = probs.argmax(1)
    om = np.isin(labels, OBJECT)
    micro = float((pred == labels).mean())
    obj = float((pred[om] == labels[om]).mean())
    mot = float((pred[~om] == labels[~om]).mean())
    tag_note = " [TRAIN-ON-TEST, not comparable to fold OOF]" if args.all_train else ""
    print(f"{args.tag} OUTER-ONCE{tag_note}: micro={micro:.5f} object={obj:.5f} "
          f"({int((pred[om]==labels[om]).sum())}/{int(om.sum())}) "
          f"gross_motion={mot:.5f}", flush=True)

    os.makedirs(CKPT, exist_ok=True)
    torch.save({"state_dict": ema.state_dict(), "args": vars(args),
                "micro": micro, "object": obj}, os.path.join(CKPT, f"{args.tag}.pt"))
    np.savez_compressed(os.path.join(ART, f"oof_{args.tag}.npz"),
                        probs=probs.astype(np.float32),
                        sids=np.array(val_sids, dtype="<U15"), labels=labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
