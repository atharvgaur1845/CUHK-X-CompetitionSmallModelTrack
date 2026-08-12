#!/usr/bin/env python3
"""EXP-080: fine-tune an ImageNet-pretrained ResNet18 visual member.

Legality (research/RULES_VERIFIED.md R-1, discussion topic 711665, 2026-06-25):
    "Small, standard pretrained CNNs such as ImageNet-pretrained ResNet18 (~44MB)
     are perfectly acceptable in the Small Model Track."
A prior session recorded the opposite as fact, cancelled the pretrained probe, and
built an SSL programme as a substitute for an init that was legal all along.

Evidence this is the right target (EXP-079, frozen linear probe, fold 2, one
classifier, weights the only difference):
    random-init  ResNet18 : micro 0.18558  object 0.09603 (46/479)
    ImageNet     ResNet18 : micro 0.26074  object 0.16284 (78/479)
i.e. +70% relative on the 26 sedentary classes that hold 75% of our error -- frozen,
mean-pooled, no temporal model at all.

Reference points on the same fold-2 split, from-scratch trunk with full temporal MIL:
    micro 0.35123   object 0.24843 (119/479)      [EXP-062/064]
and the deployed 4-fold from-scratch member scores 0.38805 public solo.

Architecture: one shared ResNet18 over every (modality, frame) image; mean+max
temporal pooling per modality; concat; dropout; linear head.  The backbone is the only
representation-learning component, so this sits inside the CNN family the rules allow,
and 11.69M params = 46.8 MB fp32 / 23.4 MB fp16 fits the single-file 100 MB budget.
"""
from __future__ import annotations

import argparse
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
CACHE = os.path.join(ROOT, "cache", "visual_mil_v1")
ART = os.path.join(ROOT, "research", "artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OBJECT = np.array(sorted(set(range(28)) | {37, 38, 39}))
MODALITIES = ("ir", "depth", "thermal")


class ClipSet(Dataset):
    def __init__(self, sids, labels, split, stride, train, crop_min=0.7,
                 modalities=MODALITIES, resize=None):
        self.sids, self.labels, self.split = sids, labels, split
        self.stride, self.train, self.crop_min = stride, train, crop_min
        self.modalities = tuple(modalities)
        self.resize = resize

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        d = np.load(os.path.join(CACHE, self.split, f"{self.sids[i]}.npz"))
        mods = []
        for name in self.modalities:
            a = np.asarray(d[name])[:: self.stride]
            if a.ndim == 3:
                a = np.repeat(a[..., None], 3, axis=-1)
            mods.append(a)
        x = np.stack(mods, 0).astype(np.float32) / 255.0        # (M,T,H,W,3)
        x = torch.from_numpy(x).permute(0, 1, 4, 2, 3)          # (M,T,3,H,W)
        if self.train:
            m, t, c, h, w = x.shape
            scale = np.random.uniform(self.crop_min, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
            x = x[..., top:top + ch, left:left + cw]
            x = F.interpolate(x.reshape(m * t, c, ch, cw), size=(h, w),
                              mode="bilinear", align_corners=False).reshape(m, t, c, h, w)
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=[-1])
        if self.resize is not None:
            m, t, c, h, w = x.shape
            x = F.interpolate(x.reshape(m * t, c, h, w), size=self.resize,
                              mode="bilinear", align_corners=False
                              ).reshape(m, t, c, *self.resize)
        label = -1 if self.labels is None else int(self.labels[i])
        return x, label


class PretrainedVisual(nn.Module):
    def __init__(self, n_classes=40, dropout=0.4, pretrained=True,
                 n_modalities=len(MODALITIES), arch="resnet18"):
        super().__init__()
        import torchvision.models as tvm
        spec = {"resnet18": (tvm.resnet18, tvm.ResNet18_Weights, 512),
                "resnet34": (tvm.resnet34, tvm.ResNet34_Weights, 512),
                "resnet50": (tvm.resnet50, tvm.ResNet50_Weights, 2048)}[arch]
        ctor, wenum, dim = spec
        backbone = ctor(weights=wenum.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = dim
        self.drop = nn.Dropout(dropout)
        self.n_modalities = n_modalities
        self.head = nn.Linear(self.feat_dim * 2 * n_modalities, n_classes)

    def forward(self, x):                                        # (B,M,T,3,H,W)
        b, m, t = x.shape[:3]
        f = self.backbone(x.reshape(b * m * t, *x.shape[3:])).reshape(b, m, t, -1)
        pooled = torch.cat([f.mean(2), f.max(2).values], dim=-1)  # (B,M,1024)
        return self.head(self.drop(pooled.reshape(b, -1)))


def evaluate(model, loader, device, log_prior):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = ((x.to(device) - MEAN.to(device)) / STD.to(device))
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="pre_r18_f2")
    p.add_argument("--fold-oof", default="oof_v2_scratch_f2_f2.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--scratch", action="store_true", help="control: random init")
    p.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                   help=("train a per-modality model instead of one shared trunk. "
                         "The paper trains each modality separately and reports "
                         "Thermal 92.57 > Depth 90.5 > IR 90.2, while our single "
                         "shared backbone sees all three under one set of weights."))
    p.add_argument("--arch", default="resnet18",
                   choices=("resnet18", "resnet34", "resnet50"))
    p.add_argument("--resize", type=int, nargs=2, default=None,
                   help=("H W to resample the cached 192x256 frames to. Tests "
                         "whether spatial resolution is the binding constraint "
                         "BEFORE paying to build a higher-resolution cache: the "
                         "26 hand-object classes need to resolve a held object, "
                         "and the cache already downsamples 640x480 by 2.5x."))
    p.add_argument("--freeze-bn", action="store_true",
                   help=("keep BatchNorm in eval mode so ImageNet running statistics "
                         "survive. IR/depth/thermal are grayscale-replicated colormaps "
                         "whose channel statistics differ sharply from natural images, "
                         "so 2281 clips of BN re-estimation can wash out the very "
                         "features EXP-079 measured as +70% relative on object."))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(20260730)
    np.random.seed(20260730)
    device = torch.device("cuda")

    ref = np.load(os.path.join(ART, args.fold_oof), allow_pickle=True)
    val_sids = [str(s) for s in ref["sids"]]
    val_labels = np.asarray(ref["labels"])
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        lab = {r["sample_id"]: int(r["class_id"]) for r in csv.DictReader(h)}
    vset = set(val_sids)
    tr_sids = sorted(s for s in lab if s not in vset
                     and os.path.exists(os.path.join(CACHE, "train", f"{s}.npz")))
    tr_labels = np.array([lab[s] for s in tr_sids])

    counts = np.bincount(tr_labels, minlength=40).astype(np.float64)
    log_prior = torch.tensor(np.log(np.maximum(counts / counts.sum(), 1e-9)),
                             dtype=torch.float32, device=device)

    mods = tuple(args.modalities)
    train_loader = DataLoader(ClipSet(tr_sids, tr_labels, "train", args.stride, True,
                                     modalities=mods, resize=tuple(args.resize) if args.resize else None),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(ClipSet(val_sids, val_labels, "train", args.stride, False,
                                   modalities=mods, resize=tuple(args.resize) if args.resize else None),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = PretrainedVisual(dropout=args.dropout, pretrained=not args.scratch,
                             n_modalities=len(mods), arch=args.arch).to(device)
    n = sum(q.numel() for q in model.parameters())
    print(f"{args.tag}: train={len(tr_sids)} outer={len(val_sids)}  "
          f"params={n:,}  fp32={n*4/1e6:.1f}MB  fp16={n*2/1e6:.1f}MB  "
          f"init={'random' if args.scratch else 'IMAGENET'}  mods={','.join(mods)}", flush=True)

    head = [q for k, q in model.named_parameters() if not k.startswith("backbone")]
    back = [q for k, q in model.named_parameters() if k.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": back, "lr": args.lr * args.backbone_lr_mult},
                             {"params": head, "lr": args.lr}], weight_decay=0.05)
    steps = args.epochs * max(1, len(train_loader) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr * args.backbone_lr_mult, args.lr],
        total_steps=steps, pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            for module in model.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        total, seen, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader):
            x = ((x.to(device, non_blocking=True) - MEAN.to(device)) / STD.to(device))
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                # balanced softmax: train with +log prior, infer without, so the
                # member's probabilities live in the same uniform-prior space the
                # existing fusion (fuse_prior_visual.py) expects.
                loss = F.cross_entropy(model(x) + log_prior, y,
                                       label_smoothing=args.label_smoothing)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
            total += float(loss) * y.numel(); seen += y.numel()
        if epoch % 3 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  epoch {epoch:02d}/{args.epochs}: loss={total/seen:.5f} "
                  f"lr={sched.get_last_lr()[1]:.2e} [{time.time()-t0:.0f}s]", flush=True)

    probs, labels = evaluate(model, val_loader, device, log_prior)
    pred = probs.argmax(1)
    om = np.isin(labels, OBJECT)
    micro = float((pred == labels).mean())
    obj = float((pred[om] == labels[om]).mean())
    mot = float((pred[~om] == labels[~om]).mean())
    print(f"{args.tag} OUTER-ONCE: micro={micro:.5f} object={obj:.5f} "
          f"({int((pred[om]==labels[om]).sum())}/{int(om.sum())}) "
          f"gross_motion={mot:.5f}", flush=True)

    os.makedirs(CKPT, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "micro": micro, "object": obj},
               os.path.join(CKPT, f"{args.tag}.pt"))
    np.savez_compressed(os.path.join(ART, f"oof_{args.tag}.npz"),
                        probs=probs.astype(np.float32),
                        sids=np.array(val_sids, dtype="<U15"), labels=labels)
    with open(os.path.join(ART, f"run_{args.tag}.json"), "w") as h:
        json.dump({"micro": micro, "object": obj, "motion": mot,
                   "args": {k: str(v) for k, v in vars(args).items()}}, h, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
