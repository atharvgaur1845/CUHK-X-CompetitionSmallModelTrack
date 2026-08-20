#!/usr/bin/env python3
"""EXP-094: MotionBERT (pretrained) on our skeletons — the unfixed half of R-1.

Why this exists
---------------
R-1 (2026-08-10) established that pretrained backbones are legal. We applied it to
the VISUAL branch and got +23.9 micro, the largest gain of the campaign. The
SKELETON branch was never touched: `code/har_models.py` is STGCN / AdaptiveSTGCN /
CTRGCN / TCN / BiGRU, all from-scratch, scoring 0.594 solo OOF while carrying 0.35 —
the largest single weight in the champion fusion. The dataset paper's own skeleton
baseline is MotionBERT at 79.1, a *pretrained* model, against our ~73 from-scratch.

Checkpoint: `FT_MB_release_MB_ft_NTU60_xsub` — MotionBERT finetuned on NTU-60
**cross-subject** skeleton action recognition. Same task, same protocol, same
17-joint H36M topology as our data. R-2 names NTU RGB+D as permitted external data.

THE AXIS CONVENTION IS LOAD-BEARING — measured, not assumed
-----------------------------------------------------------
MotionBERT's action model was trained on 2D HRNet keypoints in IMAGE coordinates
(x right, y DOWN) plus a confidence channel, normalised to [-1,1] by `crop_scale`.

Our skeletons are pelvis-centred and floor-aligned, and probing 400 clips shows:

    head minus foot:   x +0.086   y -0.119   z +1.131   <- z is VERTICAL
    per-axis spread:   x  0.204   y  0.092   z  0.378   <- y is depth, least informative

So our (x, y) is a TOP-DOWN floor-plane view. Feeding that as "2D keypoints" would
be meaningless to a frontally-pretrained model, and the failure would have looked
like "MotionBERT does not transfer" rather than "we fed it the wrong plane".
The correct mapping is image_x = x, image_y = -z, conf = 1.

`--depth-channel` instead writes normalised depth into the confidence slot: we have
genuine 3D where NTU HRNet had only a detector score, so that channel is free information
— but it is a distribution shift from pretraining, so it is an option, not the default.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MB = os.path.join(ROOT, "third_party", "motionbert")
ART = os.path.join(ROOT, "research", "artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
OBJECT = np.array(sorted(set(range(28)) | {37, 38, 39}))
sys.path.insert(0, MB)


def crop_scale(motion: np.ndarray) -> np.ndarray:
    """Port of MotionBERT's utils_data.crop_scale: normalise x,y to [-1,1] with a
    single isotropic scale so the aspect ratio the backbone expects is preserved."""
    out = motion.copy()
    valid = motion[motion[..., 2] != 0][:, :2]
    if len(valid) < 4:
        return np.zeros_like(motion)
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros_like(motion)
    xs, ys = (xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2
    out[..., :2] = ((motion[..., :2] - [xs, ys]) / scale - 0.5) * 2
    return np.clip(out, -1, 1)


def resample_ids(ori_len: int, target_len: int, randomness: bool) -> np.ndarray:
    if ori_len <= 0:
        return np.zeros(target_len, dtype=int)
    if not randomness:
        return np.linspace(0, ori_len - 1, target_len).round().astype(int)
    even = np.linspace(0, ori_len, num=target_len, endpoint=False)
    low = np.floor(even).astype(int)
    high = np.clip(low + 1, a_min=0, a_max=ori_len - 1)
    sel = np.random.rand(target_len) < (even - low)
    return np.clip(np.where(sel, high, low), 0, ori_len - 1)


class SkelSet(Dataset):
    def __init__(self, sids, labels, split, n_frames, train, max_person=2,
                 depth_channel=False):
        self.sids, self.labels, self.split = sids, labels, split
        self.n_frames, self.train = n_frames, train
        self.max_person, self.depth_channel = max_person, depth_channel

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, i):
        path = os.path.join(ROOT, "cache", self.split, f"{self.sids[i]}.npz")
        with np.load(path) as d:
            pos = np.asarray(d["skel_pos"], np.float32) if "skel_pos" in d.files else None
        M = self.max_person
        out = np.zeros((M, self.n_frames, 17, 3), np.float32)
        if pos is not None and pos.size and pos.shape[0] > 0:
            ids = resample_ids(pos.shape[0], self.n_frames, self.train)
            pos = pos[ids]                                     # (T,M0,17,3)
            for m in range(min(M, pos.shape[1])):
                p = pos[:, m]                                  # (T,17,3)
                motion = np.zeros((self.n_frames, 17, 3), np.float32)
                motion[..., 0] = p[..., 0]                     # image x  <- world x
                motion[..., 1] = -p[..., 2]                    # image y  <- world -z (z is up)
                if self.depth_channel:
                    dep = p[..., 1]
                    rng = np.ptp(dep)
                    motion[..., 2] = 0.5 + 0.5 * (dep - dep.mean()) / (rng + 1e-6) if rng > 0 else 1.0
                else:
                    motion[..., 2] = 1.0
                out[m] = crop_scale(motion)
        y = -1 if self.labels is None else int(self.labels[i])
        return torch.from_numpy(out), y


def build_actionnet(args, n_classes=40):
    from lib.model.DSTformer import DSTformer
    from lib.model.model_action import ActionNet
    backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=512,
                         depth=5, num_heads=8, mlp_ratio=2, num_joints=17,
                         maxlen=243, att_fuse=True)
    net = ActionNet(backbone=backbone, dim_rep=512, num_classes=n_classes,
                    dropout_ratio=args.dropout, version="class", hidden_dim=2048,
                    num_joints=17)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = state.get("model") or state.get("model_pos") or state.get("state_dict") or state
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # The NTU head is 60-way; ours is 40-way. Drop the final classifier only —
        # everything before it (backbone + fc1 + bn) is what carries the transfer.
        own = net.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        dropped = sorted(set(sd) - set(keep))
        net.load_state_dict(keep, strict=False)
        print(f"  loaded {len(keep)}/{len(own)} tensors from checkpoint; "
              f"dropped {len(dropped)} (shape/name mismatch)", flush=True)
        if dropped[:6]:
            print(f"    e.g. {dropped[:6]}", flush=True)
    return net


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(x).float(), -1)
            xf = x.clone(); xf[..., 0] *= -1          # horizontal flip in image x
            p = 0.5 * (p + torch.softmax(model(xf).float(), -1))
        P.append(p.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="skel_mb_f2")
    p.add_argument("--fold-oof", default="oof_visual_mil_v1_f2.npz")
    p.add_argument("--checkpoint",
                   default=os.path.join(MB, "checkpoint", "MB_ft_NTU60_xsub.bin"))
    p.add_argument("--dim-feat", type=int, default=512)
    p.add_argument("--n-frames", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4,
                   help=("4, not the config's 32: DSTformer at 60.4M params over "
                         "M=2 persons x T frames x 17 joints OOMs an 8GB card at 16. "
                         "--accum restores the effective batch."))
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--depth-channel", action="store_true")
    p.add_argument("--scratch", action="store_true",
                   help="random init — the control that isolates what pretraining buys")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.scratch:
        args.checkpoint = ""

    torch.manual_seed(20260730); np.random.seed(20260730)
    device = torch.device("cuda")

    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        meta = [r for r in csv.DictReader(h)
                if os.path.exists(os.path.join(ROOT, "cache", "train", f"{r['sample_id']}.npz"))]
    lab = {r["sample_id"]: int(r["class_id"]) for r in meta}
    ref = np.load(os.path.join(ART, args.fold_oof), allow_pickle=True)
    val_sids = [s for s in (str(x) for x in ref["sids"]) if s in lab]
    vset = set(val_sids)
    tr_sids = [r["sample_id"] for r in meta if r["sample_id"] not in vset]

    tr_lab = np.array([lab[s] for s in tr_sids])
    va_lab = np.array([lab[s] for s in val_sids])
    counts = np.bincount(tr_lab, minlength=40).astype(np.float64)
    log_prior = torch.tensor(np.log(np.maximum(counts / counts.sum(), 1e-9)),
                             dtype=torch.float32, device=device)

    mk = lambda s, l, t: SkelSet(s, l, "train", args.n_frames, t,
                                 depth_channel=args.depth_channel)
    tl = DataLoader(mk(tr_sids, tr_lab, True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=False)
    vl = DataLoader(mk(val_sids, va_lab, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=False)

    model = build_actionnet(args).to(device)
    n = sum(q.numel() for q in model.parameters())
    print(f"{args.tag}: train={len(tr_sids)} outer={len(val_sids)} frames={args.n_frames} "
          f"params={n:,} fp16={n*2/1e6:.1f}MB pretrained={not args.scratch}", flush=True)

    head_ids = {id(q) for q in model.head.parameters()}
    opt = torch.optim.AdamW(
        [{"params": [q for q in model.parameters() if id(q) not in head_ids],
          "lr": args.lr_backbone},
         {"params": list(model.head.parameters()), "lr": args.lr_head}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr_backbone, args.lr_head],
        total_steps=args.epochs * max(1, len(tl)), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)

    resume_path = os.path.join(CKPT, f"{args.tag}_resume.pt")
    fingerprint = {k: v for k, v in vars(args).items() if k != "resume"}
    start = 1
    if args.resume and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if st.get("fingerprint") == fingerprint:
            model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"])
            start = int(st["epoch"]) + 1
            for _ in range(min((start - 1) * len(tl), args.epochs * len(tl) - 1)):
                sched.step()
            print(f"  resumed from epoch {st['epoch']}", flush=True)

    for epoch in range(start, args.epochs + 1):
        model.train(); tot = seen = 0; t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tl):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x) + log_prior, y,
                                       label_smoothing=args.label_smoothing)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum:
                tot += float(loss.detach()) * y.numel(); seen += y.numel()
                continue
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if sched.last_epoch < args.epochs * len(tl) - 1:
                sched.step()
            with torch.no_grad():
                for e, m in zip(ema.parameters(), model.parameters()):
                    e.mul_(args.ema).add_(m.detach(), alpha=1 - args.ema)
                for e, m in zip(ema.buffers(), model.buffers()):
                    e.copy_(m)
            tot += float(loss.detach()) * y.numel(); seen += y.numel()
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "epoch": epoch, "fingerprint": fingerprint}, resume_path + ".tmp")
        os.replace(resume_path + ".tmp", resume_path)
        if epoch % 3 == 0 or epoch in (1, args.epochs):
            print(f"  epoch {epoch:02d}/{args.epochs}: loss={tot/seen:.5f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    probs, labels = predict(ema, vl, device)
    pred = probs.argmax(1); om = np.isin(labels, OBJECT)
    micro = float((pred == labels).mean())
    obj = float((pred[om] == labels[om]).mean())
    mot = float((pred[~om] == labels[~om]).mean())
    print(f"{args.tag} OUTER-ONCE: micro={micro:.5f} object={obj:.5f} "
          f"({int((pred[om]==labels[om]).sum())}/{int(om.sum())}) gross_motion={mot:.5f}",
          flush=True)
    print(f"  reference: from-scratch skeleton stack world25 solo OOF = 0.594", flush=True)
    torch.save({"state_dict": ema.state_dict(), "args": vars(args),
                "micro": micro, "object": obj}, os.path.join(CKPT, f"{args.tag}.pt"))
    np.savez_compressed(os.path.join(ART, f"oof_{args.tag}.npz"),
                        probs=probs.astype(np.float32),
                        sids=np.array(val_sids, dtype="<U15"), labels=labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
