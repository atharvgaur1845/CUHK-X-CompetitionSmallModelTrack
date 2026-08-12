#!/usr/bin/env python3
"""Single-frame appearance model: the direct attack on EXP-068's diagnosis.

EXP-068 established that the visual MIL branch is a pure MOTION model. Destroying
94% of its pixels costs nothing (96x128 scores +0.45 on sedentary classes), while
halving its frames costs 4.1 points and a single frame drops the whole-body motion
classes to exactly 0.0000. It never learned what is in the hand -- and EXP-067
showed that fine-grained hand-object discrimination is 75% of the remaining error.

A model that sees ONE frame at a time cannot use motion, so it must learn
appearance or score at chance. That makes it complementary to the motion branch by
construction rather than by hope, which is the property every previous member
lacked (visual-right/base-wrong was only 6.8% on sedentary classes).

Trains on (clip, frame) pairs -- 2933 x 16 = ~47k samples -- and averages frame
logits at inference. Roughly 20x cheaper per epoch than the 16-frame MIL model.

    python3 code/train_frame_appearance.py train --fold 2 --epochs 12
    python3 code/train_frame_appearance.py infer --tag frame_app --folds 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "visual_mil_v1"
META = ROOT / "cache" / "meta_train.csv"
FOLDS = ROOT / "research" / "artifacts" / "cv_folds_all18.json"
ARTIFACTS = ROOT / "research" / "artifacts"
CKPT = ROOT / "checkpoints"
N_CLASSES = 40
FRAMES = 16
# EXP-068 measured that spatial detail below half resolution is unused, so the
# appearance model runs at 96x128: same information, a quarter of the compute.
HEIGHT, WIDTH = 96, 128
SED = (0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24,
       25, 26, 27, 37, 38, 39)


def read_meta() -> dict[str, int]:
    out = {}
    with META.open() as fh:
        header = fh.readline().strip().split(",")
        ci, si = header.index("class_id"), header.index("sample_id")
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) > max(ci, si):
                out[parts[si]] = int(parts[ci])
    return out


def read_folds(fold: int) -> tuple[list[str], list[str]]:
    records = json.loads(FOLDS.read_text())
    records = records["folds"] if isinstance(records, dict) else records
    val_users = set()
    for rec in records:
        if int(rec.get("fold", rec.get("index", -1))) == fold:
            val_users = set(map(str, rec.get("validation_users", rec.get("val_users", []))))
    if not val_users:
        raise ValueError(f"fold {fold}: no validation users in {FOLDS}")
    return sorted(val_users), records


class FrameDataset(Dataset):
    """One item per CLIP, returning all 16 frames as independent samples.

    Indexing per (clip, frame) instead re-opens and re-decodes the whole npz for
    every single frame -- 16x redundant I/O, which pinned the first run at 160 s
    per epoch on a GPU that was mostly idle.
    """

    def __init__(self, ids: list[str], labels: dict[str, int], split: str,
                 augment: bool, frames: int = FRAMES):
        self.ids = list(ids)
        self.labels = labels
        self.split = split
        self.augment = augment
        self.frames = frames

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        sid = self.ids[index]
        with np.load(CACHE / self.split / f"{sid}.npz", allow_pickle=True) as z:
            ir = z["ir"].astype(np.float32) / 255.0
            depth = z["depth"].astype(np.float32) / 255.0
            thermal = z["thermal"].astype(np.float32).mean(-1) / 255.0
            masks = np.stack([z["ir_frame_mask"], z["depth_frame_mask"],
                              z["thermal_frame_mask"]], 1).astype(np.float32)
        x = torch.from_numpy(np.stack([ir, depth, thermal], 1))   # (T,3,192,256)
        x = F.interpolate(x, size=(HEIGHT, WIDTH), mode="area")
        if self.augment:
            scale = np.random.uniform(0.85, 1.0)
            h, w = int(HEIGHT * scale), int(WIDTH * scale)
            top = np.random.randint(0, HEIGHT - h + 1)
            left = np.random.randint(0, WIDTH - w + 1)
            x = F.interpolate(x[:, :, top:top + h, left:left + w],
                              size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False)
            x = x * np.float32(np.random.uniform(0.9, 1.1))
            drop = np.random.rand(3) < 0.15                       # modality dropout
            if drop.any():
                x[:, drop] = 0.0
                masks[:, drop] = 0.0
        x = x * torch.from_numpy(masks)[:, :, None, None]
        # Training draws only a few frames per clip so that one batch spans many
        # clips.  Taking all 16 leaves just 8 distinct labels in a 128-sample
        # batch and the model barely moves (clip-acc 0.028 by epoch 3).
        if self.frames < x.shape[0]:
            pick = np.sort(np.random.choice(x.shape[0], self.frames, replace=False))
            x = x[pick]
        y = self.labels.get(sid, -1)
        return x, y, sid


class MemmapFrameDataset(Dataset):
    """Per-(clip, frame) indexing over the packed memmap: global frame shuffling.

    This is the configuration the throwaway first run used -- a batch spans as many
    distinct clips as it has samples -- restored now that per-frame access is free.
    """

    MEMMAP = ROOT / "cache" / "frame_memmap"

    def __init__(self, ids: list[str], labels: dict[str, int], split: str,
                 augment: bool):
        all_sids = json.loads((self.MEMMAP / f"{split}_sids.json").read_text())
        pos = {s: i for i, s in enumerate(all_sids)}
        self.rows = np.array([pos[s] for s in ids])
        self.ids = list(ids)
        self.labels = np.array([labels.get(s, -1) for s in ids])
        self.split = split
        self.augment = augment
        self.frames = np.load(self.MEMMAP / f"{split}_frames.npy", mmap_mode="r")
        self.masks = np.load(self.MEMMAP / f"{split}_masks.npy")

    def __len__(self) -> int:
        return len(self.rows) * FRAMES

    def __getitem__(self, index: int):
        clip, f = index // FRAMES, index % FRAMES
        row = self.rows[clip]
        x = torch.from_numpy(np.asarray(self.frames[row, f], dtype=np.float32) / 255.0)
        mask = self.masks[row, f].astype(np.float32)
        if self.augment:
            scale = np.random.uniform(0.85, 1.0)
            h, w = int(HEIGHT * scale), int(WIDTH * scale)
            top = np.random.randint(0, HEIGHT - h + 1)
            left = np.random.randint(0, WIDTH - w + 1)
            x = F.interpolate(x[None, :, top:top + h, left:left + w],
                              size=(HEIGHT, WIDTH), mode="bilinear",
                              align_corners=False)[0]
            x = x * np.float32(np.random.uniform(0.9, 1.1))
            drop = np.random.rand(3) < 0.15
            if drop.any():
                x[drop] = 0.0
                mask = mask * (~drop).astype(np.float32)
        x = x * torch.from_numpy(mask)[:, None, None]
        return x, int(self.labels[clip]), self.ids[clip]


def collate(batch):
    """Flatten (clips, T, 3, H, W) into independent frame samples."""
    xs, ys, sids = zip(*batch)
    x = torch.cat(list(xs), 0)
    t = xs[0].shape[0]
    y = torch.tensor([v for v in ys for _ in range(t)], dtype=torch.long)
    ids = [s for s in sids for _ in range(t)]
    return x, y, ids


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.GroupNorm(min(8, cout), cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
        )
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                   nn.GroupNorm(min(8, cout), cout)))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class FrameNet(nn.Module):
    """Small from-scratch 2D CNN. No temporal input of any kind, by design."""

    def __init__(self, classes: int = N_CLASSES, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 5, 2, 2, bias=False),
            nn.GroupNorm(8, width), nn.SiLU(inplace=True))
        self.stages = nn.Sequential(
            Block(width, width * 2, 2), Block(width * 2, width * 2, 1),
            Block(width * 2, width * 4, 2), Block(width * 4, width * 4, 1),
            Block(width * 4, width * 8, 2), Block(width * 8, width * 8, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(width * 8 * 2, classes))

    def forward(self, x):
        h = self.stages(self.stem(x))
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.head(pooled)


def evaluate(model, loader, device, labels_map, allowed: np.ndarray | None = None):
    """Balanced softmax trains with +log(prior) and infers without it, which is
    correct while every class has support.  A specialist trained on a subset gives
    the absent classes prior log(1e-9) = -20.7, so training never penalises their
    logits and raw inference then predicts nothing else: the 26-class run scored
    exactly 0.0000 on sedentary clips and 0.1803 on motion ones.  Classes with no
    training support are therefore masked out at inference as well.
    """
    model.eval()
    acc: dict[str, np.ndarray] = {}
    block = None
    if allowed is not None:
        block = np.where(allowed, 0.0, -np.inf).astype(np.float32)
    with torch.no_grad():
        for x, y, sids in loader:
            logits = model(x.to(device, non_blocking=True)).float().cpu().numpy()
            if block is not None:
                logits = logits + block
            for s, lg in zip(sids, logits):
                acc[s] = acc.get(s, 0.0) + lg
    sids = sorted(acc)
    probs = np.stack([acc[s] for s in sids])
    probs = np.exp(probs - probs.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    y = np.array([labels_map.get(s, -1) for s in sids])
    return probs, y, sids


def train_command(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    labels = read_meta()
    val_users, _ = read_folds(args.fold)
    all_ids = sorted(p.stem for p in (CACHE / "train").glob("*.npz"))

    def user_of(sid: str) -> str:
        return sid.split("_")[1]

    if args.all_users:
        # Full-data checkpoint: SEARCH_MAP.md:72 has carried
        # "Validated full-data checkpoint strategy ?" untested for the whole
        # campaign, while every deployed member is a 4-fold ensemble whose models
        # each saw only 13-14 of the 18 users.  Training the final member on all
        # 18 is the standard post-CV move, and unlike every OOF-fitted lever that
        # failed to transfer here it is a mechanism rather than a correlation:
        # more training subjects generalises better across subjects.
        # There is no honest held-out set left, so the epoch budget is fixed at
        # the value the fold runs converged on and the LAST epoch is kept -- no
        # checkpoint selection is performed.
        tr = [s for s in all_ids if s in labels]
        va = [s for s in all_ids if user_of(s) in val_users and s in labels]
    else:
        tr = [s for s in all_ids if user_of(s) not in val_users and s in labels]
        va = [s for s in all_ids if user_of(s) in val_users and s in labels]
    if args.sedentary_only:
        # Specialist: spend all capacity on the 26 hand-object classes that carry
        # 75% of the error (EXP-067).  The coarse sedentary/motion split is already
        # 96% solved by the stack, so the 14 whole-body classes are wasted capacity.
        # Validation stays FULL so the member still emits a row for every clip and
        # can be stacked alongside the generalists.
        tr = [s for s in tr if labels[s] in SED]
        print(f"sedentary-only specialist: {len(tr)} training clips over "
              f"{len(SED)} classes", flush=True)
    print(f"fold {args.fold}: train {len(tr)} clips / {len(tr)*FRAMES} frames | "
          f"val {len(va)} clips", flush=True)

    if args.memmap:
        tl = DataLoader(MemmapFrameDataset(tr, labels, "train", True),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
        vl = DataLoader(MemmapFrameDataset(va, labels, "train", False),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    else:
        tl = DataLoader(FrameDataset(tr, labels, "train", True, frames=args.frames_per_clip),
                        batch_size=max(1, args.batch_size // args.frames_per_clip),
                        shuffle=True, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers > 0,
                        collate_fn=collate)
        vl = DataLoader(FrameDataset(va, labels, "train", False),
                        batch_size=max(1, args.batch_size // FRAMES),
                        shuffle=False, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, collate_fn=collate)

    model = FrameNet(width=args.width, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters {n_params:,} ({n_params*4/1e6:.1f} MB fp32)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps,
                                                pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = np.bincount([labels[s] for s in tr], minlength=N_CLASSES).astype(np.float32)
    prior = torch.tensor(np.log(counts / counts.sum() + 1e-9), device=device)
    allowed = counts > 0

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y, _ in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                # balanced softmax: the class counts span 12..319
                loss = F.cross_entropy(model(x) + prior, y, label_smoothing=args.smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += float(loss.detach()) * len(y); seen += len(y)
        probs, yv, sids = evaluate(model, vl, device, labels, allowed)
        correct = probs.argmax(1) == yv
        sed = np.isin(yv, SED)
        print(f"epoch {epoch+1:2d}/{args.epochs} loss {total/max(seen,1):.4f} "
              f"clip-acc {correct.mean():.4f} sedentary {correct[sed].mean():.4f} "
              f"motion {correct[~sed].mean():.4f} [{time.time()-t0:.0f}s]", flush=True)
        if args.all_users or correct.mean() > best:
            best = correct.mean()
            CKPT.mkdir(exist_ok=True)
            name = f"{args.tag}_all{args.seed}" if args.all_users else f"{args.tag}_f{args.fold}"
            torch.save({"state_dict": model.state_dict(), "fold": args.fold,
                        "acc": float(best), "args": vars(args)}, CKPT / f"{name}.pt")
            ARTIFACTS.mkdir(exist_ok=True)
            np.savez(ARTIFACTS / f"oof_{args.tag}_f{args.fold}.npz" if not args.all_users
                     else ARTIFACTS / f"trainfit_{args.tag}_all{args.seed}.npz",
                     probs=probs.astype(np.float32), labels=yv.astype(np.int64),
                     sids=np.array(sids))
    print(f"BEST fold {args.fold}: {best:.4f}", flush=True)


def infer_command(args):
    """Average frame logits over the 16 frames and the four fold checkpoints."""
    device = torch.device(args.device)
    memmap = ROOT / "cache" / "frame_memmap"
    sids = json.loads((memmap / "test_sids.json").read_text())
    frames = np.load(memmap / "test_frames.npy", mmap_mode="r")
    masks = np.load(memmap / "test_masks.npy")
    stems = ([f"{args.tag}_{x}" for x in args.checkpoints.split(",")]
             if args.checkpoints else
             [f"{args.tag}_f{f}" for f in args.folds.split(",")])

    total = np.zeros((len(sids), N_CLASSES), dtype=np.float64)
    used = 0
    for stem in stems:
        path = CKPT / f"{stem}.pt"
        if not path.is_file():
            print(f"  missing {path.name}, skipping", flush=True)
            continue
        package = torch.load(path, map_location="cpu", weights_only=False)
        model = FrameNet(width=package["args"].get("width", 32),
                         dropout=package["args"].get("dropout", 0.3)).to(device).eval()
        model.load_state_dict(package["state_dict"])
        # A specialist must be masked at inference for the same reason as in
        # evaluate(): balanced softmax leaves unsupported logits unconstrained.
        block = np.zeros(N_CLASSES, dtype=np.float32)
        if package["args"].get("sedentary_only"):
            block[[c for c in range(N_CLASSES) if c not in SED]] = -np.inf
        acc = np.zeros((len(sids), N_CLASSES), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(sids), args.chunk):
                stop = min(start + args.chunk, len(sids))
                block_frames = np.asarray(frames[start:stop], dtype=np.float32) / 255.0
                m = masks[start:stop].astype(np.float32)
                x = torch.from_numpy(block_frames * m[:, :, :, None, None])
                n, t = x.shape[:2]
                logits = model(x.reshape(n * t, 3, HEIGHT, WIDTH).to(device))
                logits = logits.float().cpu().numpy().reshape(n, t, N_CLASSES)
                acc[start:stop] = logits.mean(1) + block
        total += acc
        used += 1
        print(f"  {stem} done", flush=True)
    if not used:
        raise SystemExit("no checkpoints found")
    total /= used
    probs = np.exp(total - total.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    out = ARTIFACTS / f"testprobs_{args.tag}.npz"
    np.savez(out, probs=np.nan_to_num(probs).astype(np.float32), sids=np.array(sids))
    print(f"wrote {out.name}: {probs.shape} from {used} folds")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--tag", default="frame_app_mm")
    i.add_argument("--folds", default="0,1,2,3")
    i.add_argument("--checkpoints", default="")
    i.add_argument("--chunk", type=int, default=16)
    i.add_argument("--device", default="cuda")
    i.set_defaults(func=infer_command)
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, default=2)
    t.add_argument("--tag", default="frame_app")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch-size", type=int, default=96)
    t.add_argument("--frames-per-clip", type=int, default=4)
    t.add_argument("--memmap", action="store_true")
    t.add_argument("--sedentary-only", action="store_true")
    t.add_argument("--all-users", action="store_true")
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--smoothing", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.set_defaults(func=train_command)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
