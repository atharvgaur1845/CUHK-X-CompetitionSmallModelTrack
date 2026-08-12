#!/usr/bin/env python3
"""Cross-modal masked pretraining for the visual trunk — the legal substitute for
the pretrained init the published baselines depend on.

Why this exists
---------------
The dataset paper reports Depth 90.5 / IR 90.2 / Thermal 92.6, but those are
**pretrained ResNet-50, random-split, in-domain** numbers, and pretrained weights
are banned by the organizer ruling.  EXP-015 measured both discounts on this data:
ImageNet init roughly *doubles* from-scratch visual accuracy at this scale, and
cross-subject costs ~20 points even with pretraining.  So "visual reaches 0.90 from
scratch" was never supported by the evidence — the init is the binding constraint,
and self-supervision on the competition's own clips is the only legal way to buy it.

EXP-021 already measured a cross-modal masked pretext at **+3.0 over from-scratch**.
It was abandoned because the *transformer fusion trunk* it fed lost by 8 points for
unrelated reasons, and the objective was never re-attached to a CNN.  This script is
that re-attachment, and trunk identity is the whole point: the encoder here is
literally `VisualMILNet`'s adapters + `SharedVisualTrunk`, so the learned weights
load into the fine-tune model without a shape or semantics mismatch.

The objective
-------------
Mask whole modalities and spatial patches, encode what survives, fuse across the
surviving modalities, and reconstruct the masked content with a light deconv head.
L1 is taken on masked regions only, so the loss cannot be satisfied by copying
visible input.  Reconstructing depth/thermal from IR (and vice versa) requires
representing *what the object is*, which is exactly the capability EXP-068 proved
the supervised branch never learned (it is motion-only: flat to 96x128 resolution,
but a single frame drops the whole-body classes to exactly 0.000).

Deliberately NOT used: temporal-order or temporal-contrastive pretexts.  They would
reinforce the motion shortcut this is meant to break.

Corpus: all 3,338 clips (2,933 train + 405 test).  Using unlabelled test clips is
test-time transductive processing, which the organizers ruled legal (OBJECTIVE.md);
no label is read and no test sample is inspected by hand.

    CUHKX_CACHE_SEGMENTS=32 CUHKX_CACHE_HEIGHT=96 CUHKX_CACHE_WIDTH=128 \
    python3 code/ssl_pretrain_visual.py --cache-dir cache/visual_mil_v2 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import train_visual_mil as T

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
ARTIFACTS = ROOT / "research" / "artifacts"
MODALITIES = ("ir", "depth", "thermal")
CHANNELS = {"ir": 2, "depth": 3, "thermal": 4}


class SSLClipDataset(Dataset):
    """Every clip in a split, unlabelled, reusing the trainer's own derivations."""

    def __init__(self, cache_dir: Path, splits=("train", "test"), frames=None):
        self.frames = frames
        self.items: list[tuple[Path, str]] = []
        for split in splits:
            for path in sorted((cache_dir / split).glob("*.npz")):
                self.items.append((path, split))
        if not self.items:
            raise SystemExit(f"no cache files under {cache_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, _ = self.items[index]
        sample = T.load_visual_sample(path)
        # Reuse the supervised path's exact feature construction so the SSL trunk
        # sees the same tensors it will see at fine-tune time.
        return T.derive_features(sample)


def _fallback_features(path: Path, frames: int | None = None):
    """Build the 2/3/4-channel stacks directly if the trainer exposes no helper.

    ``frames`` subsamples the clip.  The reconstruction target is the frame-averaged
    image and the trunk is frame-wise (its only temporal mixing is a +/-1 channel
    shift), so SSL does not need the full 32-frame stack to learn appearance, and
    the resulting weights load into a 32-frame fine-tune unchanged.  Full-length
    clips at batch 4 exhaust the 8 GB card.
    """
    with np.load(path, allow_pickle=True) as z:
        ir = torch.from_numpy(z["ir"].astype(np.float32) / 255.0)[:, None]
        depth = torch.from_numpy(z["depth"].astype(np.float32) / 255.0)[:, None]
        thermal = torch.from_numpy(
            z["thermal"].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        masks = torch.from_numpy(
            np.stack([z["ir_frame_mask"], z["depth_frame_mask"],
                      z["thermal_frame_mask"]], 0).astype(np.float32)
        )
    if frames is not None and frames < ir.shape[0]:
        idx = torch.linspace(0, ir.shape[0] - 1, frames).round().long()
        ir, depth, thermal, masks = ir[idx], depth[idx], thermal[idx], masks[:, idx]
    ir_m, depth_m, thermal_m = masks[0], masks[1], masks[2]
    d_ir = torch.zeros_like(ir)
    d_ir[1:] = (ir[1:] - ir[:-1]).abs()
    depth_valid = (depth > 0).float() * depth_m[:, None, None, None]
    d_depth = torch.zeros_like(depth)
    d_depth[1:] = (depth[1:] - depth[:-1]).abs() * depth_valid[1:] * depth_valid[:-1]
    luma = 0.2126 * thermal[:, 0:1] + 0.7152 * thermal[:, 1:2] + 0.0722 * thermal[:, 2:3]
    d_luma = torch.zeros_like(luma)
    d_luma[1:] = (luma[1:] - luma[:-1]).abs()
    return {
        "ir": torch.cat([ir * ir_m[:, None, None, None], d_ir], 1),
        "depth": torch.cat([(2 * depth - 1) * depth_valid, depth_valid, d_depth], 1),
        "thermal": torch.cat([thermal * thermal_m[:, None, None, None], d_luma], 1),
        "frame_masks": masks,
    }


class Decoder(nn.Module):
    """Light deconv head: 1/8-resolution trunk features -> masked pixels."""

    def __init__(self, channels_in: int, channels_out: int, width: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, width, 1, bias=False),
            nn.GroupNorm(8, width), nn.SiLU(inplace=True),
            nn.ConvTranspose2d(width, width // 2, 4, 2, 1, bias=False),
            nn.GroupNorm(8, width // 2), nn.SiLU(inplace=True),
            nn.ConvTranspose2d(width // 2, width // 4, 4, 2, 1, bias=False),
            nn.GroupNorm(8, width // 4), nn.SiLU(inplace=True),
            nn.ConvTranspose2d(width // 4, channels_out, 4, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalMAE(nn.Module):
    """VisualMILNet's adapters + SharedVisualTrunk, plus per-modality decoders."""

    def __init__(self):
        super().__init__()
        host = T.VisualMILNet()
        self.ir_adapter = host.ir_adapter
        self.depth_adapter = host.depth_adapter
        self.thermal_adapter = host.thermal_adapter
        self.shared_trunk = host.shared_trunk
        c = self.shared_trunk.channels_out
        self.decoders = nn.ModuleDict(
            {m: Decoder(c, CHANNELS[m]) for m in MODALITIES}
        )

    def adapters(self):
        return {"ir": self.ir_adapter, "depth": self.depth_adapter,
                "thermal": self.thermal_adapter}

    def encode(self, features, frame_mask, modality):
        b, t, c, h, w = features.shape
        hidden = self.adapters()[modality](features.reshape(b * t, c, h, w))
        hidden = hidden.reshape(b, t, *hidden.shape[1:])
        return self.shared_trunk(hidden, frame_mask)

    def forward(self, batch, drop, patch_mask):
        """drop: (B, 3) 1 = modality withheld. patch_mask: (B,1,H/8,W/8) 1 = hidden."""
        pooled, n = None, 0
        for i, m in enumerate(MODALITIES):
            keep = (1.0 - drop[:, i])[:, None, None, None, None]
            if float(keep.sum()) == 0:
                continue
            feats = self.encode(batch[m] * keep, batch["frame_masks"][:, i], m)
            # features are (B, T, C, H/8, W/8); the patch mask is (B, H/8, W/8)
            feats = feats * (1.0 - patch_mask)[:, None, None]
            pooled = feats if pooled is None else pooled + feats
            n += 1
        pooled = pooled / max(n, 1)
        # temporal mean: reconstruction is per-frame-averaged appearance content
        context = pooled.mean(1)
        return {m: self.decoders[m](context) for m in MODALITIES}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "visual_mil_v2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames", type=int, default=12,
                    help="frames per clip for SSL; the trunk is frame-wise so "
                         "these weights still load into a 32-frame fine-tune")
    ap.add_argument("--modality-drop", type=float, default=0.5)
    ap.add_argument("--patch-drop", type=float, default=0.5)
    ap.add_argument("--tag", default="ssl_trunk_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    SSLClipDataset.__getitem__ = (
        lambda self, i: _fallback_features(self.items[i][0], self.frames))
    ds = SSLClipDataset(args.cache_dir, frames=args.frames)
    print(f"SSL corpus: {len(ds)} clips from {args.cache_dir} "
          f"({args.frames} of {T.CACHE_STEPS} frames at "
          f"{T.INPUT_HEIGHT}x{T.INPUT_WIDTH})", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)

    model = CrossModalMAE().to(device)
    trunk_params = sum(p.numel() for n, p in model.named_parameters()
                       if not n.startswith("decoders"))
    print(f"trunk parameters {trunk_params:,} (decoders are discarded after SSL)",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * len(loader), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    fh, fw = T.INPUT_HEIGHT // 8, T.INPUT_WIDTH // 8

    for epoch in range(args.epochs):
        model.train()
        total, seen, t0 = 0.0, 0, time.time()
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            b = batch["ir"].shape[0]
            # Withhold at least one modality per clip, so the task is always
            # cross-modal and never solvable by autoencoding one stream.
            drop = (torch.rand(b, 3, device=device) < args.modality_drop).float()
            forced = torch.randint(0, 3, (b,), device=device)
            drop[torch.arange(b, device=device), forced] = 1.0
            drop[drop.sum(1) == 3] = 0.0
            drop[torch.arange(b, device=device), forced] = 1.0
            patch = (torch.rand(b, fh, fw, device=device) < args.patch_drop).float()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(batch, drop, patch)
                loss, terms = 0.0, 0
                up = F.interpolate(patch[:, None], size=(T.INPUT_HEIGHT, T.INPUT_WIDTH),
                                   mode="nearest")
                for i, m in enumerate(MODALITIES):
                    target = batch[m].mean(1)          # frame-averaged appearance
                    # Supervise only where content was actually withheld.
                    region = torch.maximum(up, drop[:, i][:, None, None, None])
                    denom = region.sum().clamp_min(1.0) * target.shape[1]
                    loss = loss + ((out[m] - target).abs() * region).sum() / denom
                    terms += 1
                loss = loss / terms
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += float(loss.detach()) * b; seen += b
        print(f"epoch {epoch+1:3d}/{args.epochs} masked-L1 {total/max(seen,1):.5f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

        CKPT.mkdir(exist_ok=True)
        trunk = {n: p.detach().cpu() for n, p in model.state_dict().items()
                 if not n.startswith("decoders")}
        torch.save({"state_dict": trunk, "epoch": epoch + 1, "args": vars(args),
                    "cache_geometry": [T.CACHE_STEPS, T.INPUT_HEIGHT, T.INPUT_WIDTH],
                    "model_config": T.model_config()},
                   CKPT / f"{args.tag}.pt")
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"{args.tag}_manifest.json").write_text(json.dumps(
        {"corpus": len(ds), "cache_dir": str(args.cache_dir),
         "args": {k: str(v) for k, v in vars(args).items()}},
        indent=1))
    print(f"wrote {CKPT / (args.tag + '.pt')}", flush=True)


if __name__ == "__main__":
    main()
