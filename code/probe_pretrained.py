#!/usr/bin/env python3
"""EXP-079: frozen-encoder linear probe — ImageNet vs random vs SSL, one classifier.

Why this experiment and not a fine-tune
---------------------------------------
Organizer ruling R-1 (research/RULES_VERIFIED.md, discussion topic 711665) says
ImageNet-pretrained small CNNs are permitted; a prior session recorded the opposite
as fact and cancelled the pretrained probe on that basis.  Before spending GPU-days
re-architecting around a pretrained backbone, establish the cheapest discriminating
measurement the directive asks for: freeze each encoder, train the SAME linear
classifier on the SAME split, and compare.

This separates two things our fine-tuning experiments cannot:
  * information ACCESSIBILITY -- does an ImageNet representation expose class
    structure our from-scratch trunk never found?
  * model CAPABILITY -- is our trunk/optimization the limit rather than the data?

Random-init ResNet18 is the control: identical architecture, identical probe, so any
gap is attributable to the weights alone and not to depth, width, or feature dim.

Split is fold 2, taken from an existing OOF artifact so it is bit-identical to the
partition every visual experiment has used.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache", "visual_mil_v1")
ART = os.path.join(ROOT, "research", "artifacts")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ClipDataset(Dataset):
    """Yields (3*T, 3, H, W) -- every modality's frames stacked as RGB-like images."""

    def __init__(self, sids: list[str], split: str, stride: int):
        self.sids = sids
        self.split = split
        self.stride = stride

    def __len__(self) -> int:
        return len(self.sids)

    def __getitem__(self, index: int):
        data = np.load(os.path.join(CACHE, self.split, f"{self.sids[index]}.npz"))
        frames = []
        for name in ("ir", "depth", "thermal"):
            arr = np.asarray(data[name])[:: self.stride]
            if arr.ndim == 3:                      # (T,H,W) grey -> replicate to 3ch
                arr = np.repeat(arr[..., None], 3, axis=-1)
            frames.append(arr)
        stack = np.concatenate(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(stack).permute(0, 3, 1, 2)   # (N,3,H,W)


def build_encoder(kind: str, device: torch.device) -> tuple[nn.Module, int]:
    from torchvision.models import resnet18, ResNet18_Weights

    if kind == "imagenet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif kind == "random":
        torch.manual_seed(0)
        model = resnet18(weights=None)
    else:
        raise SystemExit(f"unknown encoder {kind}")
    model.fc = nn.Identity()
    return model.eval().to(device), 512


@torch.no_grad()
def extract(sids, split, kind, device, stride, batch):
    encoder, dim = build_encoder(kind, device)
    loader = DataLoader(ClipDataset(sids, split, stride), batch_size=batch,
                        num_workers=4, shuffle=False)
    out = np.zeros((len(sids), dim * 3), dtype=np.float32)
    row = 0
    start = time.time()
    for clips in loader:
        b, n = clips.shape[0], clips.shape[1]
        flat = clips.reshape(b * n, *clips.shape[2:]).to(device, non_blocking=True)
        flat = (flat - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        feats = encoder(flat).reshape(b, n, dim)
        # n frames split equally across the three modalities; mean-pool within each
        per_modality = feats.reshape(b, 3, n // 3, dim).mean(2)
        out[row:row + b] = per_modality.reshape(b, dim * 3).float().cpu().numpy()
        row += b
    print(f"    extracted {kind:9s} {split:5s} {out.shape} in {time.time()-start:.0f}s")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=["imagenet", "random"])
    parser.add_argument("--stride", type=int, default=2, help="frame subsample")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--oof", default="oof_v2_scratch_f2_f2.npz",
                        help="artifact defining the fold-2 outer split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = np.load(os.path.join(ART, args.oof), allow_pickle=True)
    val_sids = [str(s) for s in ref["sids"]]
    val_labels = np.asarray(ref["labels"])

    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as handle:
        labels = {r["sample_id"]: int(r["class_id"]) for r in csv.DictReader(handle)}
    val_set = set(val_sids)
    train_sids = sorted(s for s in labels if s not in val_set
                        and os.path.exists(os.path.join(CACHE, "train", f"{s}.npz")))
    train_labels = np.array([labels[s] for s in train_sids])
    print(f"fold 2: train={len(train_sids)} outer={len(val_sids)}  device={device}")

    OBJECT = np.array(sorted(set(range(28)) | {37, 38, 39}))
    obj_mask = np.isin(val_labels, OBJECT)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    for kind in args.encoders:
        print(f"\n=== encoder: {kind} ===")
        xtr = extract(train_sids, "train", kind, device, args.stride, args.batch)
        xva = extract(val_sids, "train", kind, device, args.stride, args.batch)
        scaler = StandardScaler().fit(xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf.fit(scaler.transform(xtr), train_labels)
        pred = clf.predict(scaler.transform(xva))
        micro = (pred == val_labels).mean()
        obj = (pred[obj_mask] == val_labels[obj_mask]).mean()
        mot = (pred[~obj_mask] == val_labels[~obj_mask]).mean()
        print(f"  >>> {kind:9s} micro={micro:.5f}  object={obj:.5f} "
              f"({int((pred[obj_mask]==val_labels[obj_mask]).sum())}/{int(obj_mask.sum())})"
              f"  motion={mot:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
