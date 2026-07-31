#!/usr/bin/env python3
"""Dump validation probabilities for a checkpoint on one outer fold.

This exists to build a *partition-matched* baseline for the visual MIL screen.
The historical `oof_astgcn_world25.npz` cannot serve that role on all-18 fold 2:
it omits users 5 and 21 entirely, and for the clips it does cover its
predictions come from models trained on a different subject partition (old
fold 2 trained on user6; old fold 3 trained on users 22 and 24; user5 was an
`always_train` user in every old split). Fusing across that mismatch would
bias the visual member's marginal.

Usage
-----
    CUHKX_FOLD_FILE=research/artifacts/cv_folds_all18.json \
    python3 code/baseline_all18_probs.py \
        --checkpoint checkpoints/astgcn_all18_f2.pt \
        --fold 2 --modality skelg --arch adaptive \
        --output research/artifacts/oof_astgcn_all18_f2_best.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import har_data
import har_models

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--modality", default="skelg")
    ap.add_argument("--arch", default="adaptive")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--feat", default="jv")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--t-skel", type=int, default=32)
    ap.add_argument("--skel-time", default="stretch")
    ap.add_argument("--person", default="first")
    ap.add_argument("--imu-feat", default="raw")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw2 = dict(
        feat=args.feat,
        n_frames=args.n_frames,
        person=args.person,
        imu_feat=args.imu_feat,
        t_skel=args.t_skel,
        skel_time=args.skel_time,
    )
    va = har_data.HARDataset(
        "train", args.modality, args.fold, "val", aug=False, **kw2
    )
    loader = DataLoader(
        va, args.bs, shuffle=False, num_workers=args.workers,
        pin_memory=dev == "cuda",
    )

    in_ch = va.skel_dim() if args.modality == "skel" else (
        va.imu_dim() if args.modality == "imu" else None
    )
    kw = {"width": args.width} if args.width else {}
    if args.arch:
        kw["arch"] = args.arch
    model = har_models.build(args.modality, in_ch=in_ch, **kw).to(dev)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x.to(dev))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.append(y.numpy())
    probs = np.concatenate(probs).astype(np.float32)
    labels = np.concatenate(labels).astype(np.int64)
    sids = np.array(va.ids, dtype="<U15")

    if len(sids) != len(probs):
        raise RuntimeError(
            f"row mismatch: {len(sids)} ids vs {len(probs)} predictions"
        )

    fold_file = har_data.FOLDS
    with open(fold_file) as handle:
        protocol = json.load(handle)
    val_users = sorted(protocol["folds"][args.fold]["val_users"])

    accuracy = float((probs.argmax(1) == labels).mean())
    np.savez_compressed(
        args.output,
        probs=probs,
        labels=labels,
        sids=sids,
        profile=np.array(os.path.basename(args.checkpoint), dtype="<U64"),
        fold=np.array(args.fold),
        fold_file=np.array(os.path.basename(fold_file), dtype="<U32"),
        val_users=np.array(val_users, dtype="<U16"),
    )
    print(
        f"wrote {args.output}: {len(sids)} clips, fold {args.fold} "
        f"({','.join(val_users)}), micro accuracy {accuracy:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
