#!/usr/bin/env python3
"""Test inference for the EXP-086 person-crop video member.

The member trains with `cross_entropy(logits + log_prior)` (logit adjustment), so
its raw softmax at inference is already in UNIFORM-prior space -- the same space
`code/fuse_prior_visual.py` expects on its `--visual` slot, which is why no prior
division happens here. Adding log_prior back at inference would double-count the
train skew and is the single easiest way to silently break this member.

Emits the canonical 405-row order taken from sample_submission.csv, not from the
cache index, so a row can never shift against the submission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np
import torch

import torch.nn as nn

from train_video_crop import CropClips, build_model, predict, KIN_MEAN, KIN_STD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache", "crop_v1")
ART = os.path.join(ROOT, "research", "artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
SAMPLE = os.path.join(ROOT, "Small-Model-Track", "Testing", "sample_submission.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="vid_r2p1d_f2")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", default=None)
    p.add_argument("--submission", default=None)
    p.add_argument("--no-flip-tta", action="store_true")
    p.add_argument("--adabn", action="store_true",
                   help=("re-estimate BatchNorm running statistics from the UNLABELED "
                         "test clips before predicting (AdaBN, EXP-099). Parameter-free "
                         "test-time adaptation: +1.53 micro / +11 object clips measured "
                         "on held-out fold 2 with vid_ig65m_f2. Labels are never used."))
    p.add_argument("--cache", default="crop_v1",
                   help="cache directory under cache/ (crop_v1 = IR+depth 4ch, "
                        "thermal_v1 = thermal 3ch)")
    args = p.parse_args()

    device = torch.device("cuda")
    package = torch.load(os.path.join(CKPT, f"{args.tag}.pt"),
                         map_location="cpu", weights_only=False)
    saved = package["args"]
    # Channel count comes from the cache the checkpoint was trained on, and the
    # normalization must match it or every prediction is silently miscalibrated.
    cache_dir = os.path.join(ROOT, "cache", args.cache)
    n_ch = int(json.load(open(os.path.join(cache_dir, "test_index.json")))["channels"])
    mean, std = (None, None) if n_ch == 4 else (
        torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1),
        torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))
    model = build_model(str(saved.get("arch", "r2plus1d_18")), in_channels=n_ch)
    model.load_state_dict(package["state_dict"])
    model = model.to(device).eval()
    print(f"{args.tag}: arch={saved.get('arch')}  fold-2 OOF "
          f"micro={package.get('micro'):.5f} object={package.get('object'):.5f}")

    with open(SAMPLE) as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    paths = [r[0] for r in body]
    sids = [q.strip("/").split("/")[-1] for q in paths]

    index = json.load(open(os.path.join(cache_dir, "test_index.json")))
    row_of = {s: i for i, s in enumerate(index["sids"])}
    missing = [s for s in sids if s not in row_of]
    assert not missing, f"{len(missing)} submission clips absent from the crop cache: {missing[:5]}"
    order = np.array([row_of[s] for s in sids])

    memmap = np.load(os.path.join(cache_dir, "test.npy"), mmap_mode="r")
    loader = torch.utils.data.DataLoader(
        CropClips(memmap, order, None, False),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    if args.adabn:
        # Cumulative-average re-estimation over the target inputs. momentum=None makes
        # each BN layer average over every batch seen, so the result is independent of
        # batch order; only the running buffers change, never a weight.
        bn = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for m in bn:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        amean = (KIN_MEAN if mean is None else mean).to(device)
        astd = (KIN_STD if std is None else std).to(device)
        with torch.no_grad():
            for x, _ in loader:
                x = (x.to(device) - amean) / astd
                with torch.autocast("cuda", dtype=torch.float16):
                    model(x)
        for m in bn:
            m.eval()
        print(f"  AdaBN: re-estimated {len(bn)} BatchNorm layers on {len(sids)} "
              f"unlabeled test clips")

    probs, _ = predict(model, loader, device, flip_tta=not args.no_flip_tta,
                       mean=mean, std=std)
    probs = probs.astype(np.float32)
    assert probs.shape == (len(sids), 40), probs.shape

    suffix = "_adabn" if args.adabn else ""
    output = args.output or os.path.join(ART, f"testprobs_{args.tag}{suffix}.npz")
    np.savez_compressed(output, probs=probs, sids=np.array(sids, dtype="<U15"))
    with open(output, "rb") as handle:
        print(f"wrote {output}\n  sha256={hashlib.sha256(handle.read()).hexdigest()}")

    if args.submission:
        pred = probs.argmax(1)
        with open(args.submission, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for path, k in zip(paths, pred):
                writer.writerow([path, int(k)])
        print(f"wrote submission {args.submission}  "
              f"distinct classes={len(set(pred.tolist()))}/40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
