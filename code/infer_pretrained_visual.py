#!/usr/bin/env python3
"""Test-set inference for the EXP-080 pretrained visual member.

Exists because OOF has misestimated public 5/5 times on this competition (GBDT
stacker, structure decoder, cohort weights, learned gate, SSL-v2 -- the last one
inverted sign). The pretrained member ties the from-scratch trunk on fold-2 object
accuracy while its frozen features beat random init by +70% relative on the same
classes (EXP-079), so the two available signals disagree and only the leaderboard
settles it.

Emits probabilities in the canonical 405-row test order and in the same NPZ schema
the fusion scripts consume, so `code/fuse_prior_visual.py` can take it directly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os

import numpy as np
import torch

from train_pretrained_visual import ClipSet, PretrainedVisual, MEAN, STD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
SAMPLE = os.path.join(ROOT, "Small-Model-Track", "Testing", "sample_submission.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="pre_r18_f2")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", default=None)
    p.add_argument("--submission", default=None)
    p.add_argument("--no-flip-tta", action="store_true",
                   help="disable the horizontal-flip test-time average")
    args = p.parse_args()

    device = torch.device("cuda")
    package = torch.load(os.path.join(CKPT, f"{args.tag}.pt"),
                         map_location="cpu", weights_only=False)
    saved = package["args"]
    stride = int(saved.get("stride", 2))
    # Per-modality members carry a different head width, so the checkpoint's own
    # modality list -- not the module default -- has to rebuild the architecture.
    mods = tuple(saved.get("modalities") or ("ir", "depth", "thermal"))
    # Backbone family and resize both live in the checkpoint's args; rebuilding from
    # module defaults silently constructs a resnet18 and fails on every shape.
    arch = str(saved.get("arch") or "resnet18")
    resize = saved.get("resize")
    model = PretrainedVisual(dropout=float(saved.get("dropout", 0.4)), pretrained=False,
                             n_modalities=len(mods), arch=arch)
    model.load_state_dict(package["state_dict"])
    model = model.to(device).eval()
    print(f"{args.tag}: stride={stride} arch={arch} mods={','.join(mods)}  "
          f"OOF micro={package.get('micro'):.5f} object={package.get('object'):.5f}")

    with open(SAMPLE) as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    # The submission's first column is a path ("small_model_track_test/SM_test_0001/"),
    # not the cache sample id; keep both so rows stay in the canonical order.
    paths = [r[0] for r in body]
    sids = [p.strip("/").split("/")[-1] for p in paths]

    loader = torch.utils.data.DataLoader(
        ClipSet(sids, None, "test", stride, False, modalities=mods,
                resize=tuple(resize) if resize else None),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    out = []
    with torch.no_grad():
        for x, _ in loader:
            x = (x.to(device) - MEAN.to(device)) / STD.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                probs = torch.softmax(model(x).float(), -1)
                if not args.no_flip_tta:
                    # Training applies a p=0.5 horizontal flip but inference never
                    # did, and the model is NOT flip-invariant: it disagrees with its
                    # own mirrored input on ~30% of clips, concentrated exactly on the
                    # handedness/hand-object classes (Put_on_clothes 0.667, Write
                    # 0.600, Drink_water 0.548) and ~0 on gross motion. Averaging the
                    # two views is worth +50/2933 outer-fold clips, positive on 4/4.
                    probs = 0.5 * (probs + torch.softmax(
                        model(torch.flip(x, dims=[-1])).float(), -1))
            out.append(probs.cpu().numpy())
    probs = np.concatenate(out).astype(np.float32)
    assert probs.shape == (len(sids), 40), probs.shape

    output = args.output or os.path.join(ART, f"testprobs_{args.tag}.npz")
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
