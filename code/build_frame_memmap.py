#!/usr/bin/env python3
"""Pack the visual cache into a uint8 memmap for cheap per-frame random access.

The appearance model wants batches of frames drawn from as many different clips as
possible -- a batch of 96 random frames spanning 96 clips learns markedly faster
(clip-acc 0.193 after two epochs) than 32 clips contributing 4 correlated frames
each (0.120 after ten).  But sampling per (clip, frame) against the npz cache
re-decodes a whole 16-frame archive for every single frame: 160 s per epoch with
the GPU mostly idle.

Packing to a flat memmap resolves the conflict.  At 96x128 -- the resolution
EXP-068 showed is already more than the model uses -- train is
2933 x 16 x 3 x 96 x 128 bytes = 1.73 GB and test 239 MB, so the OS page cache
holds it and per-frame access costs nothing.

    python3 code/build_frame_memmap.py --split train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "visual_mil_v1"
OUT = ROOT / "cache" / "frame_memmap"
FRAMES, HEIGHT, WIDTH = 16, 96, 128


def build(split: str) -> None:
    files = sorted((CACHE / split).glob("*.npz"))
    if not files:
        raise SystemExit(f"no cache files in {CACHE / split}")
    sids = [f.stem for f in files]
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.lib.format.open_memmap(
        OUT / f"{split}_frames.npy", mode="w+", dtype=np.uint8,
        shape=(len(sids), FRAMES, 3, HEIGHT, WIDTH))
    masks = np.zeros((len(sids), FRAMES, 3), dtype=np.uint8)

    for i, path in enumerate(files):
        with np.load(path, allow_pickle=True) as z:
            ir = z["ir"].astype(np.float32)
            depth = z["depth"].astype(np.float32)
            thermal = z["thermal"].astype(np.float32).mean(-1)
            masks[i] = np.stack([z["ir_frame_mask"], z["depth_frame_mask"],
                                 z["thermal_frame_mask"]], 1).astype(np.uint8)
        stack = torch.from_numpy(np.stack([ir, depth, thermal], 1))
        small = F.interpolate(stack, size=(HEIGHT, WIDTH), mode="area")
        data[i] = small.clamp(0, 255).round().to(torch.uint8).numpy()
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(sids)}", flush=True)
    data.flush()
    np.save(OUT / f"{split}_masks.npy", masks)
    (OUT / f"{split}_sids.json").write_text(json.dumps(sids))
    size = (OUT / f"{split}_frames.npy").stat().st_size
    print(f"{split}: {len(sids)} clips -> {size/1e9:.2f} GB at {HEIGHT}x{WIDTH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=("train", "test", "both"))
    args = ap.parse_args()
    for split in (("train", "test") if args.split == "both" else (args.split,)):
        build(split)


if __name__ == "__main__":
    main()
