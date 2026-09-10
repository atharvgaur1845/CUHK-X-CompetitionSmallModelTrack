#!/usr/bin/env python3
"""EXP-149 -- int8 the YOLO detectors so they fit inside the 100 MB package.

WHY THIS IS REQUIRED, not an optimisation. Organiser ruling 2026-09-07 (topic 738333)
allows YOLO person cropping "with the detector weights included in your <100 MB checkpoint
package". `stage2_ship3.pth` contains no detector at all, and the on-site stage runs on a
brand-new private dataset where crops must be computed from raw frames -- so a package
without a detector cannot run. We need 11.87 MB of detector against 3.53 MB of headroom.

WHY THIS NEEDS ITS OWN GATE, unlike the video quantisation. Quantising a classifier is
verifiable against a reference: EXP-144 checked the dequantised tensors were bit-identical
and the argmaxes reproduced 405/405. A DETECTOR is different -- its output is a crop
window, which changes the model's INPUT, so a shifted box propagates into every downstream
member and nothing about the classifier check would catch it. The gate here is therefore
on the WINDOWS themselves, against the cached ones every existing member was trained and
scored on.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def quantize_sd(sd, bits=8):
    """Symmetric per-output-channel, weight-only, dim>=2 -- the video branch's scheme."""
    qmax = 2 ** (bits - 1) - 1
    out, nq = {}, 0
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point() and v.dim() >= 2:
            w = v.reshape(v.shape[0], -1).float()
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            c = (w / s).round().clamp(-qmax - 1, qmax)
            out[k] = (c * s).reshape(v.shape).to(v.dtype)
            nq += 1
        else:
            out[k] = v
    return out, nq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--cache", default="cache/crop_224",
                    help="cache whose windows.json is the reference")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--bits", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "kaggle"))
    import cuhkx_224_kaggle as K
    from ultralytics import YOLO

    ref = json.loads((ROOT / a.cache / "windows.json").read_text())
    paths = K.find_paths(str(ROOT), "crop_224", str(ROOT / a.cache))
    train_jobs, test_jobs = K.build_jobs(paths)
    jobs = (test_jobs if a.split == "test" else train_jobs)
    jobs = [j for j in jobs if j[0] in ref][: a.limit]
    print(f"comparing windows on {len(jobs)} {a.split} clips from {a.weights}")

    y = YOLO(a.weights)
    sd = y.model.state_dict()
    qsd, nq = quantize_sd(sd, a.bits)
    print(f"  quantised {nq} tensors to int{a.bits} (dequantised back in place)")

    from PIL import Image
    def windows_with(model):
        out = {}
        for sid, ir_dir, _d, _c, _u in jobs:
            frames = K.sorted_frames(Path(ir_dir))
            w, h = 640, 480
            for cand in frames:
                try:
                    with Image.open(cand) as im: w, h = im.size
                    break
                except Exception: continue
            out[sid] = K.crop_window_for(model, frames, w, h) if hasattr(K, "crop_window_for") else None
        return out

    # reuse the trainer's own window routine so this measures the SHIPPED path
    def run(model_obj):
        y.model.load_state_dict(model_obj)
        got = {}
        for n, j in enumerate(jobs):
            wins = K.compute_windows([j], Path("/tmp/qwin"), "cpu", "person")
            got[j[0]] = wins[j[0]]
        return got

    import shutil, os
    shutil.rmtree("/tmp/qwin", ignore_errors=True); os.makedirs("/tmp/qwin", exist_ok=True)
    base = run(sd)
    shutil.rmtree("/tmp/qwin", ignore_errors=True); os.makedirs("/tmp/qwin", exist_ok=True)
    quant = run(qsd)

    same = ident = 0
    ious, shifts = [], []
    for sid in base:
        b, q = base[sid], quant[sid]
        if b == q: ident += 1
        if b is None or q is None:
            continue
        bx = np.array(b, float); qx = np.array(q, float)
        shifts.append(np.abs(bx - qx).max())
        ix = max(0, min(b[2], q[2]) - max(b[0], q[0])); iy = max(0, min(b[3], q[3]) - max(b[1], q[1]))
        inter = ix * iy
        ab = (b[2]-b[0])*(b[3]-b[1]); aq = (q[2]-q[0])*(q[3]-q[1])
        ious.append(inter / max(ab + aq - inter, 1e-9))
        # also compare to the CACHED window every member was built on
        if sid in ref and ref[sid] == q: same += 1
    ious = np.array(ious); shifts = np.array(shifts)
    print(f"\n  identical windows fp16 vs int{a.bits}: {ident}/{len(base)}")
    print(f"  IoU  mean {ious.mean():.5f}  min {ious.min():.5f}  "
          f"below 0.98: {(ious<0.98).sum()}")
    print(f"  max corner shift (px): mean {shifts.mean():.2f}  max {shifts.max():.1f}")
    print(f"  int{a.bits} windows matching the CACHED windows exactly: {same}/{len(base)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
