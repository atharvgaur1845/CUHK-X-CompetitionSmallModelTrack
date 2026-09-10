#!/usr/bin/env python3
"""EXP-149 gate -- do the quantised detectors produce the SAME crop windows, at scale?

The person window survives a lot of detector jitter by construction: it is the
max-confidence box over 8 frames, squared, expanded by margin 1.4, clipped to the frame
and rounded to integer pixels. The WRIST window is not built that way -- it comes from
yolo11n-pose keypoints -- so it must be gated separately rather than assumed to inherit
the person view's robustness.

Gate: every window identical to the fp16 baseline, on every clip of the split. Not "IoU
above a threshold" -- a shifted window changes the model's INPUT, and no downstream check
in unpack_stage2.py would catch it.
"""
from __future__ import annotations
import argparse, json, shutil, sys, os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "kaggle"))
import bitpack                                            # noqa: E402
from quantize_detector import quantize_sd                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, default=5)
    ap.add_argument("--crop", default="person", choices=("person", "wrist"))
    ap.add_argument("--split", default="test", choices=("test", "train"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import cuhkx_224_kaggle as K
    from ultralytics import YOLO

    paths = K.find_paths(str(ROOT), "crop_224", str(ROOT / "cache" / "crop_224"))
    train_jobs, test_jobs = K.build_jobs(paths)
    jobs = test_jobs if a.split == "test" else train_jobs
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{a.crop} windows, {a.split}: {len(jobs)} clips, int{a.bits}")

    det = YOLO("yolo11n.pt"); pose = YOLO("yolo11n-pose.pt")
    base_sd = {"det": {k: v.clone() for k, v in det.model.state_dict().items()},
               "pose": {k: v.clone() for k, v in pose.model.state_dict().items()}}

    def run(tag):
        d = Path(f"/tmp/winchk_{tag}")
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        return K.compute_windows(jobs, d, "cpu", a.crop)

    base = run("base")
    qd, _ = quantize_sd(base_sd["det"], a.bits)
    qp, _ = quantize_sd(base_sd["pose"], a.bits)
    # patch the classes YOLO() will construct inside compute_windows
    _orig = YOLO.__init__
    def patched(self, *args, **kw):
        _orig(self, *args, **kw)
        name = str(args[0]) if args else ""
        if "pose" in name:
            self.model.load_state_dict(qp)
        elif "yolo11n" in name:
            self.model.load_state_dict(qd)
    YOLO.__init__ = patched
    quant = run("quant")
    YOLO.__init__ = _orig

    ident = sum(1 for s in base if base[s] == quant.get(s))
    bad = [s for s in base if base[s] != quant.get(s)]
    print(f"\n  identical windows: {ident}/{len(base)}")
    if bad:
        print(f"  DIFFER on {len(bad)}: {bad[:5]}")
        for s in bad[:3]:
            print(f"    {s}: fp16 {base[s]}  int{a.bits} {quant[s]}")
    print(f"  GATE: {'PASS' if not bad else 'FAIL'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
