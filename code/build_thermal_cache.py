#!/usr/bin/env python3
"""EXP-088: YOLO person-crop cache for THERMAL, the modality both public notebooks discard.

Why thermal, and why now
------------------------
The dataset paper's own per-modality baselines rank thermal FIRST: Thermal 92.57 >
Depth 90.5 > IR 90.2 > Skeleton 79.1 > mmWave 46.6 > IMU 45.5. Both public notebooks
we ported from (`lb-0-711-yolo-person-crop-r2plus1d-100mb`, `lb0-667-baseline-with-
yolo-person-crop`) feed only Depth_Color + IR and throw thermal away. It covers
395/405 test clips.

Our own thermal member already exists and is WORTHLESS in the current ensemble:
`pre_thermal_f*` (ImageNet ResNet on full thermal frames) scores 0.34783 solo and
*monotonically degrades* the fusion from weight 0.05 upward. That refutes the member,
not the modality -- it is the same 2D-ImageNet family that the crop+video recipe beat
by +23.9 micro on IR/depth (EXP-086). The untested claim is thermal THROUGH that recipe.

The alignment problem does not apply here
-----------------------------------------
`code/visual_mil_cache.py:518` samples thermal on independent normalized centers and
never matches it to the IR anchor timestamps, so ~22% of clips are >0.5 s out of sync.
That bug only bites when thermal shares a tensor with IR/depth. A thermal-ONLY member
reads nothing but thermal, so endpoint-uniform sampling over the clip's own frames is
exactly correct and no cross-modal alignment is needed. The modality enters the
ensemble through the fusion instead, where EXP-086 measured that a decorrelated member
earns weight far above its solo strength (IMU: solo 0.36, fusion weight 0.45).

Detection runs on the thermal frames themselves rather than reusing the IR crop window:
thermal is a different camera (320x240 vs 640x480, different FOV), so an IR-derived box
does not map into thermal coordinates without a homography we have not measured. People
are high-contrast against room temperature, which is the favourable case for a person
detector; clips with no detection fall back to the full frame, as in the IR builder.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from build_crop_cache import (CROP_MARGIN, DETECTION_FRAMES, IMAGE_SIZE, MIN_SIDE_FRACTION,
                              N_FRAMES, PERSON_CONFIDENCE, crop_window, endpoint_uniform)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = ROOT / "Small-Model-Track" / "Training" / "data" / "HAR" / "data"
TEST_ROOT = ROOT / "Small-Model-Track" / "Testing" / "data" / "small_model_track_test"
OUT_ROOT = ROOT / "cache" / "thermal_v1"
CHANNELS = 3                       # ironbow JPG kept as RGB; it is not cleanly invertible
THERMAL_RE = re.compile(r"(\d+)")


def thermal_frames(directory: Path) -> list[Path]:
    """Sorted by the session frame counter in `frame_NNNNNN.jpg`, numerically."""
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    def key(p: Path) -> int:
        m = THERMAL_RE.search(p.stem)
        return int(m.group(1)) if m else -1

    return sorted(files, key=key)


def load_clip(model, thermal_dir: Path, full_frame: bool = False) -> np.ndarray:
    """full_frame=True skips person detection entirely and keeps the whole thermal view.

    EXP-118: the team tied with us at 166 runs a thermal pipeline on UNCROPPED frames.
    The hypothesis is that in thermal the discriminative cue for an OBJECT class is the
    object's own heat signature -- a kettle, a laptop, a stove, a running tap -- rather
    than the subject's pose, and cropping to the person deletes it. 75% of our residual
    error is OBJECT classes, so this is exactly where it would show.
    """
    frames = thermal_frames(thermal_dir)
    out = np.zeros((N_FRAMES, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if not frames:
        return out
    width, height = 320, 240
    for candidate in frames:
        try:
            with Image.open(candidate) as im:
                width, height = im.size
            break
        except Exception:
            continue
    window = None if full_frame else crop_window(model, frames, width, height)
    for slot, path in enumerate(endpoint_uniform(frames, N_FRAMES)):
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        if window is not None:
            image = image.crop(window)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        out[slot] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
    return out


def main() -> int:
    global OUT_ROOT, IMAGE_SIZE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full-frame", action="store_true",
                    help="no person crop; keep the whole thermal view (EXP-118)")
    ap.add_argument("--out-name", default="",
                    help="cache dir under cache/ (default thermal_v1, or "
                         "thermal_full when --full-frame)")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = ap.parse_args()

    IMAGE_SIZE = args.image_size
    OUT_ROOT = ROOT / "cache" / (args.out_name or
                                 ("thermal_full" if args.full_frame else "thermal_v1"))

    model = None
    if not args.full_frame:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")   # only needed when we actually crop

    jobs: list[tuple[str, Path]] = []
    if args.split == "train":
        # Enumerate from IR so the sample ids and their ORDER match cache/crop_v1
        # exactly; a thermal-only enumeration would silently drop the 103 clips that
        # differ in coverage and shift every row against the other cache.
        for class_dir in sorted((TRAIN_ROOT / "IR").iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = int(class_dir.name.split("_")[0])
            for user_dir in sorted(class_dir.iterdir()):
                for trial_dir in sorted(user_dir.iterdir()):
                    sid = f"{class_id:02d}_{user_dir.name}_{trial_dir.name}"
                    jobs.append((sid, Path(str(trial_dir).replace("/IR/", "/Thermal/"))))
    else:
        for clip_dir in sorted(TEST_ROOT.glob("SM_test_*")):
            if clip_dir.is_dir():
                jobs.append((clip_dir.name, clip_dir / "Thermal"))
    if args.limit:
        jobs = jobs[: args.limit]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{args.split}.npy"
    memmap = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8,
        shape=(len(jobs), N_FRAMES, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    nonempty = 0
    for i, (sid, thermal_dir) in enumerate(jobs):
        memmap[i] = load_clip(model, thermal_dir, args.full_frame)
        if memmap[i].any():
            nonempty += 1
        if i % 100 == 0 or i == len(jobs) - 1:
            print(f"  {args.split} {i+1}/{len(jobs)} nonempty={nonempty}", flush=True)
    memmap.flush()
    with open(OUT_ROOT / f"{args.split}_index.json", "w") as handle:
        json.dump({"sids": [j[0] for j in jobs], "n_frames": N_FRAMES,
                   "channels": CHANNELS, "image_size": IMAGE_SIZE,
                   "full_frame": bool(args.full_frame),
                   "crop_margin": CROP_MARGIN,
                   "min_side_fraction": MIN_SIDE_FRACTION,
                   "detection_frames": DETECTION_FRAMES,
                   "person_confidence": PERSON_CONFIDENCE}, handle, indent=2)
    print(f"wrote {path}  clips={len(jobs)}  nonempty={nonempty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
