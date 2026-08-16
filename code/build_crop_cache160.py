#!/usr/bin/env python3
"""EXP-085: YOLO person-crop cache -- ports the published LB 0.711 preprocessing.

Why this exists
---------------
Kaggle notebook `phuongncn/lb-0-711-yolo-person-crop-r2plus1d-100mb` scores
0.71144 = 143/201 against our 131, and `welshonionman/lb0-667-baseline-with-yolo-
person-crop` (an author of the rank-7 team) scores 0.667. Both hinge on the same
preprocessing: crop to the person before the model ever sees the frame. Their
stated rationale matches what we measured independently -- "the background differs
systematically between rooms. Cropping around the person removes that background
shortcut" -- and our own EXP-083 arm A showed spatial resolution IS binding
(96x128 costs -3.34 object vs 192x256) while we still downsample 640x480 by 2.5x
over a mostly-empty frame.

Faithful to the published recipe, because it is verified and we are not:
  * probe DETECTION_FRAMES endpoint-uniform IR frames with YOLO11n, person class
    only, conf 0.25; take the highest-confidence box per probe
  * UNION those boxes, expand by CROP_MARGIN, square it in 640x480 geometry, and
    floor the side at MIN_SIDE_FRACTION * max(W, H)
  * apply ONE fixed window to all N_FRAMES -- a per-frame moving crop would cancel
    part of the action motion, which is the signal
  * no detection anywhere -> full frame
  * frames chosen endpoint-uniform over a NUMERIC sort of the frame index, so
    _10.png cannot sort between _1 and _2

Channels are Depth_Color (3) + IR (1) = 4, matching the notebook. Thermal is
deliberately excluded HERE only to reproduce their baseline exactly; it is the
paper's best modality (92.57) and both public notebooks discard it, so it is our
clearest edge once this baseline is confirmed.

Output: uint8 memmap (N, N_FRAMES, 4, IMAGE_SIZE, IMAGE_SIZE) + an index JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = ROOT / "Small-Model-Track" / "Training" / "data" / "HAR" / "data"
TEST_ROOT = ROOT / "Small-Model-Track" / "Testing" / "data" / "small_model_track_test"
OUT_ROOT = ROOT / "cache" / "crop_v160"

N_FRAMES = 16
DETECTION_FRAMES = 8
IMAGE_SIZE = 160
CHANNELS = 4
PERSON_CONFIDENCE = 0.25
CROP_MARGIN = 1.4
MIN_SIDE_FRACTION = 0.35
FRAME_RE = re.compile(r"_(\d+)(?:_Color)?\.png$")


def frame_index(path: Path) -> int:
    m = FRAME_RE.search(path.name)
    return int(m.group(1)) if m else -1


def sorted_frames(directory: Path, suffix: str = "") -> list[Path]:
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix == ".png"]
    return sorted(files, key=frame_index)


def endpoint_uniform(items: list, count: int) -> list:
    """Evenly spaced INCLUDING both endpoints; repeats when the clip is short."""
    if not items:
        return []
    if len(items) == 1:
        return items * count
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in idx]


def crop_window(model, ir_frames: list[Path], width: int, height: int):
    """Union of per-probe top-confidence person boxes -> one fixed square window."""
    probes = endpoint_uniform(ir_frames, min(DETECTION_FRAMES, len(ir_frames)))
    boxes = []
    for path in dict.fromkeys(probes):
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        result = model.predict(image, classes=[0], conf=PERSON_CONFIDENCE,
                               verbose=False, device="cpu")[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        conf = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        boxes.append(xyxy[int(conf.argmax())])
    if not boxes:
        return None
    boxes = np.stack(boxes)
    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * CROP_MARGIN
    side = max(side, MIN_SIDE_FRACTION * max(width, height))
    side = min(side, float(min(width, height)))
    half = side / 2.0
    cx = float(np.clip(cx, half, width - half))
    cy = float(np.clip(cy, half, height - half))
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def load_clip(model, ir_dir: Path, depth_dir: Path) -> np.ndarray:
    ir_frames = sorted_frames(ir_dir)
    depth_frames = sorted_frames(depth_dir)
    out = np.zeros((N_FRAMES, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if not ir_frames and not depth_frames:
        return out
    # Four test clips ship all-zero placeholder IR PNGs that PIL cannot even open,
    # so the geometry probe must survive an unreadable frame rather than abort the
    # whole build. Fall through IR -> depth -> the sensor's native 640x480.
    width, height = 640, 480
    for candidate in (*ir_frames, *depth_frames):
        try:
            with Image.open(candidate) as im:
                width, height = im.size
            break
        except Exception:
            continue
    window = crop_window(model, ir_frames, width, height) if ir_frames else None

    def render(paths, channels_out, start):
        for slot, path in enumerate(endpoint_uniform(paths, N_FRAMES)):
            try:
                image = Image.open(path)
                image = image.convert("RGB" if channels_out == 3 else "L")
            except Exception:
                continue                      # all-zero placeholder PNGs exist in test
            if window is not None:
                image = image.crop(window)
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.uint8)
            if channels_out == 3:
                out[slot, start:start + 3] = arr.transpose(2, 0, 1)
            else:
                out[slot, start] = arr

    if depth_frames:
        render(depth_frames, 3, 0)
    if ir_frames:
        render(ir_frames, 1, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")

    jobs: list[tuple[str, Path, Path]] = []
    if args.split == "train":
        for class_dir in sorted((TRAIN_ROOT / "IR").iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = int(class_dir.name.split("_")[0])
            for user_dir in sorted(class_dir.iterdir()):
                for trial_dir in sorted(user_dir.iterdir()):
                    sid = f"{class_id:02d}_{user_dir.name}_{trial_dir.name}"
                    depth = Path(str(trial_dir).replace("/IR/", "/Depth_Color/"))
                    jobs.append((sid, trial_dir, depth))
    else:
        # Match the clip name explicitly: the test root also contains unrelated
        # directories (e.g. a stray .claude), and picking one up shifts every row
        # against the canonical submission order.
        for clip_dir in sorted(TEST_ROOT.glob("SM_test_*")):
            if clip_dir.is_dir():
                jobs.append((clip_dir.name, clip_dir / "IR", clip_dir / "Depth_Color"))
    if args.limit:
        jobs = jobs[: args.limit]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{args.split}.u8"
    memmap = np.lib.format.open_memmap(
        path.with_suffix(".npy"), mode="w+", dtype=np.uint8,
        shape=(len(jobs), N_FRAMES, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    cropped = 0
    for i, (sid, ir_dir, depth_dir) in enumerate(jobs):
        memmap[i] = load_clip(model, ir_dir, depth_dir)
        if memmap[i].any():
            cropped += 1
        if i % 100 == 0 or i == len(jobs) - 1:
            print(f"  {args.split} {i+1}/{len(jobs)} nonempty={cropped}", flush=True)
    memmap.flush()
    with open(OUT_ROOT / f"{args.split}_index.json", "w") as handle:
        json.dump({"sids": [j[0] for j in jobs], "n_frames": N_FRAMES,
                   "channels": CHANNELS, "image_size": IMAGE_SIZE,
                   "crop_margin": CROP_MARGIN,
                   "min_side_fraction": MIN_SIDE_FRACTION,
                   "detection_frames": DETECTION_FRAMES,
                   "person_confidence": PERSON_CONFIDENCE}, handle, indent=2)
    print(f"wrote {path.with_suffix('.npy')}  clips={len(jobs)}  nonempty={cropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
