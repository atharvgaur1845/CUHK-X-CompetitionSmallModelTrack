#!/usr/bin/env python3
"""Q-70: person-ROI crop cache. Locate person via temporal-median background
subtraction on cached small depth; crop original 640x480 Depth/IR to a square
clip-level ROI; save cache/<split>_roi/<sid>.npz with depth_roi/ir_roi (T,112,112) u8.

Usage: python3 roi_cache.py test train [--debug sid1,sid2,...]
"""
import os
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from build_cache import TRAIN, TEST, invert_jet, fname_ts  # reuse paths + JET LUT

# Portable root: env CUHKX_ROOT wins, else the repo dir two levels up from this
# file. Was a hardcoded absolute path (with a space in it) in 11 files, which was
# the #1 blocker for running anywhere but the original laptop.
ROOT = os.environ.get("CUHKX_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
SIZE = int(os.environ.get("ROI_SIZE", "112"))
SUFFIX = "_roi" if SIZE == 112 else f"_roi{SIZE}"
FG_THRESH = 8          # depth-index units
MIN_BLOB = 40          # px at 120x160
PAD = 0.15


def find_bbox(small):  # small: (T,120,160) u8, 0=invalid
    T, H, W = small.shape
    valid = small > 0
    bg = np.zeros((H, W), np.float32)
    vc = valid.sum(0)
    with np.errstate(invalid="ignore"):
        med = np.ma.median(np.ma.masked_array(small, ~valid), axis=0).filled(0)
    bg = med.astype(np.float32)
    fg = np.zeros((H, W), np.uint8)
    for t in range(T):
        d = np.abs(small[t].astype(np.float32) - bg)
        fg |= ((d > FG_THRESH) & valid[t] & (vc > T // 2)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    boxes = [s for s in stats[1:] if s[4] >= MIN_BLOB]
    if not boxes:  # fallback: nearest 25% of valid depth (person usually nearest)
        med_all = med[med > 0]
        if med_all.size == 0:
            return None
        thr = np.percentile(med_all, 25)
        m = ((med > 0) & (med <= thr)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        boxes = [s for s in stats[1:] if s[4] >= MIN_BLOB]
        if not boxes:
            return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    # pad, square-ify in 120x160 coords
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(w, h) * (1 + 2 * PAD)
    side = min(side, min(2 * 160, 2 * 120))  # cap
    return cx / 160, cy / 120, side / 160, side / 120  # normalized


def crop_frames(dirpath, files, box, is_depth):
    cxn, cyn, swn, shn = box
    out = []
    for f in files:
        img = cv2.imread(os.path.join(dirpath, f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        H, W = img.shape[:2]
        side = int(max(swn * W, shn * H))
        cx, cy = int(cxn * W), int(cyn * H)
        x0 = max(0, min(W - side, cx - side // 2)); y0 = max(0, min(H - side, cy - side // 2))
        x0, y0 = max(0, x0), max(0, y0)
        side_x = min(side, W - x0); side_y = min(side, H - y0)
        crop = img[y0:y0 + side_y, x0:x0 + side_x]
        if is_depth:
            crop = invert_jet(crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
            crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        else:
            if crop.ndim == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out.append(crop)
    return np.stack(out) if out else None


def process(job):
    sid, split, paths, out_path = job
    if os.path.exists(out_path):
        return 0
    try:
        with np.load(os.path.join(CACHE, split, sid + ".npz")) as z:
            if "depth" not in z.files:
                return 0
            small = z["depth"]
            dt = z["depth_t"] if "depth_t" in z.files else None
    except (OSError, KeyError):
        return 0
    box = find_bbox(small)
    if box is None:
        box = (0.5, 0.5, 1.0, 1.0)  # full frame
    data = {"bbox": np.array(box, np.float32)}
    for mod, key in (("Depth_Color", "depth_roi"), ("IR", "ir_roi")):
        d = paths.get(mod)
        if not d or not os.path.isdir(d):
            continue
        fs = sorted((fname_ts(f), f) for f in os.listdir(d) if f.endswith(".png") and fname_ts(f))
        arr = crop_frames(d, [f for _, f in fs], box, mod == "Depth_Color")
        if arr is not None:
            data[key] = arr
            data[key + "_t"] = np.array([t for t, _ in fs][:len(arr)])
    if dt is not None:
        data["depth_t_small"] = dt
    np.savez_compressed(out_path + ".tmp.npz", **data)
    os.replace(out_path + ".tmp.npz", out_path)
    return 1


def jobs_for(split):
    out = []
    outdir = os.path.join(CACHE, split + SUFFIX)
    os.makedirs(outdir, exist_ok=True)
    if split == "train":
        for mod0 in ("Depth_Color",):
            mroot = os.path.join(TRAIN, mod0)
            for cls in sorted(os.listdir(mroot)):
                cid = int(cls.split("_")[0])
                for user in sorted(os.listdir(os.path.join(mroot, cls))):
                    for trial in sorted(os.listdir(os.path.join(mroot, cls, user))):
                        sid = f"{cid:02d}_{user}_{trial}"
                        paths = {m: os.path.join(TRAIN, m, cls, user, trial) for m in ("Depth_Color", "IR")}
                        out.append((sid, split, paths, os.path.join(outdir, sid + ".npz")))
    else:
        for d in sorted(os.listdir(TEST)):
            if d.startswith("SM_test"):
                paths = {m: os.path.join(TEST, d, m) for m in ("Depth_Color", "IR")}
                out.append((d, split, paths, os.path.join(outdir, d + ".npz")))
    return out


if __name__ == "__main__":
    splits = [a for a in sys.argv[1:] if not a.startswith("--")] or ["test", "train"]
    for split in splits:
        js = jobs_for(split)
        done = 0
        with Pool(6) as p:
            for i, r in enumerate(p.imap_unordered(process, js, chunksize=4)):
                done += r
                if (i + 1) % 300 == 0:
                    print(f"{split}: {i + 1}/{len(js)}", flush=True)
        print(f"{split} ROI DONE: {done} written / {len(js)}", flush=True)
