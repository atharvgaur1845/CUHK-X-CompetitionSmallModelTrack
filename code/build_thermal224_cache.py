#!/usr/bin/env python3
"""L3a — a 224 px THERMAL cache in the JPEG format the MViT trainer already reads.

The case for re-opening thermal, in three measured numbers:
  * the dataset paper ranks Thermal **first of six sensors (92.57)**; our member sits at
    **0.544** (EXP-088) and the IR+depth member at 0.715;
  * every thermal run we have ever done was **128 px through r2plus1d_18** -- EXP-088,
    EXP-119 (full frame, 0.558), EXP-120b (null at 4 folds). That is exactly the recipe the
    224 px/MViTv2-S recipe beat by **+7.5 points** on IR+depth (0.640 -> 0.715, EXP-102/103);
  * thermal agrees with the person view on only **54%** of clips, the most decorrelated
    view we own.
So the modality has never been tried through the recipe that works. Thermal frames are
320x240, so 224 is near-native and nothing is upsampled.

FULL FRAME, no person crop. EXP-120b measured cropped vs full-frame over 4 paired folds at
+0.83 (0.96 SE) -- the crop is not the defect -- and full frame needs no YOLO, so the cache
builds anywhere with just PIL. It is also the `skomuro` recipe (EXP-118).

Sample ids and their ORDER are taken from the IR tree, so this cache lines up clip-for-clip
with cache/crop_224. A thermal-only enumeration would silently drop the clips with no
thermal and shift every row against the other caches and the submission.

    python3 code/build_thermal224_cache.py --split train --workers 64
    python3 code/build_thermal224_cache.py --split test  --workers 64
"""
from __future__ import annotations
import argparse, io, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("CUHKX_ROOT") or Path(__file__).resolve().parents[1])
TRAIN_DATA = ROOT / "Small-Model-Track" / "Training" / "data" / "HAR" / "data"
TEST_ROOT = ROOT / "Small-Model-Track" / "Testing" / "data" / "small_model_track_test"
N_FRAMES = 16
NUM_RE = re.compile(r"(\d+)")


def thermal_frames(d: Path):
    if not d.is_dir():
        return []
    fs = [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return sorted(fs, key=lambda p: int(m.group(1)) if (m := NUM_RE.search(p.stem)) else -1)


def endpoint_uniform(items, count):
    if not items:
        return []
    if len(items) == 1:
        return items * count
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in idx]


def _render(task):
    sid, tdir, size, q = task
    from PIL import Image
    frames = thermal_frames(Path(tdir))
    blobs = []
    for path in endpoint_uniform(frames, N_FRAMES):
        try:
            im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=q)
            blobs.append(buf.getvalue())
        except Exception:
            blobs.append(b"")
    while len(blobs) < N_FRAMES:
        blobs.append(b"")
    return sid, blobs


def jobs_for(split, ref_cache: Path):
    """Sample ids and their ORDER come from the reference crop_224 index, not from a
    directory walk.

    The first version enumerated the IR tree to guarantee alignment with crop_224. That is
    the right invariant and the wrong source: it needs ~10 GB of raw IR frames that the
    cache already summarises, and on a machine holding only the Thermal tree it simply
    fails. The crop_224 index IS the authoritative sid list -- it is what every existing
    member and every submission row is ordered by -- so read it directly and reconstruct
    the thermal path from the sid. Alignment becomes exact by construction rather than by
    two directory walks agreeing.
    """
    idx = json.loads((ref_cache / f"{split}_index.json").read_text())
    sids = idx["sids"]
    if split == "test":
        return [(sid, str(TEST_ROOT / sid / "Thermal"), -1, "") for sid in sids]
    labels, users = idx["labels"], idx["users"]
    cls_dir = {}
    for d in (TRAIN_DATA / "Thermal").iterdir():
        if d.is_dir() and "_" in d.name:
            try:
                cls_dir[int(d.name.split("_")[0])] = d.name
            except ValueError:
                pass
    out = []
    for sid in sids:
        cid = int(labels[sid]); trial = sid.split("_", 2)[2]
        name = cls_dir.get(cid)
        path = "" if name is None else str(TRAIN_DATA / "Thermal" / name / users[sid] / trial)
        out.append((sid, path, cid, users[sid]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "test"))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4)))
    ap.add_argument("--out", default="")
    ap.add_argument("--ref-cache", default="",
                    help="cache whose sid ORDER this must match (default cache/crop_224). "
                         "Alignment with the other members is not optional: a shifted row "
                         "mis-scores every clip in the fusion.")
    a = ap.parse_args()

    cache = Path(a.out) if a.out else ROOT / "cache" / f"thermal_{a.image_size}"
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / f"{a.split}.DONE").exists():
        print(f"{a.split} already complete at {cache}")
        return 0

    ref = Path(a.ref_cache) if a.ref_cache else ROOT / "cache" / "crop_224"
    js = jobs_for(a.split, ref)
    missing = sum(1 for j in js if not j[1] or not Path(j[1]).is_dir())
    print(f"sid order from {ref}/{a.split}_index.json; "
          f"{missing} clips have no thermal directory (they render as zeros)", flush=True)
    print(f"{a.split}: {len(js)} clips -> {cache} at {a.image_size}px q{a.jpeg_quality}, "
          f"{a.workers} workers", flush=True)
    tasks = [(j[0], j[1], a.image_size, a.jpeg_quality) for j in js]
    row_of = {j[0]: i for i, j in enumerate(js)}
    index = np.zeros((len(js), 2 * N_FRAMES, 2), dtype=np.int64)
    off = nonempty = 0
    t0 = time.time()
    with open(cache / f"{a.split}.bin", "wb") as blob, \
            ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n, (sid, frames) in enumerate(pool.map(_render, tasks, chunksize=8)):
            r = row_of[sid]
            for k, raw in enumerate(frames):
                if raw:
                    blob.write(raw); index[r, k] = (off, len(raw)); off += len(raw)
            nonempty += int(any(frames))
            if (n + 1) % 250 == 0 or n + 1 == len(tasks):
                rate = (n + 1) / (time.time() - t0)
                print(f"   {n+1}/{len(tasks)} nonempty={nonempty} {rate:.1f} clip/s "
                      f"{off/1e9:.2f} GB eta {(len(tasks)-n-1)/max(rate,1e-9)/60:.1f} min",
                      flush=True)
    np.save(cache / f"{a.split}_frames.npy", index)
    (cache / f"{a.split}_index.json").write_text(json.dumps({
        "sids": [j[0] for j in js], "n_frames": N_FRAMES, "channels": 3,
        "image_size": a.image_size, "jpeg_quality": a.jpeg_quality,
        "modality": "thermal", "full_frame": True,
        "labels": {j[0]: j[2] for j in js} if a.split == "train" else {},
        "users": {j[0]: j[3] for j in js} if a.split == "train" else {}}, indent=2))
    (cache / f"{a.split}.DONE").write_text("ok")
    print(f"wrote {cache}/{a.split}.bin  {off/1e9:.2f} GB  nonempty {nonempty}/{len(js)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
