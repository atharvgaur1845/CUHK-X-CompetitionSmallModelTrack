#!/usr/bin/env python3
"""CUHK-X Small Model Track — 224px person crops + a 224-native video backbone.

WHY THIS EXISTS (EXP-101). Every YOLO person crop in this dataset is 224-480 px on
the 640x480 sensor (median 396). The local cache renders it at 128 -- a 3.1x
downsample that discards ~90% of the pixels. 75% of the remaining error mass sits in
the OBJECT classes, which are hand-object interactions, i.e. exactly the detail a
3.1x downsample destroys.

    *** RUNS LOCALLY. Kaggle is optional. ***
    The competition page hosts only sample_submission.csv and test.csv -- the ~50 GB
    of IR/Depth frames is NOT on Kaggle, so there is nothing to train against there
    unless you upload a cache yourself.
    That turned out not to matter. A 224px cache is 10.7 GB as raw uint8, which is
    why 224 looked impossible on a 15 GB / 8 GB-VRAM laptop -- but stored as JPEG
    q90 it is 1.3 GB (measured 8.1x smaller) and decodes in 10.9 ms per clip, about
    6 s per epoch across 4 workers against ~230 s of GPU time. Raw uint8 storage was
    the constraint, not 224px.
    Measured on an RTX 4060 laptop (8 GB): mvit_v2_s at batch 4 peaks at 5.18 GB and
    runs 446 ms/step = 4.2 min/epoch = ~1.4 h for 20 epochs. Batch 6 needs 7.6 GB and
    fits Kaggle's 16 GB but not this card.

The local res160 null does NOT refute the resolution hypothesis: 160 px is a 1.25x
step against a 3.1x loss, and R(2+1)D's native pretrain resolution is 112x112, so 160
pushed the input further off the backbone's distribution than the extra pixels were
worth. This script changes BOTH together -- 224 px input AND a backbone actually
pretrained at 224 (MViTv2-S, K400 top-1 80.3% at 34M params, against IG-65M
R(2+1)D-34's ~79.6% at 63M). 34M also means 34 MB int8, which fits the Stage-2 100 MB
single-file budget that the 12-member ~700 MB bag never will.

EVERYTHING ELSE IS HELD FIXED against the local recipe so the comparison is clean:
same 16 endpoint-uniform frames, same YOLO11n union-box crop geometry, same
Depth_Color(3ch)+IR(1ch) stacking, same Kinetics normalization with the IR channel
on the RGB mean, same single-window augmentation, same logit-adjusted CE, same EMA,
same last-epoch (never best-epoch) selection, same flip TTA, same AdaBN.

    *** THE LOGIT-ADJUSTED LOSS IS LOAD-BEARING. ***
    The loss is cross_entropy(logits + log_prior), so the softmax at inference is in
    UNIFORM-prior space, which is what the fusion expects on its video slot. Training
    with plain CE and fusing it scored 0.58706 against 0.62686 once already. Do not
    "simplify" it.

ONE CAVEAT. MViT and Swin3D normalise with LayerNorm, which has no running statistics,
so the AdaBN worth +2 public clips on the CNN members has nothing to adapt here and is
skipped automatically (the script says so). Not a loss: LayerNorm normalises per
sample, so a transformer already absorbs part of the subject shift AdaBN corrects. The
existing 12 CNN members keep their AdaBN, and this member is an ADDITION to that bag --
additions pay, replacements do not, confirmed three times (g4only replacing K400 scored
157, identical to K400 alone; g5 adding IG-65M scored 161). Do not swap anything out.

OUTPUTS, ready to drop straight into research/artifacts/:
    oof_<tag>.npz        fold-2 holdout: probs, sids, labels  -> compare vs 0.67638
    testprobs_<tag>.npz  405 rows in sample_submission order   -> add to the bag

USAGE — locally, from the repo root:

    python3 kaggle/cuhkx_224_kaggle.py --stage cache     # ~35 min CPU, once, 1.3 GB
    python3 kaggle/cuhkx_224_kaggle.py --stage train --tag k224_mvit_f2   # ~1.4 h
    python3 kaggle/cuhkx_224_kaggle.py --stage infer --tag k224_mvit_f2

USAGE — on Kaggle, only worthwhile to run folds in parallel with the local card.
Build the cache locally first, upload the ~1.3 GB cache directory as a private
Dataset, attach it, then paste this file into a cell (it detects the kernel and runs
every stage) or call run(stage="train", data_root="/kaggle/input/<your-cache>").

Training is resumable: rerun the same command and it continues from the last epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------
# Geometry and recipe constants. These MUST match code/build_crop_cache.py except
# for IMAGE_SIZE, which is the whole point of the experiment.
# --------------------------------------------------------------------------------
N_FRAMES = 16
DETECTION_FRAMES = 8
IMAGE_SIZE = 224          # local cache is 128; this is the change under test
CHANNELS = 4
PERSON_CONFIDENCE = 0.25
CROP_MARGIN = 1.4
MIN_SIDE_FRACTION = 0.35
N_CLASSES = 40
FRAME_RE = re.compile(r"_(\d+)(?:_Color)?\.png$")

# Fold 2 of the local video members, so the number this produces is directly
# comparable to vid_ig65m_f2's micro=0.67638 / object=289/479.
FOLD2_USERS = ("user22", "user24", "user5", "user6")

OBJECT_CLASSES = np.array(sorted(set(range(28)) | {37, 38, 39}))


# ================================================================================
# Environment discovery
# ================================================================================
def _walk_prune(base: Path, max_depth: int = 9):
    """os.walk that prunes clip-level directories.

    A blind rglob over /kaggle/input walks every one of the ~500k frame PNGs and
    can appear to hang. The dataset is only ~7 levels deep to a clip, so bound the
    depth and stop descending as soon as a directory looks like a clip.
    """
    base_str = str(base)
    for root, dirs, files in os.walk(base_str, followlinks=False):
        # Yield what is REALLY there; prune only what we descend into. Yielding the
        # pruned list hid every SM_test_* from the test-root check.
        listing = list(dirs)
        depth = root[len(base_str):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs
                       if not re.match(r"^SM_test_\d+$", d)
                       and not re.match(r"^\d+-\d+-\d+$", d)
                       and d != "__MACOSX"]
        yield Path(root), listing, files


def _describe(base: Path, max_depth: int = 3, limit: int = 40) -> str:
    """What we actually saw, for a failure message that is worth reading."""
    lines, n = [], 0
    base_str = str(base)
    for root, dirs, files in os.walk(base_str):
        depth = root[len(base_str):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        lines.append(f"{'  ' * depth}{os.path.basename(root) or base_str}/  "
                     f"[{len(dirs)} dirs, {len(files)} files]")
        n += 1
        if n >= limit:
            lines.append("  ...")
            break
    return "\n".join(lines)


def find_paths(data_root: str | None = None, cache_name: str = f"crop_{IMAGE_SIZE}") -> dict:
    """Locate the competition data wherever Kaggle mounted it.

    Set --data-root (or run(data_root=...)) to skip the search entirely.
    """
    bases = []
    if data_root:
        bases.append(Path(data_root))
    else:
        inp = Path("/kaggle/input")
        if inp.is_dir():
            # each attached dataset/competition is one directory under /kaggle/input
            bases.extend(sorted(p for p in inp.iterdir() if p.is_dir()))
            bases.append(inp)
        bases += [Path.cwd(), Path.cwd().parent]

    train_ir = test_root = sample_sub = None
    for base in bases:
        if not base.is_dir():
            continue
        for root, dirs, files in _walk_prune(base):
            if train_ir is None and root.name == "IR":
                classes = [d for d in dirs if re.match(r"^\d+_", d)]
                if len(classes) >= 30:
                    train_ir = root
            if test_root is None and any(re.match(r"^SM_test_\d+$", d) for d in dirs):
                test_root = root
            if sample_sub is None and "sample_submission.csv" in files:
                sample_sub = root / "sample_submission.csv"
            if train_ir and test_root and sample_sub:
                break
        if train_ir and test_root:
            break

    if train_ir is None or test_root is None:
        seen = "\n".join(_describe(b) for b in bases[:4] if b.is_dir())
        sys.exit(
            "could not locate the competition data.\n"
            f"  train IR root: {train_ir}\n  test root: {test_root}\n\n"
            "What is actually mounted:\n" + seen +
            "\n\nIf the layout above looks right, pass the path explicitly, e.g.\n"
            "  run(data_root='/kaggle/input/<name>')")

    # Depth_Color normally sits beside IR; fall back to a search if it does not.
    depth = train_ir.parent / "Depth_Color"
    if not depth.is_dir():
        for root, dirs, _ in _walk_prune(train_ir.parent.parent, max_depth=4):
            if root.name in ("Depth_Color", "Depth"):
                depth = root
                break

    # Off Kaggle, the cache belongs under cache/ — that path is already gitignored,
    # and a 1.1 GB train.bin in the repo root gets committed by a stray `git add -A`
    # and then rejected by GitHub's 100 MB file limit.
    scratch = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/kaggle/working")
    if not scratch.is_dir():
        scratch = Path.cwd() / "cache"
        scratch.mkdir(parents=True, exist_ok=True)
    out = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    paths = {"train_ir": train_ir, "train_depth": depth, "test_root": test_root,
             "sample_sub": sample_sub, "cache": scratch / cache_name, "out": out}
    print("resolved paths:")
    for k, v in paths.items():
        print(f"  {k:12s} {v}")
    if not paths["train_depth"].is_dir():
        sys.exit(f"Depth_Color not found near IR (looked at {paths['train_depth']})")
    n_test = len(list(test_root.glob("SM_test_*")))
    print(f"  test clips visible: {n_test}")
    if sample_sub is None:
        print("  NOTE: no sample_submission.csv found; inference will use sorted sid order")
    return paths


def sorted_frames(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix == ".png"]
    return sorted(files, key=lambda p: (int(m.group(1)) if (m := FRAME_RE.search(p.name)) else -1))


def endpoint_uniform(items: list, count: int) -> list:
    """Evenly spaced INCLUDING both endpoints; repeats when the clip is short."""
    if not items:
        return []
    if len(items) == 1:
        return items * count
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in idx]


def build_jobs(paths: dict) -> tuple[list, list]:
    train, test = [], []
    for class_dir in sorted(paths["train_ir"].iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name.split("_")[0])
        except ValueError:
            continue
        for user_dir in sorted(class_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            for trial_dir in sorted(user_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                sid = f"{class_id:02d}_{user_dir.name}_{trial_dir.name}"
                depth = paths["train_depth"] / class_dir.name / user_dir.name / trial_dir.name
                train.append((sid, str(trial_dir), str(depth), class_id, user_dir.name))
    # Explicit glob: the test root can contain unrelated directories, and picking
    # one up shifts every row against the submission order.
    for clip_dir in sorted(paths["test_root"].glob("SM_test_*")):
        if clip_dir.is_dir():
            test.append((clip_dir.name, str(clip_dir / "IR"), str(clip_dir / "Depth_Color"), -1, ""))
    return train, test


# ================================================================================
# Stage 1 — crop windows (YOLO, GPU) then render (CPU pool)
# ================================================================================
def compute_windows(jobs: list, cache: Path, device: str) -> dict:
    """One square crop window per clip, checkpointed so a timeout is not fatal."""
    wpath = cache / "windows.json"
    windows = json.loads(wpath.read_text()) if wpath.exists() else {}
    todo = [j for j in jobs if j[0] not in windows]
    if not todo:
        print(f"  windows: all {len(jobs)} cached")
        return windows
    from ultralytics import YOLO
    from PIL import Image
    model = YOLO("yolo11n.pt")
    print(f"  windows: {len(todo)} to compute on {device}")
    t0 = time.time()
    for n, (sid, ir_dir, _depth, _c, _u) in enumerate(todo):
        frames = sorted_frames(Path(ir_dir))
        width, height = 640, 480
        for cand in frames:
            try:
                with Image.open(cand) as im:
                    width, height = im.size
                break
            except Exception:
                continue
        win = None
        if frames:
            probes = list(dict.fromkeys(endpoint_uniform(frames, min(DETECTION_FRAMES, len(frames)))))
            images = []
            for p in probes:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
            if images:
                boxes = []
                for res in model.predict(images, classes=[0], conf=PERSON_CONFIDENCE,
                                         verbose=False, device=device):
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    conf = res.boxes.conf.cpu().numpy()
                    boxes.append(res.boxes.xyxy.cpu().numpy()[int(conf.argmax())])
                if boxes:
                    b = np.stack(boxes)
                    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
                    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                    side = max(x1 - x0, y1 - y0) * CROP_MARGIN
                    side = max(side, MIN_SIDE_FRACTION * max(width, height))
                    side = min(side, float(min(width, height)))
                    half = side / 2.0
                    cx = float(np.clip(cx, half, width - half))
                    cy = float(np.clip(cy, half, height - half))
                    win = [int(round(cx - half)), int(round(cy - half)),
                           int(round(cx + half)), int(round(cy + half))]
        windows[sid] = win
        if (n + 1) % 250 == 0 or n + 1 == len(todo):
            wpath.write_text(json.dumps(windows))
            rate = (n + 1) / (time.time() - t0)
            print(f"    {n+1}/{len(todo)}  {rate:.1f} clip/s  "
                  f"eta {(len(todo)-n-1)/max(rate,1e-9)/60:.1f} min", flush=True)
    wpath.write_text(json.dumps(windows))
    return windows


def _render(task):
    """Render one clip to 32 JPEG blobs: 16 Depth_Color (RGB) then 16 IR (L).

    Storing JPEG rather than raw uint8 is what makes 224px tractable: measured 8.1x
    smaller, so the whole cache is 1.3 GB instead of 10.7 GB and fits in RAM on an
    ordinary machine. Decode costs 10.9 ms per clip, about 6 s per epoch across 4
    workers against ~230 s of GPU time, so it is free in practice.
    """
    import io
    from PIL import Image
    sid, ir_dir, depth_dir, win, size, quality = task
    box = tuple(win) if win else None
    blobs = []

    def paint(paths_, mode):
        got = []
        for path in endpoint_uniform(paths_, N_FRAMES):
            try:
                im = Image.open(path).convert(mode)
            except Exception:
                got.append(b"")     # all-zero placeholder PNGs exist in the test split
                continue
            if box:
                im = im.crop(box)
            im = im.resize((size, size), Image.BILINEAR)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
            got.append(buf.getvalue())
        while len(got) < N_FRAMES:
            got.append(b"")
        return got

    depth_frames = sorted_frames(Path(depth_dir))
    ir_frames = sorted_frames(Path(ir_dir))
    blobs += paint(depth_frames, "RGB") if depth_frames else [b""] * N_FRAMES
    blobs += paint(ir_frames, "L") if ir_frames else [b""] * N_FRAMES
    return sid, blobs


class ClipStore:
    """Read side of the JPEG cache: one concatenated blob plus an offset table."""

    def __init__(self, cache: Path, split: str):
        self.blob = np.memmap(cache / f"{split}.bin", dtype=np.uint8, mode="r")
        self.index = np.load(cache / f"{split}_frames.npy")      # (N, 2*N_FRAMES, 2)
        self.meta = json.loads((cache / f"{split}_index.json").read_text())
        self.size = int(self.meta["image_size"])
        self.sids = self.meta["sids"]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        import io
        from PIL import Image
        s = self.size
        out = np.zeros((N_FRAMES, CHANNELS, s, s), dtype=np.uint8)
        rec = self.index[i]
        for slot in range(N_FRAMES):
            off, ln = rec[slot]                       # channels 0-2: Depth_Color
            if ln:
                im = Image.open(io.BytesIO(self.blob[off:off + ln].tobytes()))
                out[slot, :3] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
            off, ln = rec[N_FRAMES + slot]            # channel 3: IR
            if ln:
                im = Image.open(io.BytesIO(self.blob[off:off + ln].tobytes()))
                out[slot, 3] = np.asarray(im, dtype=np.uint8)
        return out


def build_cache(paths: dict, split: str, jobs: list, windows: dict,
                workers: int, size: int, quality: int):
    cache = paths["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    bin_path = cache / f"{split}.bin"
    done_marker = cache / f"{split}.DONE"
    if done_marker.exists() and bin_path.exists():
        print(f"  {split} cache already complete ({bin_path})")
        return
    free = shutil.disk_usage(cache).free / 1e9
    print(f"  {split}: {len(jobs)} clips at {size}px q{quality} (free {free:.1f} GB)")
    tasks = [(j[0], j[1], j[2], windows.get(j[0]), size, quality) for j in jobs]
    row_of = {j[0]: i for i, j in enumerate(jobs)}
    index = np.zeros((len(jobs), 2 * N_FRAMES, 2), dtype=np.int64)
    t0, nonempty, offset = time.time(), 0, 0
    with open(bin_path, "wb") as blob, ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (sid, frames) in enumerate(pool.map(_render, tasks, chunksize=8)):
            row = row_of[sid]
            for k, raw in enumerate(frames):
                if raw:
                    blob.write(raw)
                    index[row, k] = (offset, len(raw))
                    offset += len(raw)
            nonempty += int(any(frames))
            if (n + 1) % 200 == 0 or n + 1 == len(tasks):
                rate = (n + 1) / (time.time() - t0)
                print(f"    {n+1}/{len(tasks)} nonempty={nonempty} {rate:.1f} clip/s "
                      f"{offset/1e9:.2f} GB eta {(len(tasks)-n-1)/max(rate,1e-9)/60:.1f} min",
                      flush=True)
    np.save(cache / f"{split}_frames.npy", index)
    (cache / f"{split}_index.json").write_text(json.dumps({
        "sids": [j[0] for j in jobs], "n_frames": N_FRAMES, "channels": CHANNELS,
        "image_size": size, "jpeg_quality": quality, "crop_margin": CROP_MARGIN,
        "min_side_fraction": MIN_SIDE_FRACTION, "detection_frames": DETECTION_FRAMES,
        "person_confidence": PERSON_CONFIDENCE,
        "labels": {j[0]: j[3] for j in jobs} if split == "train" else {},
        "users": {j[0]: j[4] for j in jobs} if split == "train" else {}}, indent=2))
    done_marker.write_text("ok")
    print(f"  wrote {bin_path} ({offset/1e9:.2f} GB) nonempty={nonempty}/{len(jobs)}")


# ================================================================================
# Stage 2 — model
# ================================================================================
def build_model(arch: str, n_classes: int = N_CLASSES, in_channels: int = CHANNELS):
    """224-native Kinetics backbone with the stem widened to 4 channels."""
    import torch
    import torch.nn as nn
    import torchvision.models.video as V

    if arch == "mvit_v2_s":
        model = V.mvit_v2_s(weights=V.MViT_V2_S_Weights.KINETICS400_V1)
        model.head[-1] = nn.Linear(model.head[-1].in_features, n_classes)
        stem_attr = ("conv_proj",)
    elif arch == "swin3d_t":
        model = V.swin3d_t(weights=V.Swin3D_T_Weights.KINETICS400_V1)
        model.head = nn.Linear(model.head.in_features, n_classes)
        stem_attr = ("patch_embed", "proj")
    elif arch == "swin3d_s":
        model = V.swin3d_s(weights=V.Swin3D_S_Weights.KINETICS400_V1)
        model.head = nn.Linear(model.head.in_features, n_classes)
        stem_attr = ("patch_embed", "proj")
    elif arch == "s3d":
        model = V.s3d(weights=V.S3D_Weights.KINETICS400_V1)
        model.classifier[1] = nn.Conv3d(1024, n_classes, kernel_size=1)
        stem_attr = ("features", "0", "0", "0")
    else:
        raise ValueError(f"unknown arch {arch!r}")

    if in_channels != 3:
        parent = model
        for a in stem_attr[:-1]:
            parent = parent[int(a)] if a.isdigit() else getattr(parent, a)
        name = stem_attr[-1]
        old = parent[int(name)] if name.isdigit() else getattr(parent, name)
        new = nn.Conv3d(in_channels, old.out_channels, old.kernel_size, old.stride,
                        old.padding, bias=old.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = old.weight.data
            # The IR channel starts as the mean RGB filter: a zero init would give it
            # no gradient path for the first steps and it is the sharpest modality.
            for extra in range(3, in_channels):
                new.weight[:, extra:extra + 1] = old.weight.data.mean(dim=1, keepdim=True)
            if old.bias is not None:
                new.bias.copy_(old.bias.data)
        if name.isdigit():
            parent[int(name)] = new
        else:
            setattr(parent, name, new)
    return model


def make_dataset(store, rows, labels, train):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset

    class Clips(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            x = torch.from_numpy(store[int(rows[i])]).float() / 255.0
            x = x.permute(1, 0, 2, 3)                      # (T,C,H,W) -> (C,T,H,W)
            if train:
                c, t, h, w = x.shape
                scale = float(np.random.uniform(0.75, 1.0))
                ch, cw = int(h * scale), int(w * scale)
                top = int(np.random.randint(0, h - ch + 1))
                left = int(np.random.randint(0, w - cw + 1))
                # ONE window for the whole clip: per-frame jitter injects fake motion.
                x = x[:, :, top:top + ch, left:left + cw]
                x = F.interpolate(x.permute(1, 0, 2, 3), size=(h, w), mode="bilinear",
                                  align_corners=False).permute(1, 0, 2, 3)
                if np.random.rand() < 0.5:
                    x = torch.flip(x, dims=[-1])
            return x, (-1 if labels is None else int(labels[i]))

    return Clips()


def norm_tensors(device):
    import torch
    mean = torch.tensor([0.43216, 0.394666, 0.37645, 0.401092]).view(1, 4, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989, 0.222156]).view(1, 4, 1, 1, 1)
    return mean.to(device), std.to(device)


def predict(model, loader, device, flip_tta=True):
    import torch
    mean, std = norm_tensors(device)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = (x.to(device) - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.softmax(model(x).float(), -1)
                if flip_tta:
                    p = 0.5 * (p + torch.softmax(model(torch.flip(x, dims=[-1])).float(), -1))
            probs.append(p.float().cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def apply_adabn(model, loader, device):
    """Re-estimate BN/LN-free running stats from UNLABELED target inputs (EXP-099/100).

    Confirmed on public: +2 clips. momentum=None makes each layer a cumulative
    average, so the result does not depend on batch order. Only buffers change.
    """
    import torch
    import torch.nn as nn
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        print("  AdaBN: no BatchNorm layers in this arch (transformers use LayerNorm) — skipped")
        return False
    mean, std = norm_tensors(device)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    with torch.no_grad():
        for x, _ in loader:
            x = (x.to(device) - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
    for m in bns:
        m.eval()
    print(f"  AdaBN: re-estimated {len(bns)} BatchNorm layers")
    return True


# ================================================================================
# Stage 2 — train
# ================================================================================
def stage_train(args, paths):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import copy

    cache = paths["cache"]
    idx = json.loads((cache / "train_index.json").read_text())
    sids = idx["sids"]
    lab = {k: int(v) for k, v in idx["labels"].items()}
    usr = idx["users"]
    row_of = {s: i for i, s in enumerate(sids)}

    val_sids = [s for s in sids if usr[s] in FOLD2_USERS]
    tr_sids = [s for s in sids if usr[s] not in FOLD2_USERS] if not args.all_train else list(sids)
    tr_rows = np.array([row_of[s] for s in tr_sids])
    va_rows = np.array([row_of[s] for s in val_sids])
    tr_lab = np.array([lab[s] for s in tr_sids])
    va_lab = np.array([lab[s] for s in val_sids])

    device = torch.device("cuda")
    counts = np.bincount(tr_lab, minlength=N_CLASSES).astype(np.float64)
    log_prior = torch.tensor(np.log(np.maximum(counts / counts.sum(), 1e-9)),
                             dtype=torch.float32, device=device)

    store = ClipStore(cache, "train")
    tl = DataLoader(make_dataset(store, tr_rows, tr_lab, True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=False, persistent_workers=args.workers > 0)
    vl = DataLoader(make_dataset(store, va_rows, va_lab, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=False)

    model = build_model(args.arch).to(device)
    n = sum(q.numel() for q in model.parameters())
    print(f"{args.tag}: arch={args.arch} px={store.size} train={len(tr_rows)} "
          f"outer={len(va_rows)} params={n:,} fp16={n*2/1e6:.1f}MB int8={n/1e6:.1f}MB",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = max(1, len(tl) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps,
        pct_start=max(args.warmup / max(args.epochs, 1), 0.05))
    scaler = torch.amp.GradScaler("cuda")
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)

    ckpt = paths["out"] / f"{args.tag}_resume.pt"
    fingerprint = {k: v for k, v in vars(args).items() if k not in ("resume", "stage")}
    start_epoch = 1
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        if st.get("fingerprint") == fingerprint:
            model.load_state_dict(st["model"])
            ema.load_state_dict(st["ema"])
            start_epoch = int(st["epoch"]) + 1
            # Optimizer state is deliberately not persisted (it tripled the file and
            # killed the write on the dev box). Fast-forward the LR schedule instead.
            for _ in range(min((start_epoch - 1) * steps, args.epochs * steps - 1)):
                sched.step()
            print(f"  resumed from epoch {st['epoch']}", flush=True)
        else:
            print("  resume recipe differs; starting fresh", flush=True)

    mean, std = norm_tensors(device)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total, seen, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tl):
            x = (x.to(device, non_blocking=True) - mean) / std
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                # LOGIT-ADJUSTED: keeps the softmax in uniform-prior space for fusion.
                loss = F.cross_entropy(model(x) + log_prior, y,
                                       label_smoothing=args.label_smoothing)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < args.epochs * steps - 1:
                    sched.step()
                with torch.no_grad():
                    for e, m in zip(ema.parameters(), model.parameters()):
                        e.mul_(args.ema).add_(m.detach(), alpha=1 - args.ema)
                    for e, m in zip(ema.buffers(), model.buffers()):
                        e.copy_(m)
            total += float(loss.detach()) * y.numel()
            seen += y.numel()
        tmp = str(ckpt) + ".tmp"
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "epoch": epoch, "fingerprint": fingerprint}, tmp)
        os.replace(tmp, ckpt)
        print(f"  epoch {epoch:02d}/{args.epochs}: loss={total/seen:.5f} "
              f"lr={sched.get_last_lr()[0]:.2e} [{time.time()-t0:.0f}s]", flush=True)

    # LAST epoch, never the best: selecting on the val curve is worth ~6 clips of
    # noise on this competition and does not survive to the private split.
    probs, labels = predict(ema, vl, device)
    pred = probs.argmax(1)
    om = np.isin(labels, OBJECT_CLASSES)
    micro = float((pred == labels).mean())
    obj = float((pred[om] == labels[om]).mean())
    mot = float((pred[~om] == labels[~om]).mean())
    print(f"\n{args.tag} OUTER-ONCE: micro={micro:.5f} object={obj:.5f} "
          f"({int((pred[om]==labels[om]).sum())}/{int(om.sum())}) gross_motion={mot:.5f}")
    print(f"  local reference to beat — vid_ig65m_f2: micro=0.67638 object=0.60334 (289/479)")
    np.savez_compressed(paths["out"] / f"oof_{args.tag}.npz", probs=probs.astype(np.float32),
                        sids=np.array(val_sids, dtype="<U20"), labels=labels.astype(np.int64))
    torch.save({"state_dict": ema.state_dict(), "args": vars(args),
                "micro": micro, "object": obj}, paths["out"] / f"{args.tag}.pt")
    print(f"  wrote oof_{args.tag}.npz and {args.tag}.pt")


# ================================================================================
# Stage 3 — test inference in canonical submission order
# ================================================================================
def stage_infer(args, paths):
    import torch
    from torch.utils.data import DataLoader

    cache = paths["cache"]
    idx = json.loads((cache / "test_index.json").read_text())
    row_of = {s: i for i, s in enumerate(idx["sids"])}

    if paths["sample_sub"] is None:
        sids = sorted(row_of)
        header, paths_col = ["path", "class_id"], sids
        print("  WARNING: no sample_submission.csv found; using sorted sids order")
    else:
        rows = list(csv.reader(open(paths["sample_sub"])))
        header, body = rows[0], rows[1:]
        paths_col = [r[0] for r in body]
        sids = [q.strip("/").split("/")[-1] for q in paths_col]
    missing = [s for s in sids if s not in row_of]
    assert not missing, f"{len(missing)} submission clips absent from the cache: {missing[:5]}"
    order = np.array([row_of[s] for s in sids])

    device = torch.device("cuda")
    pkg = torch.load(paths["out"] / f"{args.tag}.pt", map_location="cpu", weights_only=False)
    model = build_model(pkg["args"]["arch"]).to(device)
    model.load_state_dict(pkg["state_dict"])
    model.eval()
    print(f"{args.tag}: fold-2 micro={pkg['micro']:.5f} object={pkg['object']:.5f}")

    store = ClipStore(cache, "test")
    dl = DataLoader(make_dataset(store, order, None, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=False)
    tag = args.tag
    if args.adabn and apply_adabn(model, dl, device):
        tag = f"{args.tag}_adabn"
    probs, _ = predict(model, dl, device)
    assert probs.shape == (len(sids), N_CLASSES), probs.shape
    out = paths["out"] / f"testprobs_{tag}.npz"
    np.savez_compressed(out, probs=probs.astype(np.float32),
                        sids=np.array(sids, dtype="<U20"))
    print(f"  wrote {out}")
    sub = paths["out"] / f"sub_{tag}_argmax.csv"
    with open(sub, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(header)
        for p, k in zip(paths_col, probs.argmax(1)):
            w.writerow([p, int(k)])
    print(f"  wrote {sub} (video member alone — the real gain comes from fusing at home)")


# ================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=("cache", "train", "infer", "all"))
    ap.add_argument("--tag", default="k224_mvit_f2")
    ap.add_argument("--arch", default="mvit_v2_s",
                    choices=("mvit_v2_s", "swin3d_t", "swin3d_s", "s3d"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="measured peak VRAM at 224px/16f: mvit_v2_s bs4=5.18GB (446 ms/step "
                         "on an RTX 4060), bs6=7.60GB, swin3d_t bs4=5.81GB. 4 fits an 8GB "
                         "card; raise to 6-8 on Kaggle's 16GB.")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--all-train", action="store_true",
                    help="train on all 18 users; the reported fold-2 number is then "
                         "TRAIN-ON-TEST and must never be compared with an honest OOF")
    ap.add_argument("--adabn", action="store_true", default=True)
    ap.add_argument("--no-adabn", dest="adabn", action="store_false")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                    help="crop render size. 128 reproduces the local cache; 224 is the "
                         "experiment. Person crops are 224-480px so 224 is near-lossless.")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0,
                    help="cache only the first N clips of each split — a smoke test, "
                         "not a runnable cache. Delete the cache dir before a real run.")
    ap.add_argument("--data-root", default=None,
                    help="skip path discovery; e.g. /kaggle/input/<dataset-name>")
    ap.add_argument("--cache-workers", type=int, default=max(2, (os.cpu_count() or 4)))
    args = ap.parse_args(argv)

    paths = find_paths(args.data_root, f"crop_{args.image_size}")
    if args.stage in ("cache", "all"):
        train_jobs, test_jobs = build_jobs(paths)
        print(f"jobs: train={len(train_jobs)} test={len(test_jobs)}")
        if args.limit:
            train_jobs, test_jobs = train_jobs[:args.limit], test_jobs[:args.limit]
            print(f"  --limit {args.limit}: SMOKE TEST ONLY, this cache is not trainable")
        paths["cache"].mkdir(parents=True, exist_ok=True)
        dev = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "" else "cpu"
        windows = compute_windows(train_jobs + test_jobs, paths["cache"], dev)
        build_cache(paths, "train", train_jobs, windows, args.cache_workers,
                    args.image_size, args.jpeg_quality)
        build_cache(paths, "test", test_jobs, windows, args.cache_workers,
                    args.image_size, args.jpeg_quality)
    if args.stage in ("train", "all"):
        stage_train(args, paths)
    if args.stage in ("infer", "all"):
        stage_infer(args, paths)
    return 0


def _in_notebook() -> bool:
    """True inside a Jupyter/Kaggle kernel, where sys.argv belongs to the kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def run(**overrides):
    """Entry point for a Kaggle notebook cell.

    Kaggle notebooks are .ipynb, so there is no argv to parse -- paste this whole
    file into one cell and it calls every stage on import. Override any flag by
    keyword, using the argparse dest name:

        run()                                   # cache -> train -> infer, defaults
        run(arch="swin3d_t", epochs=30)
        run(stage="train", tag="k224_mvit_f2")  # one stage at a time
        run(batch_size=4, accum=6)              # if CUDA OOMs
        run(adabn=False)

    Booleans map to the store_true flags: True passes --flag, False passes
    --no-flag (only --no-adabn exists, so adabn=False is the only valid False).
    """
    argv = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False:
            argv.append("--no-" + key.replace("_", "-"))
        else:
            argv += [flag, str(value)]
    return main(argv)


if __name__ == "__main__":
    if _in_notebook():
        # Pasted into a Kaggle cell. sys.argv here is the kernel's (-f kernel.json),
        # which argparse would reject, so parse an empty argv and run every stage.
        # Edit the call below, or delete it and call run(...) from the next cell.
        raise SystemExit(run())
    raise SystemExit(main())
