#!/usr/bin/env python3
"""L1/T1 — frozen video-foundation features over the existing 224 crop caches.

The teacher is never shipped, so R-1's size rule does not touch it and R-3 explicitly
permits distilling from larger models. What IS constrained is the student, and EXP-122
already measured the machinery: distilling an oracle target into the 34 MB MViT gave
+4.91 on fold 2 over a leak-free control. That teacher was capped at oracle-any-member
0.877 because it was built from our own five members. A foundation model is the first
teacher that is not.

Why FROZEN and not fine-tuned: EXP-115 measured VideoMAE-B fine-tuned as 0/4 folds against
MViTv2-S -- 86.7M params cannot be fine-tuned on 2,281 clips. A frozen extractor plus a
small probe is the opposite regime and is how V-JEPA 2 is evaluated in its own paper.

Emits one npz per (model, view, split) holding pooled features, so the probe is seconds.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import cuhkx_224_kaggle as K            # noqa: E402

VIEW_CACHE = {"person": "crop_224", "wrist": "crop_wrist224"}
# ImageNet stats: the foundation encoders were trained with these, not Kinetics' .43/.39/.37
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def build(name):
    """Returns (callable(x_bcthw)->(B,D), description). x is float in [0,1], 3-channel."""
    if name.startswith("vjepa2"):
        from transformers import AutoModel
        repo = {"vjepa2_l": "facebook/vjepa2-vitl-fpc64-256"}[name]
        m = AutoModel.from_pretrained(repo, dtype=torch.float16).cuda().eval()

        def fwd(x):                       # (B,C,T,H,W) -> (B,T,C,H,W)
            out = m.get_vision_features(pixel_values_videos=x.permute(0, 2, 1, 3, 4).half())
            return out.float().mean(1)    # mean over tokens
        return fwd, repo
    if name.startswith("videomae"):
        from transformers import VideoMAEModel
        repo = {"videomae_l": "MCG-NJU/videomae-large-finetuned-kinetics",
                "videomae_b": "MCG-NJU/videomae-base-finetuned-kinetics"}[name]
        m = VideoMAEModel.from_pretrained(repo, dtype=torch.float16).cuda().eval()

        def fwd(x):
            out = m(pixel_values=x.permute(0, 2, 1, 3, 4).half()).last_hidden_state
            return out.float().mean(1)
        return fwd, repo
    raise SystemExit(f"unknown model {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vjepa2_l")
    ap.add_argument("--view", default="person", choices=list(VIEW_CACHE))
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--stream", default="depth", choices=("depth", "ir"),
                    help="depth = Depth_Color RGB channels 0-2; ir = channel 3 as gray-RGB")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "research" / "artifacts" / "teacher"))
    a = ap.parse_args()

    cache = ROOT / "cache" / VIEW_CACHE[a.view]
    store = K.ClipStore(cache, a.split)
    idx = json.loads((cache / f"{a.split}_index.json").read_text())
    sids = idx["sids"]
    rows = np.arange(len(sids))
    if a.limit:
        rows, sids = rows[:a.limit], sids[:a.limit]

    fwd, repo = build(a.model)
    print(f"{repo}  view={a.view} split={a.split} stream={a.stream} n={len(rows)}", flush=True)
    mean, std = IMNET_MEAN.cuda(), IMNET_STD.cuda()

    from torch.utils.data import DataLoader
    dl = DataLoader(K.make_dataset(store, rows, None, False), batch_size=a.batch,
                    shuffle=False, num_workers=0, pin_memory=False)
    F, t0 = [], time.time()
    with torch.no_grad():
        for i, (x, _) in enumerate(dl):
            x = x.cuda()
            x = x[:, :3] if a.stream == "depth" else x[:, 3:4].repeat(1, 3, 1, 1, 1)
            x = (x - mean) / std
            F.append(fwd(x).cpu().numpy())
            if (i + 1) % 100 == 0:
                done = (i + 1) * a.batch
                r = done / (time.time() - t0)
                print(f"   {done}/{len(rows)}  {r:.1f} clip/s  "
                      f"eta {(len(rows)-done)/max(r,1e-9)/60:.1f} min", flush=True)
    F = np.concatenate(F)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{a.model}_{a.view}_{a.stream}_{a.split}.npz"
    np.savez_compressed(p, feats=F.astype(np.float16), sids=np.array(sids, dtype="<U24"))
    print(f"wrote {p}  shape={F.shape}  [{(time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
