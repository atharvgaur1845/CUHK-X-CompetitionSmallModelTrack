#!/usr/bin/env python3
"""EXP-107: how far below int8 can MViTv2-S go before it stops being the same model?

R-6 (organiser, topic 729056) allows "fp16 / int8 or lower" and encourages it explicitly
to fit multiple models in the 100 MB budget. The champion video slot is four MViT models
= 137 MB at int8, which does not fit alongside the skeleton and IMU branches. At int4 it
is 68.6 MB and does. This measures what that costs.

Weight-only, symmetric, per-output-channel. Evaluated on the fold-2 held-out clips, the
same 652 the member was scored on, so the number is comparable to its logged micro.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle"))
import cuhkx_224_kaggle as K
import torch
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="k224_mvit_f2")
ap.add_argument("--fold", type=int, default=2)
ap.add_argument("--crop", default="person")
ap.add_argument("--bits", default="32,8,6,4")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--device", default="cpu")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

cache_name = "crop_224" if a.crop == "person" else "crop_wrist224"
paths = K.find_paths(cache_name=cache_name)
cache = paths["cache"]
idx = json.loads((cache / "train_index.json").read_text())
sids, lab, usr = idx["sids"], {k: int(v) for k, v in idx["labels"].items()}, idx["users"]
row_of = {s: i for i, s in enumerate(sids)}
val = [s for s in sids if usr[s] in K.FOLDS[a.fold]]
if a.limit:
    val = val[:a.limit]
rows = np.array([row_of[s] for s in val]); y = np.array([lab[s] for s in val])
store = K.ClipStore(cache, "train")
loader = DataLoader(K.make_dataset(store, rows, y, False), batch_size=a.batch_size,
                    shuffle=False, num_workers=0, pin_memory=False)
print(f"{a.tag}: evaluating {len(rows)} fold-{a.fold} held-out clips on {a.device}", flush=True)

ckpt = torch.load(f"checkpoints/{a.tag}.pt", map_location="cpu", weights_only=False)
base = {k: v.clone() for k, v in ckpt["state_dict"].items()}
device = torch.device(a.device)
model = K.build_model("mvit_v2_s").to(device).eval()
mean, std = K.norm_tensors(device)
OBJ = np.isin(y, K.OBJECT_CLASSES)


def quantize(sd, bits):
    """Symmetric per-output-channel weight-only quantization; returns a NEW state dict.

    Only >=2-D floating tensors are touched: norm scales, biases and positional tables
    are small and disproportionately sensitive, so leaving them at fp32 costs almost no
    bytes and avoids the failure mode where the whole model collapses on a rounding cliff.
    """
    if bits >= 32:
        return {k: v.clone() for k, v in sd.items()}
    qmax = 2 ** (bits - 1) - 1
    out = {}
    for k, v in sd.items():
        if v.dtype.is_floating_point and v.dim() >= 2:
            w = v.reshape(v.shape[0], -1).float()
            scale = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            out[k] = ((w / scale).round().clamp(-qmax - 1, qmax) * scale).reshape(v.shape).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def payload_mb(sd, bits):
    n = sum(v.numel() for k, v in sd.items()
            if v.dtype.is_floating_point and v.dim() >= 2)
    rest = sum(v.numel() for k, v in sd.items()
               if not (v.dtype.is_floating_point and v.dim() >= 2))
    return (n * bits / 8 + rest * 4) / 1e6


ref = None
print(f"\n{'bits':>5s} {'MB':>7s} {'micro':>8s} {'object':>8s} {'agree_fp32':>11s} {'sec':>6s}")
for bits in [int(b) for b in a.bits.split(",")]:
    t0 = time.time()
    model.load_state_dict(quantize(base, bits))
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = (x.to(device) - mean) / std
            p = torch.softmax(model(x).float(), -1)
            p = 0.5 * (p + torch.softmax(model(torch.flip(x, dims=[-1])).float(), -1))
            probs.append(p.cpu().numpy())
    probs = np.concatenate(probs); pred = probs.argmax(1)
    if ref is None:
        ref = pred
    print(f"{bits:5d} {payload_mb(base, bits):7.2f} {(pred==y).mean():8.5f} "
          f"{(pred[OBJ]==y[OBJ]).mean():8.5f} {(pred==ref).mean():11.4f} {time.time()-t0:6.0f}",
          flush=True)
    np.savez_compressed(f"research/artifacts/quantprobe_{a.tag}_b{bits}.npz",
                        probs=probs.astype(np.float32),
                        sids=np.array(val, dtype="<U20"), labels=y)
