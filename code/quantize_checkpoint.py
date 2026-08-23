#!/usr/bin/env python3
"""EXP-108: write a bit-width-reduced copy of a trained checkpoint, ready for inference.

R-6 permits "fp16 / int8 or lower" and encourages it to fit multiple models under the
100 MB cap. EXP-107 measured MViTv2-S at 32/8/6/4 bits on 652 held-out clips: int6 is
accuracy-identical to int8 (micro 0.71319 both) at 75% of the bytes, and int4 is where
it breaks (-7 clips, argmax agreement 0.914).

Symmetric, per-output-channel, weight-only, matching quant_probe.py exactly. Norm scales,
biases and positional tables stay fp32: they are a rounding-sensitive fraction of a
percent of the parameters, so quantizing them buys nothing and risks a collapse.
"""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--bits", type=int, required=True)
ap.add_argument("--out", required=True, help="new tag; writes <out>.pt in the repo root")
a = ap.parse_args()

pkg = torch.load(f"checkpoints/{a.tag}.pt", map_location="cpu", weights_only=False)
sd = pkg["state_dict"]
qmax = 2 ** (a.bits - 1) - 1
quant = big = small = 0
out = {}
for k, v in sd.items():
    if v.dtype.is_floating_point and v.dim() >= 2:
        w = v.reshape(v.shape[0], -1).float()
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
        out[k] = ((w / scale).round().clamp(-qmax - 1, qmax) * scale).reshape(v.shape).to(v.dtype)
        quant += 1; big += v.numel()
    else:
        out[k] = v.clone(); small += v.numel()
pkg["state_dict"] = out
pkg["quantization"] = {"bits": a.bits, "scheme": "symmetric_per_output_channel_weight_only",
                       "source_tag": a.tag}
torch.save(pkg, f"{a.out}.pt")
mb = (big * a.bits / 8 + small * 4) / 1e6
print(f"{a.tag} -> {a.out}: {quant} tensors at int{a.bits}, {big:,} quantized + "
      f"{small:,} fp32 params = {mb:.2f} MB payload")
