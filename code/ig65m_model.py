#!/usr/bin/env python3
"""R(2+1)D-34 with IG-65M -> Kinetics-400 weights (moabitcoin/ig65m-pytorch).

Why not torchvision's r2plus1d_18: the published LB-0.711 notebook uses R(2+1)D-34
pretrained on IG-65M (65M Instagram videos) then finetuned on Kinetics-400, and its
single model scores 143/201 against our 121. Kinetics-400 alone is ~240k videos, so
this is a ~270x larger video pretraining corpus -- the last identified, untested
explanation for that 22-clip gap.

THE MIDPLANE BUG THIS FILE EXISTS TO FIX
----------------------------------------
torchvision's BasicBlock computes ONE midplanes value from (inplanes, planes) and
reuses it for both convs:

    midplanes = (inplanes*planes*27) // (inplanes*9 + 3*planes)
    conv1 = conv_builder(inplanes, planes, midplanes, stride)
    conv2 = conv_builder(planes,   planes, midplanes)          # <-- wrong for conv2

The original R(2+1)D derives conv2's midplanes from (planes, planes). In the
downsampling block of each layer they differ:

    layer2.0.conv2   checkpoint 288    torchvision 230
    layer3.0.conv2   checkpoint 576    torchvision 460
    layer4.0.conv2   checkpoint 1152   torchvision 921

Built naively, 18 of 416 tensors mismatch and a strict=False load leaves every
downsampling path randomly initialised -- which would look exactly like "IG-65M does
not transfer to this data" rather than "we built the wrong architecture".
`build_ig65m` asserts a complete load so that failure mode cannot happen silently.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models.video.resnet as R


def _mid(cin: int, cout: int) -> int:
    return (cin * cout * 3 * 3 * 3) // (cin * 3 * 3 + 3 * cout)


class BasicBlockR21D(nn.Module):
    """BasicBlock whose conv2 midplanes are derived from (planes, planes)."""
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, _mid(inplanes, planes), stride),
            nn.BatchNorm3d(planes), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, _mid(planes, planes)),
            nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


def build_ig65m(checkpoint: str, n_classes: int = 40, in_channels: int = 4) -> nn.Module:
    model = R.VideoResNet(block=BasicBlockR21D, conv_makers=[R.Conv2Plus1D] * 4,
                          layers=[3, 4, 6, 3], stem=R.R2Plus1dStem, num_classes=400)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in sd if k in model.state_dict()
           and model.state_dict()[k].shape != sd[k].shape]
    assert not missing and not unexpected and not bad, (
        f"incomplete load: {len(missing)} missing, {len(unexpected)} unexpected, "
        f"{len(bad)} shape-mismatched -- refusing to train on a partly random backbone")

    if in_channels != 3:
        stem = model.stem[0]
        old = stem.weight.data
        new = nn.Conv3d(in_channels, stem.out_channels, stem.kernel_size,
                        stem.stride, stem.padding, bias=stem.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = old
            for extra in range(3, in_channels):
                new.weight[:, extra:extra + 1] = old.mean(dim=1, keepdim=True)
        model.stem[0] = new
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
