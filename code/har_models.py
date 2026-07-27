#!/usr/bin/env python3
"""Compact per-modality models. GroupNorm throughout (cross-subject; B-001/Q-15)."""
import torch
import torch.nn as nn


def conv1d_block(ci, co, k=5, s=1):
    return nn.Sequential(
        nn.Conv1d(ci, co, k, s, k // 2, bias=False),
        nn.GroupNorm(8, co),
        nn.SiLU(),
    )


class SeqTCN(nn.Module):
    """Temporal conv net over per-step feature vectors. skel: in=102, imu: in=85."""

    def __init__(self, in_ch, n_cls=40, width=128, depth=4, drop=0.3):
        super().__init__()
        self.in_ch = in_ch
        chs = [width * min(2 ** (i // 2), 2) for i in range(depth)]
        layers, ci = [], in_ch
        for i, co in enumerate(chs):
            layers.append(conv1d_block(ci, co, s=2 if i in (1, 3) else 1))
            ci = co
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(ci * 2, n_cls)

    def forward(self, x):  # (B, T, C) or (B, C, T)
        if x.shape[1] != self.in_ch:
            x = x.transpose(1, 2)
        h = self.body(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], -1)
        return self.head(self.drop(h))


def conv2d_block(ci, co, s=1):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, s, 1, bias=False),
        nn.GroupNorm(8, co),
        nn.SiLU(),
    )


class FrameCNN(nn.Module):
    """TSN-style: shared 2D CNN per frame, temporal mean+max pool. ~2.6M params @ width 32."""

    def __init__(self, n_cls=40, width=32, drop=0.4):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            conv2d_block(1, w, 2),          # 60x80
            conv2d_block(w, w),
            conv2d_block(w, 2 * w, 2),      # 30x40
            conv2d_block(2 * w, 2 * w),
            conv2d_block(2 * w, 4 * w, 2),  # 15x20
            conv2d_block(4 * w, 4 * w),
            conv2d_block(4 * w, 8 * w, 2),  # 8x10
            conv2d_block(8 * w, 8 * w),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(8 * w * 2, n_cls)

    def forward(self, x):  # (B, F, H, W)
        B, F, H, W = x.shape
        h = self.stem(x.reshape(B * F, 1, H, W)).reshape(B, F, -1)
        h = torch.cat([h.mean(1), h.amax(1)], -1)
        return self.head(self.drop(h))


def build(modality, in_ch=None, **kw):
    if modality == "skel":
        return SeqTCN(in_ch or 102, **kw)
    if modality == "imu":
        return SeqTCN(in_ch or 85, **kw)
    if modality in ("depth", "ir"):
        return FrameCNN(**kw)
    raise ValueError(modality)
