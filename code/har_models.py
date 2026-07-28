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


H36M_EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9),
              (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]


class STGCNBlock(nn.Module):
    def __init__(self, ci, co, A, stride=1):
        super().__init__()
        self.register_buffer("A", A)
        self.edge = nn.Parameter(torch.ones_like(A))
        self.gcn = nn.Conv2d(ci, co, 1)
        self.tcn = nn.Sequential(
            nn.GroupNorm(8, co), nn.SiLU(),
            nn.Conv2d(co, co, (5, 1), (stride, 1), (2, 0), bias=False),
            nn.GroupNorm(8, co))
        self.res = (nn.Identity() if ci == co and stride == 1
                    else nn.Conv2d(ci, co, 1, (stride, 1)))
        self.act = nn.SiLU()

    def forward(self, x):  # (B,C,T,V)
        h = self.gcn(x)
        h = torch.einsum("bctv,vw->bctw", h, self.A * self.edge)
        return self.act(self.tcn(h) + self.res(x))


class STGCN(nn.Module):
    """ST-GCN-lite on H36M topology. Input (B, 9, T, 17): xyz+vel+bone per joint."""

    def __init__(self, n_cls=40, width=64, drop=0.3):
        super().__init__()
        A = torch.eye(17)
        for i, j in H36M_EDGES:
            A[i, j] = A[j, i] = 1.0
        A = A / A.sum(1, keepdim=True)
        w = width
        self.blocks = nn.Sequential(
            STGCNBlock(9, w, A), STGCNBlock(w, w, A),
            STGCNBlock(w, 2 * w, A, stride=2), STGCNBlock(2 * w, 2 * w, A),
            STGCNBlock(2 * w, 4 * w, A, stride=2), STGCNBlock(4 * w, 4 * w, A))
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(4 * w * 2, n_cls)

    def forward(self, x):  # (B,9,T,17)
        h = self.blocks(x)
        h = torch.cat([h.mean((2, 3)), h.amax(3).amax(2)], -1)
        return self.head(self.drop(h))


class ConvLSTM1D(nn.Module):
    """DeepConvLSTM-style: conv1d stem + LSTM over time. Diversity member for IMU."""

    def __init__(self, in_ch=85, n_cls=40, width=96, drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(conv1d_block(in_ch, width), conv1d_block(width, width, s=2))
        self.lstm = nn.LSTM(width, width, 2, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(width, n_cls)
        self.in_ch = in_ch

    def forward(self, x):
        if x.shape[1] != self.in_ch:
            x = x.transpose(1, 2)
        h = self.stem(x).transpose(1, 2)  # (B,T,W)
        o, _ = self.lstm(h)
        return self.head(self.drop(o.mean(1)))


def build(modality, in_ch=None, arch="", **kw):
    if modality == "imu" and arch == "lstm":
        return ConvLSTM1D(in_ch or 85, **kw)
    if modality == "skelg":
        return STGCN(**kw)
    if modality == "skel":
        return SeqTCN(in_ch or 102, **kw)
    if modality == "imu":
        return SeqTCN(in_ch or 85, **kw)
    if modality in ("depth", "ir", "droi", "iroi", "droi224", "iroi224", "hroi"):
        return FrameCNN(**kw)
    raise ValueError(modality)
