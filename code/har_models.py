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


class MultiStreamTCN(nn.Module):
    """Separate joint, motion, and bone TCN branches before feature fusion.

    The input is the existing jvb tensor flattened as (B,T,17*9). Keeping the
    three physical views separate prevents the first convolution from collapsing
    them and supplies architecture diversity without changing preprocessing.
    """

    def __init__(self, in_ch=153, n_cls=40, width=128, depth=4, drop=0.3):
        super().__init__()
        if in_ch != 17 * 9:
            raise ValueError(f"MultiStreamTCN requires 153-channel jvb input, got {in_ch}")
        self.in_ch = in_ch
        chs = [width * min(2 ** (i // 2), 2) for i in range(depth)]
        self.branches = nn.ModuleList()
        for _ in range(3):
            layers, ci = [], 17 * 3
            for i, co in enumerate(chs):
                layers.append(conv1d_block(ci, co, s=2 if i in (1, 3) else 1))
                ci = co
            self.branches.append(nn.Sequential(*layers))
        self.drop = nn.Dropout(drop)
        self.feature_dim = 3 * ci * 2
        self.head = nn.Linear(self.feature_dim, n_cls)

    def forward_features(self, x):
        if x.shape[-1] != self.in_ch:
            x = x.transpose(1, 2)
        b, t, _ = x.shape
        # _skel concatenates xyz, velocity, and bone inside each joint.
        x = x.reshape(b, t, 17, 9)
        views = [x[..., 0:3], x[..., 3:6], x[..., 6:9]]
        pooled = []
        for view, branch in zip(views, self.branches):
            h = branch(view.reshape(b, t, 17 * 3).transpose(1, 2))
            pooled.extend((h.mean(-1), h.amax(-1)))
        return torch.cat(pooled, -1)

    def forward(self, x):
        return self.head(self.drop(self.forward_features(x)))


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


class DANNMultiStreamTCN(MultiStreamTCN):
    """Multi-stream classifier with a training-only subject-adversarial head."""

    def __init__(self, in_ch=153, n_cls=40, n_domains=25, width=128, depth=4, drop=0.3):
        super().__init__(in_ch=in_ch, n_cls=n_cls, width=width, depth=depth, drop=drop)
        self.domain_head = nn.Sequential(
            nn.Linear(self.feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, n_domains),
        )

    def forward_domain(self, x, strength):
        features = self.forward_features(x)
        logits = self.head(self.drop(features))
        domain = self.domain_head(_GradientReverse.apply(features, strength))
        return logits, domain


class MaskedSeqTCN(nn.Module):
    """TCN with validity-aware pooling for real-time padded skeleton sequences."""

    def __init__(self, in_ch=154, n_cls=40, width=256, depth=4, drop=0.3):
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

    def forward(self, x):
        if x.shape[-1] == self.in_ch:
            mask = x[..., -1]
            x = x.transpose(1, 2)
        elif x.shape[1] == self.in_ch:
            mask = x[:, -1]
        else:
            raise ValueError(f"expected {self.in_ch} input channels, got {tuple(x.shape)}")
        h = self.body(x)
        m = torch.nn.functional.adaptive_max_pool1d(mask[:, None], h.shape[-1])
        mean = (h * m).sum(-1) / m.sum(-1).clamp_min(1.0)
        maxv = h.masked_fill(m == 0, torch.finfo(h.dtype).min).amax(-1)
        maxv = torch.where(m.sum(-1) > 0, maxv, torch.zeros_like(maxv))
        return self.head(self.drop(torch.cat([mean, maxv], -1)))


class SkeletonBiGRU(nn.Module):
    """Order-sensitive recurrent skeleton encoder for ensemble diversity."""

    def __init__(self, in_ch=153, n_cls=40, width=128, depth=2, drop=0.3):
        super().__init__()
        self.in_ch = in_ch
        emb = 2 * width
        self.proj = nn.Sequential(
            nn.Linear(in_ch, emb),
            nn.LayerNorm(emb),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            emb,
            width,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
            dropout=drop if depth > 1 else 0.0,
        )
        self.attn = nn.Linear(2 * width, 1)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(4 * width, n_cls)

    def forward(self, x):
        if x.shape[-1] != self.in_ch:
            x = x.transpose(1, 2)
        h, _ = self.gru(self.proj(x))
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=-1)
        attended = (h * weights[..., None]).sum(1)
        pooled = torch.cat([attended, h.amax(1)], -1)
        return self.head(self.drop(pooled))


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


def h36m_adjacency_partitions():
    """Self, inward, and outward normalized H36M adjacency matrices."""
    a = torch.zeros(3, 17, 17)
    a[0] = torch.eye(17)
    for parent, child in H36M_EDGES:
        a[1, child, parent] = 1.0
        a[2, parent, child] = 1.0
    degree = a.sum(-1, keepdim=True).clamp_min(1.0)
    return a / degree


class AdaptiveSTGCNBlock(nn.Module):
    """Partitioned adaptive graph conv plus multi-scale temporal conv."""

    def __init__(self, ci, co, A, stride=1):
        super().__init__()
        self.register_buffer("A", A)
        self.PA = nn.Parameter(torch.zeros_like(A))
        self.gconv = nn.Conv2d(ci, len(A) * co, 1, bias=False)
        self.pre = nn.Sequential(nn.GroupNorm(8, co), nn.SiLU())
        q = co // 4
        self.temporal = nn.ModuleList([
            nn.Conv2d(co, q, (3, 1), (stride, 1), (1, 0), bias=False),
            nn.Conv2d(co, q, (3, 1), (stride, 1), (2, 0),
                      dilation=(2, 1), bias=False),
            nn.Sequential(
                nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                nn.Conv2d(co, q, 1, bias=False),
            ),
            nn.Conv2d(co, q, 1, (stride, 1), bias=False),
        ])
        self.post = nn.GroupNorm(8, co)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(co, max(8, co // 8), 1),
            nn.SiLU(),
            nn.Conv2d(max(8, co // 8), co, 1),
            nn.Sigmoid(),
        )
        self.res = (nn.Identity() if ci == co and stride == 1
                    else nn.Conv2d(ci, co, 1, (stride, 1), bias=False))
        self.act = nn.SiLU()

    def forward(self, x):
        b, _, t, v = x.shape
        h = self.gconv(x).reshape(b, len(self.A), -1, t, v)
        h = torch.einsum("bkctv,kvw->bctw", h, self.A + self.PA)
        h = self.pre(h)
        h = torch.cat([branch(h) for branch in self.temporal], 1)
        h = self.post(h)
        h = h * self.channel_gate(h)
        return self.act(h + self.res(x))


class AdaptiveSTGCN(nn.Module):
    """Adaptive multi-partition ST-GCN for a stronger graph-family member."""

    def __init__(self, n_cls=40, width=64, drop=0.3):
        super().__init__()
        A = h36m_adjacency_partitions()
        w = width
        self.input_norm = nn.GroupNorm(3, 9)
        self.blocks = nn.Sequential(
            AdaptiveSTGCNBlock(9, w, A),
            AdaptiveSTGCNBlock(w, w, A),
            AdaptiveSTGCNBlock(w, 2 * w, A, stride=2),
            AdaptiveSTGCNBlock(2 * w, 2 * w, A),
            AdaptiveSTGCNBlock(2 * w, 4 * w, A, stride=2),
            AdaptiveSTGCNBlock(4 * w, 4 * w, A),
        )
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(8 * w, n_cls)

    def forward(self, x):
        h = self.blocks(self.input_norm(x))
        h = torch.cat([h.mean((2, 3)), h.amax(3).amax(2)], -1)
        return self.head(self.drop(h))


class CTRGraphConv(nn.Module):
    """Channel-wise, sample-dependent topology refinement (CTR-GCN style)."""

    def __init__(self, ci, co, A):
        super().__init__()
        rel = max(8, co // 8)
        self.register_buffer("A", A)
        self.query = nn.ModuleList([nn.Conv2d(ci, rel, 1) for _ in range(len(A))])
        self.key = nn.ModuleList([nn.Conv2d(ci, rel, 1) for _ in range(len(A))])
        self.value = nn.ModuleList([
            nn.Conv2d(ci, co, 1, bias=False) for _ in range(len(A))
        ])
        self.relation = nn.ModuleList([
            nn.Conv2d(rel, co, 1, bias=False) for _ in range(len(A))
        ])
        # Starting from the physical graph keeps the first optimization steps
        # stable; each subset learns how much sample-dependent topology to add.
        self.alpha = nn.Parameter(torch.zeros(len(A)))

    def forward(self, x):
        out = 0.0
        for k, (query, key, value, relation) in enumerate(
            zip(self.query, self.key, self.value, self.relation)
        ):
            q = query(x).mean(2)  # (B,R,V)
            v_key = key(x).mean(2)
            pairwise = torch.tanh(q.unsqueeze(-1) - v_key.unsqueeze(-2))
            adaptive = relation(pairwise)
            adjacency = self.A[k][None, None] + self.alpha[k] * adaptive
            features = value(x)
            out = out + torch.einsum("bcuv,bctv->bctu", adjacency, features)
        return out


class CTRGCNBlock(nn.Module):
    """CTR graph refinement followed by multi-scale temporal aggregation."""

    def __init__(self, ci, co, A, stride=1):
        super().__init__()
        if co % 8 or co % 4:
            raise ValueError("CTRGCN block width must be divisible by 8")
        self.graph = CTRGraphConv(ci, co, A)
        self.pre = nn.Sequential(nn.GroupNorm(8, co), nn.SiLU())
        q = co // 4
        self.temporal = nn.ModuleList([
            nn.Conv2d(co, q, (3, 1), (stride, 1), (1, 0), bias=False),
            nn.Conv2d(
                co, q, (3, 1), (stride, 1), (2, 0),
                dilation=(2, 1), bias=False,
            ),
            nn.Sequential(
                nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                nn.Conv2d(co, q, 1, bias=False),
            ),
            nn.Conv2d(co, q, 1, (stride, 1), bias=False),
        ])
        self.post = nn.GroupNorm(8, co)
        self.res = (
            nn.Identity()
            if ci == co and stride == 1
            else nn.Conv2d(ci, co, 1, (stride, 1), bias=False)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.pre(self.graph(x))
        h = self.post(torch.cat([branch(h) for branch in self.temporal], 1))
        return self.act(h + self.res(x))


class CTRGCN(nn.Module):
    """Compact sample-adaptive graph network over joint/motion/bone channels."""

    def __init__(self, n_cls=40, width=64, drop=0.3):
        super().__init__()
        if width % 8:
            raise ValueError("CTRGCN width must be divisible by 8")
        A = h36m_adjacency_partitions()
        w = width
        self.input_norm = nn.GroupNorm(3, 9)
        self.blocks = nn.Sequential(
            CTRGCNBlock(9, w, A),
            CTRGCNBlock(w, w, A),
            CTRGCNBlock(w, 2 * w, A, stride=2),
            CTRGCNBlock(2 * w, 2 * w, A),
            CTRGCNBlock(2 * w, 4 * w, A, stride=2),
            CTRGCNBlock(4 * w, 4 * w, A),
        )
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(8 * w, n_cls)

    def forward(self, x):
        h = self.blocks(self.input_norm(x))
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], -1)
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
    if modality == "skel" and arch == "multitcn":
        return MultiStreamTCN(in_ch or 153, **kw)
    if modality == "skel" and arch == "multitcn_dann":
        return DANNMultiStreamTCN(in_ch or 153, **kw)
    if modality == "skel" and arch == "masktcn":
        return MaskedSeqTCN(in_ch or 154, **kw)
    if modality == "skel" and arch == "bigru":
        return SkeletonBiGRU(in_ch or 153, **kw)
    if modality == "skelg" and arch == "adaptive":
        return AdaptiveSTGCN(**kw)
    if modality == "skelg" and arch == "ctr":
        return CTRGCN(**kw)
    if modality == "skelg":
        return STGCN(**kw)
    if modality == "skel":
        return SeqTCN(in_ch or 102, **kw)
    if modality == "imu":
        return SeqTCN(in_ch or 85, **kw)
    if modality in ("depth", "ir", "droi", "iroi", "droi224", "iroi224", "hroi"):
        return FrameCNN(**kw)
    raise ValueError(modality)
