#!/usr/bin/env python3
"""M-02: temporal fusion transformer over aligned modality tokens (RESET-001)."""
import math

import torch
import torch.nn as nn


def conv_block(ci, co, s=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.GroupNorm(8, co), nn.SiLU())


class VisStem(nn.Module):
    """per-frame CNN -> token; applied on a stride-2 temporal grid to bound compute"""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1, 24, 2), conv_block(24, 48, 2), conv_block(48, 96, 2),
            conv_block(96, 128, 2), conv_block(128, 160, 2), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(160, d)

    def forward(self, v):  # (B,T,H,W) -> (B,T,d)
        B, T, H, W = v.shape
        h = self.net(v.reshape(B * T, 1, H, W)).reshape(B, T, -1)
        return self.proj(h)


class MMFusion(nn.Module):
    def __init__(self, skel_dim=153, imu_dim=85, d=256, layers=4, heads=4, n_cls=40,
                 t=48, vis_stride=2, mod_drop=0.3):
        super().__init__()
        self.t, self.vs, self.mod_drop = t, vis_stride, mod_drop
        self.skel_stem = nn.Sequential(nn.Linear(skel_dim, d), nn.GELU(), nn.Linear(d, d))
        self.imu_stem = nn.Sequential(nn.Linear(imu_dim, d), nn.GELU(), nn.Linear(d, d))
        self.vis_stem = VisStem(d)
        self.dur_proj = nn.Linear(1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mod_emb = nn.Parameter(torch.zeros(4, d))  # skel/imu/vis/dur
        pe = torch.zeros(t, d)
        pos = torch.arange(t).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(1e4) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, n_cls)
        self.aux = nn.ModuleDict({m: nn.Linear(d, n_cls) for m in ("skel", "imu", "vis")})
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.mod_emb, std=0.02)

    def forward(self, b, train_mode=True):
        B = b["skel"].shape[0]
        dev = b["skel"].device
        toks, masks, spans = [], [], {}
        drop = {}
        for m in ("skel", "imu", "vis"):
            r = torch.rand(B, device=dev) < self.mod_drop
            drop[m] = r if (train_mode and self.training) else torch.zeros(B, dtype=torch.bool, device=dev)
        # avoid dropping ALL modalities for a sample
        all_drop = drop["skel"] & drop["imu"] & drop["vis"]
        drop["skel"] = drop["skel"] & ~all_drop
        sk = self.skel_stem(b["skel"]) + self.mod_emb[0] + self.pe
        skm = b["skel_m"].bool() & ~drop["skel"][:, None]
        toks.append(sk); masks.append(skm); spans["skel"] = (0, self.t)
        im = self.imu_stem(b["imu"]) + self.mod_emb[1] + self.pe
        imm = b["imu_m"].bool() & ~drop["imu"][:, None]
        toks.append(im); masks.append(imm); spans["imu"] = (self.t, 2 * self.t)
        v = b["vis"][:, ::self.vs]
        vt = self.vis_stem(v) + self.mod_emb[2] + self.pe[::self.vs]
        vtm = b["vis_m"][:, ::self.vs].bool() & ~drop["vis"][:, None]
        toks.append(vt); masks.append(vtm)
        spans["vis"] = (2 * self.t, 2 * self.t + vt.shape[1])
        dur = (self.dur_proj(b["dur"]) + self.mod_emb[3])[:, None]
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([toks[0], toks[1], toks[2], dur, cls], 1)
        keep = torch.cat([masks[0], masks[1], masks[2],
                          torch.ones(B, 2, dtype=torch.bool, device=dev)], 1)
        h = self.tr(x, src_key_padding_mask=~keep)
        out = {"logits": self.head(h[:, -1])}
        for m in ("skel", "imu", "vis"):
            s0, s1 = spans[m]
            hm = h[:, s0:s1]
            km = keep[:, s0:s1].float()[..., None]
            pooled = (hm * km).sum(1) / km.sum(1).clamp(min=1)
            out[m] = self.aux[m](pooled)
        return out
