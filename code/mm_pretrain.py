#!/usr/bin/env python3
"""M-03 / Q-94: cross-modal masked pretraining on ALL clips (train + test, no labels).

Mask contiguous spans (and sometimes whole modalities) of the aligned token grid;
predict the masked content from surviving tokens. Targets: raw skel/imu features,
14x14 downsampled vis frames. Rules-legal: no external weights, no labels.
Saves trunk weights for mm_train.py --init.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import mm_data
import mm_model

ROOT = "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
CKPT = os.path.join(ROOT, "checkpoints")


class Pretrainer(nn.Module):
    def __init__(self, d=256, layers=4, t=48):
        super().__init__()
        self.net = mm_model.MMFusion(d=d, layers=layers, t=t, mod_drop=0.0)
        self.dec_skel = nn.Linear(d, 153)
        self.dec_imu = nn.Linear(d, 85)
        self.dec_vis = nn.Linear(d, 14 * 14)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_tok, std=0.02)

    def forward(self, b, rng):
        B = b["skel"].shape[0]
        dev = b["skel"].device
        t = self.net.t
        n = self.net
        # encode stems (no drop)
        sk = n.skel_stem(b["skel"]) + n.mod_emb[0] + n.pe
        im = n.imu_stem(b["imu"]) + n.mod_emb[1] + n.pe
        v = b["vis"][:, ::n.vs]
        vt = n.vis_stem(v) + n.mod_emb[2] + n.pe[::n.vs]
        tv = vt.shape[1]
        # build masks: True = MASKED (to predict)
        def span_mask(valid, frac):
            m = torch.zeros_like(valid)
            for i in range(B):
                idx = valid[i].nonzero().flatten()
                if len(idx) < 4:
                    continue
                L = max(1, int(len(idx) * frac))
                s = int(rng.integers(0, len(idx) - L + 1))
                m[i, idx[s:s + L]] = True
            return m
        vm_s, vm_i = b["skel_m"].bool(), b["imu_m"].bool()
        vm_v = b["vis_m"][:, ::n.vs].bool()
        frac = 0.4
        ms, mi, mv = span_mask(vm_s, frac), span_mask(vm_i, frac), span_mask(vm_v, frac)
        # whole-modality masking (cross-modal prediction pressure)
        for m_, vm_ in ((ms, vm_s), (mi, vm_i), (mv, vm_v)):
            whole = torch.rand(B, device=dev) < 0.15
            m_ |= whole[:, None] & vm_
        # replace masked tokens
        sk = torch.where(ms[..., None], self.mask_tok.expand(B, t, -1), sk)
        im = torch.where(mi[..., None], self.mask_tok.expand(B, t, -1), im)
        vt = torch.where(mv[..., None], self.mask_tok.expand(B, tv, -1), vt)
        dur = (n.dur_proj(b["dur"]) + n.mod_emb[3])[:, None]
        x = torch.cat([sk, im, vt, dur], 1)
        keep = torch.cat([vm_s, vm_i, vm_v, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        h = n.tr(x, src_key_padding_mask=~keep)
        hs, hi, hv = h[:, :t], h[:, t:2 * t], h[:, 2 * t:2 * t + tv]
        loss, denom = 0.0, 0
        if ms.any():
            loss = loss + nn.functional.smooth_l1_loss(self.dec_skel(hs)[ms], b["skel"][ms])
            denom += 1
        if mi.any():
            loss = loss + nn.functional.smooth_l1_loss(self.dec_imu(hi)[mi], b["imu"][mi])
            denom += 1
        if mv.any():
            tgt = nn.functional.adaptive_avg_pool2d(b["vis"][:, ::n.vs][mv][:, None], 14).flatten(1)
            loss = loss + nn.functional.smooth_l1_loss(self.dec_vis(hv)[mv], tgt)
            denom += 1
        return loss / max(denom, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="mm_ssl_trunk")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    dss = []
    for fold in [0]:  # train split: use ALL samples via train+val parts of one fold
        dss.append(mm_data.MMDataset("train", fold, "train", aug=True))
        dss.append(mm_data.MMDataset("train", fold, "val", aug=True))
    dss.append(mm_data.MMDataset("test", aug=True))
    ds = ConcatDataset(dss)
    dl = DataLoader(ds, args.bs, shuffle=True, num_workers=4, drop_last=True,
                    persistent_workers=True, pin_memory=True)
    model = Pretrainer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler()
    for ep in range(args.epochs):
        tot = k = 0
        for b, _, _ in dl:
            b = {kk: v.to(dev, non_blocking=True) for kk, v in b.items()}
            with torch.amp.autocast("cuda"):
                loss = model(b, rng)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item(); k += 1
        print(f"ssl ep{ep + 1}: loss {tot / k:.4f}", flush=True)
    torch.save(model.net.state_dict(), os.path.join(CKPT, args.out + ".pt"))
    print(f"SSL_DONE saved {args.out}.pt")


if __name__ == "__main__":
    main()
