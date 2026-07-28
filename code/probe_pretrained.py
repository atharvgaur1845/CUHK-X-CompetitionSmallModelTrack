#!/usr/bin/env python3
"""Q-84 DIAGNOSTIC ONLY (legality pending): ImageNet ResNet-18 vs from-scratch on ROI streams.
Measures whether initialization/representation is the visual gap. Not for submission
unless organizers rule pretrained-small legal.
Usage: python3 probe_pretrained.py droi|iroi [--scratch]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import har_data

class R18Frames(nn.Module):
    def __init__(self, pretrained=True, n_cls=40):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        old = m.conv1.weight.data
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        if pretrained:
            m.conv1.weight.data = old.sum(1, keepdim=True)
        m.fc = nn.Identity()
        self.body, self.head = m, nn.Linear(512, n_cls)

    def forward(self, x):  # (B,F,H,W)
        B, F, H, W = x.shape
        h = self.body(x.reshape(B * F, 1, H, W)).reshape(B, F, -1).mean(1)
        return self.head(h)

def run(modality, fold, pretrained, epochs=25, bs=32):
    tr = har_data.HARDataset("train", modality, fold, "train", aug="basic", seed=0)
    va = har_data.HARDataset("train", modality, fold, "val")
    lt = DataLoader(tr, bs, shuffle=True, num_workers=3, drop_last=True)
    lv = DataLoader(va, 64, num_workers=2)
    dev = "cuda"
    model = R18Frames(pretrained).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.body.parameters(), "lr": 1e-4 if pretrained else 1e-3},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(lt))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y, _ in lt:
            loss = crit(model(x.to(dev)), y.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _ in lv:
                ps.append(model(x.to(dev)).argmax(1).cpu()); ys.append(y)
        yv, pv = torch.cat(ys), torch.cat(ps)
        macro = float(np.mean([(pv[yv == c] == c).float().mean().item() for c in yv.unique().tolist()]))
        best = max(best, macro)
        if ep % 5 == 4:
            print(f"  {modality} fold={fold} pre={pretrained} ep{ep+1}: macro {macro:.4f} (best {best:.4f})", flush=True)
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modality")
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    pre = not a.scratch
    r_subj = run(a.modality, 0, pre, epochs=a.epochs, bs=a.bs)   # decision variable first
    print(f"PROBE-SUBJ {a.modality} pretrained={pre}: subject-fold0 {r_subj:.4f}", flush=True)
    r_rand = run(a.modality, "rand", pre, epochs=a.epochs, bs=a.bs)
    print(f"PROBE {a.modality} pretrained={pre}: random-split {r_rand:.4f}, subject-fold0 {r_subj:.4f}")
