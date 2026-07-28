#!/usr/bin/env python3
"""OOF + test probs for arbitrary tag configs; seed-soup; nested fusion; submission.
EXP-019: skel 3-seed soup + imu + droi_big fusion."""
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import har_data
import har_models

ROOT = "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
ART = os.path.join(ROOT, "research/artifacts")
CKPT = os.path.join(ROOT, "checkpoints")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {  # tag -> (modality, build_kwargs, dataset_kwargs)
    "skel_jvb_big": ("skel", {"width": 256}, {"feat": "jvb"}),
    "skel_jvb_big_s1": ("skel", {"width": 256}, {"feat": "jvb"}),
    "skel_jvb_big_s2": ("skel", {"width": 256}, {"feat": "jvb"}),
    "skel_jvb_big_s3": ("skel", {"width": 256}, {"feat": "jvb"}),
    "skel_jvb_big_s4": ("skel", {"width": 256}, {"feat": "jvb"}),
    "stgcn_v1": ("skelg", {}, {}),
    "stgcn_s1": ("skelg", {}, {}),
    "stgcn_w96": ("skelg", {"width": 96}, {}),
    "imu_aug": ("imu", {}, {}),
    "imu_inv4": ("imu", {}, {"imu_feat": "inv"}),
    "imu_lstm": ("imu", {"arch": "lstm"}, {"imu_feat": "inv"}),
    "skel_mildaug": ("skel", {"width": 256}, {"feat": "jvb"}),
    "droi_big": ("droi", {"width": 64}, {"n_frames": 12}),
}
IMU_TAG = "imu_inv4"
SOUP_TAGS = ("skel_jvb_big", "skel_jvb_big_s1", "skel_jvb_big_s2", "skel_jvb_big_s3", "skel_jvb_big_s4", "skel_mildaug")


def gen(tag, split):
    out = os.path.join(ART, f"{'oof' if split == 'train' else 'testprobs'}_{tag}.npz")
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        return z["probs"], (z["labels"] if "labels" in z.files else None), list(z["sids"])
    mod, bkw, dkw = CFG[tag]
    P, Y, S = [], [], []
    folds = range(4) if split == "train" else [None]
    for fold in folds:
        if split == "train":
            ds = har_data.HARDataset("train", mod, fold, "val", **dkw)
        else:
            ds = har_data.HARDataset("test", mod, **dkw)
        in_ch = ds.skel_dim() if mod == "skel" else (ds.imu_dim() if mod == "imu" else None)
        m = har_models.build(mod, in_ch=in_ch, **bkw).to(DEV)
        if split == "train":
            m.load_state_dict(torch.load(f"{CKPT}/{tag}_f{fold}.pt", map_location=DEV))
        m.eval()
        acc = None
        if split == "test":  # average the 4 fold checkpoints
            acc = np.zeros((len(ds), 40))
            for f2 in range(4):
                m.load_state_dict(torch.load(f"{CKPT}/{tag}_f{f2}.pt", map_location=DEV))
                m.eval()
                ps = []
                with torch.no_grad():
                    for x, y, s in DataLoader(ds, 64, num_workers=4):
                        ps.append(torch.softmax(m(x.to(DEV)), 1).cpu().numpy())
                acc += np.concatenate(ps)
            P.append(acc / 4)
            S.extend(ds.ids)
            break
        with torch.no_grad():
            for x, y, s in DataLoader(ds, 64, num_workers=4):
                P.append(torch.softmax(m(x.to(DEV)), 1).cpu().numpy())
                Y.append(y.numpy())
                S.extend(s)
    P = np.concatenate(P)
    Y = np.concatenate(Y) if Y else None
    kw = {"probs": P, "sids": np.array(S)}
    if Y is not None:
        kw["labels"] = Y
    np.savez_compressed(out, **kw)
    print(f"{tag} {split} written", flush=True)
    return P, Y, S


def align(P, S, ref):
    o = {s: i for i, s in enumerate(S)}
    out = np.zeros((len(ref), 40))
    for i, s in enumerate(ref):
        j = o.get(s)
        if j is not None:
            out[i] = P[j]
    return out


def main():
    # OOF
    oof, labels, ref = {}, None, None
    for tag in CFG:
        P, Y, S = gen(tag, "train")
        if ref is None:
            ref, labels = S, Y
        oof[tag] = align(P, S, ref)
        labels_aligned = Y  # same order per tag construction differs; align labels via ref below
    # labels come from the first tag's order
    Y = labels
    soup = sum(oof[t] for t in SOUP_TAGS) / len(SOUP_TAGS)
    gsoup = (oof["stgcn_v1"] + oof["stgcn_s1"] + oof["stgcn_w96"]) / 3
    print(f"tcn-soup5: {(soup.argmax(1) == Y).mean():.4f} gcn-soup3: {(gsoup.argmax(1) == Y).mean():.4f}")
    for wg in (0.2, 0.25, 0.3, 0.35, 0.4):
        blk = (1 - wg) * soup + wg * gsoup
        print(f"  +{wg}*gcnsoup: {(blk.argmax(1) == Y).mean():.4f}")
    soup = 0.65 * soup + 0.35 * gsoup  # skeleton block (grid still rising at 0.4; 0.35 compromise)
    imublk = 0.7 * oof[IMU_TAG] + 0.3 * oof["imu_lstm"]
    print(f"imu inv {(oof[IMU_TAG].argmax(1)==Y).mean():.4f} +lstm blk {(imublk.argmax(1)==Y).mean():.4f}")
    streams = {"skel": soup, "imu": imublk, "droi": oof["droi_big"]}
    # nested fusion weight tuning (leave-one-fold-out over user folds)
    user_of = {}
    with open(os.path.join(ROOT, "cache/meta_train.csv")) as f:
        for r in csv.DictReader(f):
            user_of[r["sample_id"]] = r["user"]
    fold_users = json.load(open(os.path.join(ART, "cv_folds.json")))["folds"]
    fmask = {}
    for i, fd in enumerate(fold_users):
        us = set(fd["val_users"])
        fmask[i] = np.array([user_of[s] in us for s in ref])
    grid = [(a / 10, b / 10, 1 - a / 10 - b / 10) for a in range(11) for b in range(11 - a)]
    nested_hits = 0
    w_all_best, best_all = None, -1
    for w in grid:
        F = w[0] * streams["skel"] + w[1] * streams["imu"] + w[2] * streams["droi"]
        a = (F.argmax(1) == Y).mean()
        if a > best_all:
            best_all, w_all_best = a, w
    for i in range(4):
        best, wbest = -1, None
        tr = ~fmask[i]
        for w in grid:
            F = w[0] * streams["skel"] + w[1] * streams["imu"] + w[2] * streams["droi"]
            a = (F[tr].argmax(1) == Y[tr]).mean()
            if a > best:
                best, wbest = a, w
        F = wbest[0] * streams["skel"] + wbest[1] * streams["imu"] + wbest[2] * streams["droi"]
        nested_hits += (F[fmask[i]].argmax(1) == Y[fmask[i]]).sum()
    print(f"NESTED fusion OOF: {nested_hits / len(Y):.4f} | all-tuned (optimistic): {best_all:.4f} w={w_all_best}")
    # test
    tp = {}
    for tag in (IMU_TAG, "droi_big"):
        P, _, S = gen(tag, "test")
        tp[tag] = (P, S)
    souptest = None
    for tag in SOUP_TAGS:
        P, _, S = gen(tag, "test")
        souptest = P if souptest is None else souptest + P
    souptest /= len(SOUP_TAGS)
    Pl, _, Sl = gen("imu_lstm", "test")
    ol = {x: i for i, x in enumerate(Sl)}
    gtest = None
    for tag in ("stgcn_v1", "stgcn_s1", "stgcn_w96"):
        Pg, _, Sg = gen(tag, "test")
        og = {x: i for i, x in enumerate(Sg)}
        a = Pg[[og[x] for x in S]]
        gtest = a if gtest is None else gtest + a
    souptest = 0.65 * souptest + 0.35 * (gtest / 3)
    tsids = S
    imutest = 0.7 * align(*tp[IMU_TAG], tsids) + 0.3 * Pl[[ol[x] for x in tsids]]
    Ft = (w_all_best[0] * souptest + w_all_best[1] * imutest
          + w_all_best[2] * align(*tp["droi_big"], tsids))
    pred = Ft.argmax(1)
    out = os.path.join(ROOT, "submissions", "sub_block10.csv")
    with open(out, "w") as f:
        f.write("path,prediction\n")
        for sid, p in zip(tsids, pred):
            f.write(f"small_model_track_test/{sid}/,{p}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
