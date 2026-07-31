#!/usr/bin/env python3
"""Held-out OOF gate for temporal-jitter TTA.

The probe averages one center pass with N-1 temporal-only jitter passes for each
validation fold. It must be run before using TTA in a leaderboard submission.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import har_data
import har_models
from fuse_submit import ART, CFG, CKPT


def checkpoint_signature(tag):
    parts = []
    for fold in range(4):
        path = os.path.join(CKPT, f"{tag}_f{fold}.pt")
        st = os.stat(path)
        parts.append(f"{os.path.basename(path)}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def predict(model, dataset, batch_size, workers, device):
    probs, labels, sids = [], [], []
    with torch.no_grad():
        for x, y, sid in DataLoader(dataset, batch_size, num_workers=workers,
                                    pin_memory=device.type == "cuda"):
            probs.append(torch.softmax(model(x.to(device, non_blocking=True)), 1).cpu().numpy())
            labels.append(y.numpy())
            sids.extend(sid)
    return np.concatenate(probs), np.concatenate(labels), sids


def run_tag(tag, passes, batch_size, workers, force):
    if tag not in CFG:
        raise KeyError(f"unknown tag {tag!r}; add its config to fuse_submit.CFG")
    signature = checkpoint_signature(tag)
    out = os.path.join(ART, f"oof_tjitter_v2_p{passes}_{tag}.npz")
    if os.path.exists(out) and not force:
        z = np.load(out, allow_pickle=True)
        if str(z["checkpoint_signature"]) == signature:
            return z["clean_probs"], z["probs"], z["labels"], list(z["sids"])

    modality, build_kw, data_kw = CFG[tag]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_all, tta_all, labels_all, sids_all = [], [], [], []
    for fold in range(4):
        clean_ds = har_data.HARDataset("train", modality, fold, "val", aug=False, **data_kw)
        in_ch = (clean_ds.skel_dim() if modality == "skel"
                 else clean_ds.imu_dim() if modality == "imu" else None)
        model = har_models.build(modality, in_ch=in_ch, **build_kw).to(device)
        model.load_state_dict(torch.load(os.path.join(CKPT, f"{tag}_f{fold}.pt"),
                                         map_location=device))
        model.eval()
        clean, labels, sids = predict(model, clean_ds, batch_size, workers, device)
        total = clean.astype(np.float64)
        for seed in range(1, passes):
            jitter_ds = har_data.HARDataset(
                "train", modality, fold, "val", aug="tjit", seed=seed, **data_kw)
            jitter, jitter_labels, jitter_sids = predict(
                model, jitter_ds, batch_size, workers, device)
            if jitter_sids != sids or not np.array_equal(jitter_labels, labels):
                raise RuntimeError(f"sample order changed for {tag}, fold {fold}, pass {seed}")
            total += jitter
        clean_all.append(clean)
        tta_all.append((total / passes).astype(np.float32))
        labels_all.append(labels)
        sids_all.extend(sids)

    clean = np.concatenate(clean_all)
    tta = np.concatenate(tta_all)
    labels = np.concatenate(labels_all)
    np.savez_compressed(
        out,
        clean_probs=clean,
        probs=tta,
        labels=labels,
        sids=np.asarray(sids_all),
        passes=np.asarray(passes),
        checkpoint_signature=np.asarray(signature),
    )
    return clean, tta, labels, sids_all


def run_test_tag(tag, passes, batch_size, workers, force):
    if tag not in CFG:
        raise KeyError(f"unknown tag {tag!r}; add its config to fuse_submit.CFG")
    signature = checkpoint_signature(tag)
    out = os.path.join(ART, f"testprobs_tjitter_v2_p{passes}_{tag}.npz")
    if os.path.exists(out) and not force:
        z = np.load(out, allow_pickle=True)
        if str(z["checkpoint_signature"]) == signature:
            return z["clean_probs"], z["probs"], list(z["sids"])

    modality, build_kw, data_kw = CFG[tag]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_ds = har_data.HARDataset("test", modality, aug=False, **data_kw)
    in_ch = (clean_ds.skel_dim() if modality == "skel"
             else clean_ds.imu_dim() if modality == "imu" else None)
    model = har_models.build(modality, in_ch=in_ch, **build_kw).to(device)
    clean_total = np.zeros((len(clean_ds), 40), np.float64)
    tta_total = np.zeros_like(clean_total)
    ref_sids = None
    for fold in range(4):
        model.load_state_dict(torch.load(os.path.join(CKPT, f"{tag}_f{fold}.pt"),
                                         map_location=device))
        model.eval()
        clean, _, sids = predict(model, clean_ds, batch_size, workers, device)
        if ref_sids is None:
            ref_sids = sids
        elif sids != ref_sids:
            raise RuntimeError(f"sample order changed for {tag}, fold {fold}")
        clean_total += clean
        fold_total = clean.astype(np.float64)
        for seed in range(1, passes):
            jitter_ds = har_data.HARDataset(
                "test", modality, aug="tjit", seed=seed, **data_kw)
            jitter, _, jitter_sids = predict(model, jitter_ds, batch_size, workers, device)
            if jitter_sids != ref_sids:
                raise RuntimeError(
                    f"sample order changed for {tag}, fold {fold}, pass {seed}")
            fold_total += jitter
        tta_total += fold_total / passes

    clean = (clean_total / 4).astype(np.float32)
    tta = (tta_total / 4).astype(np.float32)
    np.savez_compressed(
        out,
        clean_probs=clean,
        probs=tta,
        sids=np.asarray(ref_sids),
        passes=np.asarray(passes),
        checkpoint_signature=np.asarray(signature),
    )
    return clean, tta, ref_sids


def report(tag, clean, tta, labels):
    p0, p1 = clean.argmax(1), tta.argmax(1)
    changed = p0 != p1
    rescued = changed & (p0 != labels) & (p1 == labels)
    harmed = changed & (p0 == labels) & (p1 != labels)
    print(
        f"{tag}: center={(p0 == labels).mean():.5f} "
        f"tta={(p1 == labels).mean():.5f} "
        f"delta={((p1 == labels).mean() - (p0 == labels).mean()):+.5f} "
        f"changed={changed.sum()} rescued={rescued.sum()} harmed={harmed.sum()}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="skel_jvb_big,imu_inv4,droi_big")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", choices=("train", "test", "both"), default="train")
    args = ap.parse_args()
    if args.passes < 2:
        raise ValueError("--passes must be at least 2")
    for tag in args.tags.split(","):
        tag = tag.strip()
        if args.split in ("train", "both"):
            clean, tta, labels, _ = run_tag(
                tag, args.passes, args.bs, args.workers, args.force)
            report(tag, clean, tta, labels)
        if args.split in ("test", "both"):
            clean, tta, sids = run_test_tag(
                tag, args.passes, args.bs, args.workers, args.force)
            print(
                f"{tag} test: rows={len(sids)} "
                f"center-vs-tta changes={(clean.argmax(1) != tta.argmax(1)).sum()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
