#!/usr/bin/env python3
"""Dump the MViT penultimate feature for every clip. Substrate for P0 / 2a / 2c.

EXP-131 showed the rank-1-vs-rank-2 decision is NOT recoverable from the five members'
posteriors. EXP-132 showed the error is a per-SUBJECT bias (229 of 658 errors repeat the
same true->pred confusion inside one subject; 242 of 290 rank-2 errors have a correctly
predicted clip of the true class from the same subject). Both point at the same place: the
information that is missing from the posterior is still present in the FEATURE, and the
conditioning variable is the subject.

So dump f(x) in R^768 -- the input to model.head[-1] -- for
  * every train clip, from the fold model that HELD IT OUT (honest, never trained on it)
  * every test clip, from the all-train model (what actually ships)
for both views. One (view, fold) per process invocation: the first attempt loaded several
137 MB checkpoints in one process alongside the memmap and was OOM-killed at 15 GB.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
import cuhkx_224_kaggle as K            # noqa: E402  (build_model/ClipStore/make_dataset)

VIEWS = {"person": ("crop_224", "k224_mvit"), "wrist": ("crop_wrist224", "k224_mvitwrist")}


def dump(view: str, fold, batch: int, out_dir: Path):
    cache_name, tag = VIEWS[view]
    cache = ROOT / "cache" / cache_name
    split = "test" if fold == "test" else "train"
    store = K.ClipStore(cache, split)
    idx = json.loads((cache / f"{split}_index.json").read_text())
    sids_all = idx["sids"]

    if fold == "test":
        ckpt, rows, sids = f"{tag}_all", np.arange(len(sids_all)), sids_all
        labels = users = None
    else:
        hold = K.FOLDS[int(fold)]
        usr, lab = idx["users"], idx["labels"]
        row_of = {s: i for i, s in enumerate(sids_all)}
        sids = [s for s in sids_all if usr[s] in hold]
        rows = np.array([row_of[s] for s in sids])
        labels = np.array([int(lab[s]) for s in sids])
        users = np.array([usr[s] for s in sids])
        ckpt = f"{tag}_f{fold}"

    out = out_dir / f"emb_{view}_{fold}.npz"
    if out.exists():
        print(f"  {out.name} exists, skip")
        return

    dev = torch.device("cuda")
    pkg = torch.load(ROOT / "checkpoints" / f"{ckpt}.pt", map_location="cpu", weights_only=False)
    model = K.build_model(pkg["args"]["arch"], image_size=store.size).to(dev)
    model.load_state_dict(pkg["state_dict"])
    model.eval()

    grab = {}
    h = model.head[-1].register_forward_hook(lambda m, i, o: grab.__setitem__("f", i[0].detach()))
    mean, std = K.norm_tensors(dev)
    dl = DataLoader(K.make_dataset(store, rows, None, False), batch_size=batch,
                    shuffle=False, num_workers=0, pin_memory=False)

    F, L, t0 = [], [], time.time()
    with torch.no_grad():
        for x, _ in dl:
            x = (x.to(dev) - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                lg1 = model(x).float(); f1 = grab["f"].float()
                lg2 = model(torch.flip(x, dims=[-1])).float(); f2 = grab["f"].float()
            F.append((0.5 * (f1 + f2)).cpu().numpy())
            L.append((0.5 * (lg1 + lg2)).cpu().numpy())
    h.remove()
    F = np.concatenate(F); L = np.concatenate(L)
    kw = dict(feats=F.astype(np.float32), logits=L.astype(np.float32),
              sids=np.array(sids, dtype="<U24"))
    if labels is not None:
        kw.update(labels=labels, users=users)
        acc = float((L.argmax(1) == labels).mean())
        print(f"  {view} fold {fold}: n={len(sids)} dim={F.shape[1]} acc={acc:.5f} "
              f"[{time.time()-t0:.0f}s]")
    else:
        print(f"  {view} test: n={len(sids)} dim={F.shape[1]} [{time.time()-t0:.0f}s]")
    np.savez_compressed(out, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, choices=list(VIEWS))
    ap.add_argument("--fold", required=True, help="0|1|2|3|test")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "research" / "artifacts" / "emb"))
    a = ap.parse_args()
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    dump(a.view, a.fold if a.fold == "test" else int(a.fold), a.batch, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
