#!/usr/bin/env python3
"""EXP-121: combine seed replicates of one config, in WEIGHT space and in PROBABILITY
space, and measure both against the individual seeds on held-out clips.

WHY THIS AND NOT PREDICTION AVERAGING ALONE. R-6 caps every weight loaded at inference,
ensemble members included, at 100 MB in ONE file. Averaging the PROBABILITIES of three
MViT seeds means shipping three checkpoints: 3 x 34.3 MB int8 = 102.9 MB for the person
view alone, before the wrist view, the skeleton stack or the IMU branch. It is
unshippable, so its score is of academic interest only.

Averaging the WEIGHTS costs nothing: the soup IS one checkpoint, 34.3 MB, byte-identical
in size to a single seed. If it recovers even part of the prediction-averaging gain it is
the only member-strength lever in this campaign that is free under the size cap -- which
matters because T-PKG is the project's highest risk and every other member-strength idea
makes it worse.

Weight averaging is not generally valid -- two independently initialised networks are not
in a common loss basin and their mean is garbage. It IS expected to work here, which is
the whole reason this is worth an experiment: every seed fine-tunes from the SAME
Kinetics-400 initialisation and differs only in data order and augmentation draws. That
is precisely the regime "model soups" (Wortsman et al., 2022) reports as souppable.
The script measures it rather than assuming it, and prints prediction averaging beside it
as the upper bound weight-space is trying to reach.

Both are reported against the seed mean, NOT against the best seed: picking the best of
three and calling the soup's margin over it a gain is selection bias, and with a seed
sigma of 1.16 (EXP-120a) the best of three runs about +1.3 high by construction.

Usage (checkpoints are {tag}.pt as written by stage_train):
    python3 code/soup_seeds.py --ckpt k224_mvit_f2_s1.pt k224_mvit_f2_s2.pt k224_mvit_f2_s3.pt \
                               --fold 2 --cache-dir /kaggle/working/cache/crop_224
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import cuhkx_224_kaggle as K   # noqa: E402


def load_ckpts(paths):
    import torch
    out = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        out.append(d)
        print(f"  {Path(p).name}: micro={d.get('micro', float('nan')):.5f} "
              f"arch={d['args']['arch']} seed={d['args'].get('seed')}")
    archs = {d["args"]["arch"] for d in out}
    px = {d["args"]["image_size"] for d in out}
    assert len(archs) == 1 and len(px) == 1, f"cannot soup across configs: {archs} {px}"
    return out


def soup(state_dicts):
    """Uniform mean of float tensors; non-float buffers (counts, indices) taken from
    the first checkpoint -- averaging an integer step counter is meaningless."""
    import torch
    keys = set(state_dicts[0])
    for sd in state_dicts[1:]:
        assert set(sd) == keys, "checkpoints disagree on parameter names"
    out, averaged, copied = {}, 0, 0
    for k in state_dicts[0]:
        vals = [sd[k] for sd in state_dicts]
        if vals[0].is_floating_point():
            out[k] = torch.stack([v.float() for v in vals], 0).mean(0).to(vals[0].dtype)
            averaged += 1
        else:
            out[k] = vals[0].clone()
            copied += 1
    print(f"  soup: averaged {averaged} float tensors, copied {copied} non-float")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=2, choices=(0, 1, 2, 3))
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None, help="write the soup checkpoint here")
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    print("checkpoints:")
    ck = load_ckpts(a.ckpt)
    arch = ck[0]["args"]["arch"]
    px = ck[0]["args"]["image_size"]

    cache = Path(a.cache_dir)
    idx = json.loads((cache / "train_index.json").read_text())
    sids, lab, usr = idx["sids"], idx["labels"], idx["users"]
    row_of = {s: i for i, s in enumerate(sids)}
    holdout = K.FOLDS[a.fold]
    val_sids = [s for s in sids if usr[s] in holdout]
    va_rows = np.array([row_of[s] for s in val_sids])
    va_lab = np.array([int(lab[s]) for s in val_sids])
    print(f"fold {a.fold}: {len(val_sids)} held-out clips")

    store = K.ClipStore(cache, "train")
    vl = DataLoader(K.make_dataset(store, va_rows, va_lab, False),
                    batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, pin_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(sd, name):
        model = K.build_model(arch, image_size=px).to(device)
        model.load_state_dict(sd)
        probs, labels = K.predict(model, vl, device)
        pred = probs.argmax(1)
        om = np.isin(labels, K.OBJECT_CLASSES)
        micro = float((pred == labels).mean())
        obj = float((pred[om] == labels[om]).mean())
        print(f"  {name:22s} micro={micro:.5f}  object={obj:.5f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return micro, probs

    print("\nindividual seeds (re-evaluated here, so every number below shares one path):")
    singles, plist = [], []
    for p, d in zip(a.ckpt, ck):
        m, pr = evaluate(d["state_dict"], Path(p).stem)
        singles.append(m)
        plist.append(pr)

    print("\ncombinations:")
    sm = soup([d["state_dict"] for d in ck])
    soup_micro, _ = evaluate(sm, "WEIGHT soup (34.3MB)")

    pa = np.mean(np.stack(plist, 0), 0)
    pred = pa.argmax(1)
    om = np.isin(va_lab, K.OBJECT_CLASSES)
    pavg_micro = float((pred == va_lab).mean())
    print(f"  {'PROB average (%.1fMB)' % (34.3*len(ck)):22s} micro={pavg_micro:.5f}  "
          f"object={float((pred[om]==va_lab[om]).mean()):.5f}")

    mean_seed = float(np.mean(singles))
    print("\n================ EXP-121 RESULT ================")
    print(f"  seed mean            {mean_seed:.5f}   (n={len(singles)}, the honest baseline)")
    print(f"  best single seed     {max(singles):.5f}   (selection-biased, do not compare against this)")
    print(f"  WEIGHT soup          {soup_micro:.5f}   {100*(soup_micro-mean_seed):+.2f} pts vs seed mean   [SHIPPABLE: 34.3 MB]")
    print(f"  PROB average         {pavg_micro:.5f}   {100*(pavg_micro-mean_seed):+.2f} pts vs seed mean   [{34.3*len(ck):.1f} MB -- breaks R-6]")
    print("  Adoption bar: this is ONE fold. Seed sigma 1.16 => a paired single-fold")
    print("  delta carries 1.64, so the 2-SE bar here is 3.28 (EXP-120a). Treat anything")
    print("  smaller as a lead to replicate on more folds, not as a result.")

    if a.out:
        torch.save({"state_dict": sm, "args": ck[0]["args"],
                    "micro": soup_micro, "souped_from": [str(p) for p in a.ckpt]}, a.out)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
