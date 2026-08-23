#!/usr/bin/env python3
"""EXP-109: does averaging two interleaved 16-frame views beat one? Pooled OOF.

50.9% of raw frames are discarded at N_FRAMES=16 (median clip holds 24, 32.4% hold >32).
A 32-frame cache stores twice as many uniform samples; taking every other frame from
phase p yields 16 uniform samples over the same clip -- the distribution the models were
trained on, offset by half a step. So this buys temporal averaging on the EXISTING
checkpoints: no retraining, and no package bytes, which matters because EXP-108 measured
extra video models as the least byte-efficient thing we can buy.

Each fold's model is run only on the clips it held out, so the pooled number is honest.
"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle"))
import cuhkx_224_kaggle as K
import torch
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="k224_mvit")
ap.add_argument("--crop", default="person")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--out", default="oof_k224_mvit_t32")
a = ap.parse_args()

name = ("crop_224" if a.crop == "person" else "crop_wrist224") + "_t32"
paths = K.find_paths(cache_name=name)
cache = paths["cache"]
idx = json.loads((cache / "train_index.json").read_text())
sids, lab, usr = idx["sids"], {k: int(v) for k, v in idx["labels"].items()}, idx["users"]
row_of = {s: i for i, s in enumerate(sids)}
store = K.ClipStore(cache, "train")
phases = max(1, store.n_frames // K.N_FRAMES)
device = torch.device("cuda")
print(f"cache {name}: n_frames={store.n_frames} -> {phases} interleaved 16-frame views")

allS, allP1, allPT, allY = [], [], [], []
for fold in range(4):
    tag = f"{a.prefix}_f{fold}"
    val = [s for s in sids if usr[s] in K.FOLDS[fold]]
    rows = np.array([row_of[s] for s in val]); y = np.array([lab[s] for s in val])
    pkg = torch.load(f"checkpoints/{tag}.pt", map_location="cpu", weights_only=False)
    model = K.build_model(pkg["args"]["arch"]).to(device)
    model.load_state_dict(pkg["state_dict"]); model.eval()
    per = []
    for ph in range(phases):
        dl = DataLoader(K.make_dataset(store, rows, y, False, phase=ph),
                        batch_size=a.batch_size, shuffle=False, num_workers=0,
                        pin_memory=False)
        p, _ = K.predict(model, dl, device)
        per.append(p)
    p1, pt = per[0], np.mean(per, 0)
    print(f"  {tag}: n={len(rows)} phase0={float((p1.argmax(1)==y).mean()):.5f} "
          f"avg{phases}={float((pt.argmax(1)==y).mean()):.5f}", flush=True)
    allS += val; allP1.append(p1); allPT.append(pt); allY.append(y)
    del model
    torch.cuda.empty_cache()

P1 = np.concatenate(allP1); PT = np.concatenate(allPT); Y = np.concatenate(allY)
OBJ = np.isin(Y, K.OBJECT_CLASSES)
print(f"\npooled {len(Y)} clips")
for n, p in [("single 16-frame view", P1), (f"average of {phases} views", PT)]:
    k = (p.argmax(1) == Y)
    print(f"  {n:24s} micro={k.mean():.5f} object={k[OBJ].mean():.5f} clips={k.sum()}")
d = int((PT.argmax(1) == Y).sum()) - int((P1.argmax(1) == Y).sum())
print(f"\ntemporal TTA: {d:+d} clips / {len(Y)} -> {d/len(Y)*201:+.1f} public (member level)")
np.savez_compressed(f"research/artifacts/{a.out}.npz", probs=PT.astype(np.float32),
                    sids=np.array(allS, dtype="<U20"), labels=Y.astype(np.int64))
print(f"wrote research/artifacts/{a.out}.npz")
