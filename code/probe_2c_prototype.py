#!/usr/bin/env python3
"""2c — decide the fused top-2 by comparing the clip to the SUBJECT'S OWN prototypes.

EXP-132: 242 of the 290 rank-2 errors have a correctly-predicted clip of the true class
from the same subject, and 232 have one of the wrong (rank-1) class. So for most contested
clips the subject supplies both references. EXP-131 killed arbitration from posteriors;
this arbitrates in FEATURE space (B-028) using only the subject's other clips, and fits
nothing.

Rule (fixed in advance, no tuning):
    for clip x with fused top-2 (a, b):
      p_a = mean feature of the subject's OTHER clips predicted a with margin > MARGIN
      p_b = same for b
      if both exist:  s = cos(f(x), p_b) - cos(f(x), p_a)
      swap to b iff s > TAU
Everything else is left alone. Reported as rescued / broken, never net.

The failure mode to watch: prototypes are built from the model's own confident predictions,
so a subject-systematic bias contaminates them. If that dominates, this cannot work and the
measurement will say so.
"""
from __future__ import annotations
import argparse, collections, csv, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "research" / "artifacts"
EMB = ART / "emb"


def load():
    z = np.load(ART / "exp132_champion_oof.npz", allow_pickle=True)
    fused = {s: (zz, int(y), str(u)) for s, zz, y, u
             in zip([str(x) for x in z["sids"]], z["z"], z["y"], z["users"])}
    feats, blocks = {}, {}
    for view in ("person", "wrist"):
        for f in range(4):
            d = np.load(EMB / f"emb_{view}_{f}.npz", allow_pickle=True)
            for s, ff in zip([str(x) for x in d["sids"]], d["feats"]):
                feats.setdefault(s, {})[view] = ff
    info = {}
    with open(ROOT / "cache" / "meta_train.csv") as h:
        for r in csv.DictReader(h):
            if r.get("t0"):
                info[r["sample_id"]] = (r["user"], float(r["t0"]), float(r["t1"]))
    return fused, feats, info


def blocks_of(sids, info, gap=300.0):
    have = sorted([s for s in sids if s in info], key=lambda s: info[s][1])
    b, cur = {}, 0
    for i, s in enumerate(have):
        if i and info[s][1] - info[have[i - 1]][2] > gap:
            cur += 1
        b[s] = cur
    for s in sids:
        b.setdefault(s, (cur := cur + 1))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="person", choices=("person", "wrist", "both"))
    ap.add_argument("--group", default="user", choices=("user", "block"))
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pool-cap", type=int, default=0,
                    help="subsample each group to N clips, to mimic the test-side pool size")
    a = ap.parse_args()

    fused, feats, info = load()
    sids = [s for s in sorted(fused) if s in feats]
    print(f"clips with both fused posterior and embedding: {len(sids)}")
    blk = blocks_of(sids, info, 300.0)

    def vec(s):
        if a.view == "both":
            v = np.concatenate([feats[s]["person"], feats[s]["wrist"]])
        else:
            v = feats[s][a.view]
        return v / (np.linalg.norm(v) + 1e-9)

    Z = np.stack([fused[s][0] for s in sids])
    Y = np.array([fused[s][1] for s in sids])
    G = np.array([fused[s][2] if a.group == "user" else f"b{blk[s]}" for s in sids])
    V = np.stack([vec(s) for s in sids])
    order = np.argsort(-Z, axis=1)
    top1, top2 = order[:, 0], order[:, 1]
    marg = Z[np.arange(len(Z)), top1] - Z[np.arange(len(Z)), top2]
    base = (top1 == Y).mean()
    print(f"base (champion argmax, pre-decoder): {base:.5f}   "
          f"rank-2 available: {((top1 != Y) & (top2 == Y)).sum()}")

    rng = np.random.default_rng(0)
    idx_by_g = collections.defaultdict(list)
    for i, g in enumerate(G):
        idx_by_g[g].append(i)
    if a.pool_cap:
        for g in idx_by_g:
            if len(idx_by_g[g]) > a.pool_cap:
                idx_by_g[g] = list(rng.choice(idx_by_g[g], a.pool_cap, replace=False))

    print(f"\ngroup={a.group}  view={a.view}  margin>{a.margin}  "
          f"pool_cap={a.pool_cap or 'none'}  (median pool "
          f"{int(np.median([len(v) for v in idx_by_g.values()]))})")
    print("  tau   swaps  rescued  broken   net    acc")
    for tau in (0.0, 0.02, 0.05, 0.10, 0.15):
        pred = top1.copy()
        resc = brok = swaps = 0
        for i in range(len(sids)):
            pool = [j for j in idx_by_g[G[i]] if j != i and marg[j] > a.margin]
            if not pool:
                continue
            pa = [j for j in pool if top1[j] == top1[i]]
            pb = [j for j in pool if top1[j] == top2[i]]
            if not pa or not pb:
                continue
            ca = V[pa].mean(0); cb = V[pb].mean(0)
            ca /= np.linalg.norm(ca) + 1e-9; cb /= np.linalg.norm(cb) + 1e-9
            if float(V[i] @ cb - V[i] @ ca) > tau:
                swaps += 1
                pred[i] = top2[i]
                if Y[i] == top2[i]:
                    resc += 1
                elif Y[i] == top1[i]:
                    brok += 1
        acc = (pred == Y).mean()
        print(f"  {tau:4.2f}  {swaps:5d}  {resc:7d}  {brok:6d}  {resc-brok:+5d}  "
              f"{acc:.5f} ({100*(acc-base):+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
