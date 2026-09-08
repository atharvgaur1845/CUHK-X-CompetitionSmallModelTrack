#!/usr/bin/env python3
"""P0 + 2a — can we recover the SUBJECT on test, and does centering on it pay?

EXP-132: 229 of 658 champion errors repeat the same (true -> pred) confusion inside one
subject, and 55 of 70 subject-pair cells are strictly one-directional. That is a
subject-constant offset in feature space, which is invisible to any per-clip model and to
any re-ranker over posteriors (EXP-131).

P0  Can test clips be grouped by subject? Validated on TRAIN, where the answer is known.
    A group is only useful if it is both PURE and LARGE: EXP-099 measured per-block AdaBN
    (100% pure, median 10 clips) at 0.67945, WORSE than pooled 0.69172, while per-subject
    reached 0.70092. Purity alone is not the criterion.

2a  Does f~ = f - mean_S(f) + mu_train raise held-out accuracy? Measured two ways:
      ORACLE   subject = the true user      -> the ceiling if recovery were perfect
      BLOCK    subject = 300 s timestamp block -> what actually ships
    Adoption bar: 4-fold paired mean > 1.64 points (2 SE at sigma=1.16, EXP-120a).
"""
from __future__ import annotations
import argparse, collections, csv, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "research" / "artifacts"
EMB = ART / "emb"
FOLDS = {0: ("user1", "user19", "user21", "user8", "user9"),
         1: ("user18", "user20", "user23", "user3", "user7"),
         2: ("user22", "user24", "user5", "user6"),
         3: ("user16", "user17", "user2", "user4")}


def head_of(tag):
    p = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu", weights_only=False)
    sd = p["state_dict"]
    w = [k for k in sd if k.startswith("head") and k.endswith("weight")][-1]
    b = w[:-6] + "bias"
    return sd[w].float().numpy(), sd[b].float().numpy()


def meta():
    m = {}
    with open(ROOT / "cache" / "meta_train.csv") as h:
        for r in csv.DictReader(h):
            if r.get("t0"):
                m[r["sample_id"]] = (r["user"], float(r["t0"]), float(r["t1"]))
    t = {}
    with open(ROOT / "cache" / "meta_test.csv") as h:
        for r in csv.DictReader(h):
            t[r["sample_id"]] = (None, float(r["t0"]), float(r["t1"]))
    return m, t


def blocks_of(sids, info, gap=300.0):
    """Contiguous timestamp blocks. Two train clips carry no t0 (measured); they get
    singleton blocks rather than being dropped, so every clip keeps a group id."""
    have = [s for s in sids if s in info]
    o = sorted(have, key=lambda s: info[s][1])
    b, cur = {}, 0
    for i, s in enumerate(o):
        if i and info[s][1] - info[o[i - 1]][2] > gap:
            cur += 1
        b[s] = cur
    for s in sids:
        if s not in b:
            cur += 1
            b[s] = cur
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="person")
    ap.add_argument("--gap", type=float, default=300.0)
    a = ap.parse_args()
    tr_info, te_info = meta()
    tag = "k224_mvit" if a.view == "person" else "k224_mvitwrist"

    # ---------------------------------------------------------------- P0
    print("=" * 78)
    print(f"P0 — subject recovery from block-mean embeddings ({a.view} view)")
    print("=" * 78)
    F, S, U = [], [], []
    for f in range(4):
        d = np.load(EMB / f"emb_{a.view}_{f}.npz", allow_pickle=True)
        F.append(d["feats"]); S += [str(x) for x in d["sids"]]; U.append(d["users"])
    F = np.concatenate(F); U = np.concatenate(U); S = np.array(S)
    blk = blocks_of(S, tr_info, a.gap)
    bid = np.array([blk[s] for s in S])
    print(f"train: {len(S)} clips, {len(set(bid))} blocks at gap>{a.gap:.0f}s")

    # raw block means, and class-residual block means (remove what the block was DOING)
    cls = np.concatenate([np.load(EMB / f"emb_{a.view}_{f}.npz")["logits"].argmax(1)
                          for f in range(4)])
    cmean = np.stack([F[cls == c].mean(0) if (cls == c).any() else F.mean(0) for c in range(40)])
    R = F - cmean[cls]

    for name, M in (("raw", F), ("class-residual", R)):
        bm = np.stack([M[bid == b].mean(0) for b in sorted(set(bid))])
        bu = [collections.Counter(U[bid == b]).most_common(1)[0][0] for b in sorted(set(bid))]
        bn = np.array([(bid == b).sum() for b in sorted(set(bid))])
        users = sorted(set(U))
        cent = np.stack([M[U == u].mean(0) for u in users])
        pred = [users[i] for i in np.linalg.norm(bm[:, None] - cent[None], axis=-1).argmin(1)]
        nn = np.average(np.array(pred) == np.array(bu), weights=bn)
        from scipy.cluster.hierarchy import linkage, fcluster
        lab = fcluster(linkage(bm, "ward"), len(users), "maxclust")
        df = collections.defaultdict(lambda: collections.Counter())
        for l, u, n in zip(lab, bu, bn):
            df[l][u] += n
        pur = sum(c.most_common(1)[0][1] for c in df.values()) / bn.sum()
        print(f"  {name:15s} block->known-user nearest-centroid {nn:.3f} | "
              f"unsupervised 18-cluster clip-weighted purity {pur:.3f}")

    print(f"  [reference] the timestamp block ALONE is {0.944:.3f} pure with median "
          f"{int(np.median([ (bid==b).sum() for b in set(bid)]))} clips")

    # ---------------------------------------------------------------- 2a
    print()
    print("=" * 78)
    print("2a — subject mean-centering of the feature, then re-apply the head")
    print("=" * 78)
    print("fold   base    ORACLE(true user)   BLOCK(300s)   n")
    rows = []
    for f in range(4):
        d = np.load(EMB / f"emb_{a.view}_{f}.npz", allow_pickle=True)
        Ff, y = d["feats"].astype(np.float64), d["labels"]
        uu = d["users"]; ss = np.array([str(x) for x in d["sids"]])
        W, b = head_of(f"{tag}_f{f}")
        mu_tr = np.concatenate([np.load(EMB / f"emb_{a.view}_{g}.npz")["feats"]
                                for g in range(4) if g != f]).astype(np.float64).mean(0)
        base = ((Ff @ W.T + b).argmax(1) == y).mean()

        def centered(groups):
            G = np.array(groups)
            out = Ff.copy()
            for g in set(G):
                m = G == g
                if m.sum() >= 5:
                    out[m] = Ff[m] - Ff[m].mean(0) + mu_tr
            return ((out @ W.T + b).argmax(1) == y).mean()

        orc = centered(uu)
        bl = blocks_of(ss, tr_info, a.gap)
        blkacc = centered([bl[s] for s in ss])
        rows.append((base, orc, blkacc))
        print(f"  {f}   {base:.5f}    {orc:.5f} ({100*(orc-base):+.2f})   "
              f"{blkacc:.5f} ({100*(blkacc-base):+.2f})   {len(y)}")
    r = np.array(rows) * 100
    do, db = r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]
    print(f"\n  mean delta  ORACLE {do.mean():+.2f} (sd {do.std(ddof=1):.2f})   "
          f"BLOCK {db.mean():+.2f} (sd {db.std(ddof=1):.2f})")
    print(f"  adoption bar = +1.64 (2 SE over 4 paired folds, sigma=1.16 EXP-120a)")
    for nm, v in (("ORACLE", do), ("BLOCK", db)):
        print(f"  {nm:7s} -> {'PASS' if v.mean() > 1.64 else 'FAIL'} "
              f"({sum(v > 0)}/4 folds positive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
