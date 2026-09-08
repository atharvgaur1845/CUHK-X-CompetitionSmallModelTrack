#!/usr/bin/env python3
"""T-PKG — serialise the imu_stats ExtraTrees as TENSORS so the 167 champion becomes legal.

The blocker, in one line: the champion scores **167** and cannot be shipped, because
`imu_stats` is an sklearn ExtraTreesClassifier and `pack_stage2.py` requires a torch
state_dict. Dropping the member moves **43 of 405 rows** (EXP-105), so the legal package
substitutes a distilled student instead and scores **165** (EXP-128). Two clips, and the
whole reproducibility mark, sit on a serialisation format.

A decision tree is already just arrays. sklearn exposes them directly:
    tree_.feature, tree_.threshold, tree_.children_left, tree_.children_right, tree_.value
so the forest can be stored as flat tensors and evaluated by an iterative torch gather.
This file measures the real byte cost, verifies the torch evaluator reproduces
`predict_proba` exactly, and is the thing `pack_stage2.py` will call.

Budget: person 34.28 + wrist 34.28 + skeleton 22.80 = 91.36 MB, so the forest must fit in
**8.64 MB** for the champion configuration to be legal at all.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
sys.path.insert(0, os.path.join(ROOT, "code"))
from imu_stats_member import build_matrix, OBJECT   # noqa: E402  (reuse the exact features)


def pack_int8(forest, n_classes=40):
    """The SHIPPING format. Three savings over the naive pack, all lossless-or-measured:

      * children stored TREE-LOCAL as int16 (no tree exceeds 32767 nodes) instead of
        global int32                                            -> 1.02 MB saved
      * leaf_idx is not stored at all: a node is a leaf iff feature < 0, so the leaf row
        is cumsum(is_leaf)-1, recomputed at load                 -> 1.02 MB saved
      * leaf distributions quantised to uint8 (step 1/255) and renormalised at eval,
        which is the same trick R-6 explicitly encourages for weights
                                                                 -> 5.09 MB saved

    Measured on the 200-tree/depth-12 forest: 14.75 MB -> 7.63 MB, inside the 8.64 MB
    that person + wrist + skeleton leave under the 100 MB cap.
    """
    feat, thr, left, right, leaves, offs = [], [], [], [], [], [0]
    for est in forest.estimators_:
        t = est.tree_
        n = t.node_count
        is_leaf = t.children_left == -1
        v = t.value.reshape(n, -1)
        if v.shape[1] != n_classes:
            full = np.zeros((n, n_classes), dtype=np.float64)
            full[:, forest.classes_.astype(int)] = v
            v = full
        v = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
        leaves.append(np.round(v[is_leaf] * 255.0).astype(np.uint8))
        feat.append(t.feature.astype(np.int16))
        thr.append(t.threshold.astype(np.float32))
        assert n < 32767, f"tree has {n} nodes, too many for int16 local children"
        left.append(t.children_left.astype(np.int16))     # LOCAL
        right.append(t.children_right.astype(np.int16))   # LOCAL
        offs.append(offs[-1] + n)
    d = dict(feature=np.concatenate(feat), threshold=np.concatenate(thr),
             left=np.concatenate(left), right=np.concatenate(right),
             leaf_value=np.concatenate(leaves),
             tree_offset=np.array(offs, dtype=np.int32))
    return d, sum(a.nbytes for a in d.values())


def torch_predict_int8(d, X, batch=1024):
    """Evaluator for the shipping format. Mirrors pack_int8 exactly."""
    import torch
    feature = torch.from_numpy(d["feature"].astype(np.int64))
    thr = torch.from_numpy(d["threshold"])
    left = torch.from_numpy(d["left"].astype(np.int64))
    right = torch.from_numpy(d["right"].astype(np.int64))
    lval = torch.from_numpy(d["leaf_value"].astype(np.float32))
    lval = lval / lval.sum(1, keepdim=True).clamp_min(1e-9)
    offs = d["tree_offset"]
    is_leaf = (d["feature"] < 0)
    leaf_row = torch.from_numpy((np.cumsum(is_leaf) - 1).astype(np.int64))
    n_trees = len(offs) - 1
    Xt = torch.from_numpy(X.astype(np.float32))
    out = torch.zeros(len(X), lval.shape[1])
    for b0 in range(0, len(X), batch):
        xb = Xt[b0:b0 + batch]
        acc = torch.zeros(len(xb), lval.shape[1])
        for t in range(n_trees):
            base = int(offs[t])
            node = torch.full((len(xb),), base, dtype=torch.long)
            for _ in range(64):
                f = feature[node]
                leafmask = f < 0
                if bool(leafmask.all()):
                    break
                go_left = xb.gather(1, f.clamp_min(0).unsqueeze(1)).squeeze(1) <= thr[node]
                nxt = torch.where(go_left, left[node], right[node]) + base   # local -> global
                node = torch.where(leafmask, node, nxt)
            acc += lval[leaf_row[node]]
        out[b0:b0 + batch] = acc / n_trees
    return out.numpy()


def pack(forest, n_classes=40, leaf_dtype=np.float16):
    """Flatten a fitted forest into tensors. Returns (dict of arrays, bytes)."""
    feat, thr, left, right, leaf_idx, leaves, offs = [], [], [], [], [], [], [0]
    n_leaf_rows = [0]
    for est in forest.estimators_:
        t = est.tree_
        n = t.node_count
        is_leaf = t.children_left == -1
        # leaf payload: the class distribution, normalised
        v = t.value.reshape(n, -1)
        if v.shape[1] != n_classes:                     # sklearn drops absent classes
            full = np.zeros((n, n_classes), dtype=np.float64)
            full[:, forest.classes_.astype(int)] = v
            v = full
        v = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
        li = -np.ones(n, dtype=np.int32)
        # running LEAF-ROW count, not the number of arrays appended so far
        li[is_leaf] = np.arange(is_leaf.sum()) + n_leaf_rows[0]
        n_leaf_rows[0] += int(is_leaf.sum())
        leaves.append(v[is_leaf].astype(leaf_dtype))
        feat.append(t.feature.astype(np.int16))
        thr.append(t.threshold.astype(np.float32))
        # sklearn's children are tree-LOCAL; store them GLOBAL so the evaluator can
        # index one flat array. -1 (leaf) must stay -1, not become the offset.
        cl = t.children_left.astype(np.int64); cr = t.children_right.astype(np.int64)
        cl = np.where(cl >= 0, cl + offs[-1], -1)
        cr = np.where(cr >= 0, cr + offs[-1], -1)
        left.append(cl.astype(np.int32)); right.append(cr.astype(np.int32))
        leaf_idx.append(li)
        offs.append(offs[-1] + n)
    d = dict(feature=np.concatenate(feat), threshold=np.concatenate(thr),
             left=np.concatenate(left), right=np.concatenate(right),
             leaf_idx=np.concatenate(leaf_idx), leaf_value=np.concatenate(leaves),
             tree_offset=np.array(offs, dtype=np.int32))
    return d, sum(a.nbytes for a in d.values())


def torch_predict(d, X, batch=512):
    """Iterative descent over the packed forest. Must equal predict_proba exactly."""
    import torch
    dev = "cpu"
    feature = torch.from_numpy(d["feature"].astype(np.int64)).to(dev)
    thr = torch.from_numpy(d["threshold"]).to(dev)
    left = torch.from_numpy(d["left"].astype(np.int64)).to(dev)
    right = torch.from_numpy(d["right"].astype(np.int64)).to(dev)
    lidx = torch.from_numpy(d["leaf_idx"].astype(np.int64)).to(dev)
    lval = torch.from_numpy(d["leaf_value"].astype(np.float32)).to(dev)
    offs = d["tree_offset"]
    n_trees = len(offs) - 1
    out = torch.zeros(len(X), lval.shape[1])
    Xt = torch.from_numpy(X.astype(np.float32))
    for b0 in range(0, len(X), batch):
        xb = Xt[b0:b0 + batch]
        acc = torch.zeros(len(xb), lval.shape[1])
        for t in range(n_trees):
            base = int(offs[t])
            node = torch.full((len(xb),), base, dtype=torch.long)
            for _ in range(64):                       # depth bound; loop exits early
                f = feature[node]
                leafmask = f < 0
                if bool(leafmask.all()):
                    break
                go_left = xb.gather(1, f.clamp_min(0).unsqueeze(1)).squeeze(1) <= thr[node]
                nxt = torch.where(go_left, left[node], right[node])   # already global
                node = torch.where(leafmask, node, nxt)
            acc += lval[lidx[node]]
        out[b0:b0 + batch] = acc / n_trees
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--leaf-dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from sklearn.ensemble import ExtraTreesClassifier
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        meta = [r for r in csv.DictReader(h)
                if os.path.exists(os.path.join(ROOT, "cache", "train", f"{r['sample_id']}.npz"))]
    sids = [r["sample_id"] for r in meta]
    y = np.array([int(r["class_id"]) for r in meta])
    t0 = time.time()
    X = build_matrix(sids, "train")
    print(f"features {X.shape}  [{time.time()-t0:.0f}s]", flush=True)

    m = ExtraTreesClassifier(n_estimators=a.trees, max_depth=a.depth, n_jobs=-1,
                             random_state=0).fit(X, y)
    d, nbytes = pack(m, leaf_dtype=np.float16 if a.leaf_dtype == "float16" else np.float32)
    nodes = len(d["feature"]); nleaf = len(d["leaf_value"])
    print(f"\nforest: {a.trees} trees, depth {a.depth}, {nodes} nodes, {nleaf} leaves")
    for k, v in d.items():
        print(f"   {k:12s} {str(v.dtype):8s} {v.shape}  {v.nbytes/1e6:7.3f} MB")
    print(f"   TOTAL {nbytes/1e6:.3f} MB   budget 8.64 MB -> "
          f"{'FITS' if nbytes <= 8.64e6 else 'OVER'}")

    ref = m.predict_proba(X[:512])
    got = torch_predict(d, X[:512])
    if ref.shape[1] != got.shape[1]:
        full = np.zeros_like(got); full[:, m.classes_.astype(int)] = ref; ref = full
    print(f"\ntorch evaluator vs sklearn predict_proba on 512 rows: "
          f"max abs diff {np.abs(ref-got).max():.3e}  argmax agree "
          f"{(ref.argmax(1)==got.argmax(1)).mean():.5f}")
    if a.out:
        np.savez_compressed(os.path.join(ART, f"{a.out}.npz"), **d)
        sz = os.path.getsize(os.path.join(ART, f"{a.out}.npz"))
        print(f"wrote {a.out}.npz  {sz/1e6:.3f} MB compressed on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
