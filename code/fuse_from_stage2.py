#!/usr/bin/env python3
"""Fuse the branch outputs of a Stage-2 package using ONLY what its manifest declares.

Reads `fusion.weights` from the manifest, the per-branch `testprobs_pkg_<role>.npz` files
that `unpack_stage2.py --check infer` and `infer_stage2_skel_imu.py` wrote, and the train
class prior from cache/meta_train.csv (shipped in inference_assets/). Skeleton members are
plain-CE (train-prior posteriors) and are divided by the prior; every other branch is
uniform-prior. Nothing here is a tunable; if the number is not in the manifest it is not
used.
"""
from __future__ import annotations
import argparse, csv, json, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
L = lambda a: np.log(np.maximum(a, 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True)
    ap.add_argument("--artifacts", required=True, help="dir holding testprobs_pkg_*.npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.loads(zipfile.ZipFile(a.package).read("manifest.json"))
    W = man["fusion"]["weights"]
    counts = np.zeros(40)
    for r in csv.DictReader(open(ROOT / "cache" / "meta_train.csv")):
        counts[int(r["class_id"])] += 1
    prior = counts / counts.sum()
    art = Path(a.artifacts)
    sids, z = None, None
    for role, w in W.items():
        if role == "prior":
            continue
        d = np.load(art / f"testprobs_pkg_{role}.npz", allow_pickle=True)
        s = [str(x) for x in d["sids"]]
        if sids is None:
            sids, z = s, np.zeros((len(s), 40))
        assert s == sids, f"sid order differs for {role}"
        P = np.asarray(d["probs"], np.float64)
        z += w * (L(P / prior) if role == "skel" else L(P))
        print(f"  + {role:10s} w={w:.4f}")
    z += float(W.get("prior", 0.0)) * L(prior)[None, :]
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    np.savez_compressed(a.out, probs=p.astype(np.float32), sids=np.array(sids, dtype="<U15"))
    print(f"wrote {a.out}  rows={len(sids)}  formula: {man['fusion']['formula']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
