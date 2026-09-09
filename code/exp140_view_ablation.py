#!/usr/bin/env python3
"""EXP-140 -- thermal is IN (172/201). Which view does it displace, at zero new bytes?

EXP-139's thermal member is adopted: sub_r2th20 scored 0.85572 = 172/201 against the
champion's 167, +5 clips, above the pre-registered centre. That settles accuracy. It
opens the question B-033 says a score can never answer: SIZE.

    person 34.28 + wrist 34.28 + thermal 34.28 + skeleton 22.80 + trees 7.63 = 133.3 MB

against a hard 100 MB cap (R-6). Thermal cannot simply be added. Three ways out, and
only measurement chooses between them:

  keep4    person + wrist + thermal + skel + imu   -- what scored 172, UNSHIPPABLE
  swapw    person + thermal + skel + imu           -- thermal REPLACES wrist. Exactly
                                                      byte-neutral, and the wrist view
                                                      is the cheapest thing on the
                                                      shelf: EXP-109 added it to a
                                                      4-fold slot and measured +0.
  prunesk  person + wrist + thermal + skel(3 arch) + imu -- keeps every member, pays
                                                      for it out of the 48-member
                                                      skeleton stack.

This writes each variant's fused OOF so the shipped decoder can score it, and reports
rescued/broken against the champion rather than net (repo convention).
"""
from __future__ import annotations
import argparse, csv, itertools, json, os, subprocess, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
EPS = 1e-12
L = lambda a: np.log(np.maximum(a, EPS))

# (tag, weight, prior_space); "train" members are divided by the train prior first.
CHAMPION = [
    ("k224_mvit_pooled",      0.2925 / 2, "uniform"),
    ("k224_mvitwrist_pooled", 0.2925 / 2, "uniform"),
    ("astgcn_world25",        0.35,       "train"),
    ("imu_stats_t200_d12",    0.3575,     "uniform"),
]
THERMAL = ("th224_pooled", "uniform")
PRIOR_W = 0.25


def load(tag):
    d = np.load(os.path.join(ART, f"oof_{tag}.npz"), allow_pickle=True)
    return {str(s): np.asarray(p, np.float64) for s, p in zip(d["sids"], d["probs"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thermal-weights", type=float, nargs="+",
                    default=[0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    ap.add_argument("--variants", nargs="+", default=["keep4", "swapw"])
    ap.add_argument("--decode", action="store_true", help="also run the shipped decoder")
    args = ap.parse_args()

    meta = {}
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        for r in csv.DictReader(h):
            meta[r["sample_id"]] = (int(r["class_id"]), r["user"])
    counts = np.bincount([v[0] for v in meta.values()], minlength=40).astype(np.float64)
    prior = counts / counts.sum()

    tags = [t for t, _, _ in CHAMPION] + [THERMAL[0]]
    mem = {t: load(t) for t in tags}
    sids = sorted(set.intersection(*[set(mem[t]) for t in tags]))
    print(f"pooled OOF clips common to all {len(tags)} members: {len(sids)}")
    y = np.array([meta[s][0] for s in sids])

    P = {t: np.stack([mem[t][s] for s in sids]) for t in tags}
    logs = {t: L(P[t] / prior if sp == "train" else P[t])
            for t, _, sp in CHAMPION}
    logs[THERMAL[0]] = L(P[THERMAL[0]])
    base_prior = PRIOR_W * L(prior)[None, :]

    def fuse(members, wth):
        z = base_prior.copy()
        for tag, w in members:
            z = z + w * logs[tag]
        if wth:
            z = z + wth * logs[THERMAL[0]]
        return z

    VARIANTS = {
        # thermal ADDED next to the wrist -- what scored 172 but does not fit
        "keep4": [(t, w) for t, w, _ in CHAMPION],
        # thermal REPLACES the wrist view; person keeps its own half-weight
        "swapw": [(t, w) for t, w, _ in CHAMPION if t != "k224_mvitwrist_pooled"],
    }

    champ_pred = fuse(VARIANTS["keep4"], 0.0).argmax(1)
    print(f"\nchampion (no thermal) pre-decoder top-1: {(champ_pred == y).mean():.5f}"
          f"  correct={int((champ_pred == y).sum())}/{len(y)}\n")

    rows = []
    for name in args.variants:
        for wth in args.thermal_weights:
            z = fuse(VARIANTS[name], wth)
            pred = z.argmax(1)
            ok = pred == y
            resc = int((ok & ~(champ_pred == y)).sum())
            brok = int((~ok & (champ_pred == y)).sum())
            rows.append((name, wth, ok.mean(), int(ok.sum()), resc, brok))
            print(f"{name:8s} w_th={wth:.2f}  top1={ok.mean():.5f}  "
                  f"n={int(ok.sum()):4d}  rescued={resc:3d}  broken={brok:3d}")
            out = os.path.join(ART, f"oof_fused_{name}_th{int(wth*100):02d}.npz")
            np.savez_compressed(out, probs=np.exp(z - z.max(1, keepdims=True)),
                                sids=np.array(sids, dtype="<U15"), labels=y)
    print("\nfused OOF written to research/artifacts/oof_fused_<variant>_th<NN>.npz")
    print("decode each with: python3 code/ordered_transition_decoder.py oof --probs "
          "<file> --transition-weight 0.5 --transition-score conditional --backoff "
          "unigram --distinctness penalty --distinctness-penalty 2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
