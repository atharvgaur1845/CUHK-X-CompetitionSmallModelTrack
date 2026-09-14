#!/usr/bin/env python3
"""EXP-160 gate -- does person-448 beat person-384 in the SHIPPED fusion, fold-safe?

Rebuilds the shipped fusion (EXP-158 centroid weights, skeleton divided by the train prior,
IMU Laplace-smoothed) on the 2,700 pooled OOF clips from the member OOF files, once with
the 384 person folds and once with the 448 folds, and reports the pooled count and the
per-fold paired delta with its SE. The harness reconstructs `oof_p384w_fused.npz` to
2109/2700 against the recorded 2106 (99.8% argmax agreement) -- close enough for a PAIRED
read, which is all the gate needs. Then writes both fused files so the decoder's `oof`
mode can give the decoded number (EXP-155: the decoder absorbs most of a member gain).

Adoption bar (CLAUDE.md): mean paired delta > 2 SE on the 4-fold mean.

  python3 code/exp160_gate.py                # needs research/artifacts/oof_k224_mvit448_f{0..3}.npz
  python3 code/ordered_transition_decoder.py oof --probs research/artifacts/oof_p448w_fused.npz \
      --transition-weight 0.5 --transition-score conditional --backoff unigram \
      --distinctness penalty --distinctness-penalty 2.0 --output research/artifacts/oof_p448w_decoded.npz
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "research" / "artifacts"
W = json.loads((ART / "weights_p384.json").read_text()); W["prior"] = 0.25
EPS, SM = 1e-12, 0.001


def load(tag):
    z = np.load(ART / f"oof_{tag}.npz")
    return dict(zip(z["sids"], z["probs"]))


def stack(tags, sids):
    d = {}
    for t in tags:
        d.update(load(t))
    return np.stack([d[s] for s in sids])


def main() -> int:
    ref = np.load(ART / "oof_p384w_fused.npz")
    sids, y = list(ref["sids"]), ref["labels"]
    rows = list(csv.DictReader(open(ROOT / "cache" / "meta_train.csv")))
    cls = np.array([int(r["class_id"]) for r in rows])
    prior = np.bincount(cls, minlength=40) / len(cls)
    wr = stack([f"k224_mvitwrist_f{f}" for f in range(4)], sids)
    th = stack([f"k224_mvit_th_f{f}" for f in range(4)], sids)
    sk = stack(["astgcn_world25"], sids)
    im = (stack(["imu_stats_t200_d12"], sids) + SM) / (1 + 40 * SM)
    base = (W["wrist"] * np.log(wr + EPS) + W["thermal"] * np.log(th + EPS)
            + W["skel"] * np.log(sk / prior + EPS) + W["imu"] * np.log(im)
            + W["prior"] * np.log(prior))

    def fused(person_tag):
        P = stack([f"{person_tag}_f{f}" for f in range(4)], sids)
        L = base + W["person"] * np.log(P + EPS)
        S = np.exp(L - L.max(1, keepdims=True)); S /= S.sum(1, keepdims=True)
        return P, S

    P4, S4 = fused("k224_mvit384")
    P8, S8 = fused("k224_mvit448")
    ok4, ok8 = S4.argmax(1) == y, S8.argmax(1) == y
    m4, m8 = P4.argmax(1) == y, P8.argmax(1) == y
    print(f"member  384 {m4.sum()}/2700 = {m4.mean():.5f}   448 {m8.sum()}/2700 = {m8.mean():.5f}   delta {100*(m8.mean()-m4.mean()):+.2f}")
    print(f"fusion  384 {ok4.sum()}/2700 = {ok4.mean():.5f}   448 {ok8.sum()}/2700 = {ok8.mean():.5f}   delta {ok8.sum()-ok4.sum():+d} clips")
    # per-fold paired deltas from the member files' own fold membership
    d = []
    for f in range(4):
        pos = {s: i for i, s in enumerate(sids)}   # user5/user21 clips are not in the 2,700 pooled set
        z = np.load(ART / f"oof_k224_mvit448_f{f}.npz"); idx = [pos[s] for s in z["sids"] if s in pos]
        a, b = ok4[idx].mean(), ok8[idx].mean(); d.append(100 * (b - a))
        print(f"  fold {f}: n={len(idx)}  fusion 384 {a:.5f}  448 {b:.5f}  delta {100*(b-a):+.2f}   member {100*(m8[idx].mean()-m4[idx].mean()):+.2f}")
    d = np.array(d); se = d.std(ddof=1) / 2
    print(f"paired fusion delta: mean {d.mean():+.2f}  SE {se:.2f}  t {d.mean()/se if se else float('nan'):.2f}   bar: mean > 2 SE -> {'PASS' if d.mean() > 2*se and d.mean() > 0 else 'FAIL'}")
    print(f"rescued {int((ok8 & ~ok4).sum())} / broken {int((~ok8 & ok4).sum())}")
    np.savez_compressed(ART / "oof_p448w_fused.npz", probs=S8, sids=np.array(sids), labels=y)
    np.savez_compressed(ART / "oof_p384w_fused_rebuilt.npz", probs=S4, sids=np.array(sids), labels=y)
    print("wrote oof_p448w_fused.npz and oof_p384w_fused_rebuilt.npz -- decode both with the shipped decoder settings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
