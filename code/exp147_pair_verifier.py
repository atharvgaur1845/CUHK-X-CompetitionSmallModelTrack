#!/usr/bin/env python3
"""EXP-147 / L4 -- the pair verifier: decide the fused top-2 from TRUNK FEATURES, not
from posteriors.

The plan's fallback, and its own decision rule says to build it: L1 (privileged teacher)
failed its gate at 0.419 against 0.712, and L2's P0 (subject recovery) failed at 0.339
purity, so the branch that was conditioned on "if T1 fails and L2 fails P0" is live.

What makes it not-already-refuted. EXP-131 showed the rank-1-vs-rank-2 decision is not
recoverable from the five members' POSTERIORS (0.8759 against a 0.8785 base rate). A
verifier over trunk features is a different question only to the extent that the 768-d
pooled feature carries something the 40-d posterior threw away. That is a real gap -- the
head is a rank-40 projection of a rank-768 space -- but it is NOT the "raw input" the
brief asked for: a genuinely raw verifier would fine-tune the trunk, and EXP-115 measured
that 2,281 clips cannot fine-tune 86M parameters. **This is L4-lite, and the honest
statement of what it tests is: does the discarded 728 dimensions of trunk feature contain
the pair decision?**

Design. Antisymmetry is structural, not a penalty: we learn a score s(x, c) over
(clip, candidate class) and take argmax over the two candidates, so g(x,a,b) = s(x,a) -
s(x,b) = -g(x,b,a) exactly. Two scorers:

  proto   hand-built similarity features to each class prototype + logistic regression
  metric  low-rank bilinear s(x,c) = <Pf(x), Q mu_c>, trained with a pairwise logistic
          loss on (correct, incorrect) candidate pairs

Everything is subject-grouped: prototypes are rebuilt from the training subjects of each
split, so a test subject's own clips never define the class centres it is scored against.

Pre-registered gate (from the plan): accuracy on the decidable top-2 clips > 0.893, i.e.
the 0.8785 base rate plus 2 SE. Below that, L4 is dead and modelling is over.
"""
from __future__ import annotations
import csv, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
EMB = os.path.join(ART, "emb")
L = lambda a: np.log(np.maximum(a, 1e-12))
SHIP = {"person": 0.14625, "wrist": 0.14625, "thermal": 0.20, "skel": 0.35, "imu": 0.3575}
SRC = {"person": "k224_mvit_pooled", "wrist": "k224_mvitwrist_pooled",
       "thermal": "th224_pooled", "skel": "astgcn_world25", "imu": "imu_stats_t200_d12"}


def load_oof(tag):
    d = np.load(os.path.join(ART, f"oof_{tag}.npz"), allow_pickle=True)
    return {str(s): np.asarray(v, np.float64) for s, v in zip(d["sids"], d["probs"])}


def load_emb(view):
    F, S = [], []
    for f in range(4):
        d = np.load(os.path.join(EMB, f"emb_{view}_{f}.npz"), allow_pickle=True)
        F.append(np.asarray(d["feats"], np.float32)); S += [str(x) for x in d["sids"]]
    return dict(zip(S, np.concatenate(F)))


def main() -> int:
    meta = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "cache", "meta_train.csv"))):
        meta[r["sample_id"]] = (int(r["class_id"]), r["user"])
    counts = np.bincount([v[0] for v in meta.values()], minlength=40).astype(float)
    prior = counts / counts.sum()

    mem = {k: load_oof(v) for k, v in SRC.items()}
    emb = {v: load_emb(v) for v in ("person", "wrist")}
    sids = sorted(set.intersection(*[set(m) for m in mem.values()],
                                   *[set(e) for e in emb.values()]))
    y = np.array([meta[s][0] for s in sids])
    u = np.array([meta[s][1] for s in sids])
    z = 0.25 * L(prior)[None, :]
    for k, w in SHIP.items():
        P = np.stack([mem[k][s] for s in sids])
        z = z + w * L(P / prior if k == "skel" else P)
    order = np.argsort(-z, axis=1)
    a, b = order[:, 0], order[:, 1]

    dec = (y == a) | (y == b)                 # the decidable set: truth is in the top-2
    base = (a[dec] == y[dec]).mean()
    print(f"clips {len(sids)}   decidable (truth in top-2) {int(dec.sum())}   "
          f"base rate (keep rank-1) {base:.5f}")
    print(f"pre-registered gate: > 0.893\n")

    F = np.concatenate([np.stack([emb[v][s] for s in sids]) for v in ("person", "wrist")], 1)
    F = F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-9)
    users = sorted(set(u)); fold = {x: i % 4 for i, x in enumerate(users)}
    fu = np.array([fold[x] for x in u])

    from sklearn.linear_model import LogisticRegression
    pred_proto = np.zeros(len(sids), dtype=int)
    pred_metric = np.zeros(len(sids), dtype=int)

    for f in range(4):
        tr, te = (fu != f), (fu == f)
        mu = np.zeros((40, F.shape[1]), np.float32)
        for c in range(40):
            m = tr & (y == c)
            if m.any():
                mu[c] = F[m].mean(0)
        mu = mu / np.linalg.norm(mu, axis=1, keepdims=True).clip(1e-9)

        def feats(idx, cand):
            """similarity of each clip to ONE candidate class prototype"""
            g = F[idx]; p = mu[cand]
            cos = (g * p).sum(1)
            d2 = ((g - p) ** 2).sum(1)
            return np.stack([cos, -d2, cos ** 2], 1)

        # --- proto: logistic on the DIFFERENCE of candidate features ---------------
        itr = np.flatnonzero(tr & dec)
        Xtr = feats(itr, a[itr]) - feats(itr, b[itr])
        ytr = (y[itr] == a[itr]).astype(int)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        ite = np.flatnonzero(te)
        Xte = feats(ite, a[ite]) - feats(ite, b[ite])
        keep = clf.predict(Xte).astype(bool)
        pred_proto[ite] = np.where(keep, a[ite], b[ite])

        # --- metric: low-rank bilinear, pairwise logistic loss ---------------------
        import torch
        dev = "cpu"; rank = 64
        Pm = torch.nn.Linear(F.shape[1], rank, bias=False)
        Qm = torch.nn.Linear(F.shape[1], rank, bias=False)
        opt = torch.optim.AdamW(list(Pm.parameters()) + list(Qm.parameters()), 1e-3,
                                weight_decay=1e-2)
        Ft = torch.from_numpy(F); Mt = torch.from_numpy(mu)
        ia = torch.from_numpy(a[itr]); ib = torch.from_numpy(b[itr])
        gi = torch.from_numpy(itr); lab = torch.from_numpy(ytr).float()
        for ep in range(300):
            g = Pm(Ft[gi]); sa = (g * Qm(Mt[ia])).sum(1); sb = (g * Qm(Mt[ib])).sum(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(sa - sb, lab)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            gt = Pm(Ft[torch.from_numpy(ite)])
            sa = (gt * Qm(Mt[torch.from_numpy(a[ite])])).sum(1)
            sb = (gt * Qm(Mt[torch.from_numpy(b[ite])])).sum(1)
            k2 = (sa > sb).numpy()
        pred_metric[ite] = np.where(k2, a[ite], b[ite])
        print(f"  fold {f}: proto {(pred_proto[ite][dec[ite]]==y[ite][dec[ite]]).mean():.5f}  "
              f"metric {(pred_metric[ite][dec[ite]]==y[ite][dec[ite]]).mean():.5f}  "
              f"(n={int(dec[ite].sum())}, loss {loss.item():.4f})")

    print()
    for name, pr in (("proto ", pred_proto), ("metric", pred_metric)):
        acc = (pr[dec] == y[dec]).mean()
        flips = int((pr[dec] != a[dec]).sum())
        won = int(((pr[dec] == y[dec]) & (a[dec] != y[dec])).sum())
        lost = int(((pr[dec] != y[dec]) & (a[dec] == y[dec])).sum())
        print(f"{name} on decidable: {acc:.5f}  (base {base:.5f}, {acc-base:+.5f})  "
              f"flips {flips}, of which right {won} / wrong {lost}   "
              f"-> {'PASS' if acc > 0.893 else 'FAIL'} vs gate 0.893")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
