#!/usr/bin/env python3
"""EXP-132 — is the rank-2 error a per-CLIP ambiguity or a per-SUBJECT bias?

Recomputes the champion fusion on the 2,700 pooled OOF clips from artifacts already on
disk (no training), then measures whether the errors repeat *within* a subject.

Why it matters: EXP-131 showed the rank-1-vs-rank-2 decision is not recoverable from the
five members' posteriors (0.8759 vs a 0.8785 base rate). That closes per-clip arbitration.
It does NOT close arbitration conditioned on the subject's OTHER clips -- and if a
subject's errors are the same (true -> pred) confusion over and over, then the missing
variable is a subject-constant offset, which no per-clip model can see by construction.
"""
from __future__ import annotations
import collections, csv, json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
EPS = 1e-12
L = lambda a: np.log(np.maximum(a, EPS))

MEMBERS = {  # tag -> (weight, prior_space)  "train" members get divided by the prior
    "k224_mvit_pooled":      (0.2925 / 2, "uniform"),
    "k224_mvitwrist_pooled": (0.2925 / 2, "uniform"),
    "astgcn_world25":        (0.35,       "train"),
    "imu_stats_t200_d12":    (0.3575,     "uniform"),
}


def main() -> int:
    meta = {}
    with open(os.path.join(ROOT, "cache", "meta_train.csv")) as h:
        for r in csv.DictReader(h):
            meta[r["sample_id"]] = (int(r["class_id"]), r["user"], r["station"])
    counts = collections.Counter(v[0] for v in meta.values())
    prior = np.array([counts[i] for i in range(40)], dtype=np.float64)
    prior /= prior.sum()

    mem = {}
    for tag in MEMBERS:
        d = np.load(os.path.join(ART, f"oof_{tag}.npz"), allow_pickle=True)
        mem[tag] = {str(s): np.asarray(p, np.float64) for s, p in zip(d["sids"], d["probs"])}
    sids = sorted(set.intersection(*[set(v) for v in mem.values()]))
    print(f"pooled OOF clips: {len(sids)}")

    z = np.zeros((len(sids), 40))
    for tag, (w, space) in MEMBERS.items():
        P = np.stack([mem[tag][s] for s in sids])
        z += w * L(P / prior if space == "train" else P)
    z += 0.25 * L(prior)[None, :]

    y = np.array([meta[s][0] for s in sids])
    u = np.array([meta[s][1] for s in sids])
    order = np.argsort(-z, axis=1)
    top1, top2 = order[:, 0], order[:, 1]
    err = top1 != y
    r2 = err & (top2 == y)
    margin = z[np.arange(len(z)), top1] - z[np.arange(len(z)), top2]

    print(f"top-1 {1 - err.mean():.5f}   top-2 {((~err) | r2).mean():.5f}   "
          f"errors {err.sum()}   rank-2 {r2.sum()}")

    # (1) do a subject's errors repeat the SAME confusion?
    cells = collections.Counter(zip(u[err], y[err], top1[err]))
    big = {k: n for k, n in cells.items() if n >= 3}
    print(f"\n(subject,true,pred) error cells: {len(cells)}   with n>=3: {len(big)}   "
          f"errors inside them: {sum(big.values())}/{err.sum()} "
          f"({sum(big.values()) / err.sum():.3f})")

    per_user = collections.defaultdict(lambda: [0, 0])
    for (usr, a, b), n in cells.items():
        per_user[usr][0] += n if n >= 2 else 0
        per_user[usr][1] += n
    frac = {k: v[0] / v[1] for k, v in sorted(per_user.items())}
    print("share of each subject's errors inside a repeated (true,pred) cell:")
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(frac.items(), key=lambda x: -x[1])))

    # (2) are those confusions DIRECTIONAL within the subject?
    pair = collections.Counter()
    for usr, a, b in zip(u[err], y[err], top1[err]):
        pair[(usr, min(a, b), max(a, b), a < b)] += 1
    seen, one_dir, tot = set(), 0, 0
    for usr, a, b, _ in pair:
        if (usr, a, b) in seen:
            continue
        seen.add((usr, a, b))
        n1, n2 = pair[(usr, a, b, True)], pair[(usr, a, b, False)]
        if n1 + n2 >= 3:
            tot += 1
            one_dir += int(min(n1, n2) == 0)
    print(f"\n(subject,class-pair) cells with >=3 errors: {tot}   "
          f"strictly one-directional: {one_dir}")

    # (3) is a within-subject REFERENCE available for the rank-2 decision?
    have_true = have_wrong = 0
    for i in np.where(r2)[0]:
        same = (u == u[i]) & (top1 == y)
        have_true += int((same & (y == y[i]) & (np.arange(len(y)) != i)).any())
        have_wrong += int((same & (y == top1[i])).any())
    print(f"rank-2 errors with a correctly-predicted same-class clip from the SAME "
          f"subject: {have_true}/{r2.sum()}")
    print(f"...and one of the WRONG (rank-1) class from the same subject: "
          f"{have_wrong}/{r2.sum()}")

    print(f"\nmedian top-2 margin: correct {np.median(margin[~err]):.3f}  "
          f"rank-2 error {np.median(margin[r2]):.3f}  "
          f"other error {np.median(margin[err & ~r2]):.3f}")

    acc = {usr: 1 - err[u == usr].mean() for usr in sorted(set(u))}
    print("\nper-subject accuracy (the quantity private/on-site actually sample):")
    for k, v in sorted(acc.items(), key=lambda x: x[1]):
        print(f"  {k:8s} n={(u == k).sum():4d}  acc={v:.4f}  errors={err[u == k].sum():3d}  "
              f"rank2={r2[u == k].sum():3d}")
    a = np.array(list(acc.values()))
    print(f"  between-subject sd {a.std(ddof=1) * 100:.2f} points   "
          f"worst-4 mean {np.sort(a)[:4].mean():.4f}   mean {a.mean():.4f}")

    out = os.path.join(ART, "exp132_champion_oof.npz")
    np.savez_compressed(out, z=z.astype(np.float32), y=y, users=u,
                        sids=np.array(sids, dtype="<U24"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
