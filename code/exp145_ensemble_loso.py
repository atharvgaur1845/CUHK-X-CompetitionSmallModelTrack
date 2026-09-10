#!/usr/bin/env python3
"""EXP-145 -- the ENSEMBLE per-subject table under leave-one-subject-out, and the
final-selection statistic.

Why not the 4-fold OOF. A 4-fold model trains on 13-14 subjects and holds out 4. The
deployed model trains on all 18. LOSO trains on 17, which is both closer to deployment and
gives 18 per-subject numbers instead of 4 -- and Kaggle private (8 unseen subjects) and the
on-site stage (8 more, 30% of the grade) are 8-subject draws from a distribution whose sd
EXP-141 measured at 6.89 points. The WORST subjects decide those draws, so the statistic is
the lower quartile, never the mean and never public.

One honest gap, stated rather than hidden. Video views (person, wrist, thermal) and the IMU
member have true LOSO. The SKELETON does not: only its 4-fold OOF exists, so for subject X
it comes from a model trained on 13-14 subjects rather than 17. That is leak-free -- no
model ever saw its own held-out subject -- but it understates the skeleton, which makes
every number here mildly CONSERVATIVE rather than optimistic. Fixing it needs a skeleton
LOSO run, which is the one piece of this table still owed.
"""
from __future__ import annotations
import csv, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "research", "artifacts")
L = lambda a: np.log(np.maximum(a, 1e-12))
USERS = ["user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9",
         "user16", "user17", "user18", "user19", "user20", "user21", "user22", "user23",
         "user24"]


def loso_probs(prefix):
    """sid -> posterior, from the 18 per-subject holdout files."""
    out = {}
    for u in USERS:
        p = os.path.join(ART, f"oof_{prefix}_{u}.npz")
        if not os.path.isfile(p):
            return None
        d = np.load(p, allow_pickle=True)
        for s, v in zip(d["sids"], np.asarray(d["probs"], np.float64)):
            out[str(s)] = v
    return out


def pooled_probs(tag):
    d = np.load(os.path.join(ART, f"oof_{tag}.npz"), allow_pickle=True)
    return {str(s): np.asarray(v, np.float64) for s, v in zip(d["sids"], d["probs"])}


def main() -> int:
    meta = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "cache", "meta_train.csv"))):
        meta[r["sample_id"]] = (int(r["class_id"]), r["user"])
    counts = np.bincount([v[0] for v in meta.values()], minlength=40).astype(float)
    prior = counts / counts.sum()

    # (name, source, weight, prior space).  Weights are the SHIPPED ones (EXP-144).
    MEMBERS = [
        ("person",  loso_probs("loso"),        0.14625, "uniform"),
        ("wrist",   loso_probs("losowrist"),   0.14625, "uniform"),
        ("thermal", loso_probs("losoth"),      0.20,    "uniform"),
        ("imu",     loso_probs("loso_imu_stats") or pooled_probs("loso_imu_stats"),
                                                0.3575,  "uniform"),
        ("skel",    pooled_probs("astgcn_world25"), 0.35, "train"),
    ]
    for n, m, _, _ in MEMBERS:
        print(f"  {n:8s} {'MISSING' if m is None else f'{len(m)} clips'}"
              f"{'   <- 4-fold OOF, not LOSO' if n == 'skel' else ''}")
    if any(m is None for _, m, _, _ in MEMBERS):
        raise SystemExit("a member is missing; cannot build the table")

    sids = sorted(set.intersection(*[set(m) for _, m, _, _ in MEMBERS]))
    y = np.array([meta[s][0] for s in sids])
    u = np.array([meta[s][1] for s in sids])
    print(f"\nclips common to all members: {len(sids)}")

    z = 0.25 * L(prior)[None, :]
    for name, m, w, space in MEMBERS:
        P = np.stack([m[s] for s in sids])
        z = z + w * L(P / prior if space == "train" else P)
    ok = z.argmax(1) == y

    rows = sorted(((u == x).sum(), ok[u == x].mean(), x) for x in sorted(set(u)))
    rows = sorted(rows, key=lambda r: r[1])
    print(f"\n{'subject':>9s} {'clips':>6s} {'acc':>8s}")
    for n, a, x in rows:
        print(f"{x:>9s} {n:6d} {a:8.5f}  {'#' * int(round(a * 40))}")
    acc = np.array([r[1] for r in rows])
    q = int(np.ceil(len(acc) / 4))
    print(f"\npooled           {ok.mean():.5f}  ({int(ok.sum())}/{len(ok)})")
    print(f"subject mean     {acc.mean():.5f}")
    print(f"between-subj sd  {acc.std(ddof=1):.5f}  ({len(acc)-1} df)")
    print(f"8-subject draw SE {acc.std(ddof=1)/np.sqrt(8):.5f} "
          f"({100*acc.std(ddof=1)/np.sqrt(8):.2f} points)")
    print(f"\nLOWER QUARTILE (worst {q})  {acc[:q].mean():.5f}   <- SELECTION STATISTIC")
    print(f"upper quartile  (best {q})  {acc[-q:].mean():.5f}")
    print(f"spread                     {acc[-1]-acc[0]:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
