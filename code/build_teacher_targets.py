#!/usr/bin/env python3
"""T3 — build soft teacher targets for distillation from existing OOF artifacts.

Why this needs no teacher inference run: every member's OOF prediction was produced by a
fold model on clips it did NOT train on. That is precisely the honest signal distillation
wants — a teacher that has memorised the clip gives a near-one-hot target and teaches the
student nothing. So the 2,933-clip pooled OOFs we already own ARE the teacher targets.

What the student is being asked to learn is the thing our fusion cannot express. Over the
five members, fused accuracy is 0.763 while oracle-any-member is 0.877 — 309 clips of
2,700. A global geometric weight has to pick one compromise for every clip; a soft target
carries the full posterior of each member on THAT clip, so the student can learn a
per-clip arbitration no scalar weight can represent. Fitted arbitration has failed 4/4 on
public (GBDT stacker, structure decoder, cohort weights, learned gate) -- all of them fit
in probability space on 2,700 rows. This differs in kind: the student fits in FEATURE
space against 2,933 dense 40-dim targets, which is B-028's distinction.

Outputs research/artifacts/teacher_targets.npz with probs (N,40), sids, labels, plus the
per-member stack so ablations do not need a rebuild.
"""
import argparse, collections, csv, os
import numpy as np

ART = "research/artifacts"
EPS = 1e-12
L = lambda a: np.log(np.maximum(a, EPS))

ap = argparse.ArgumentParser()
ap.add_argument("--members", default="k224_mvit_pooled,k224_mvitwrist_pooled,"
                                     "astgcn_world25,imu_stats,pre_thermal_pooled")
ap.add_argument("--weights", default="",
                help="comma list matching --members; default = the champion recipe "
                     "(video .2925 split across video views, skel .35, imu .3575)")
ap.add_argument("--temperature", type=float, default=1.0,
                help="softens the fused target. T>1 spreads mass onto runner-up classes, "
                     "which is where the oracle headroom lives")
ap.add_argument("--target", choices=("fused", "oracle"), default="fused",
                help="fused = the champion geometric mix (accuracy 0.763). "
                     "oracle = average only the members that were RIGHT on that clip, "
                     "falling back to fused where none were. Uses TRAIN labels, which is "
                     "legal for constructing a training target, and lifts the target to "
                     "~0.877 -- the whole point, since 309 of 2,700 clips are winnable by "
                     "some member and no global weight can reach them.")
ap.add_argument("--out", default="teacher_targets")
a = ap.parse_args()

names = [m for m in a.members.split(",") if m]
mem = {}
for n in names:
    d = np.load(f"{ART}/oof_{n}.npz", allow_pickle=True)
    mem[n] = {str(s): (np.asarray(p, np.float64), int(l))
              for s, p, l in zip(d["sids"], d["probs"], d["labels"])}

# Intersect: a clip is usable only where every teacher has an honest prediction.
sids = sorted(set.intersection(*[set(v) for v in mem.values()]))
y = np.array([mem[names[0]][s][1] for s in sids])
P = {n: np.stack([mem[n][s][0] for s in sids]) for n in names}
print(f"teachers: {len(names)}  clips: {len(sids)} (intersection of all members)")

tr = collections.Counter(int(r["class_id"]) for r in csv.DictReader(open("cache/meta_train.csv")))
tot = sum(tr.values())
prior = np.array([tr[k] / tot for k in range(40)])

if a.weights:
    w = {n: float(x) for n, x in zip(names, a.weights.split(","))}
else:
    vid = [n for n in names if n.startswith("k224_")]
    w = {n: 0.2925 / max(len(vid), 1) for n in vid}
    for n in names:
        if n in w:
            continue
        w[n] = 0.35 if "astgcn" in n or "world25" in n else (0.3575 if "imu" in n else 0.0)
    if all(w[n] == 0.0 for n in names if n not in vid):
        raise SystemExit("could not infer weights; pass --weights")
print("  weights: " + ", ".join(f"{n}={w[n]:.4f}" for n in names))

lg = np.zeros((len(sids), 40))
for n in names:
    # world25 emits TRAIN-prior posteriors; the video members are trained with
    # logit-adjusted CE and emit UNIFORM-prior ones. Reconcile before mixing -- getting
    # this backwards scored 0.58706 vs 0.62686 once already (CLAUDE.md).
    base = P[n] / prior if ("astgcn" in n or "world25" in n) else P[n]
    lg += w[n] * L(base)
lg += 0.25 * L(prior)[None, :]

if a.target == "oracle":
    # Per clip, mix only the members that got it right. This is the sharpest expression
    # of the selection hypothesis: instead of asking the student to imitate one global
    # compromise, ask it to imitate whichever teacher was correct HERE. If the student's
    # inputs carry enough signal to tell those cases apart, the oracle gap is reachable;
    # if they do not, this will fail cleanly and tell us the gap is not input-recoverable.
    correct = np.stack([(P[n].argmax(1) == y) for n in names])          # (M, N)
    fused_lg = lg.copy()
    stack = np.stack([L(P[n] / prior) if ("astgcn" in n or "world25" in n) else L(P[n])
                      for n in names])                                   # (M, N, 40)
    any_right = correct.any(0)
    wsum = correct.sum(0, keepdims=True).astype(np.float64)              # (1, N)
    mix = (stack * correct[:, :, None]).sum(0) / np.maximum(wsum.T, 1.0)
    lg = np.where(any_right[:, None], mix + 0.25 * L(prior)[None, :], fused_lg)
    print(f"  oracle target: {any_right.sum()} of {len(y)} clips had a correct member "
          f"({any_right.mean():.4f}); {len(y)-any_right.sum()} fall back to fused")

lg /= max(a.temperature, 1e-6)
lg -= lg.max(1, keepdims=True)
soft = np.exp(lg)
soft /= soft.sum(1, keepdims=True)

acc = float((soft.argmax(1) == y).mean())
oracle = np.zeros(len(y), bool)
for n in names:
    oracle |= (P[n].argmax(1) == y)
ent = float(-(soft * L(soft)).sum(1).mean())
print(f"  fused teacher accuracy : {acc:.5f}")
print(f"  oracle any-member      : {oracle.mean():.5f}   <- the ceiling the student chases")
print(f"  mean target entropy    : {ent:.4f} nats (T={a.temperature})")
print(f"  mean target confidence : {soft.max(1).mean():.4f}")

out = f"{ART}/{a.out}.npz"
np.savez_compressed(out, probs=soft.astype(np.float32),
                    sids=np.array(sids, dtype="<U20"), labels=y.astype(np.int64),
                    members=np.array(names), member_probs=np.stack([P[n] for n in names]).astype(np.float32))
print(f"wrote {out}")
