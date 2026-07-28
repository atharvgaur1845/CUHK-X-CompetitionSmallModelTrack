#!/usr/bin/env python3
"""DA-001 zero-GPU diagnostics:
A) Test prior correction: logit-adjust saved test probs by train prior; also Sinkhorn
   to uniform marginal. Writes two new submissions (no Hungarian).
B) Oracle-fusion ceiling on OOF: is >0.75 even reachable with current streams?
C) Per-user OOF decomposition + all C(16,4) 4-user subset accuracies -> quantiles
   (does subject sampling variance explain the -9pt CV->LB offset?)
"""
import itertools
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = "/home/atharv/Desktop/projects/KAggle /CUHK-X-CompetitionSmallModelTrack"
ART = os.path.join(ROOT, "research", "artifacts")
SUBS = os.path.join(ROOT, "submissions")
W = {"skel_noaug": 0.6, "imu_aug": 0.2, "depth_aug": 0.2}

# ---------- load OOF ----------
oof, ref = {}, None
for t in W:
    z = np.load(os.path.join(ART, f"oof_{t}.npz"), allow_pickle=True)
    sids = list(z["sids"])
    if ref is None:
        ref = sids
    order = {s: i for i, s in enumerate(sids)}
    idx = [order[s] for s in ref]
    oof[t] = (z["probs"][idx], z["labels"][idx])
Y = oof["skel_noaug"][1]
P_oof = sum(w * oof[t][0] for t, w in W.items())

# train prior from OOF labels + full train meta
import csv
cnt = Counter()
with open(os.path.join(ROOT, "cache", "meta_train.csv")) as f:
    for r in csv.DictReader(f):
        cnt[int(r["class_id"])] += 1
prior = np.array([cnt[c] for c in range(40)], np.float64)
prior /= prior.sum()

# ---------- A) test prior correction ----------
Pt = None
tsids = None
for t, w in W.items():
    z = np.load(os.path.join(ART, f"testprobs_{t}.npz"), allow_pickle=True)
    if tsids is None:
        tsids = list(z["sids"])
    Pt = z["probs"] * w if Pt is None else Pt + z["probs"] * w
print("=== A) test prior correction ===")
print("argmax class histogram (top5):", Counter(Pt.argmax(1)).most_common(5))
P_adj = Pt / prior[None, :]
P_adj /= P_adj.sum(1, keepdims=True)
print("logit-adjusted top5:", Counter(P_adj.argmax(1)).most_common(5))
# Sinkhorn to uniform marginal
S = Pt.copy() / Pt.sum(1, keepdims=True)
target = np.full(40, len(tsids) / 40.0)
for _ in range(200):
    S /= S.sum(1, keepdims=True)          # rows to 1
    S *= (target / S.sum(0))[None, :]     # cols to target
print("sinkhorn top5:", Counter(S.argmax(1)).most_common(5))
for name, arr in (("sub_fuse3_prioradj.csv", P_adj), ("sub_fuse3_sinkhorn.csv", S)):
    with open(os.path.join(SUBS, name), "w") as f:
        f.write("path,prediction\n")
        for sid, p in zip(tsids, arr.argmax(1)):
            f.write(f"small_model_track_test/{sid}/,{p}\n")
    print("wrote", name)
# how much would prior adjustment have helped on OOF? (OOF val dist is also imbalanced
# -> simulate balanced val by per-class mean accuracy)
base_acc = (P_oof.argmax(1) == Y).mean()
adj_oof = P_oof / prior[None, :]
adj_acc = (adj_oof.argmax(1) == Y).mean()
percls = [ (P_oof.argmax(1) == Y)[Y == c].mean() for c in range(40) if (Y == c).any()]
percls_adj = [ (adj_oof.argmax(1) == Y)[Y == c].mean() for c in range(40) if (Y == c).any()]
print(f"OOF: argmax {base_acc:.4f} -> adj {adj_acc:.4f} | balanced-mean {np.mean(percls):.4f} -> {np.mean(percls_adj):.4f}")

# ---------- B) oracle fusion ceiling ----------
print("\n=== B) oracle fusion ceiling (OOF) ===")
correct_any = np.zeros(len(Y), bool)
for t in W:
    correct_any |= oof[t][0].argmax(1) == Y
print(f"oracle pick-best-stream: {correct_any.mean():.4f}")
for k in (2, 3, 5):
    topk = np.zeros(len(Y), bool)
    for t in W:
        rk = np.argsort(-oof[t][0], 1)[:, :k]
        topk |= (rk == Y[:, None]).any(1)
    print(f"oracle any-stream top-{k}: {topk.mean():.4f}")

# ---------- C) per-user decomposition + subset quantiles ----------
print("\n=== C) per-user + 4-user subset quantiles (fused OOF) ===")
user_of = {}
with open(os.path.join(ROOT, "cache", "meta_train.csv")) as f:
    for r in csv.DictReader(f):
        user_of[r["sample_id"]] = r["user"]
hits = P_oof.argmax(1) == Y
by_user = defaultdict(list)
for i, s in enumerate(ref):
    by_user[user_of[s]].append(hits[i])
uaccs = {u: (np.mean(v), len(v)) for u, v in by_user.items()}
for u in sorted(uaccs, key=lambda x: uaccs[x][0]):
    print(f"  {u}: {uaccs[u][0]:.3f} (n={uaccs[u][1]})")
users = sorted(by_user)
accs4 = []
early = [u for u in users if int(u[4:]) <= 9]
late = [u for u in users if int(u[4:]) >= 16]
cons = []
for combo in itertools.combinations(users, 4):
    v = np.concatenate([by_user[u] for u in combo])
    a = v.mean()
    accs4.append(a)
    if sum(u in early for u in combo) == 2:
        cons.append(a)
accs4, cons = np.array(accs4), np.array(cons)
print(f"all C(16,4)={len(accs4)}: mean {accs4.mean():.4f}, p5 {np.percentile(accs4,5):.4f}, p10 {np.percentile(accs4,10):.4f}, p25 {np.percentile(accs4,25):.4f}, min {accs4.min():.4f}")
print(f"cohort-matched (2E+2L) n={len(cons)}: mean {cons.mean():.4f}, p10 {np.percentile(cons,10):.4f}, min {cons.min():.4f}")
print(f"LB observed: 0.458 -> percentile of all-subsets dist: {(accs4 < 0.458).mean():.3f}")
