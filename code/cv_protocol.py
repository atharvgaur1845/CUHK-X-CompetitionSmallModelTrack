#!/usr/bin/env python3
"""Q-01: fixed subject-grouped CV protocol.

Test users are 10,11 (early id block) and 25,26 (late block); train has 9 early
(1-9) + 9 late (16-24). Each val fold = 2 early + 2 late users, mirroring that
structure. We search assignments maximizing the worst fold's val class coverage
(40 classes; coverage is ragged), tie-break on balanced val sizes.
Output: research/artifacts/cv_folds.json — THE protocol file for all experiments.
"""
import csv
import itertools
import json
import os
from collections import defaultdict

# Portable root: env CUHKX_ROOT wins, else the repo dir two levels up from this
# file. Was a hardcoded absolute path (with a space in it) in 11 files, which was
# the #1 blocker for running anywhere but the original laptop.
ROOT = os.environ.get("CUHKX_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "research", "artifacts", "train_index.csv")
OUT = os.path.join(ROOT, "research", "artifacts", "cv_folds.json")

user_classes = defaultdict(set)
user_samples = defaultdict(int)
seen = set()
with open(IDX) as f:
    for r in csv.DictReader(f):
        key = (r["class_id"], r["user"], r["trial"])
        if key in seen:
            continue
        seen.add(key)
        user_classes[r["user"]].add(int(r["class_id"]))
        user_samples[r["user"]] += 1

early = [f"user{i}" for i in range(1, 10)]
late = [f"user{i}" for i in range(16, 25)]
# Hold the weakest-coverage user of each block out of validation (always-train).
skip_e = min(early, key=lambda u: len(user_classes[u]))
skip_l = min(late, key=lambda u: len(user_classes[u]))
e8 = [u for u in early if u != skip_e]
l8 = [u for u in late if u != skip_l]

def eval_assign(e_pairs, l_pairs):
    folds = [sorted(e_pairs[i]) + sorted(l_pairs[i]) for i in range(4)]
    covs = [len(set().union(*(user_classes[u] for u in f))) for f in folds]
    sizes = [sum(user_samples[u] for u in f) for f in folds]
    return min(covs), sum(covs), -max(sizes) + min(sizes), folds, covs, sizes

def pairings(items):
    if not items:
        yield []
        return
    a = items[0]
    for i in range(1, len(items)):
        for rest in pairings(items[1:i] + items[i + 1:]):
            yield [(a, items[i])] + rest

best = None
e_parts = list(pairings(e8))
l_parts = list(pairings(l8))
for ep in e_parts:
    for lp in l_parts:
        for perm in itertools.permutations(range(4)):
            lp2 = [lp[i] for i in perm]
            score = eval_assign(ep, lp2)
            if best is None or score[:3] > best[:3]:
                best = score

min_cov, tot_cov, _, folds, covs, sizes = best
protocol = {
    "description": "4-fold subject-grouped CV; val fold = 2 early + 2 late users "
                   "(mirrors test users 10,11,25,26). Always-train users excluded from val.",
    "always_train": [skip_e, skip_l],
    "folds": [
        {"fold": i, "val_users": folds[i], "val_classes_covered": covs[i], "val_samples": sizes[i]}
        for i in range(4)
    ],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(protocol, f, indent=2)
print(json.dumps(protocol, indent=2))
print(f"min fold coverage {min_cov}/40, total {tot_cov}/160")
