#!/usr/bin/env python3
"""Emit a pruned world25 skeleton branch (EXP-105 / Stage-2 packaging).

world25 is a weighted probability sum over 48 members. 60% of its 84.5 MB is five seed
replicas of one architecture carrying 23% of the weight. This drops chosen tags and
redistributes their weight onto the surviving replicas of the same architecture, so a
seed collapse changes the seed count and nothing else.
"""
import argparse, json, zipfile, collections
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--drop", default="", help="comma-separated tags to drop")
ap.add_argument("--folds", default="0,1,2,3", help="folds to keep")
ap.add_argument("--merge", action="append", default=[],
                help="survivor=dropped1,dropped2 — move dropped tags' weight onto survivor")
ap.add_argument("--tag", required=True)
a = ap.parse_args()

PKG = "research/artifacts/model_astgcn_world25_int8.pth"
man = json.loads(zipfile.ZipFile(PKG).read("manifest.json"))
d = np.load("research/artifacts/world25_per_member.npz", allow_pickle=True)
ids = [str(x) for x in d["ids"]]
by_id = {m["id"]: m for m in man["members"]}
bytes_by_tag = collections.defaultdict(int)
for m in man["members"]:
    bytes_by_tag[m["tag"]] += sum(t["nbytes"] for t in m["tensors"])

drop = {t for t in a.drop.split(",") if t}
keep_folds = {int(f) for f in a.folds.split(",")}
moved = {}
for spec in a.merge:
    surv, _, src = spec.partition("=")
    for t in src.split(","):
        moved[t] = surv
        drop.add(t)

w = {i: float(by_id[i]["weight"]) for i in ids}
# redistribute merged weight onto the survivor's members of the same fold
for i in ids:
    m = by_id[i]
    if m["tag"] in moved:
        surv_id = f"{moved[m['tag']]}_f{m['fold']}"
        if surv_id in w:
            w[surv_id] += w[i]
kept = [i for i in ids
        if by_id[i]["tag"] not in drop and by_id[i]["fold"] in keep_folds]
if not kept:
    raise SystemExit("pruned everything")
tot = sum(w[i] for i in kept)
P = np.stack([d[i].astype(np.float64) for i in kept])
probs = ((np.array([w[i] for i in kept]) / tot)[:, None, None] * P).sum(0)

mb = sum(bytes_by_tag[t] for t in {by_id[i]["tag"] for i in kept}) \
     * len(keep_folds) / 4 / 1e6
out = f"research/artifacts/testprobs_{a.tag}.npz"
np.savez_compressed(out, probs=probs.astype(np.float32),
                    sids=np.array([str(s) for s in d["sids"]], dtype="<U15"))
tags = sorted({by_id[i]["tag"] for i in kept})
print(f"{a.tag}: {len(kept)} members, {len(tags)} archs, folds={sorted(keep_folds)}, "
      f"{mb:.2f} MB int8")
print(f"  kept: {','.join(tags)}")
print(f"  wrote {out}")
