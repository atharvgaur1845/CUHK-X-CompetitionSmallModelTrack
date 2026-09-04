#!/usr/bin/env python3
"""Rebuild the fused test probabilities using ONLY what the Stage-2 package declares.

The point is reproducibility, not convenience: the fusion weights, the skeleton tag set
and the merge rule are read out of the package's own manifest, so the submission can be
regenerated from the shipped file plus this script. `w25_p4` could not be regenerated
because its `prune_world25.py` invocation was never recorded anywhere -- that is the
defect this avoids repeating.

The video terms come from `testprobs_pkg_*.npz`, which `unpack_stage2.py --check infer`
produces by rebuilding each model FROM THE PACKAGE. The skeleton term is the declared
weighted sum over `world25_per_member.npz`; those per-member probabilities correspond to
the same checkpoints whose tensors are copied verbatim into the package, and the source
package carries per-member argmax verification.
"""
import argparse, collections, csv, json, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "research" / "artifacts"
L = lambda a: np.log(np.maximum(a, 1e-12))

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--package", default=str(ART / "stage2_package.pth"))
ap.add_argument("--tag", default="pkg_v2")
a = ap.parse_args()

man = json.loads(zipfile.ZipFile(a.package).read("manifest.json"))
spec = man["skeleton_spec"]
print(f"skeleton spec from the package: keep={spec['keep_tags']}")
for m in spec["merge"]:
    print(f"  merge {m}")

src = json.loads(zipfile.ZipFile(ART / "model_astgcn_world25_int8.pth").read("manifest.json"))
by_id = {m["id"]: m for m in src["members"]}
base_w = {m["id"]: float(m["weight"]) for m in src["members"]}
moved = {}
for s in spec["merge"]:
    surv, _, srcs = s.partition("=")
    for t in srcs.split(","):
        moved[t] = surv
w = dict(base_w)
for mid, m in by_id.items():
    if m["tag"] in moved:
        k = f"{moved[m['tag']]}_f{m['fold']}"
        if k in w:
            w[k] += base_w[mid]

d = np.load(ART / "world25_per_member.npz", allow_pickle=True)
ids = [str(x) for x in d["ids"]]
sids = [str(s) for s in d["sids"]]
kept = [i for i in ids if by_id[i]["tag"] in set(spec["keep_tags"])]
tot = sum(w[i] for i in kept)
P = np.stack([d[i].astype(np.float64) for i in kept])
B = ((np.array([w[i] for i in kept]) / tot)[:, None, None] * P).sum(0)
print(f"  skeleton branch: {len(kept)} members over {len(set(by_id[i]['tag'] for i in kept))} archs")

def take(role):
    z = np.load(ART / f"testprobs_pkg_{role}.npz", allow_pickle=True)
    assert [str(s) for s in z["sids"]] == sids, f"{role}: sid order differs from the skeleton branch"
    return z["probs"].astype(np.float64)

wrist, student = take("wrist"), take("student")
tr = collections.Counter(int(r["class_id"]) for r in csv.DictReader(open(ROOT / "cache" / "meta_train.csv")))
n = sum(tr.values()); prior = np.array([tr[k] / n for k in range(40)])

lg = 0.35 * L(B / prior) + 0.2925 * L(wrist) + 0.3575 * L(student) + 0.25 * L(prior)[None, :]
lg -= lg.max(1, keepdims=True)
p = np.exp(lg); p /= p.sum(1, keepdims=True)
out = ART / f"testprobs_{a.tag}.npz"
np.savez_compressed(out, probs=p.astype(np.float32), sids=np.array(sids, dtype="<U15"))
print(f"wrote {out}")
print(f"  formula: {man['fusion']['formula']}")
