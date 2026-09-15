#!/usr/bin/env python3
"""Final-hour candidates: re-fuse the PACKAGE's own member posteriors with alternative
weight sets (reproducible by a manifest `fusion.weights` change alone), decode, rowdiff.
Also scores each weight set on the 2,700 pooled OOF through the EXP-160 harness so the
candidates are ranked by evidence, not by taste."""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/"research/artifacts"; PK = ROOT/"inference_work/artifacts"
L = lambda a: np.log(np.maximum(a, 1e-12))
counts = np.zeros(40)
for r in csv.DictReader(open(ROOT/"cache/meta_train.csv")): counts[int(r["class_id"])] += 1
prior = counts/counts.sum()
roles = ["person384","wrist","thermal","skel","imu"]
T = {r: np.load(PK/f"testprobs_pkg_{r}.npz", allow_pickle=True) for r in roles}
sids = [str(x) for x in T["person384"]["sids"]]
# OOF members (fold models) for the same slots
ref = np.load(ART/"oof_p384w_fused.npz"); osids=list(ref["sids"]); y=ref["labels"]
def stack(tags):
    d={}
    for t in tags:
        z=np.load(ART/f"oof_{t}.npz"); d.update(zip(z["sids"], z["probs"]))
    return np.stack([d[s] for s in osids])
O = {"person384": stack([f"k224_mvit384_f{f}" for f in range(4)]), "wrist": stack([f"k224_mvitwrist_f{f}" for f in range(4)]),
     "thermal": stack([f"k224_mvit_th_f{f}" for f in range(4)]), "skel": stack(["astgcn_world25"]),
     "imu": (stack(["imu_stats_t200_d12"])+0.001)/(1+40*0.001)}
def fuse(P, W):
    z = sum(W[r]*(L(P[r]/prior) if r=="skel" else L(P[r])) for r in roles) + W.get("prior",0.25)*L(prior)[None,:]
    p = np.exp(z-z.max(1,keepdims=True)); return p/p.sum(1,keepdims=True)
base = json.loads((ART/"weights_p384.json").read_text()); base = {"person384":base["person"],"wrist":base["wrist"],"thermal":base["thermal"],"skel":base["skel"],"imu":base["imu"],"prior":0.25}
cands = json.loads(sys.argv[1]) if len(sys.argv)>1 else {}
cands = {"base": base, **cands}
for name, W in cands.items():
    Wf = {**base, **W}
    oof = fuse({r: O[r] for r in roles}, Wf); n = int((oof.argmax(1)==y).sum())
    p = fuse({r: np.asarray(T[r]["probs"],np.float64) for r in roles}, Wf)
    out = ART/"cand"/f"fused_{name}.npz"; np.savez_compressed(out, probs=p.astype(np.float32), sids=np.array(sids,dtype="<U15"))
    csvp = ROOT/"submissions"/f"sub_cand_{name}.csv"
    subprocess.run(["python3", str(ROOT/"code/ordered_transition_decoder.py"), "test", "--probs", str(out),
                    "--transition-weight","0.5","--transition-score","conditional","--backoff","unigram",
                    "--distinctness","penalty","--distinctness-penalty","2.0","--output", str(csvp)], capture_output=True, text=True, check=True)
    rd = subprocess.run(["python3", str(ROOT/"code/rowdiff.py"), str(ROOT/"submissions/sub_final_ship7.csv"), str(csvp)], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"{name:12s} OOF {n}/2700  weights {json.dumps({k: round(v,4) for k,v in Wf.items()})}\n             {rd}")
