#!/usr/bin/env python3
"""Selection Stage, Tier 0 -- can the FROZEN package reproduce the primary Selected submission
(sub_pkgship6.csv) by fusion coefficients alone?

The package (stage2_ship7.pth) reproduces sub_final_ship7.csv exactly; ship6 differs in two
near-tie rows because it was fused before the skeleton stack was pruned 48 -> 20 members to
meet the 100 MB rule.  This script fuses the package's OWN member posteriors
(inference_work/artifacts/testprobs_pkg_<role>.npz, produced by inference.sh from raw frames)
with candidate coefficient sets, runs the SHIPPED decoder unchanged, and reports the row
difference against the target CSV.  Tensors are never touched; only the six coefficients
declared in the manifest's `fusion.weights` vary.

Search: coordinate descent from the manifest coefficients, then pairwise moves, on a coarse
then fine grid; every candidate is decoded (the decoder is the arbiter, EXP-155/160).  A
pre-decoder screen (fused argmax vs research/artifacts/testprobs_pkgship6.npz, the posterior
that produced ship6) prunes hopeless candidates cheaply.

  python3 code/search_config_weights.py --target submissions/sub_pkgship6.csv \
      --out configs/56212349.json
"""
from __future__ import annotations
import argparse, csv, itertools, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLES = ["person384", "wrist", "thermal", "skel", "imu"]
DECODER = ["--transition-weight", "0.5", "--transition-score", "conditional", "--backoff", "unigram",
           "--distinctness", "penalty", "--distinctness-penalty", "2.0"]
L = lambda a: np.log(np.maximum(a, 1e-12))


def load_members(art: Path):
    P, sids = {}, None
    for r in ROLES:
        d = np.load(art / f"testprobs_pkg_{r}.npz", allow_pickle=True)
        s = [str(x) for x in d["sids"]]
        if sids is None:
            sids = s
        assert s == sids, r
        P[r] = np.asarray(d["probs"], np.float64)
    return sids, P


def fuse(P, W, prior):
    z = sum(W[r] * (L(P[r] / prior) if r == "skel" else L(P[r])) for r in ROLES) + W["prior"] * L(prior)[None, :]
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def read_csv(p: Path):
    rows = list(csv.reader(open(p)))[1:]
    return {r[0]: r[1] for r in rows}


class Decoder:
    def __init__(self, sids, target: Path, sample_sub: Path):
        self.sids = np.array(sids, dtype=str)
        self.target = read_csv(target)
        self.sample_sub = sample_sub
        self.tmp = Path(tempfile.mkdtemp(prefix="cfgsearch_"))
        self.calls = 0

    def rowdiff(self, probs) -> tuple[int, list[str]]:
        self.calls += 1
        npz = self.tmp / f"f{self.calls}.npz"; out = self.tmp / f"o{self.calls}.csv"
        np.savez_compressed(npz, probs=probs.astype(np.float32), sids=self.sids)
        subprocess.run([sys.executable, str(ROOT / "code/ordered_transition_decoder.py"), "test", "--probs", str(npz),
                        *DECODER, "--sample-submission", str(self.sample_sub), "--output", str(out)],
                       check=True, capture_output=True, text=True)
        got = read_csv(out)
        bad = [k for k in self.target if self.target[k] != got.get(k)]
        for f in (npz, out, Path(str(out) + ".manifest.json")):
            f.unlink(missing_ok=True)
        return len(bad), bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default=str(ROOT / "inference_work/artifacts"))
    ap.add_argument("--target", default=str(ROOT / "submissions/sub_pkgship6.csv"))
    ap.add_argument("--screen", default=str(ROOT / "research/artifacts/testprobs_pkgship6.npz"),
                    help="pre-decoder posterior that produced the target; argmax agreement screens candidates")
    ap.add_argument("--sample-submission", default=str(ROOT / "Small-Model-Track/Testing/sample_submission.csv"))
    ap.add_argument("--package", default=str(ROOT / "research/artifacts/stage2_ship7.pth"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-decodes", type=int, default=400)
    a = ap.parse_args()

    import zipfile
    man = json.loads(zipfile.ZipFile(a.package).read("manifest.json"))
    W0 = dict(man["fusion"]["weights"])
    counts = np.zeros(40)
    for r in csv.DictReader(open(ROOT / "cache/meta_train.csv")):
        counts[int(r["class_id"])] += 1
    prior = counts / counts.sum()
    sids, P = load_members(Path(a.artifacts))
    screen = np.load(a.screen, allow_pickle=True)
    sidx = {str(s): i for i, s in enumerate(screen["sids"])}
    screen_arg = np.asarray(screen["probs"]).argmax(1)[[sidx[s] for s in sids]]
    dec = Decoder(sids, Path(a.target), Path(a.sample_submission))

    seen = {}
    def evaluate(W):
        key = tuple(round(W[k], 5) for k in sorted(W))
        if key in seen:
            return seen[key]
        p = fuse(P, W, prior)
        pre = int((p.argmax(1) != screen_arg).sum())
        if pre > 12 or dec.calls >= a.max_decodes:     # hopeless or budget exhausted: skip the decoder
            seen[key] = (999, pre, [])
            return seen[key]
        n, bad = dec.rowdiff(p)
        seen[key] = (n, pre, bad)
        return seen[key]

    base = evaluate(W0)
    print(f"manifest weights {json.dumps({k: round(v, 4) for k, v in W0.items()})} -> rowdiff {base[0]} pre-screen {base[1]} rows {[b.split('/')[-2] for b in base[2]]}")
    best = (base[0], W0)
    keys = ["person384", "wrist", "thermal", "skel", "imu", "prior"]
    # pass 1: single-coefficient scans on a coarse grid
    for k in keys:
        for f in (0.0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0):
            W = dict(W0); W[k] = round(W0[k] * f, 5)
            n, pre, bad = evaluate(W)
            if n < best[0]:
                best = (n, W); print(f"  {k}={W[k]:.4f}: rowdiff {n} pre {pre} rows {[b.split('/')[-2] for b in bad]}")
            if n == 0:
                break
        if best[0] == 0:
            break
    # pass 2: pairwise moves around the best
    if best[0] > 0:
        Wb = best[1]
        for k1, k2 in itertools.combinations(keys, 2):
            for f1, f2 in itertools.product((0.6, 0.8, 1.25, 1.5), repeat=2):
                W = dict(Wb); W[k1] = round(Wb[k1] * f1, 5); W[k2] = round(Wb[k2] * f2, 5)
                n, pre, bad = evaluate(W)
                if n < best[0]:
                    best = (n, W); print(f"  {k1}={W[k1]:.4f},{k2}={W[k2]:.4f}: rowdiff {n} pre {pre} rows {[b.split('/')[-2] for b in bad]}")
                if n == 0:
                    break
            if best[0] == 0:
                break
    # pass 3: fine scan of the single coefficient(s) that got closest
    if best[0] > 0:
        Wb = best[1]
        for k in keys:
            for f in np.linspace(0.5, 1.5, 21):
                W = dict(Wb); W[k] = round(Wb[k] * float(f), 5)
                n, pre, bad = evaluate(W)
                if n < best[0]:
                    best = (n, W); print(f"  fine {k}={W[k]:.4f}: rowdiff {n} pre {pre} rows {[b.split('/')[-2] for b in bad]}")
                if n == 0:
                    break
            if best[0] == 0:
                break
    n, W = best
    print(f"\nBEST rowdiff {n}  weights {json.dumps({k: round(v, 5) for k, v in W.items()})}  decoder calls {dec.calls}")
    exact = [k for k, v in seen.items() if v[0] == 0]
    print(f"exact settings found: {len(exact)}")
    if a.out and n == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"fusion_weights": {k: round(v, 5) for k, v in W.items()}, "rowdiff_vs_target": 0,
                   "target": Path(a.target).name}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0 if n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
