#!/usr/bin/env python3
"""Cross-machine / cross-GPU comparison of two inference.sh runs (gate G8, README section 7).

    python3 code/compare_runs.py <artifacts_dir_A> <artifacts_dir_B> [--csv-a A.csv --csv-b B.csv]

For every branch posterior file testprobs_pkg_<role>.npz present in both directories (aligned
by clip id): max and mean |dp|, number of clips whose argmax differs; then the fused posterior;
then the final CSVs: row count, order, per-row equality, sha256.  Bit-equality is not the
claim -- predictions are.
"""
from __future__ import annotations
import argparse, csv, hashlib, sys
from pathlib import Path
import numpy as np

ROLES = ["person384", "wrist", "thermal", "skel", "imu"]


def load(p):
    z = np.load(p, allow_pickle=True)
    return {str(s): np.asarray(v, np.float64) for s, v in zip(z["sids"], z["probs"])}


def compare(pa, pb, name):
    common = sorted(set(pa) & set(pb))
    if not common:
        print(f"  {name:10s} no common clips"); return
    A = np.stack([pa[s] for s in common]); B = np.stack([pb[s] for s in common])
    d = np.abs(A - B); flips = int((A.argmax(1) != B.argmax(1)).sum())
    print(f"  {name:10s} clips {len(common):4d}  max|dp| {d.max():.2e}  mean|dp| {d.mean():.2e}  argmax changes {flips}"
          + (f"  ({[common[i] for i in np.where(A.argmax(1) != B.argmax(1))[0][:5]]})" if flips else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--csv-a"); ap.add_argument("--csv-b")
    a = ap.parse_args()
    A, B = Path(a.a), Path(a.b)
    print(f"A = {A}\nB = {B}")
    for r in ROLES:
        fa, fb = A / f"testprobs_pkg_{r}.npz", B / f"testprobs_pkg_{r}.npz"
        if fa.is_file() and fb.is_file():
            compare(load(fa), load(fb), r)
    if (A / "testprobs_fused.npz").is_file() and (B / "testprobs_fused.npz").is_file():
        compare(load(A / "testprobs_fused.npz"), load(B / "testprobs_fused.npz"), "fused")
    if a.csv_a and a.csv_b:
        ra = list(csv.reader(open(a.csv_a))); rb = list(csv.reader(open(a.csv_b)))
        ha = hashlib.sha256(open(a.csv_a, "rb").read()).hexdigest(); hb = hashlib.sha256(open(a.csv_b, "rb").read()).hexdigest()
        same_order = [r[0] for r in ra] == [r[0] for r in rb]
        diff = [r[0] for r, q in zip(ra[1:], rb[1:]) if r[1] != q[1]] if same_order else None
        print(f"  CSV rows {len(ra)-1} vs {len(rb)-1}; header {ra[0]} vs {rb[0]}; same order {same_order}; "
              f"prediction changes {len(diff) if diff is not None else 'n/a'}{(' ' + str(diff[:5])) if diff else ''}; "
              f"sha256 {'identical' if ha == hb else 'DIFFERENT'} ({ha[:12]} vs {hb[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
