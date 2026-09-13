#!/usr/bin/env python3
"""Run the SKELETON and IMU branches of a Stage-2 package from the npz cache.

The video branch already runs from the package (`unpack_stage2.py --check infer`), and the
detectors are verified to reproduce every crop window. These two branches were the gap:
`fuse_from_package.py` read PRECOMPUTED skeleton probabilities (`world25_per_member.npz`),
and the IMU trees were only ever exercised by hand. Nothing ran either from the shipped file
on data it had never seen -- which is exactly what Stage 2 and the on-site test do.

Skeleton: manifest order, weighted ARITHMETIC mean of per-member softmax, identical to
`infer_packaged.fuse_members` and `fuse_from_package.py`. Inputs come from `har_data`
via `infer_packaged.load_test_inputs`, so the feature pipeline is the one the members were
trained on. IMU: `imu_stats_member.clip_features` on the same npz cache, then the packed
tree tensors through `imu_trees_to_tensors.torch_predict_int8`, then the mandatory Laplace
smoothing declared in the manifest.

Gate: `--verify` compares against the artifacts every submission was scored with.
"""
from __future__ import annotations
import argparse, csv, json, sys, zipfile
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import infer_packaged as IP                        # noqa: E402
import package_ensemble as PE                      # noqa: E402
from unpack_stage2 import load_member              # noqa: E402


def skeleton(z, man, batch, workers, device):
    members = [m for m in man["members"] if m["branch"] == "skeleton"]
    inputs, sids = IP.load_test_inputs(batch, workers, members)
    probs = np.zeros((len(sids), 40), np.float64)
    with torch.no_grad():
        for i, m in enumerate(members):
            model = PE.build_model(m["model"]).to(device).eval()
            model.load_state_dict(load_member(z, m))
            x = inputs[IP.dataset_key(m)]
            out = np.concatenate([torch.softmax(model(x[s:s + batch].to(device)), 1).cpu().numpy()
                                  for s in range(0, len(x), batch)])
            probs += float(m["weight"]) * out
            print(f"  [{i+1:02d}/{len(members)}] {m['id']:22s} w={m['weight']:.4f}", flush=True)
            del model
    return sids, probs


def imu(z, man):
    import imu_stats_member as ISM
    import imu_trees_to_tensors as ITT
    m = next(x for x in man["members"] if x["branch"] == "imu")
    d = {t["name"]: np.frombuffer(z.read(t["code_entry"]), dtype=np.dtype(t["dtype"])).reshape(t["shape"])
         for t in m["tensors"]}
    sample = ROOT / "Small-Model-Track" / "Testing" / "sample_submission.csv"
    sids = [r[0].strip("/").split("/")[-1] for r in list(csv.reader(open(sample)))[1:]]
    X = ISM.build_matrix(sids, "test")
    P = ITT.torch_predict_int8(d, X)
    P = (P.numpy() if torch.is_tensor(P) else np.asarray(P)).astype(np.float64)
    sm = float(m["model"]["laplace_smooth"])
    P = (P + sm) / (1.0 + 40.0 * sm)
    assert (P > 0).all(), "zero cell after smoothing"
    return sids, P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "research" / "artifacts"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--verify", action="store_true",
                    help="compare against testprobs_astgcn_world25 / testprobs_imu_stats_t200_d12")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = zipfile.ZipFile(a.package); man = json.loads(z.read("manifest.json"))
    out = Path(a.out_dir)

    print("skeleton branch:")
    sids, sk = skeleton(z, man, a.batch, a.workers, dev)
    np.savez_compressed(out / "testprobs_pkg_skel.npz", probs=sk.astype(np.float32), sids=np.array(sids, dtype="<U15"))
    print("imu branch:")
    isids, im = imu(z, man)
    assert isids == list(sids), "sid order differs between skeleton and imu"
    np.savez_compressed(out / "testprobs_pkg_imu.npz", probs=im.astype(np.float32), sids=np.array(sids, dtype="<U15"))
    print(f"wrote testprobs_pkg_skel.npz, testprobs_pkg_imu.npz ({len(sids)} clips)")

    if a.verify:
        ok = True
        for name, got, ref in (("skeleton", sk, "testprobs_astgcn_world25.npz"),
                               ("imu", im, "testprobs_imu_stats_t200_d12.npz")):
            r = np.load(out / ref, allow_pickle=True); rs = [str(s) for s in r["sids"]]
            assert rs == list(sids), f"{name}: sid order differs from {ref}"
            rp = np.asarray(r["probs"], np.float64)
            agree = int((got.argmax(1) == rp.argmax(1)).sum()); md = np.abs(got - rp).max()
            print(f"  {name:9s} vs {ref}: argmax {agree}/{len(sids)}  max|dp| {md:.2e}"
                  f"  -> {'PASS' if agree == len(sids) else 'FAIL'}")
            ok &= agree == len(sids)
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
