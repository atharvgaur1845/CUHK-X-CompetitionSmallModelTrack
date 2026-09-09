#!/usr/bin/env python3
"""T-PKG: build ONE Stage-2 weight file under the R-6 100 MB cap.

WHY A NEW PACKER. `package_ensemble.py` works and its format is good -- zip, per-tensor
SHA-256, streamed load -- but `build_model` is a closed registry that cannot construct
`mvit_v2_s`, and `infer_packaged.dataset_key` is a role whitelist with no video path. The
members we now ship are two MViT video models plus a skeleton/IMU stack, so the registry
is the wrong shape. This reuses the FORMAT and replaces the registry.

WHAT GOES IN (measured, EXP-128: this configuration scored 165/201, one clip above the
top-15 cut, against the unpackageable champion's 167):

  skeleton   5 archs x 4 folds, copied verbatim out of model_astgcn_world25_int8.pth
             (already symmetric_int8_per_tensor)                          22.80 MB
  student    distil_oracle_all, the oracle-distilled MViTv2-S             34.28 MB
  wrist      k224_mvitwrist_all, the second video view                    34.28 MB
                                                                        ---------
                                                                          91.36 MB

QUANTISATION, and why int6 codes are stored in int8 containers. The deployed video
probabilities came from `quantize_checkpoint.py --bits 6`: symmetric, PER-OUTPUT-CHANNEL,
weight-only, applied to every floating tensor with dim >= 2; norm scales, biases and
positional tables stay fp32 because they are a rounding-sensitive fraction of the
parameters. Six-bit codes live in [-32, 31] and fit an int8 container exactly, so storing
them as int8 is LOSSLESS with respect to the weights that produced those probabilities --
we reproduce the deployed model, not an int8 approximation of it. Bit-packing to 6/8 of a
byte would save 8.6 MB per view and is unnecessary at 91.36 MB; it is not implemented
rather than implemented and unused.

⚠ PROVENANCE GAP FOUND AND FIXED HERE. `w25_p4`'s composition was never recorded --
`prune_world25.py` takes `--drop`/`--merge` and no invocation survives in any ledger, so
its 22.80 MB probabilities cannot be regenerated. The tag set is recoverable from the byte
total (5 archs = 22.80 MB, exact), but the weight redistribution is not: the closest
reconstruction still differs on 14 of 405 rows. **This package therefore declares its own
skeleton spec in the manifest and derives weights from it**, so what ships is reproducible
from the file alone. Stage 2 is a reproduction stage; an unrecorded invocation is a defect.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "research" / "artifacts"
WORLD25 = ART / "model_astgcn_world25_int8.pth"
CAP_BYTES = 100 * 10 ** 6


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# role -> (cache directory, input channels). Hardcoding these to the IR+Depth values was
# safe only while every packaged view WAS IR+Depth. Thermal is 3-channel ironbow and reads
# cache/thermal_224; a wrong in_channels builds a model the weights cannot load into.
VIEW_SPEC = {
    "student": ("crop_224", 4),
    "person":  ("crop_224", 4),
    "wrist":   ("crop_wrist224", 4),
    "thermal": ("thermal_224", 3),
}


def quantize_video(tag: str, bits: int):
    """Reproduce quantize_checkpoint.py exactly, but keep the CODES instead of
    round-tripping them to fp32. Returns (entries, blobs, n_quant, n_fp32)."""
    import torch
    pkg = torch.load(ROOT / "checkpoints" / f"{tag}.pt", map_location="cpu",
                     weights_only=False)
    sd = pkg["state_dict"]
    qmax = 2 ** (bits - 1) - 1
    entries, blobs, nq, nf = [], {}, 0, 0
    for name, v in sd.items():
        if v.dtype.is_floating_point and v.dim() >= 2:
            w = v.reshape(v.shape[0], -1).float()
            scale = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
            code = (w / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
            cb = code.numpy().tobytes()
            sb = scale.reshape(-1).numpy().astype(np.float32).tobytes()
            entries.append({"name": name, "shape": list(v.shape),
                            "encoding": f"symmetric_int{bits}_per_output_channel",
                            "container": "int8", "qmax": qmax,
                            "dtype": str(v.dtype).replace("torch.", ""),
                            "code_entry": f"{name}.code", "scale_entry": f"{name}.scale",
                            "nbytes": len(cb) + len(sb),
                            "sha256": sha(cb), "scale_sha256": sha(sb)})
            blobs[f"{name}.code"] = cb
            blobs[f"{name}.scale"] = sb
            nq += 1
        else:
            b = v.float().numpy().astype(np.float32).tobytes()
            entries.append({"name": name, "shape": list(v.shape), "encoding": "float32",
                            "dtype": str(v.dtype).replace("torch.", ""),
                            "code_entry": f"{name}.raw", "nbytes": len(b), "sha256": sha(b)})
            blobs[f"{name}.raw"] = b
            nf += 1
    return entries, blobs, nq, nf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ART / "stage2_package.pth"))
    ap.add_argument("--skel-tags",
                    default="astgcn_v1,imu_inv4,imu_world,skel_jvb_big,stgcn_v1",
                    help="world25 archs to keep; the default set is 22.80 MB over 4 folds")
    ap.add_argument("--merge", action="append",
                    default=["skel_jvb_big=skel_jvb_big_s1,skel_jvb_big_s2,"
                             "skel_jvb_big_s3,skel_jvb_big_s4",
                             "stgcn_v1=stgcn_s1,stgcn_w96"],
                    help="survivor=dropped,... — dropped tags' weight moves to the "
                         "survivor of the same fold, so a seed collapse changes the seed "
                         "count and nothing else")
    # NOTE: default is None, not a list. argparse "append" APPENDS to a default list
    # rather than replacing it, so a --video on the command line silently produced the
    # defaults PLUS the requested members -- 4 video branches and a 130 MB package.
    ap.add_argument("--video", action="append", default=None,
                    help="role=checkpoint-tag (file must be checkpoints/<tag>.pt). "
                         "Defaults to student+wrist when omitted.")
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--weight", action="append", default=None,
                    help="role=weight in the log-linear pool, e.g. thermal=0.20. "
                         "Roles are the video roles plus skel, imu and prior. Declared "
                         "in the manifest so the package is self-describing; the loader "
                         "must not re-derive them.")
    ap.add_argument("--imu-trees", default="",
                    help="npz from code/imu_trees_to_tensors.py (pack_int8). Embeds the "
                         "ExtraTrees IMU member as TENSORS. Without it the sklearn member "
                         "cannot be represented at all and must be dropped, which moves 43 "
                         "of 405 rows (EXP-105) and costs the 2 clips between the 167 "
                         "champion and the 165 legal package (EXP-128).")
    ap.add_argument("--imu-smooth", type=float, default=1e-3,
                    help="Laplace smoothing applied after tree averaging. MANDATORY: the "
                         "raw forest emits exact zeros (2,382 cells over 392/405 rows) and "
                         "a zero in a geometric-mean fusion vetoes that class for the whole "
                         "ensemble -- the mechanism behind an earlier 0.29 submission.")
    a = ap.parse_args()

    if a.video is None:
        a.video = ["student=distil_oracle_all", "wrist=k224_mvitwrist_all"]
    for spec in a.video:
        role = spec.partition("=")[0]
        if role not in VIEW_SPEC:
            raise SystemExit(f"unknown video role {role!r}; add it to VIEW_SPEC "
                             f"(known: {sorted(VIEW_SPEC)})")
    fusion_w = {"skel": 0.35, "imu": 0.3575, "prior": 0.25}
    for spec in a.video:
        fusion_w[spec.partition("=")[0]] = 0.2925 / len(a.video)
    for spec in (a.weight or []):
        k, _, v = spec.partition("=")
        fusion_w[k] = float(v)
    a.fusion = " + ".join(
        f"{fusion_w[k]}*log({'skel/prior' if k == 'skel' else k})"
        for k in sorted(fusion_w))
    print(f"  fusion    {a.fusion}")
    keep = {t for t in a.skel_tags.split(",") if t}
    moved = {}
    for spec in a.merge:
        surv, _, src = spec.partition("=")
        for t in src.split(","):
            moved[t] = surv

    src_zip = zipfile.ZipFile(WORLD25)
    src_man = json.loads(src_zip.read("manifest.json"))
    by_id = {m["id"]: m for m in src_man["members"]}
    base_w = {m["id"]: float(m["weight"]) for m in src_man["members"]}
    w = dict(base_w)
    for mid, m in by_id.items():
        if m["tag"] in moved:
            surv = f"{moved[m['tag']]}_f{m['fold']}"
            if surv in w:
                w[surv] += base_w[mid]
    kept = [mid for mid, m in by_id.items() if m["tag"] in keep]
    tot = sum(w[i] for i in kept)

    members, blobs = [], {}
    for mid in sorted(kept):
        m = by_id[mid]
        tensors = []
        for t in m["tensors"]:
            b = src_zip.read(t["entry"])
            assert sha(b) == t["sha256"], f"{mid}/{t['name']}: source blob is corrupt"
            key = f"skel/{mid}/{t['name']}"
            blobs[key] = b
            tensors.append({**{k: v for k, v in t.items() if k != "entry"},
                            "code_entry": key})
        members.append({"id": mid, "role": m["role"], "tag": m["tag"], "fold": m["fold"],
                        "model": m["model"], "weight": w[mid] / tot,
                        "branch": "skeleton", "tensors": tensors,
                        "source_checkpoint": m.get("source_checkpoint")})

    for spec in a.video:
        role, _, tag = spec.partition("=")
        entries, vb, nq, nf = quantize_video(tag, a.bits)
        for k, v in vb.items():
            blobs[f"video/{role}/{k}"] = v
        members.append({
            "id": f"{role}_{tag}", "role": role, "tag": tag, "fold": None,
            "branch": "video", "weight": None,
            "model": {"builder": "kaggle/cuhkx_224_kaggle.py:build_model",
                      "arch": "mvit_v2_s", "n_classes": 40,
                      "in_channels": VIEW_SPEC[role][1],
                      "image_size": 224, "n_frames": 16,
                      "cache": VIEW_SPEC[role][0]},
            "tensors": [{**e, "code_entry": f"video/{role}/{e['code_entry']}",
                         **({"scale_entry": f"video/{role}/{e['scale_entry']}"}
                            if "scale_entry" in e else {})} for e in entries],
            "source_checkpoint": f"checkpoints/{tag}.pt"})
        print(f"  video {role:8s} <- {tag}: {nq} tensors at int{a.bits}, {nf} kept fp32")

    if a.imu_trees:
        tp = Path(a.imu_trees)
        if not tp.is_file():
            tp = ART / a.imu_trees
        td = np.load(tp)
        tensors = []
        for name in ("feature", "threshold", "left", "right", "leaf_value", "tree_offset"):
            arr = np.ascontiguousarray(td[name])
            b = arr.tobytes()
            key = f"imu_trees/{name}"
            blobs[key] = b
            tensors.append({"name": name, "shape": list(arr.shape),
                            "encoding": f"raw_{arr.dtype}", "dtype": str(arr.dtype),
                            "code_entry": key, "nbytes": len(b), "sha256": sha(b)})
        members.append({
            "id": "imu_trees", "role": "imu", "tag": "imu_stats_t200_d12", "fold": None,
            "branch": "imu", "weight": None,
            "model": {"builder": "code/imu_trees_to_tensors.py:torch_predict_int8",
                      "kind": "extra_trees_int8_leaves", "n_classes": 40,
                      "features": "code/imu_stats_member.py:clip_features (545-d)",
                      "laplace_smooth": a.imu_smooth},
            "tensors": tensors, "source_checkpoint": str(tp.name)})
        mb_ = sum(t["nbytes"] for t in tensors) / 1e6
        print(f"  imu       <- {tp.name}: {len(tensors)} tensors, {mb_:.2f} MB "
              f"(deflated in the archive)")

    manifest = {
        "format": "cuhkx.stage2-package", "format_version": 2, "byte_order": "little",
        "cap_bytes": CAP_BYTES,
        "quantization": {"skeleton": "symmetric_int8_per_tensor (copied verbatim)",
                         "video": f"symmetric_int{a.bits}_per_output_channel, "
                                  "weight-only, codes stored in int8 containers"},
        "skeleton_spec": {"keep_tags": sorted(keep), "merge": a.merge,
                          "note": "declared here because w25_p4's invocation was never "
                                  "recorded and its weights are not recoverable"},
        "fusion": {"formula": a.fusion,
                   "weights": dict(fusion_w),
                   "video_views": [spec.partition("=")[0] for spec in a.video],
                   "note": "weights are declared here, not inferred by the loader; "
                           "code/fuse_test_views.py reproduces them from members"},
        "loader": "code/unpack_stage2.py",
        "members": members,
    }
    mb = json.dumps(manifest, indent=1, sort_keys=True).encode()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Weight blobs are int8 codes with near-maximal entropy, so STORED is right for them.
    # The tree arrays are the exception: leaf distributions are highly repetitive and
    # deflate ~4.8x (7.63 MB -> 1.59 MB), which is what buys the headroom for the champion
    # configuration. Readers decompress transparently; the manifest SHA is over the RAW
    # bytes either way, so verification is unaffected.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", mb)
        for k, v in blobs.items():
            ct = zipfile.ZIP_DEFLATED if k.startswith("imu_trees/") else zipfile.ZIP_STORED
            z.writestr(zipfile.ZipInfo(k), v, compress_type=ct)

    size = out.stat().st_size
    payload = sum(len(v) for v in blobs.values())
    by_branch = collections.Counter()
    for m in members:
        by_branch[m["branch"]] += sum(t["nbytes"] for t in m["tensors"])
    print(f"\nwrote {out}")
    for b, n in by_branch.items():
        print(f"  {b:9s} {n/1e6:7.2f} MB")
    print(f"  payload   {payload/1e6:7.2f} MB   manifest {len(mb)/1e6:.2f} MB")
    print(f"  FILE ON DISK {size/1e6:.2f} MB / cap {CAP_BYTES/1e6:.0f} MB "
          f"-> {'PASS' if size <= CAP_BYTES else 'FAIL'}")
    if size > CAP_BYTES:
        raise SystemExit("package exceeds the R-6 cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
