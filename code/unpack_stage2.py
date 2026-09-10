#!/usr/bin/env python3
"""Load and VERIFY the Stage-2 package written by code/pack_stage2.py.

Three levels of check, cheapest first, because each one can fail independently:

  --check integrity   every blob's SHA-256 matches the manifest, and the file is under
                      the R-6 cap. Catches a corrupt or truncated archive.
  --check weights     the DEQUANTISED video tensors are bit-identical to what
                      `quantize_checkpoint.py --bits 6` produces from the source
                      checkpoint. This is the load-bearing one: it proves the package
                      contains the deployed model rather than an int8 approximation of
                      it, and it is exact arithmetic, so "close" is a failure.
  --check infer       rebuild each video model from the package and re-run test
                      inference, comparing against the reference testprobs npz.
                      Needs the crop caches and a GPU; minutes, not seconds.

B-033: a score can close an accuracy question and NEVER a serialization one. Only this
script can close the serialization one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

import bitpack

ROOT = Path(__file__).resolve().parents[1]
CAP_BYTES = 100 * 10 ** 6


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_member(z, m):
    """Reconstruct one member's state_dict from its blobs."""
    import torch
    sd = {}
    for t in m["tensors"]:
        raw = z.read(t["code_entry"])
        enc = t["encoding"]
        if enc.startswith("raw_"):
            arr = np.frombuffer(raw, dtype=np.dtype(enc[4:])).reshape(t["shape"]).copy()
            sd[t["name"]] = torch.from_numpy(arr)
        elif enc == "float32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(t["shape"]).copy()
            sd[t["name"]] = torch.from_numpy(arr)
        elif enc.startswith("symmetric_int") and "per_output_channel" in enc:
            cont = t.get("container", "int8")
            if cont.startswith("bitpacked"):
                code = bitpack.unpack_codes(raw, int(t["n_codes"]),
                                            int(cont[len("bitpacked"):])).astype(np.float32)
            else:
                code = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
            scale = np.frombuffer(z.read(t["scale_entry"]), dtype=np.float32)
            out = t["shape"][0]
            sd[t["name"]] = torch.from_numpy(
                (code.reshape(out, -1) * scale[:, None]).reshape(t["shape"]).copy())
        elif enc == "symmetric_int8_per_tensor":
            code = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
            sd[t["name"]] = torch.from_numpy(
                (code * float(t["scale"])).reshape(t["shape"]).copy())
        else:
            raise SystemExit(f"unknown encoding {enc!r} for {m['id']}/{t['name']}")
    return sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default="research/artifacts/stage2_package.pth")
    ap.add_argument("--check", default="integrity,weights",
                    help="comma list of integrity,weights,infer,detector")
    ap.add_argument("--detector-clips", type=int, default=0,
                    help="limit the detector window check to the first N test clips "
                         "(0 = all 405)")
    ap.add_argument("--cache-root", default="cache")
    a = ap.parse_args()
    checks = {c for c in a.check.split(",") if c}

    pkg = Path(a.package)
    size = pkg.stat().st_size
    z = zipfile.ZipFile(pkg)
    man = json.loads(z.read("manifest.json"))
    print(f"{pkg.name}: {size/1e6:.2f} MB on disk, cap {CAP_BYTES/1e6:.0f} MB -> "
          f"{'PASS' if size <= CAP_BYTES else 'FAIL'}")
    print(f"  format {man['format']} v{man['format_version']}, {len(man['members'])} members")
    fail = size > CAP_BYTES

    if "integrity" in checks:
        bad = n = 0
        for m in man["members"]:
            for t in m["tensors"]:
                n += 1
                if sha(z.read(t["code_entry"])) != t["sha256"]:
                    print(f"  CORRUPT {m['id']}/{t['name']}"); bad += 1
                if "scale_entry" in t and sha(z.read(t["scale_entry"])) != t["scale_sha256"]:
                    print(f"  CORRUPT scale {m['id']}/{t['name']}"); bad += 1
        print(f"  integrity: {n} tensors, {bad} mismatches -> {'FAIL' if bad else 'PASS'}")
        fail |= bad > 0

    if "weights" in checks:
        import torch
        ok = True
        for m in man["members"]:
            if m["branch"] != "video":
                continue
            src = torch.load(ROOT / m["source_checkpoint"], map_location="cpu",
                             weights_only=False)["state_dict"]
            bits = int(m["tensors"][0]["encoding"].split("int")[1].split("_")[0])
            qmax = 2 ** (bits - 1) - 1
            got = load_member(z, m)
            worst, worst_name = 0.0, ""
            for k, v in src.items():
                if v.dtype.is_floating_point and v.dim() >= 2:
                    w = v.reshape(v.shape[0], -1).float()
                    s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
                    ref = ((w / s).round().clamp(-qmax - 1, qmax) * s).reshape(v.shape)
                else:
                    ref = v.float()
                e = float((got[k].float() - ref).abs().max())
                if e > worst:
                    worst, worst_name = e, k
            status = "PASS" if worst == 0.0 else "FAIL"
            print(f"  weights {m['id']:28s} max abs diff vs quantize_checkpoint "
                  f"--bits {bits}: {worst:.3e} ({worst_name or 'n/a'}) -> {status}")
            ok &= worst == 0.0
        fail |= not ok

    if "infer" in checks:
        import sys, torch
        sys.path.insert(0, str(ROOT / "kaggle"))
        import cuhkx_224_kaggle as K
        from torch.utils.data import DataLoader
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for m in man["members"]:
            if m["branch"] != "video":
                continue
            cache = ROOT / a.cache_root / m["model"]["cache"]
            store = K.ClipStore(cache, "test")
            idx = json.loads((cache / "test_index.json").read_text())
            order = np.arange(len(idx["sids"]))
            dl = DataLoader(K.make_dataset(store, order, None, False), batch_size=4,
                            shuffle=False, num_workers=0, pin_memory=False)
            # in_channels comes from the MANIFEST, not from a constant. Every view
            # packaged before EXP-140 was 4-channel IR+Depth, so a hardcoded 4 was
            # invisible until thermal (3-channel ironbow) was packaged and the state
            # dict would not load. Trust the file, which is the point of the file.
            ch = int(m["model"].get("in_channels", store.channels))
            assert ch == store.channels, (
                f"{m['id']}: manifest says in_channels={ch} but cache "
                f"{m['model']['cache']} has {store.channels}")
            model = K.build_model(m["model"]["arch"], image_size=store.size,
                                  in_channels=ch).to(dev)
            model.load_state_dict(load_member(z, m))
            probs, _ = K.predict(model, dl, dev)
            ref_tag = m["tag"] + "_q6"
            ref_p = ROOT / "research" / "artifacts" / f"testprobs_{ref_tag}.npz"
            if ref_p.is_file():
                ref = np.load(ref_p)["probs"]
                same = int((probs.argmax(1) == ref.argmax(1)).sum())
                print(f"  infer {m['id']:28s} argmax {same}/{len(ref)} vs {ref_tag}, "
                      f"max|dp| {np.abs(probs-ref).max():.3e}")
            np.savez_compressed(
                ROOT / "research" / "artifacts" / f"testprobs_pkg_{m['role']}.npz",
                probs=probs.astype(np.float32),
                sids=np.array([str(s) for s in idx["sids"]], dtype="<U15"))
            print(f"    wrote testprobs_pkg_{m['role']}.npz")
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    if "detector" in checks:
        # The compliance check. A package that cannot rebuild its detector cannot crop,
        # and the on-site stage runs on a brand-new dataset where no cached windows exist.
        # Gate is window IDENTITY against the source .pt, on every clip -- EXP-149 showed
        # a detector's output is an argmax over frames, so "close weights" is not "same
        # window".
        import sys, shutil, torch
        sys.path.insert(0, str(ROOT / "kaggle"))
        import cuhkx_224_kaggle as K
        from ultralytics import YOLO
        paths = K.find_paths(str(ROOT), "crop_224", str(ROOT / a.cache_root / "crop_224"))
        _tr, te = K.build_jobs(paths)
        jobs = te[: a.detector_clips] if a.detector_clips else te
        for m in man["members"]:
            if m["branch"] != "detector":
                continue
            src = m["source_checkpoint"]
            sd_pkg = load_member(z, m)
            y = YOLO(src)
            ref_sd = y.model.state_dict()
            miss = [k for k in ref_sd if k not in sd_pkg]
            bad = [k for k in ref_sd
                   if k in sd_pkg and not torch.equal(
                       ref_sd[k].float(), sd_pkg[k].to(ref_sd[k].dtype).float())]
            print(f"  detector {m['role']:16s} tensors {len(sd_pkg)}, missing {len(miss)}, "
                  f"differing {len(bad)} -> {'PASS' if not miss and not bad else 'FAIL'}")
            fail |= bool(miss or bad)
            crop = "wrist" if "wrist" in m["role"] else "person"
            _orig = YOLO.__init__
            def patched(self, *args, **kw):
                _orig(self, *args, **kw)
                if Path(str(args[0])).stem == m["tag"]:
                    self.model.load_state_dict({k: v.to(dict(self.model.state_dict())[k].dtype)
                                                for k, v in sd_pkg.items()})
            def wins(tag, init):
                YOLO.__init__ = init
                d = Path(f"/tmp/pkgwin_{tag}"); shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True)
                r = K.compute_windows(jobs, d, "cpu", crop)
                YOLO.__init__ = _orig
                return r
            base = wins("base", _orig); got = wins("pkg", patched)
            ident = sum(1 for k in base if base[k] == got.get(k))
            print(f"    windows from the PACKAGE vs {src}: {ident}/{len(base)} identical"
                  f" -> {'PASS' if ident == len(base) else 'FAIL'}")
            fail |= ident != len(base)

    print("\nRESULT:", "FAIL" if fail else "PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
