#!/usr/bin/env python3
"""Rebuild the YOLO detector .pt files FROM THE STAGE-2 PACKAGE, for inference.sh.

The organisers require the detector weights inside the single <=100 MB checkpoint (topic
738333). `pack_stage2.py` stores them as raw fp16 tensors; at inference the pipeline calls
`YOLO("yolo11n.pt")` / `YOLO("yolo11n-pose.pt")` relative to the working directory, so this
writes those two files from the package's tensors and nothing else. The architecture is
rebuilt from ultralytics' bundled yaml -- with nc=1 for the pose model, which is how the
released weights were trained (the yaml defaults to nc=80 and 51 head tensors change shape).

Gate: every tensor of the rebuilt model equals the package tensor (strict load), and the
package's own `--check detector` then confirms 405/405 identical crop windows.
"""
from __future__ import annotations
import argparse, json, zipfile, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from unpack_stage2 import load_member  # noqa: E402

ARCH = {  # source checkpoint stem -> (yaml, model class, kwargs)
    "yolo11n":      ("yolo11n.yaml",      "DetectionModel", {}),
    "yolo11n-pose": ("yolo11n-pose.yaml", "PoseModel",      {"nc": 1}),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--test-root", default=None,
                    help="clip directory of the run; shipped windows are used only for clips whose "
                         "IR-filename fingerprint matches inference_assets/window_fingerprints.json")
    ap.add_argument("--fingerprints", default=str(ROOT / "inference_assets" / "window_fingerprints.json"))
    a = ap.parse_args()
    from ultralytics.nn import tasks
    z = zipfile.ZipFile(a.package); man = json.loads(z.read("manifest.json"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for m in man["members"]:
        if m["branch"] != "detector":
            continue
        stem = m["tag"]
        yaml, cls, kw = ARCH[stem]
        model = getattr(tasks, cls)(yaml, verbose=False, **kw)
        sd = load_member(z, m)
        ref = model.state_dict()
        sd = {k: v.to(ref[k].dtype) for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.args = {"task": "pose" if cls == "PoseModel" else "detect"}
        model = model.half().eval()
        # ultralytics loads checkpoints as {"model": <nn.Module>, ...}
        torch.save({"model": model, "train_args": {}, "epoch": -1, "date": None,
                    "version": "packaged", "license": "AGPL-3.0", "docs": ""},
                   out / f"{stem}.pt")
        print(f"  wrote {out / (stem + '.pt')}  ({len(sd)} tensors, strict load OK)")
        n += 1
    if n != 2:
        raise SystemExit(f"expected 2 detectors in the package, found {n}")
    # Crop-window cache for the Kaggle test clips (see pack_stage2 --windows). A shipped
    # window is written ONLY for a clip that exists under --test-root AND whose IR-filename
    # fingerprint matches the one recorded for that id: the same clip id on a different
    # dataset (a held-out set in the same layout) must go through the packaged detector.
    fp = json.load(open(a.fingerprints))["clips"] if a.test_root else {}
    from window_fingerprints import clip_fingerprint
    for m in man["members"]:
        if m["branch"] != "windows":
            continue
        shipped = json.loads(z.read(m["tensors"][0]["code_entry"]))
        keep, rejected = {}, []
        for sid, win in shipped.items():
            if a.test_root is None:
                keep[sid] = win
                continue
            cd = Path(a.test_root) / sid
            if cd.is_dir() and fp.get(sid) == clip_fingerprint(cd):
                keep[sid] = win
            else:
                rejected.append(sid)
        d = out / "cache" / m["tag"]; d.mkdir(parents=True, exist_ok=True)
        (d / "windows.json").write_text(json.dumps(keep))
        n_live = "n/a" if a.test_root is None else sum(
            1 for c in Path(a.test_root).iterdir() if c.is_dir() and not c.name.startswith(".") and c.name != "__MACOSX") - len(keep)
        print(f"  windows {m['tag']}: {len(shipped)} shipped, {len(keep)} accepted by IR-filename fingerprint, "
              f"{len(rejected)} rejected; clips to be detected live by the packaged detector: {n_live}"
              + (f"  (first rejected: {rejected[:5]})" if rejected else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
