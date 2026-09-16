#!/usr/bin/env python3
"""Clip fingerprints that gate the use of the crop windows shipped inside the package.

The package carries the person/wrist crop windows the packaged detectors produced for the
405 Kaggle test clips, because window computation drifts across GPUs (EXP-158: 22 rows).
Those windows are keyed by clip id, and a held-out dataset in the same layout may reuse the
same ids (SM_test_0001, ...).  A shipped window must therefore be used ONLY when the clip is
the same recording.  The fingerprint is the sha1 of the sorted list of IR frame filenames --
they embed the recording timestamps -- which is exactly the frame list the detector probed.
Different recording, different filenames, no match, live detector.  No pixel is read.

    python3 code/window_fingerprints.py --test-root <Kaggle small_model_track_test> \
        --out inference_assets/window_fingerprints.json
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ALGORITHM = "sha1 of '\\n'.join(sorted IR/*.png filenames); empty IR -> sha1('')"


def clip_fingerprint(clip_dir: Path) -> str:
    ir = clip_dir / "IR"
    names = sorted(p.name for p in ir.iterdir() if p.suffix.lower() == ".png") if ir.is_dir() else []
    return hashlib.sha1("\n".join(names).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.test_root)
    clips = {d.name: clip_fingerprint(d) for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")}
    json.dump({"algorithm": ALGORITHM, "source": "Kaggle Testing set, 2026-09-16", "n_clips": len(clips), "clips": clips},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(clips)} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
