#!/usr/bin/env python3
"""EXP-143 setup: a windows.json that means FULL FRAME, so no trainer change is needed.

Why this exists rather than a --crop full flag. `kaggle/cuhkx_224_kaggle.py` fingerprints
a run with `vars(args)`, so ADDING AN ARGPARSE FIELD INVALIDATES --resume for every job in
flight -- and two 18-task LOSO arrays are running. The cache stage already has the hook:
`_render` does `box = tuple(win) if win else None`, so a FALSY window renders the whole
frame, and `compute_windows` skips any sid already present in windows.json. Writing every
sid -> [] therefore selects full-frame rendering AND skips YOLO, touching no argparse.

The hypothesis (EXP-139/140). Thermal's fusion gain arrived at 224 px on FULL FRAMES, and
the leave-one-out then found the person CROP to be the most droppable member -- consistent
with thermal supplying scene/object context that cropping deletes. 75% of the residual
error is OBJECT classes and EXP-068 measured that the objects are in the pixels. IR+Depth
is our strongest modality (0.7116 cropped) and has never been seen uncropped at 224 px.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import cuhkx_224_kaggle as K  # noqa: E402


def main() -> int:
    cache = Path(sys.argv[1] if len(sys.argv) > 1 else "cache/full224")
    paths = K.find_paths(None, "crop_224", str(cache))
    train_jobs, test_jobs = K.build_jobs(paths)
    sids = [j[0] for j in train_jobs] + [j[0] for j in test_jobs]
    assert len(sids) == len(set(sids)), "duplicate sample id"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "windows.json").write_text(json.dumps({s: [] for s in sids}))
    print(f"wrote {cache/'windows.json'}: {len(sids)} sids, all full-frame "
          f"(train {len(train_jobs)}, test {len(test_jobs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
