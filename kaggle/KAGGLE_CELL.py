# ============================================================================
# CUHK-X — Kaggle notebook bootstrap.  Paste this as the FIRST cell.
#
# BEFORE RUNNING, attach the data in the notebook sidebar:
#   Add Input -> Datasets -> search "cuhkx-smt-derived-caches" -> Add
#   (and keep the competition attached, for sample_submission.csv)
#
# Why this cell exists: Kaggle mounts ONLY sample_submission.csv and test.csv for
# this competition -- the raw IR/Depth/Thermal trees are not there. So the cache
# cannot be built on Kaggle; it is built locally and uploaded as a dataset. Our
# upload is flattened (crop224_train.bin, thermalfull_train.npy, ...) because
# `kaggle datasets create` skips subfolders, so this cell rebuilds the directory
# layout the trainer expects, using symlinks (no copying, no extra disk).
# ============================================================================
import os, sys, glob, json, shutil
from pathlib import Path

DSET = next(iter(glob.glob("/kaggle/input/cuhkx-smt-derived-caches")), None)
if DSET is None:
    hits = glob.glob("/kaggle/input/*/crop224_train.bin")
    DSET = str(Path(hits[0]).parent) if hits else None
assert DSET, ("dataset not attached. Sidebar -> Add Input -> Datasets -> "
              "'cuhkx-smt-derived-caches'. Mounted now: "
              + str(sorted(os.listdir("/kaggle/input"))))
print("dataset:", DSET)

WORK = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/kaggle/working")
CACHE = WORK / "cache"

def unflatten(prefix: str, name: str) -> Path:
    """Rebuild cache/<name>/ from the flattened <prefix>_* files, via symlinks."""
    d = CACHE / name
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(glob.glob(f"{DSET}/{prefix}_*")):
        dst = d / Path(src).name[len(prefix) + 1:]
        if not dst.exists():
            os.symlink(src, dst)
        n += 1
    print(f"  {name}: {n} files -> {d}")
    return d

CROP224 = unflatten("crop224", "crop_224")
THERMAL = unflatten("thermalfull", "thermal_full")

# The repo. Attach it as a dataset too, or clone it if the notebook has internet on.
REPO = next((p for p in ["/kaggle/input/cuhkx-repo", "/kaggle/working/cuhkx"]
             if Path(p, "kaggle/cuhkx_224_kaggle.py").is_file()), None)
if REPO is None:
    raise SystemExit(
        "Repo not found. Either attach it as a dataset named 'cuhkx-repo', or with\n"
        "internet enabled run:  !git clone <your-repo-url> /kaggle/working/cuhkx")
sys.path.insert(0, str(Path(REPO, "kaggle")))
os.chdir(REPO)
print("repo:", REPO)

import cuhkx_224_kaggle as K

# ---------------------------------------------------------------------------
# Train.  --cache-dir is the important part: it lets train/infer run WITHOUT the
# raw IR/depth trees, which is the error this cell fixes.
#
# Kaggle gives a 16 GB card (T4/P100) against the 8 GB laptop, so the batch size
# can go up. Keep accum so the EFFECTIVE batch stays 16, matching every previous
# run -- otherwise the OneCycleLR schedule changes and the comparison is confounded.
# ---------------------------------------------------------------------------
K.run(stage="train", fold=2, tag="k224_mvit_f2_kaggle",
      cache_dir=str(CROP224), batch_size=8, accum=2, workers=2)
