# ============================================================================
# CUHK-X — Kaggle notebook bootstrap.  Paste this as the FIRST cell.
#
# BEFORE RUNNING, attach three inputs in the notebook sidebar (Add Input):
#   1. Competition -> CUHK-X Competition Small Model Track   (sample_submission.csv)
#   2. Dataset     -> cuhkx-smt-derived-caches               (the prebuilt caches)
#   3. Dataset     -> cuhkx-repo                             (the code tarball)
#
# Why this cell exists: Kaggle mounts ONLY sample_submission.csv and test.csv for
# this competition -- the raw IR/Depth/Thermal trees are not there, so the cache
# cannot be built here. It is built locally and uploaded. Both uploads are flat
# (crop224_train.bin, cuhkx_code.tar.gz, ...) because `kaggle datasets create`
# skips subfolders; this cell restores the layout the trainer expects.
# ============================================================================
import os, sys, glob, json, shutil, tarfile
from pathlib import Path

WORK = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/kaggle/working")

def find(pattern: str):
    """Search /kaggle/input recursively.

    Kaggle does not use one fixed layout. Some sessions mount datasets flat at
    /kaggle/input/<slug>/, others nest them as /kaggle/input/datasets/<user>/<slug>/
    with competitions under /kaggle/input/competitions/<slug>/. A one-level glob
    silently misses the nested form and reports "not attached" when it plainly is,
    so match at any depth instead of guessing the layout.
    """
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))

def _tree(root="/kaggle/input", depth=3):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        d = dirpath[len(root):].count(os.sep)
        if d >= depth:
            dirnames[:] = []
        out.append(f"    {dirpath}  [{len(dirnames)} dirs, {len(filenames)} files]")
    return "\n".join(out[:40])

# ---- 1. code -------------------------------------------------------------
# Kaggle AUTO-EXTRACTS uploaded archives, so cuhkx_code.tar.gz may not exist as a file
# -- the dataset lists code/*.py and kaggle/*.py directly. Handle both, and copy the
# tree somewhere writable: /kaggle/input is read-only and the trainer writes beside cwd.
REPO = WORK / "cuhkx"
if REPO.exists():
    shutil.rmtree(REPO)

extracted = find("cuhkx_224_kaggle.py")
tarballs = find("cuhkx_code.tar.gz")
if extracted:
    src = Path(extracted[0]).parent.parent          # <root>/kaggle/cuhkx_224_kaggle.py
    shutil.copytree(src, REPO)
    print(f"code: copied extracted dataset {src} -> {REPO}")
elif tarballs:
    REPO.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarballs[0]) as t:
        t.extractall(REPO)
    print(f"code: extracted {tarballs[0]} -> {REPO}")
else:
    raise SystemExit(
        "cuhkx-repo not found under /kaggle/input.\n"
        "If you JUST uploaded it, Kaggle may still be processing -- the dataset can\n"
        "mount with 0 files. Wait for 'ready to use', then use the refresh icon next to\n"
        "the dataset in the sidebar (or restart the session) so it re-mounts.\n"
        "What is mounted:\n" + _tree())
assert (REPO / "kaggle" / "cuhkx_224_kaggle.py").is_file(), \
    f"unexpected layout under {REPO}: {sorted(p.name for p in REPO.iterdir())}"

# ---- 2. caches -----------------------------------------------------------
cache_hits = find("crop224_train.bin")
assert cache_hits, ("cuhkx-smt-derived-caches not found under /kaggle/input.\n"
                    "Sidebar -> Add Input -> Datasets -> 'cuhkx-smt-derived-caches'.\n"
                    "What is mounted:\n" + _tree())
DSET = str(Path(cache_hits[0]).parent)
print("caches:", DSET)

def unflatten(prefix: str, name: str) -> Path:
    """Rebuild cache/<name>/ from the flattened <prefix>_* files, via symlinks."""
    d = WORK / "cache" / name
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(glob.glob(f"{DSET}/{prefix}_*")):  # DSET is now an exact dir
        dst = d / Path(src).name[len(prefix) + 1:]
        if not dst.exists():
            os.symlink(src, dst)
        n += 1
    print(f"  {name}: {n} files -> {d}")
    return d

CROP224 = unflatten("crop224", "crop_224")
THERMAL = unflatten("thermalfull", "thermal_full")

# ---- 3. go ---------------------------------------------------------------
sys.path.insert(0, str(REPO / "kaggle"))
os.chdir(REPO)
import cuhkx_224_kaggle as K

# --cache-dir is the important part: it lets train/infer run WITHOUT the raw
# IR/depth trees, which is the error this cell fixes.
#
# Kaggle gives a 16 GB card (T4/P100) against the 8 GB laptop, so batch_size can go
# up -- but accum drops to match, keeping the EFFECTIVE batch at 16 as in every
# previous run. Otherwise the OneCycleLR schedule shifts and every comparison
# against our existing folds is confounded.
K.run(stage="train", fold=2, tag="k224_mvit_f2_kaggle",
      cache_dir=str(CROP224), batch_size=8, accum=2, workers=2)

# Then, to bring the result home:
#   from IPython.display import FileLink; FileLink('oof_k224_mvit_f2_kaggle.npz')
#
# Reference to reproduce (parity check before trusting any Kaggle number):
#   k224_mvit_f2  micro = 0.71472   (laptop, EXP-102)
