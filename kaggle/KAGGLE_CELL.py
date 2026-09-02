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

# The notebook pins a dataset VERSION. Editing the trainer locally and pushing a new
# version does nothing until this notebook is pointed at it -- and the failure mode is
# an argparse "unrecognized arguments: --seed" dump 3 lines into a 6-hour run. Fail
# here instead, with the fix in the message.
if "\"--seed\"" not in Path(K.__file__).read_text():
    raise SystemExit(
        "The mounted cuhkx-repo is an OLD VERSION: its cuhkx_224_kaggle.py has no "
        "--seed.\n"
        "Fix: sidebar -> the cuhkx-repo input -> refresh/update it to the latest "
        "version\n"
        "(or remove and re-add it), then restart the session and rerun this cell.\n"
        f"Mounted copy: {K.__file__}")

# EXP-120a — MEASURE THE VISUAL BRANCH'S SEED SIGMA.  It has never been measured
# (research/LOG.md:2069, :2161); it cost 9.3 h per seed on the laptop and costs ~2 h
# here.  The 2.80 that every visual gate in this campaign quotes is a *fold* sigma;
# the skeleton branch, which actually measured itself, has a seed sigma of 0.18.
#
# This also settles the parity question.  The old gate demanded this run reproduce
# micro = 0.71472 exactly.  It cannot: the trainer seeds NOTHING, so 0.71472 is one
# draw and the laptop fails its own gate too.  Kaggle's first run returned 0.68252
# with the split, params, lr schedule, step count and loss curve all matching, and
# with gross_motion identical to five decimals -- a weaker draw, not a broken
# environment.  Replicate under --seed and ask whether 0.71472 sits inside the spread.
#
# --cache-dir is what lets this run at all: Kaggle mounts only sample_submission.csv
# for this competition, never the raw IR/Depth trees.
#
# batch_size 8 x accum 2 keeps steps = len(tl)//accum = 142, identical to the
# laptop's bs4 x accum4.  The OneCycleLR schedule is therefore unchanged -- confirmed
# by epoch-1 lr matching at 5.23e-05 -- so the bigger batch is NOT a confound.
#
# Outputs land in /kaggle/working (persisted, downloadable); /kaggle/temp is not.
# ~2 h per seed.  If the session dies partway, the finished seeds are still there and
# two replicates already bound the spread usefully -- just rerun the missing one.
REF_LAPTOP = 0.71472      # laptop k224_mvit_f2, EXP-102 -- one unseeded draw
REF_KAGGLE = 0.68252      # this notebook, unseeded

import numpy as np
got = {}
for seed in (1, 2, 3):
    tag = f"k224_mvit_f2_s{seed}"
    print(f"\n=========== seed {seed} -> {tag} ===========", flush=True)
    K.run(stage="train", fold=2, tag=tag, cache_dir=str(CROP224),
          batch_size=8, accum=2, workers=2, seed=seed)
    d = np.load(f"/kaggle/working/oof_{tag}.npz", allow_pickle=True)
    got[seed] = float((d["probs"].argmax(1) == d["labels"]).mean())
    print(f"  seed {seed}: micro={got[seed]:.5f}", flush=True)

m = np.array(list(got.values()))
print("\n================ EXP-120a RESULT ================")
for s, v in got.items():
    print(f"  seed {s}: {v:.5f}")
if len(m) >= 2:
    mu, sd = float(m.mean()), float(m.std(ddof=1))
    print(f"  mean {mu:.5f}   VISUAL SEED SIGMA = {100*sd:.2f} points")
    print(f"  2sd band [{mu-2*sd:.5f}, {mu+2*sd:.5f}]")
    print(f"  laptop {REF_LAPTOP:.5f} inside: {mu-2*sd <= REF_LAPTOP <= mu+2*sd}")
    print(f"  kaggle unseeded {REF_KAGGLE:.5f} inside: {mu-2*sd <= REF_KAGGLE <= mu+2*sd}")
    print("  If both are inside, parity is settled and sigma replaces the inherited")
    print("  2.80 as the adoption bar for every visual member-strength change.")

# To bring results home: the files are already in /kaggle/working, so use the
# notebook's Output tab, or:
#   from IPython.display import FileLink; FileLink('/kaggle/working/oof_k224_mvit_f2_s1.npz')
