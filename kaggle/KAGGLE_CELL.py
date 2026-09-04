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
    """Rebuild cache/<name>/ from the flattened <prefix>_* files, via symlinks.

    Searches ALL of /kaggle/input, not one dataset directory. Each cache lives in its
    own Kaggle dataset -- crop_224 and thermal_full in `cuhkx-smt-derived-caches`,
    crop_wrist224 in `cuhkx-wrist-cache` -- and an earlier version of this globbed
    inside the first dataset only, so an attached wrist cache reported "0 files" and
    the run died at a SystemExit that blamed the sidebar. The prefixes do not collide:
    `crop224_*` does not match `cropwrist224_*`.
    """
    d = WORK / "cache" / name
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in find(f"{prefix}_*"):
        dst = d / Path(src).name[len(prefix) + 1:]
        if not dst.exists():
            os.symlink(src, dst)
        n += 1
    print(f"  {name}: {n} files -> {d}")
    return d

CROP224 = unflatten("crop224", "crop_224")
THERMAL = unflatten("thermalfull", "thermal_full")

# ---- 3. import the trainer, defeating BOTH staleness traps ---------------
# There are two independent ways to end up running OLD trainer code here, and they
# look identical from the outside: argparse prints "unrecognized arguments" a few
# lines into what should be a multi-hour run.
#
#   Trap 1 — Kaggle pins a dataset VERSION. Pushing a new version of cuhkx-repo does
#            nothing until this notebook's input is refreshed to point at it.
#   Trap 2 — a re-run cell does NOT re-import. `import x` returns the object already
#            in sys.modules, so after `shutil.rmtree(REPO)` + copytree replaces the
#            file on disk, the NEW file is on disk and the OLD code is still in
#            memory. Refreshing the dataset input mid-session lands you here.
#
# Trap 2 is why a file-text guard is not enough: it reads the (new) file while K is
# the (old) module. Check the LOADED object's bytecode instead.
import importlib
sys.path.insert(0, str(REPO / "kaggle"))
os.chdir(REPO)

for _stale in [m for m in list(sys.modules) if m.split(".")[0] == "cuhkx_224_kaggle"]:
    del sys.modules[_stale]
importlib.invalidate_caches()
import cuhkx_224_kaggle as K

for _need in ("--seed", "--teacher", "--distill-alpha"):
    if _need not in K.main.__code__.co_consts:
        raise SystemExit(
            f"The trainer in memory has no {_need}.\n"
            "The forced re-import above rules out a stale module, so this is Trap 1: the\n"
            "mounted cuhkx-repo is an OLD VERSION.\n"
            "Fix: sidebar -> cuhkx-repo -> refresh/update to the latest version (or remove\n"
            "and re-add it), then Run -> Restart session, and rerun this cell.\n"
            f"Mounted copy: {K.__file__}")
TEACHERS = REPO / "research" / "artifacts"
assert (TEACHERS / "teacher_fused.npz").is_file(), \
    f"teacher targets missing from the dataset: {sorted(p.name for p in TEACHERS.iterdir())}"
print(f"trainer OK (--seed, --teacher, --distill-alpha present): {K.__file__}")

# ============================================================================
# EXP-130 — A DISTILLED WRIST STUDENT: upgrade a slot without disturbing what works.
#
# ⚠ NEEDS A FOURTH INPUT: Add Input -> Datasets -> "cuhkx-wrist-cache".
# ⚠ Session options -> Persistence -> "Files only", and run via
#    Save Version -> "Save & Run All (Commit)". Two sessions have already lost their
#    checkpoints; the compute succeeded both times and only the files died.
#
# WHY THIS RUN AND NOT ANOTHER. Today's three submissions taught two things that
# together pick this experiment:
#
#   * EXP-128: swapping the sklearn `imu_stats` (accuracy 0.3605) for the far more
#     accurate student (~0.73) at the same 0.3575 weight LOST 4 clips. That member earns
#     the heaviest slot in the fusion through error PLACEMENT, not accuracy. So do not
#     touch it.
#   * EXP-127/128: the person view is redundant with a person-crop-distilled student --
#     dropping it GAINED 2 clips. Adding video members is a documented graveyard.
#
# So the move is neither "add a member" nor "replace the IMU branch". It is to UPGRADE an
# existing video slot in place: the champion's wrist view is `k224_mvitwrist_all` at about
# 0.7065, and distillation lifted the person member by +4.91 over a leak-free control
# (EXP-122). A wrist student is trained on a DIFFERENT crop, so it should not inherit the
# person student's redundancy.
#
# PREDICTION, recorded before the run: the champion with its wrist view replaced scores
# 164-171, centre 168. Honest EV is modest -- the video slot is a graveyard and two
# video-slot changes lost clips today -- but this one keeps imu_stats and swaps a member
# for a stronger version of itself rather than adding to the bag.
#
# Also a package upgrade: the shipped 93.37 MB file carries this same wrist checkpoint,
# so a better one improves Stage 2 as well as the leaderboard.
# ============================================================================
WRISTC = unflatten("cropwrist224", "crop_wrist224")
if not (WRISTC / "train_index.json").is_file():
    raise SystemExit(
        "cuhkx-wrist-cache is not attached.\n"
        "Sidebar -> Add Input -> Datasets -> 'cuhkx-wrist-cache', then rerun.\n"
        "What is mounted:\n" + _tree())

TAG = "distil_oracle_wrist_all"
import numpy as np
if (WORKOUT / f"{TAG}.pt").is_file():
    print(f"\n{TAG}.pt exists -- skipping training")
else:
    print(f"\n=========== TRAIN {TAG} (oracle teacher, alpha=0.7, wrist crop) ===========",
          flush=True)
    K.run(stage="train", all_train=True, tag=TAG, cache_dir=str(WRISTC), crop="wrist",
          teacher=str(TEACHERS / "teacher_oracle.npz"), distill_alpha=0.7,
          distill_temp=2.0, seed=1, batch_size=8, accum=2, workers=2)

if (WORKOUT / f"testprobs_{TAG}.npz").is_file():
    print(f"testprobs_{TAG}.npz exists -- skipping inference")
else:
    print(f"\n=========== INFER {TAG} ===========", flush=True)
    K.run(stage="infer", all_train=True, tag=TAG, cache_dir=str(WRISTC), crop="wrist",
          teacher=str(TEACHERS / "teacher_oracle.npz"), distill_alpha=0.7,
          distill_temp=2.0, seed=1, batch_size=8, accum=2, workers=2)

print("\n================ EXP-130 OUTPUT ================")
for f in sorted(WORKOUT.iterdir()):
    if f.suffix in (".npz", ".pt", ".csv"):
        print(f"  {f.name:36s} {f.stat().st_size/1e6:8.2f} MB")
print("""
DOWNLOAD BEFORE THE SESSION ENDS:
  testprobs_distil_oracle_wrist_all.npz   <- fuse at home as the wrist view, then decode
  distil_oracle_wrist_all.pt              <- 137 MB; replaces the wrist member in the
                                             93.37 MB Stage-2 package
""")
