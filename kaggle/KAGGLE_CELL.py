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
# T3 / EXP-126 — THE ALL-TRAIN DISTILLED STUDENT: the artifact that ships.
#
# EXP-122 established the effect on fold 2 (Kaggle, seed 1):
#     distil_ctrl  (alpha=0, leak-free) 0.71012      <- baseline seed mean 0.70501
#     distil_fused (alpha=0.7)          0.74387      +3.38 vs ctrl
#     distil_oracle(alpha=0.7, oracle)  0.75920      +4.91 vs ctrl,  +21 OBJECT clips
# Both clear the 3.28 single-fold bar. The absolute numbers are optimistic (the teacher
# saw fold-2's val users), but a student trained on ALL clips and run on TEST is NOT
# leaked -- test subjects appear in no member's training set. So the honest read is a
# submission, and that is what this cell produces.
#
# ⚠⚠ BEFORE RUNNING, TWO SETTINGS. The last two sessions lost every artifact to these.
#
#   1. Session options -> Persistence -> "Files only"  (or "Variables and Files")
#      Without it /kaggle/working is DISCARDED when the session ends. Both previous
#      runs completed successfully and lost their checkpoints this way. The printed
#      numbers survive and look like the whole result; the models do not.
#
#   2. Run it as  Save Version -> "Save & Run All (Commit)",  NOT in this draft session.
#      A draft session is tied to the browser tab -- it stops shortly after you close
#      or lose the tab, which is what kills a multi-hour run. A commit run executes
#      headless on Kaggle's servers and snapshots /kaggle/working as version output.
#
# This cell is RESUMABLE: any run whose oof_/testprobs_ file already exists is skipped.
# With persistence on, a session that dies mid-way costs only the unfinished run.
# ============================================================================
WORKOUT = Path("/kaggle/working")
prior = sorted(p.name for p in WORKOUT.glob("*.npz")) + sorted(p.name for p in WORKOUT.glob("*.pt"))
print(f"/kaggle/working already holds {len(prior)} artifact(s): {prior if prior else '(none)'}")
if prior:
    print("  -> persistence appears to be ON (files survived a previous session). Good.")
else:
    print("  -> EMPTY. If you have run this notebook before, persistence is OFF and the")
    print("     previous outputs were discarded. Fix it now: Session options -> Persistence")
    print("     -> 'Files only', then rerun. Otherwise this run's checkpoints are lost too.")

TAG = "distil_oracle_all"
import numpy as np

# --- train the all-train student -------------------------------------------
# --all-train uses all 18 users, so the printed fold-2 number is TRAIN-ON-TEST and is
# meaningless as validation -- ignore it. EXP-122 already measured the honest fold-2
# effect; this run exists to produce test probabilities.
if (WORKOUT / f"{TAG}.pt").is_file():
    print(f"\n{TAG}.pt exists -- skipping training")
else:
    print(f"\n=========== TRAIN {TAG} (oracle teacher, alpha=0.7, all 18 users) ===========",
          flush=True)
    K.run(stage="train", all_train=True, tag=TAG, cache_dir=str(CROP224),
          teacher=str(TEACHERS / "teacher_oracle.npz"), distill_alpha=0.7,
          distill_temp=2.0, seed=1, batch_size=8, accum=2, workers=2)

# --- test inference ---------------------------------------------------------
if (WORKOUT / f"testprobs_{TAG}.npz").is_file():
    print(f"testprobs_{TAG}.npz exists -- skipping inference")
else:
    print(f"\n=========== INFER {TAG} ===========", flush=True)
    K.run(stage="infer", all_train=True, tag=TAG, cache_dir=str(CROP224),
          teacher=str(TEACHERS / "teacher_oracle.npz"), distill_alpha=0.7,
          distill_temp=2.0, seed=1, batch_size=8, accum=2, workers=2)

print("\n================ EXP-126 OUTPUT ================")
for f in sorted(WORKOUT.iterdir()):
    if f.suffix in (".npz", ".pt", ".csv"):
        print(f"  {f.name:34s} {f.stat().st_size/1e6:8.2f} MB")
print("""
DOWNLOAD BEFORE THE SESSION ENDS -- from the notebook's Output panel, or the version
output if this was a commit run. The two that matter:

  testprobs_distil_oracle_all.npz   <- fuse this at home, then decode and submit
  distil_oracle_all.pt              <- 137 MB; THE PACKAGING CANDIDATE (34.3 MB at int8)

A single 34.3 MB student scoring 0.75920 on fold 2 is approaching the whole five-member
fusion (0.7630). That is the T-PKG win as much as it is a score lead: one architecture,
one modality, one dataset path, and it retires the sklearn ExtraTrees member that
package_ensemble.py cannot represent at all.
""")
