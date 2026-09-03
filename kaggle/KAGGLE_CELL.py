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

# ⚠ SET Persistence = "Files only" IN Session options BEFORE RUNNING.
# EXP-120a's three checkpoints were lost because a draft session wipes /kaggle/working
# at session end. The printed numbers survived and looked like the whole result; the
# models did not. Better still: Save Version -> Save & Run All (Commit).

# ============================================================================
# T3 / EXP-122 — DISTIL THE FIVE-MEMBER ENSEMBLE INTO ONE STUDENT
#
# The target. Over the five members, fused accuracy is 0.763 while oracle-any-member
# is 0.877 -- 309 clips of 2,700. A global geometric weight must pick ONE compromise
# for every clip; a soft target carries each member's full posterior on THAT clip, so
# the student can learn a per-clip arbitration no scalar weight can express. Fitted
# arbitration has failed 4/4 on public, but all four fit in PROBABILITY space on 2,700
# rows; this fits in FEATURE space against dense 40-dim targets, which is B-028's
# distinction and the reason it is not simply a fifth attempt at the same thing.
#
# It is also the packaging fix. One student is one architecture, one modality, one
# dataset path -- it retires the sklearn ExtraTrees IMU member that package_ensemble.py
# cannot represent at all, and the closed model registry. T-PKG is the only failure
# mode that costs the whole competition, and this is the cheapest route through it.
#
# THREE RUNS, and the third is the control that makes the first two readable:
#
#   ctrl   alpha=0.0  -> teacher loaded (so the SAME 2,148 clips) but zero weight on it.
#                        Isolates the cost of dropping 133 clips that lack a teacher
#                        entry from the effect of distillation itself. Without this,
#                        student-vs-baseline confounds objective with training-set size.
#   fused  alpha=0.7  -> imitate the champion mix (target top-1 0.765)
#   oracle alpha=0.7  -> imitate, per clip, only the members that were RIGHT there
#                        (target top-1 0.880). This is the selection hypothesis stated
#                        as sharply as it can be: if the inputs carry enough signal to
#                        tell those cases apart, the oracle gap is reachable.
#
# ⚠ HOW TO READ THE NUMBERS -- two limits, stated before the run, not after.
#
#  1. ABSOLUTE fold-2 numbers are OPTIMISTIC. The teacher targets for fold-2 TRAINING
#     clips were produced by member models that trained on fold-2's VALIDATION users
#     (each pooled OOF entry comes from the one fold model that held that clip out --
#     and that model saw fold 2's users). So knowledge of the val users leaks through
#     the teacher into the student. `ctrl` has alpha=0 and is therefore leak-free,
#     which is a second reason it earns its 2.2 h.
#  2. ONE FOLD IS UNDERPOWERED. Seed sigma is 1.16, a paired single-fold delta carries
#     1.64, so the 2-SE bar here is 3.28 (EXP-120a). This run can only detect a LARGE
#     effect. `oracle - fused` is the cleanest contrast on offer (identical clips,
#     identical leak structure, one changed factor); anything under ~3 points is a lead
#     to replicate on more folds, NOT a result.
# ============================================================================
RUNS = [
    ("distil_ctrl",   "teacher_fused.npz",  0.0),
    ("distil_fused",  "teacher_fused.npz",  0.7),
    ("distil_oracle", "teacher_oracle.npz", 0.7),
]
BASELINE = 0.70501        # k224_mvit_f2 seed MEAN, n=3 (EXP-120a). NOT 0.71472, which
                          # is the highest of five draws and biases every delta by -1.

import numpy as np
got = {}
for tag, teacher, alpha in RUNS:
    print(f"\n=========== {tag}  (teacher={teacher}, alpha={alpha}) ===========", flush=True)
    K.run(stage="train", fold=2, tag=tag, cache_dir=str(CROP224),
          teacher=str(TEACHERS / teacher), distill_alpha=alpha, distill_temp=2.0,
          seed=1, batch_size=8, accum=2, workers=2)
    d = np.load(f"/kaggle/working/oof_{tag}.npz", allow_pickle=True)
    got[tag] = float((d["probs"].argmax(1) == d["labels"]).mean())
    print(f"  {tag}: micro={got[tag]:.5f}", flush=True)

print("\n================ EXP-122 RESULT ================")
for k, v in got.items():
    print(f"  {k:16s} {v:.5f}   {100*(v-BASELINE):+.2f} vs baseline seed mean")
if "distil_ctrl" in got:
    c = got["distil_ctrl"]
    print(f"\n  cost of dropping 133 clips : {100*(c-BASELINE):+.2f}  (ctrl vs baseline, leak-free)")
    for k in ("distil_fused", "distil_oracle"):
        if k in got:
            print(f"  {k:16s} vs ctrl      : {100*(got[k]-c):+.2f}  (distillation effect, OPTIMISTIC)")
if {"distil_fused", "distil_oracle"} <= set(got):
    print(f"  oracle vs fused target     : {100*(got['distil_oracle']-got['distil_fused']):+.2f}"
          "  <- the CLEAN contrast: same clips, same leak, one factor")
print("\n  2-SE bar on one fold is 3.28 points. Under that = lead, not result.")
