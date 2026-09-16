#!/usr/bin/env python3
"""Build a hostile, label-free test directory from TRAINING clips to exercise inference.sh the
way the Organizing Committee's held-out dataset might: same internal layout, but new
recordings behind reused or new clip ids, missing modalities, odd frames, stray entries, and
a shuffled sample submission.  Real file copies (not symlinks) so the tree can be mounted
read-only into a container.  Nothing here is used for any model.

  python3 code/make_synthetic_testset.py --out /tmp/synth --n 24 [--ids kaggle|new] [--no-csv]

Cases (with --n 24, ids in sorted order):
  #1  empty clip: six empty modality dirs (sorts first)
  #2,#3,#4  no Radar
  #5,#6  no Thermal
  #7,#8  no Skeleton
  #9,#10 no IMU
  #11 no IR (Depth only)
  #12 IR + Depth re-encoded at 320x240
  #13 malformed: a 0-byte IR png, a truncated IMU csv
  rest: complete clips
  + a stray non-clip directory, __MACOSX/, .DS_Store files, and a sample_submission.csv in
    shuffled order (unless --no-csv) laid out as <out>/Testing/{sample_submission.csv,test.csv,data/<root>/}
"""
from __future__ import annotations
import argparse, csv, random, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "Small-Model-Track" / "Training" / "data" / "HAR" / "data"
MODS = ["IR", "Depth_Color", "Thermal", "Skeleton", "IMU", "Radar"]


def pick_trials(n: int, seed: int):
    rng = random.Random(seed)
    trials = []
    for cls in sorted((TRAIN / "IR").iterdir()):
        for user in sorted(cls.iterdir()):
            for trial in sorted(user.iterdir()):
                if all((TRAIN / m / cls.name / user.name / trial.name).is_dir() for m in MODS):
                    trials.append((cls.name, user.name, trial.name))
    rng.shuffle(trials)
    return trials[:n]


def copy_clip(dst: Path, cls: str, user: str, trial: str, mods):
    for m in mods:
        src = TRAIN / m / cls / user / trial
        shutil.copytree(src, dst / m, dirs_exist_ok=True)


def downscale(clip: Path):
    from PIL import Image
    for m in ("IR", "Depth_Color"):
        for p in sorted((clip / m).glob("*.png")):
            im = Image.open(p); im = im.resize((320, 240)); im.save(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ids", choices=("kaggle", "new"), default="kaggle",
                    help="kaggle: reuse SM_test_0001.. (the shipped-window trap); new: clip_A01..")
    ap.add_argument("--root-name", default="small_model_track_test")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    root = out / "Testing" / "data" / a.root_name
    root.mkdir(parents=True)
    trials = pick_trials(a.n, a.seed)
    ids = [f"SM_test_{i:04d}" if a.ids == "kaggle" else f"clip_{chr(65 + i // 26)}{i % 26 + 1:02d}" for i in range(1, a.n + 1)]
    truth = []
    for i, (sid, (cls, user, trial)) in enumerate(zip(ids, trials), start=1):
        d = root / sid; d.mkdir()
        if i == 1:
            for m in MODS:
                (d / m).mkdir()
            truth.append((sid, cls, "empty")); continue
        mods = list(MODS)
        tag = "complete"
        if i in (2, 3, 4): mods.remove("Radar"); tag = "no-radar"
        elif i in (5, 6): mods.remove("Thermal"); tag = "no-thermal"
        elif i in (7, 8): mods.remove("Skeleton"); tag = "no-skeleton"
        elif i in (9, 10): mods.remove("IMU"); tag = "no-imu"
        elif i == 11: mods.remove("IR"); tag = "no-ir"
        copy_clip(d, cls, user, trial, mods)
        if i == 12:
            downscale(d); tag = "320x240"
        if i == 13:
            first = sorted((d / "IR").glob("*.png"))[0]; first.write_bytes(b"")
            for c in (d / "IMU").glob("*.csv"):
                c.write_bytes(c.read_bytes()[:200]); break
            tag = "malformed"
        (d / ".DS_Store").write_bytes(b"\x00" * 8)
        truth.append((sid, cls, tag))
    (root / "not_a_clip").mkdir(); (root / "not_a_clip" / "readme.txt").write_text("stray\n")
    (root / "__MACOSX").mkdir(); (root / "__MACOSX" / "._x").write_bytes(b"\x00")
    (root / ".DS_Store").write_bytes(b"\x00" * 8)
    with open(out / "truth_for_humans_only.csv", "w", newline="") as h:
        w = csv.writer(h); w.writerow(["sid", "train_class_dir", "case"]); w.writerows(truth)
    if not a.no_csv:
        rows = [f"{a.root_name}/{sid}/" for sid in ids]
        random.Random(a.seed + 1).shuffle(rows)
        with open(out / "Testing" / "sample_submission.csv", "w", newline="") as h:
            w = csv.writer(h); w.writerow(["path", "prediction"]); w.writerows([r, 0] for r in rows)
        with open(out / "Testing" / "test.csv", "w", newline="") as h:
            w = csv.writer(h); w.writerow(["path"]); w.writerows([r] for r in rows)
    print(f"wrote {out}: {a.n} clips under {root} (ids={a.ids}, csv={'no' if a.no_csv else 'shuffled'})")
    for t in truth:
        print("  ", *t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
