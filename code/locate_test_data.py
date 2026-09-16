#!/usr/bin/env python3
"""Find the clip directory and the submission order inside whatever <data_dir> the organisers
mount.  Prints shell assignments for inference.sh:

    TEST_ROOT=/data/Testing/data/small_model_track_test
    SAMPLE_SUB=/data/Testing/sample_submission.csv        (or a generated one under --work)
    N_CLIPS=405

Rules (Selection Stage guidelines: "the official label-free Testing data in the same internal
layout as the published challenge dataset", mounted at a path of their choosing):
  * a CLIP directory holds at least one of IR/ Depth_Color/ Thermal/ Skeleton/ IMU/ Radar/;
  * the TEST ROOT is the directory with the most clip children, found by breadth-first
    search to depth 4 from <data_dir>, skipping hidden names and __MACOSX, never descending
    into a clip; ties go to a directory named small_model_track_test, then the shallowest;
  * sample_submission.csv is looked for at TEST_ROOT/../.., TEST_ROOT/.., TEST_ROOT,
    <data_dir>, then breadth-first to depth 3; the clip id is the last path component of
    its `path` column.  If none exists, one is written under --work in sorted id order and
    a WARNING is printed -- the row order of the output is then sorted-id order.
  * every id in the csv must have a directory (exit 2 listing the missing ones); directories
    absent from the csv are reported and dropped at write time.
Nothing here reads a frame or writes inside <data_dir>.
"""
from __future__ import annotations
import argparse, csv, shlex, sys
from collections import deque
from pathlib import Path

MODS = ("IR", "Depth_Color", "Thermal", "Skeleton", "IMU", "Radar")
SKIP = {"__MACOSX"}


def is_clip(d: Path) -> bool:
    try:
        return d.is_dir() and any((d / m).is_dir() for m in MODS)
    except OSError:
        return False


def children(d: Path):
    try:
        return sorted(c for c in d.iterdir() if c.is_dir() and not c.name.startswith(".") and c.name not in SKIP)
    except OSError:
        return []


def find_test_root(data_dir: Path, max_depth: int = 4):
    best = None  # (n_clips, named_bonus, -depth, path)
    q = deque([(data_dir, 0)])
    while q:
        d, depth = q.popleft()
        kids = children(d)
        clips = [c for c in kids if is_clip(c)]
        if clips:
            key = (len(clips), int(d.name == "small_model_track_test"), -depth)
            if best is None or key > best[0]:
                best = (key, d, clips)
        if depth < max_depth:
            for c in kids:
                if not is_clip(c):
                    q.append((c, depth + 1))
    return (best[1], best[2]) if best else (None, [])


def find_sample_sub(test_root: Path, data_dir: Path, max_depth: int = 3):
    for c in (test_root.parent.parent, test_root.parent, test_root, data_dir):
        p = c / "sample_submission.csv"
        if p.is_file():
            return p
    q = deque([(data_dir, 0)])
    while q:
        d, depth = q.popleft()
        p = d / "sample_submission.csv"
        if p.is_file():
            return p
        if depth < max_depth:
            for c in children(d):
                q.append((c, depth + 1))
    return None


def read_ids(sample_sub: Path):
    with open(sample_sub, newline="") as h:
        rows = list(csv.reader(h))
    if not rows or len(rows[0]) < 1:
        sys.exit(f"ERROR: {sample_sub} is empty")
    body = rows[1:]
    ids = []
    for r in body:
        if not r or not r[0].strip():
            continue
        sid = r[0].strip().strip("/").split("/")[-1]
        if not sid:
            sys.exit(f"ERROR: empty clip id in {sample_sub}: {r!r}")
        ids.append(sid)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--work", required=True, help="where to write a generated sample_submission.csv if none exists")
    a = ap.parse_args()
    data_dir = Path(a.data_dir).resolve()
    if not data_dir.is_dir():
        sys.exit(f"ERROR: data dir does not exist: {data_dir}")
    test_root, clips = find_test_root(data_dir)
    if test_root is None:
        listing = "\n".join(f"  {p}" for p in sorted(data_dir.rglob("*"))[:60])
        sys.exit(f"ERROR: no clip directories (holding IR/ Depth_Color/ Thermal/ Skeleton/ IMU/ Radar/) "
                 f"within depth 4 of {data_dir}. First entries:\n{listing}")
    clip_ids = {c.name for c in clips}
    non_clip = [c.name for c in children(test_root) if not is_clip(c)]
    sample_sub = find_sample_sub(test_root, data_dir)
    if sample_sub is None:
        gen = Path(a.work) / "sample_submission.csv"
        gen.parent.mkdir(parents=True, exist_ok=True)
        with open(gen, "w", newline="") as h:
            w = csv.writer(h); w.writerow(["path", "prediction"])
            for sid in sorted(clip_ids):
                w.writerow([f"{test_root.name}/{sid}/", 0])
        print(f"WARNING: no sample_submission.csv found under {data_dir}; wrote {gen} in sorted clip-id order "
              f"({len(clip_ids)} clips). The output row order is therefore sorted-id order.", file=sys.stderr)
        sample_sub = gen
    ids = read_ids(sample_sub)
    missing = [s for s in ids if s not in clip_ids]
    if missing:
        sys.exit(f"ERROR: {len(missing)} clip id(s) listed in {sample_sub} have no directory under {test_root}: "
                 f"{missing[:20]}{' ...' if len(missing) > 20 else ''}")
    extra = sorted(clip_ids - set(ids))
    if extra:
        print(f"WARNING: {len(extra)} clip directorie(s) under {test_root} are not in {sample_sub.name} and will be "
              f"ignored in the output: {extra[:20]}{' ...' if len(extra) > 20 else ''}", file=sys.stderr)
    if non_clip:
        print(f"note: ignoring {len(non_clip)} non-clip entr(ies) under {test_root}: {non_clip[:10]}", file=sys.stderr)
    dup = len(ids) - len(set(ids))
    if dup:
        sys.exit(f"ERROR: {dup} duplicate clip id(s) in {sample_sub}")
    print(f"TEST_ROOT={shlex.quote(str(test_root))}")     # shell-quoted: paths may contain spaces
    print(f"SAMPLE_SUB={shlex.quote(str(sample_sub))}")
    print(f"N_CLIPS={len(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
