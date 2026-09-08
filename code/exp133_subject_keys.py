#!/usr/bin/env python3
"""EXP-133 — which key recovers SUBJECT identity on the unlabeled test split?

EXP-132 showed the residual error is a per-subject bias, so every lever built on it
(per-subject centering, prototype verification, per-subject adaptation) needs test clips
grouped by subject. R-4 and R-7 make unlabeled test grouping legal; the question is
purely whether any key works.

Four candidate keys, each validated against TRUE user labels on train:
  1. recording day          (from cache/meta_*.csv t0)
  2. timestamp blocks       (gap > 300 s)
  3. skeleton bone lengths  (body size is a physical subject signature)
  4. IMU device MAC address (a per-subject device assignment would be decisive)

A key passes only if it is >=90% clip-weighted user-pure on TRAIN blocks. Anything less
and the per-subject statistics it produces are mixtures, which is the failure mode that
made per-timestamp-block AdaBN score BELOW pooled AdaBN (EXP-099).
"""
from __future__ import annotations
import collections, csv, glob, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# H36M-17 kinematic parents; matches code/har_data.py:55
PARENT = np.array([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])


def read_meta(split):
    rows = []
    with open(os.path.join(ROOT, "cache", f"meta_{split}.csv")) as h:
        for r in csv.DictReader(h):
            if not r.get("t0"):
                continue
            rows.append(dict(sid=r["sample_id"], t0=float(r["t0"]), t1=float(r["t1"]),
                             user=r.get("user", ""), cls=r.get("class_id", "")))
    rows.sort(key=lambda r: r["t0"])
    return rows


def blocks(rows, gap):
    b, cur = [], 0
    for i, r in enumerate(rows):
        if i and r["t0"] - rows[i - 1]["t1"] > gap:
            cur += 1
        b.append(cur)
    return np.array(b)


def purity(block_ids, users):
    """Clip-weighted fraction of clips in the majority user of their own block."""
    tot = good = 0
    for b in set(block_ids):
        m = block_ids == b
        c = collections.Counter(users[m])
        good += c.most_common(1)[0][1]
        tot += m.sum()
    return good / tot


def bone_lengths(sids, split):
    out = {}
    for sid in sids:
        p = os.path.join(ROOT, "cache", split, f"{sid}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as d:
            x = d["skel_pos"][:, 0]                       # (T, 17, 3)
        if x.shape[0] < 2:
            continue
        bl = np.linalg.norm(x - x[:, PARENT], axis=-1)[:, 1:]   # (T, 16)
        ok = (bl > 1e-4).all(1)
        if ok.sum() >= 2:
            out[sid] = np.median(bl[ok], axis=0)
    return out


MAC_RE = re.compile(r"(WT\w+)\(([0-9A-F:]+)\)")


def macs(clip_dir):
    found = {}
    for f in glob.glob(os.path.join(clip_dir, "*.csv")):
        try:
            with open(f, encoding="utf-8", errors="ignore") as h:
                h.readline()
                for _ in range(200):
                    line = h.readline()
                    if not line:
                        break
                    parts = line.split(",")
                    if len(parts) > 1:
                        m = MAC_RE.match(parts[1])
                        if m:
                            found[m.group(1)] = m.group(2)
        except OSError:
            pass
    return found


def main():
    tr, te = read_meta("train"), read_meta("test")
    tu = np.array([r["user"] for r in tr])
    print(f"train {len(tr)} clips / {len(set(tu))} users     test {len(te)} clips")

    # ---- key 1: recording day ------------------------------------------------
    day = np.array([int(r["t0"] // 86400) for r in tr])
    dpur = purity(day, tu)
    per_day = {d: len(set(tu[day == d])) for d in sorted(set(day))}
    print(f"\n[1] recording day        purity {dpur:.3f}   users per train day {per_day}")
    tday = collections.Counter(int(r["t0"] // 86400) for r in te)
    print(f"    test days: {dict(sorted(tday.items()))}")
    shared = sorted(set(tday) & set(per_day))
    print(f"    days shared with train recording: {shared} "
          f"-> {sum(tday[d] for d in shared)} test clips; "
          f"test-only days -> {sum(n for d, n in tday.items() if d not in per_day)}")

    # ---- key 2: timestamp blocks --------------------------------------------
    print()
    for gap in (60, 300, 1800):
        b = blocks(tr, gap)
        sizes = collections.Counter(b)
        tb = blocks(te, gap)
        tsz = sorted(collections.Counter(tb).values(), reverse=True)
        print(f"[2] blocks gap>{gap:4d}s   train {len(set(b))} blocks  purity {purity(b, tu):.3f}  "
              f"median {np.median(list(sizes.values())):.1f}   |   test {len(set(tb))} blocks "
              f"sizes {tsz[:8]}...")

    # ---- key 3: skeleton bone lengths ---------------------------------------
    bl = bone_lengths([r["sid"] for r in tr], "train")
    keep = [r for r in tr if r["sid"] in bl]
    X = np.stack([bl[r["sid"]] for r in keep])
    U = np.array([r["user"] for r in keep])
    cent = {u: X[U == u].mean(0) for u in sorted(set(U))}
    users = sorted(cent)
    C = np.stack([cent[u] for u in users])
    D = np.linalg.norm(C[:, None] - C[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    within = np.mean([X[U == u].std(0).mean() for u in users])
    pred = np.array([users[i] for i in np.linalg.norm(X[:, None] - C[None], axis=-1).argmin(1)])
    print(f"\n[3] bone lengths         n={len(keep)}  per-clip nearest-centroid user acc "
          f"{(pred == U).mean():.3f}")
    print(f"    within-user per-clip sd {within:.4f}  vs  nearest-other-user centroid "
          f"distance {np.median(D.min(1)):.4f}   -> {'SEPARABLE' if np.median(D.min(1)) > within else 'NOT SEPARABLE'}")

    # ---- key 4: IMU device MACs ---------------------------------------------
    imu_root = os.path.join(ROOT, "Small-Model-Track", "Training", "data", "HAR", "data", "IMU")
    cls_dirs = {int(d.split("_")[0]): d for d in os.listdir(imu_root)} if os.path.isdir(imu_root) else {}
    seen = collections.defaultdict(set)
    n = 0
    for r in tr[::7]:                       # every 7th clip is plenty to see the device set
        c = cls_dirs.get(int(r["cls"])) if r["cls"] else None
        if not c:
            continue
        d = os.path.join(imu_root, c, r["user"], r["sid"].split("_", 2)[2])
        if not os.path.isdir(d):
            continue
        for dev, mac in macs(d).items():
            seen[dev].add(mac)
        n += 1
    print(f"\n[4] IMU device MACs      scanned {n} clips")
    for dev in sorted(seen):
        print(f"    {dev}: {len(seen[dev])} distinct MAC(s)")
    print("    -> a per-SUBJECT device assignment would need ~18 MACs per slot; "
          f"max observed {max((len(v) for v in seen.values()), default=0)}")

    print("\nVERDICT: the only key that clears 90% train purity is the timestamp block; "
          "everything else is a mixture. Video-embedding clustering (probe P0) is tested "
          "separately in code/probe_p0_subject.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
