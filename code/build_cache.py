#!/usr/bin/env python3
"""Q-00: one-time preprocessing cache for CUHK-X Small Model Track.

Per sample (train: class/user/trial; test: SM_test_XXXX) writes one npz with:
  skel_pos (T,P,17,3) f32, skel_np (T,) u8, skel_t (T,) f64
  imu_<DEV>_t (n,) f64, imu_<DEV>_x (n,16) f32   for DEV in WTC,WTLA,WTRA,WTLL,WTRL
  depth (T,120,160) u8  (JET-inverted scalar; 0=invalid, 1..255 near->far), depth_t (T,) f64
  ir (T,120,160) u8, ir_t (T,) f64
Plus meta_{split}.csv with per-sample metadata.
Radar/Thermal deferred (B-008 / Q-06).
"""
import csv
import json
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool

import cv2
import numpy as np

# Portable root: env CUHKX_ROOT wins, else the repo dir two levels up from this
# file. Was a hardcoded absolute path (with a space in it) in 11 files, which was
# the #1 blocker for running anywhere but the original laptop.
ROOT = os.environ.get("CUHKX_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "Small-Model-Track/Training/data/HAR/data")
TEST = os.path.join(ROOT, "Small-Model-Track/Testing/data/small_model_track_test")
CACHE = os.path.join(ROOT, "cache")
H, W = 120, 160
DEVICES = ("WTC", "WTLA", "WTRA", "WTLL", "WTRL")

# ---------------- JET inversion ----------------
_jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET).reshape(256, 3)  # BGR
JET_BGR = _jet.astype(np.int32)
_lut = None  # lazily built 2^24 uint8 LUT in each worker; 255 used as "unmatched" sentinel


def _build_lut():
    lut = np.full(1 << 24, 255, np.uint8)
    keys = (JET_BGR[:, 0] << 16) | (JET_BGR[:, 1] << 8) | JET_BGR[:, 2]
    vals = (1 + np.round(np.arange(256) * 254.0 / 255.0)).astype(np.uint8)  # 1..255
    lut[keys] = vals
    lut[0] = 0  # pure black = invalid
    return lut


def invert_jet(img_bgr):
    """640x480x3 BGR -> uint8 scalar depth index (0 invalid, 1..255 near->far)."""
    global _lut
    if _lut is None:
        _lut = _build_lut()
    key = (img_bgr[:, :, 0].astype(np.int32) << 16) | (img_bgr[:, :, 1].astype(np.int32) << 8) | img_bgr[:, :, 2]
    out = _lut[key]
    miss = out == 255
    if miss.any():
        cols = np.unique(key[miss])
        b, g, r = cols >> 16, (cols >> 8) & 255, cols & 255
        d = (
            (b[:, None] - JET_BGR[None, :, 0]) ** 2
            + (g[:, None] - JET_BGR[None, :, 1]) ** 2
            + (r[:, None] - JET_BGR[None, :, 2]) ** 2
        )
        near = (1 + np.round(np.argmin(d, 1) * 254.0 / 255.0)).astype(np.uint8)
        for c, v in zip(cols, near):
            _lut[c] = v
        out = _lut[key]
    return out


# ---------------- timestamp parsing ----------------
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2}\.\d+)")


def fname_ts(name):
    m = TS_RE.search(name)
    if not m:
        return None
    d, t = m.groups()
    return datetime.strptime(d + " " + t.replace("-", ":"), "%Y-%m-%d %H:%M:%S.%f").timestamp()


def imu_ts(s):
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2}) (.+)", s)
    if m:
        y, mo, dd, rest = m.groups()
        return imu_ts(f"{y}-{int(mo):02d}-{int(dd):02d} {rest}")
    return None


# ---------------- IMU ----------------
HDR_KEYS = {
    "acc": ("加速度X", "AccX"), "acc_y": ("加速度Y", "AccY"), "acc_z": ("加速度Z", "AccZ"),
    "gyr": ("角速度X", "AsX"), "gyr_y": ("角速度Y", "AsY"), "gyr_z": ("角速度Z", "AsZ"),
    "ang": ("角度X", "AngleX"), "ang_y": ("角度Y", "AngleY"), "ang_z": ("角度Z", "AngleZ"),
    "mag": ("磁场X", "HX"), "mag_y": ("磁场Y", "HY"), "mag_z": ("磁场Z", "HZ"),
    "q0": ("四元数0", "Q0"), "q1": ("四元数1", "Q1"), "q2": ("四元数2", "Q2"), "q3": ("四元数3", "Q3"),
}
CH_ORDER = ["acc", "acc_y", "acc_z", "gyr", "gyr_y", "gyr_z", "ang", "ang_y", "ang_z",
            "mag", "mag_y", "mag_z", "q0", "q1", "q2", "q3"]


def parse_imu_file(path):
    out = {d: ([], []) for d in DEVICES}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
    except OSError:
        return out
    if len(rows) < 2:
        return out
    hdr = rows[0]
    col = {}
    for k, pats in HDR_KEYS.items():
        for i, h in enumerate(hdr):
            if any(p in h for p in pats):
                col[k] = i
                break
    if len(col) < 16:
        return out
    for r in rows[1:]:
        if len(r) <= max(col.values()):
            continue
        dev = next((d for d in ("WTLA", "WTRA", "WTLL", "WTRL", "WTC") if r[1].startswith(d)), None)
        t = imu_ts(r[0])
        if dev is None or t is None:
            continue
        try:
            x = [float(r[col[k]]) for k in CH_ORDER]
        except ValueError:
            continue
        out[dev][0].append(t)
        out[dev][1].append(x)
    return out


# ---------------- per-sample worker ----------------
def frames_sorted(d, ext):
    try:
        fs = [f for f in os.listdir(d) if f.endswith(ext)]
    except OSError:
        return []
    fs = [(fname_ts(f), f) for f in fs]
    fs = [x for x in fs if x[0] is not None]
    fs.sort()
    return fs


def process_sample(job):
    sid, paths, out_path = job
    if os.path.exists(out_path):
        return None
    data, meta = {}, {"sample_id": sid}
    # skeleton
    sk = paths.get("Skeleton")
    if sk:
        fs = frames_sorted(os.path.join(sk, "predictions"), ".json")
        pos, nps, ts = [], [], []
        for t, f in fs:
            try:
                with open(os.path.join(sk, "predictions", f)) as fh:
                    persons = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            kp = [p["keypoints"] for p in persons if "keypoints" in p and len(p["keypoints"]) == 17]
            pos.append(kp)
            nps.append(len(kp))
            ts.append(t)
        if ts:
            P = max(1, max(nps))
            arr = np.zeros((len(ts), P, 17, 3), np.float32)
            for i, kp in enumerate(pos):
                for j, p in enumerate(kp[:P]):
                    arr[i, j] = p
            data["skel_pos"] = arr
            data["skel_np"] = np.array(nps, np.uint8)
            data["skel_t"] = np.array(ts, np.float64)
    # imu
    imu = paths.get("IMU")
    if imu:
        acc = {d: ([], []) for d in DEVICES}
        for fn in os.listdir(imu):
            if not fn.endswith(".csv"):
                continue
            part = parse_imu_file(os.path.join(imu, fn))
            for d in DEVICES:
                acc[d][0].extend(part[d][0])
                acc[d][1].extend(part[d][1])
        for d in DEVICES:
            if acc[d][0]:
                o = np.argsort(acc[d][0])
                data[f"imu_{d}_t"] = np.array(acc[d][0], np.float64)[o]
                data[f"imu_{d}_x"] = np.array(acc[d][1], np.float32)[o]
    # depth / ir
    for mod, key in (("Depth_Color", "depth"), ("IR", "ir")):
        mdir = paths.get(mod)
        if not mdir:
            continue
        fs = frames_sorted(mdir, ".png")
        imgs, ts = [], []
        for t, f in fs:
            img = cv2.imread(os.path.join(mdir, f), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if key == "depth":
                s = invert_jet(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                s = cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                s = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs.append(s)
            ts.append(t)
        if ts:
            data[key] = np.stack(imgs)
            data[key + "_t"] = np.array(ts, np.float64)
    # meta
    tall = [data[k][0] for k in ("depth_t", "ir_t", "skel_t") if k in data]
    tall += [data[f"imu_{d}_t"][0] for d in DEVICES if f"imu_{d}_t" in data]
    tend = [data[k][-1] for k in ("depth_t", "ir_t", "skel_t") if k in data]
    tend += [data[f"imu_{d}_t"][-1] for d in DEVICES if f"imu_{d}_t" in data]
    meta.update(
        t0=min(tall) if tall else "", t1=max(tend) if tend else "",
        n_depth=len(data.get("depth_t", ())), n_ir=len(data.get("ir_t", ())),
        n_skel=len(data.get("skel_t", ())),
        max_persons=int(data["skel_np"].max()) if "skel_np" in data else 0,
        imu_devs="|".join(d for d in DEVICES if f"imu_{d}_t" in data),
    )
    np.savez_compressed(out_path + ".tmp.npz", **data)
    os.replace(out_path + ".tmp.npz", out_path)
    return meta


def collect_train_jobs():
    jobs = {}
    for mod in ("Skeleton", "IMU", "Depth_Color", "IR"):
        mroot = os.path.join(TRAIN, mod)
        for cls in sorted(os.listdir(mroot)):
            cid = int(cls.split("_")[0])
            for user in sorted(os.listdir(os.path.join(mroot, cls))):
                for trial in sorted(os.listdir(os.path.join(mroot, cls, user))):
                    sid = f"{cid:02d}_{user}_{trial}"
                    jobs.setdefault(sid, {"class_id": cid, "user": user, "trial": trial, "paths": {}})
                    jobs[sid]["paths"][mod] = os.path.join(mroot, cls, user, trial)
    return jobs


def collect_test_jobs():
    jobs = {}
    for d in sorted(os.listdir(TEST)):
        if not d.startswith("SM_test"):
            continue
        jobs[d] = {"paths": {}}
        for mod in ("Skeleton", "IMU", "Depth_Color", "IR"):
            p = os.path.join(TEST, d, mod)
            if os.path.isdir(p):
                jobs[d]["paths"][mod] = p
    return jobs


def run(split):
    jobs = collect_train_jobs() if split == "train" else collect_test_jobs()
    outdir = os.path.join(CACHE, split)
    os.makedirs(outdir, exist_ok=True)
    work = [(sid, j["paths"], os.path.join(outdir, sid + ".npz")) for sid, j in sorted(jobs.items())]
    metas = []
    with Pool(8) as pool:
        for i, m in enumerate(pool.imap_unordered(process_sample, work, chunksize=4)):
            if m is not None:
                sid = m["sample_id"]
                if split == "train":
                    m.update({k: jobs[sid][k] for k in ("class_id", "user", "trial")})
                    m["station"] = jobs[sid]["trial"].split("-")[0]
                metas.append(m)
            if (i + 1) % 200 == 0:
                print(f"{split}: {i + 1}/{len(work)}", flush=True)
    if metas:
        meta_path = os.path.join(CACHE, f"meta_{split}.csv")
        cols = sorted({k for m in metas for k in m})
        new = not os.path.exists(meta_path)
        with open(meta_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerows(metas)
    print(f"{split} DONE: {len(work)} samples, {len(metas)} newly written", flush=True)


if __name__ == "__main__":
    for split in sys.argv[1:] or ("test", "train"):
        run(split)
