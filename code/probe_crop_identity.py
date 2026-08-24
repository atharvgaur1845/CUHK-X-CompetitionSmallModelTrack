#!/usr/bin/env python3
"""EXP-111: does the person crop lock onto ONE person, or does it union two?

compute_windows() picks the highest-confidence person box independently in each of 8
probe frames, then takes the min/max UNION of those boxes as the clip's crop. There is no
identity association. If two frames pick different people the crop spans both, and the
subject shrinks inside it.

The repo records test at 21.2% multi-person against train's 10.2%, so any damage here
lands on test at twice the rate -- and OOF, measured on train subjects, structurally
cannot see it.

Reports, per split: how often >1 person is visible, how consistent the per-frame pick is
(IoU between consecutive picks), and how much the union inflates over the median box.
"""
import json, sys, os
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle"))
import cuhkx_224_kaggle as K
from ultralytics import YOLO
from PIL import Image

split = sys.argv[1] if len(sys.argv) > 1 else "test"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
paths = K.find_paths(cache_name="crop_224")
train_jobs, test_jobs = K.build_jobs(paths)
jobs = test_jobs if split == "test" else train_jobs
if limit:
    jobs = jobs[:limit]
model = YOLO("yolo11n.pt")

def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / ua if ua > 0 else 0.0

rows = []
for n, (sid, ir_dir, _d, _c, _u) in enumerate(jobs):
    frames = K.sorted_frames(Path(ir_dir))
    if not frames:
        continue
    probes = list(dict.fromkeys(K.endpoint_uniform(frames, min(K.DETECTION_FRAMES, len(frames)))))
    imgs = []
    for p in probes:
        try: imgs.append(Image.open(p).convert("RGB"))
        except Exception: pass
    if not imgs:
        continue
    picks, npers = [], []
    for res in model.predict(imgs, classes=[0], conf=K.PERSON_CONFIDENCE,
                             verbose=False, device="cpu"):
        if res.boxes is None or len(res.boxes) == 0:
            npers.append(0); continue
        c = res.boxes.conf.cpu().numpy(); xy = res.boxes.xyxy.cpu().numpy()
        npers.append(len(c)); picks.append(xy[int(c.argmax())])
    if not picks:
        continue
    P = np.stack(picks)
    union = [P[:,0].min(), P[:,1].min(), P[:,2].max(), P[:,3].max()]
    ua = (union[2]-union[0]) * (union[3]-union[1])
    areas = (P[:,2]-P[:,0]) * (P[:,3]-P[:,1])
    ious = [iou(P[i], P[i+1]) for i in range(len(P)-1)] or [1.0]
    rows.append(dict(sid=sid, maxpers=max(npers), multi=int(max(npers) > 1),
                     min_iou=float(np.min(ious)), mean_iou=float(np.mean(ious)),
                     inflate=float(ua / max(np.median(areas), 1e-9))))
    if (n+1) % 100 == 0: print(f"  {n+1}/{len(jobs)}", flush=True)

import statistics as st
m = np.array([r["multi"] for r in rows]); inf = np.array([r["inflate"] for r in rows])
mi = np.array([r["min_iou"] for r in rows])
print(f"\n=== {split}: {len(rows)} clips ===")
print(f"clips with >1 person in some probe frame : {m.mean()*100:5.1f}%")
print(f"union box inflates >2x over median box   : {(inf>2).mean()*100:5.1f}%")
print(f"union box inflates >4x                   : {(inf>4).mean()*100:5.1f}%")
print(f"min consecutive-pick IoU < 0.3 (identity switch): {(mi<0.3).mean()*100:5.1f}%")
print(f"min consecutive-pick IoU < 0.1                  : {(mi<0.1).mean()*100:5.1f}%")
print(f"median inflate {np.median(inf):.2f}  p90 {np.percentile(inf,90):.2f}  max {inf.max():.2f}")
json.dump(rows, open(f"research/artifacts/cropprobe_{split}.json","w"))
print(f"wrote research/artifacts/cropprobe_{split}.json")
