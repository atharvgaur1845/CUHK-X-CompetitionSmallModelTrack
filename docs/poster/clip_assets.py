"""Real sensor data of one training clip, shared by the poster and the slides (never fabricated).

load_clip(frame, th_frame) returns, for training clip 30_Do_jumping_jacks/user2/5-1-1 (samples.json):
  ir_raw / ir  the infrared frame (raw; and contrast-stretched to [0, 1] for print)
  depth        the colour-mapped depth frame of the same index
  thermal      a thermal frame (the thermal camera is not frame-synchronised with the infrared one)
  skel, bones  the 17-joint 3-D skeleton of the frame and the H36M bone list
  imu          |acceleration| per inertial unit over the whole clip (5 units)
  radar        every radar detection of the clip (timestamp, frame, x, y, z, v, snr)
Frame 18 of 61 is the arms-up apex; thermal frame 40 of 122 is the nearest arms-up thermal frame.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
SAMPLES = json.load(open(HERE.parent / "report" / "figures_src" / "samples.json"))
FRAME, TH_FRAME = 18, 40


def load_clip(frame: int = FRAME, th_frame: int = TH_FRAME, clip: str = "abstract_clip") -> dict:
    s = SAMPLES[clip]; B = Path(s["base"]); a, u, t = s["activity"], s["user"], s["trial"]
    ir = sorted(glob.glob(f"{B}/IR/{a}/{u}/{t}/*.png")); dp = sorted(glob.glob(f"{B}/Depth_Color/{a}/{u}/{t}/*.png"))
    th = sorted(glob.glob(f"{B}/Thermal/{a}/{u}/{t}/*.jpg")); sk = sorted(glob.glob(f"{B}/Skeleton/{a}/{u}/{t}/predictions/*.json"))
    raw = np.asarray(Image.open(ir[frame]).convert("L")).astype(float)
    lo, hi = np.percentile(raw, [1, 99.5]); ir_disp = np.clip((raw - lo) / (hi - lo), 0, 1)
    A = {"ir_raw": raw, "ir": ir_disp, "depth": np.asarray(Image.open(dp[frame]).convert("RGB")),
         "thermal": np.asarray(Image.open(th[th_frame]).convert("RGB")), "n_ir": len(ir), "n_th": len(th),
         "skel": np.array(json.load(open(sk[frame]))[0]["keypoints"]), "bones": SAMPLES["h36m_bones"], "clip": f"{a}/{u}/{t}"}
    dev = {}
    for f in glob.glob(f"{B}/IMU/{a}/{u}/{t}/*.csv"):
        df = pd.read_csv(f); name = df.iloc[:, 1].str[:4]
        acc = np.sqrt((df.iloc[:, 2:5].astype(float) ** 2).sum(axis=1)).to_numpy()
        for dn in name.unique(): dev[dn] = acc[(name == dn).to_numpy()]
    A["imu"] = dev
    A["radar"] = pd.read_csv(glob.glob(f"{B}/Radar/{a}/{u}/{t}/*.csv")[0])
    return A


def thermal_frames(clip: str = "thermal_clip") -> list[str]:
    s = SAMPLES[clip]; B = Path(s["base"]); a, u, t = s["activity"], s["user"], s["trial"]
    return sorted(glob.glob(f"{B}/Thermal/{a}/{u}/{t}/*.jpg"))
