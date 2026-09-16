#!/usr/bin/env python3
"""Figures for the technical report, generated from the logged numbers (research/LOG.md,
EXP-141/145/154/160/162) and the package manifest. Every number is cited by experiment id
in the caption; nothing is fitted or smoothed here."""
from __future__ import annotations
import json, re, zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
LOG = (ROOT / "research" / "LOG.md").read_text()
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"blue": "#2b5a9e", "grey": "#8a8a8a", "orange": "#d9822b", "green": "#3a8f5c", "red": "#b23a3a"}


def table(start: str, span: int = 2500):
    i = LOG.index(start); block = LOG[i:i + span]
    return re.findall(r"\| (user\d+) \|(?: \d+ \|)? \**([0-9.]+)\**", block)


# ---- Fig 1: per-subject LOSO, person view vs shipped ensemble (EXP-141 / EXP-145)
person = dict((u, float(v)) for u, v in table("## EXP-141 — ✅ 18-SUBJECT LOSO", 3000))
ens = dict((u, float(v)) for u, v in table("### The shipped ensemble, per subject"))
users = sorted(ens, key=lambda u: ens[u])
fig, ax = plt.subplots(figsize=(6.4, 2.9))
x = np.arange(len(users)); w = 0.38
ax.bar(x - w / 2, [person[u] for u in users], w, color=C["grey"], label=f"person view alone (LOSO), sd {np.std([person[u] for u in users], ddof=1):.3f}")
ax.bar(x + w / 2, [ens[u] for u in users], w, color=C["blue"], label=f"shipped ensemble (LOSO), sd {np.std([ens[u] for u in users], ddof=1):.3f}")
ax.set_xticks(x); ax.set_xticklabels([u.replace("user", "u") for u in users], rotation=0, fontsize=7)
ax.set_ylim(0.5, 0.95); ax.set_ylabel("accuracy, subject held out"); ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_title("Leave-one-subject-out accuracy: fusion raises and flattens the per-subject distribution", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_loso_subjects.pdf"); plt.close(fig)

# ---- Fig 2: resolution curve, person view (EXP-154/160) and wrist 224 -> 288 (EXP-154)
res = [224, 288, 384, 448]; acc = [0.71444, 0.72963, 0.73333, 0.72519]
fig, ax = plt.subplots(figsize=(3.2, 2.6))
ax.plot(res, acc, "o-", color=C["blue"], label="person view")
ax.plot([224, 288], [0.7200, 0.7200 - 0.0136], "s--", color=C["orange"], label="wrist view")
for r, a in zip(res, acc): ax.annotate(f"{a:.3f}", (r, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
ax.set_xticks(res); ax.set_xlabel("input resolution (px)"); ax.set_ylabel("member accuracy, pooled OOF"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Resolution helps the body crop, hurts the wrist crop", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_resolution.pdf"); plt.close(fig)

# ---- Fig 3: missing-modality ablation and coefficient sensitivity (EXP-162)
mem = ["person", "wrist", "skeleton", "thermal", "IMU"]; drop = [-48, -29, -27, -11, -13]; alone = [1980, 1944, 1668, 1711, 1017]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
a1.barh(mem, [-d for d in drop], color=C["red"]); a1.invert_yaxis()
for i, d in enumerate(drop): a1.text(-d + 1, i, f"{d} clips ({100*d/2700:+.1f} pts)", va="center", fontsize=7)
a1.set_xlim(0, 75); a1.set_xlabel("loss when the sensor is absent (clips of 2,700)"); a1.set_title("Missing-sensor cost, fused", fontsize=9)
coef = ["person", "wrist", "thermal", "skeleton", "IMU", "prior"]
sens = {"person": [2085, 2102, 2109, 2114, 2102], "wrist": [2102, 2111, 2109, 2109, 2107], "thermal": [2113, 2116, 2109, 2107, 2105],
        "skeleton": [2100, 2104, 2109, 2107, 2100], "IMU": [2105, 2109, 2109, 2107, 2105], "prior": [2109, 2110, 2109, 2108, 2111]}
scales = [0.5, 0.75, 1.0, 1.25, 1.5]
for k, v in sens.items(): a2.plot(scales, [x / 2700 for x in v], "o-", ms=3, label=k)
a2.axhline(2109 / 2700, color=C["grey"], lw=0.8, ls=":"); a2.set_ylim(0.76, 0.80); a2.set_xlabel("coefficient scale (others fixed)"); a2.set_ylabel("fused OOF accuracy")
a2.legend(frameon=False, fontsize=6, ncol=2); a2.set_title("Coefficient sensitivity: a plateau", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_ablation_sensitivity.pdf"); plt.close(fig)

# ---- Fig 4: decoder under structure loss (EXP-162c)
cases = ["intact\norder", "shuffled\nwithin group", "reversed", "uniform\ntransitions", "permuted\nlabels"]; vals = [2240, 2118, 2055, 2164, 2092]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
cols = [C["blue"], C["grey"], C["red"], C["green"], C["grey"]]
ax.bar(range(5), [v - 2109 for v in vals], color=cols); ax.axhline(0, color="k", lw=0.8)
for i, v in enumerate(vals): ax.text(i, v - 2109 + (4 if v >= 2109 else -12), f"{v-2109:+d}", ha="center", fontsize=7)
ax.set_xticks(range(5)); ax.set_xticklabels(cases, fontsize=7); ax.set_ylabel("decoded − pre-decoder (clips of 2,700)")
ax.set_title("Decoder gain and its dependence on recording order", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_decoder_structure.pdf"); plt.close(fig)

# ---- Fig 5: byte budget from the package manifest
z = zipfile.ZipFile(ROOT / "research" / "artifacts" / "stage2_ship7.pth"); man = json.loads(z.read("manifest.json"))
sizes = {i.filename: i.compress_size for i in z.infolist()}
by = {}
for m in man["members"]:
    for t in m.get("tensors", []): by[m["branch"]] = by.get(m["branch"], 0) + sizes.get(t["code_entry"], 0) + sizes.get(t.get("scale_entry", ""), 0)
lab = {"video": "3 × MViTv2-S (int5)", "skeleton": "skeleton/IMU stack (int8)", "detector": "YOLO11n ×2 (fp16)", "imu": "IMU forest", "windows": "crop windows"}
fig, ax = plt.subplots(figsize=(6.4, 1.3)); left = 0
for k in ["video", "skeleton", "detector", "imu", "windows"]:
    v = by.get(k, 0) / 1e6; ax.barh([0], [v], left=left, label=f"{lab[k]} {v:.1f} MB"); left += v
ax.axvline(100, color=C["red"], ls="--", lw=1); ax.text(100, 0.45, "100 MB rule", color=C["red"], fontsize=7, ha="right")
ax.set_xlim(0, 105); ax.set_yticks([]); ax.set_xlabel("MB in the single package file (96.37 MB total)")
ax.legend(frameon=False, fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.5))
fig.tight_layout(); fig.savefig(OUT / "fig_bytes.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.iterdir()))
