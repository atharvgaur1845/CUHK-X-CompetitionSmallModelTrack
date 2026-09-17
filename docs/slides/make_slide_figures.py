#!/usr/bin/env python3
"""Slide-only figures (not in the report). Same palette and data tables as the report.

slide_what_we_tried: the one-glance version of the experiment landscape (Fig. 9 of the report):
seven families of levers, a cross or a check, and the measured numbers in small type.
    python3 make_slide_figures.py -> figures/slide_what_we_tried.{pdf,png,svg}
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
import figstyle as st

for f in Path("/usr/share/fonts/truetype/lato").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Lato", "font.weight": "bold", "text.color": st.INK, "pdf.fonttype": 42})

ROWS = [  # (family, verdict, evidence in one line; numbers from research/LOG.md, EXP ids in FIGURE_AUDIT.md)
    ("MORE CAPACITY", False, "VideoMAE-B  -1.8 (0/4 folds)  ·  Swin3D-T  -9.8  ·  V-JEPA 2 teacher  -29"),
    ("MORE TRAINING TRICKS", False, "distillation: student 157 vs 167 public  ·  self-training: member +1.4, fusion -2  ·  LR decay +0.2 (noise)"),
    ("MORE TEST-TIME TRICKS", False, "temporal TTA  -4 fused  ·  seed soup  -2.4 (4/4 folds)  ·  pair verifier  -0.8  ·  logit centering  +5 clips (unreadable)"),
    ("MORE INPUT RESOLUTION", True, "person crop 224 to 384 px  +1.9 member, +20 fused clips  (448 px: -0.8, rejected)"),
    ("A NEW SENSOR VIEW", True, "thermal at 224 px full frame  +5.6 member, +46 decoded clips, +5 public"),
    ("BETTER FUSION", True, "prior-corrected log-linear space (wrong space costs 4 points)  ·  nested-CV weights  +15 clips"),
    ("RECORDING STRUCTURE", True, "ordered transition decoder  +131 decoded clips (+55 distinctness, +76 order)  ·  +3 to +5 public"),
]

fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.25); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 100, 56.25, fc="white", ec="none"))
ax.text(4, 52.5, "WHAT WE TRIED", fontsize=30, color=st.INK, va="top")
ax.text(4, 46.8, "every lever measured against the same bar: paired folds, 2 SE (seed sd 1.16 points)", fontsize=12, color=st.GRAY, va="top")
ax.plot([4, 96], [45.2, 45.2], color=st.GRAY_LIGHT, lw=1.2)
y = 41.5; step = 5.55
for fam, ok, ev in ROWS:
    col = st.GREEN if ok else st.RED
    ax.add_patch(Rectangle((4, y - 2.3), 92, 4.9, fc=st.BAND if ok else "white", ec="none", zorder=0))
    ax.text(6, y + 0.9, fam, fontsize=16, color=st.INK, va="center")
    ax.text(6, y - 1.35, ev, fontsize=9.5, color=st.GRAY, va="center")
    mx, msz = 92.5, 1.3
    if ok:
        ax.plot([mx - msz, mx - 0.25 * msz, mx + msz], [y, y - msz * 0.9, y + msz], color=col, lw=4, solid_capstyle="round")
    else:
        ax.plot([mx - msz, mx + msz], [y - msz, y + msz], color=col, lw=4, solid_capstyle="round"); ax.plot([mx - msz, mx + msz], [y + msz, y - msz], color=col, lw=4, solid_capstyle="round")
    y -= step
ax.text(96, 1.6, "the bottleneck is not capacity: the residual error is subject-correlated (next slide: every lever placed on one axis)", fontsize=10, color=st.INK, ha="right", va="bottom")

OUT = HERE / "figures"; OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"slide_what_we_tried.{ext}", dpi=300)
subprocess.run(["pdftocairo", "-svg", str(OUT / "slide_what_we_tried.pdf"), str(OUT / "slide_what_we_tried.svg")])
print("written", sorted(p.name for p in OUT.glob("slide_what_we_tried.*")))
