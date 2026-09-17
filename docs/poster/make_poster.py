#!/usr/bin/env python3
"""One-page public poster for the Most Popular vote: a single-screen story, not a condensed paper.
Real sensor frames of one training clip, the pipeline in four words, four facts, three findings.
Every number is the same measured value the report and its figure audit carry.

    python3 make_poster.py   -> poster.pdf, poster.png (300 dpi), poster_social.png (1080x1350)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
import figstyle as st          # palette
import figdata                 # data tables (same source as the report)
from generate_all_figures import _load_clip_assets, D  # real clip assets, tables

# public asset: a clean bold sans (Lato) instead of the paper's Times
for f in Path("/usr/share/fonts/truetype/lato").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Lato", "font.weight": "bold", "text.color": st.INK, "pdf.fonttype": 42, "figure.dpi": 150,
                     "mathtext.fontset": "custom", "mathtext.rm": "Lato:bold", "mathtext.it": "Lato:bold:italic", "mathtext.bf": "Lato:bold"})

W, H = 100.0, 125.0   # 4:5 canvas (8 x 10 in)
fig = plt.figure(figsize=(8, 10)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc=st.PAPER, ec="none", zorder=0))
T = lambda x, y, s, **k: ax.text(x, y, s, zorder=5, **k)
A = _load_clip_assets(); m = D["members"]; r = D["resolution"]


def arrow(p0, p1, color=st.INK, lw=1.6, ms=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=4, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ header
ax.add_patch(Rectangle((0, 108), W, 17, fc=st.INK, ec="none", zorder=1))
T(4, 120.6, "FIVE SENSORS.", fontsize=30, color="white", va="top", ha="left")
T(4, 114.2, "ONE SMALL MODEL.", fontsize=30, color="#9DB7E0", va="top", ha="left")
T(96, 120.0, "Can a machine recognise everyday activities\nwithout an ordinary camera?", fontsize=12.5, color="white", va="top", ha="right", linespacing=1.25)
T(96, 111.2, "CUHK-X Challenge 2026  ·  Small Model Track  ·  Team AGX", fontsize=8.5, color="#9DB7E0", va="top", ha="right")

# ------------------------------------------------------------------ five sensors, real frames
y0, y1 = 87.5, 103.0; tw = 17.6; gap = 1.4; x = 4.0
tiles = [("IR + DEPTH", st.BLUE, "sees the body\nand the scene"), ("THERMAL", st.ORANGE, "sees heat:\nbodies and objects"), ("SKELETON", st.TEAL, "tracks 17 joints"),
         ("WRIST IMU", st.TEAL_LIGHT, "feels the motion\nof the hands"), ("RADAR", st.PURPLE, "keeps the order\nof a recording")]
for i, (lab, col, sub) in enumerate(tiles):
    ax.add_patch(FancyBboxPatch((x, y0), tw, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=st.GRAY_LIGHT, lw=0.8, zorder=1))
    ax.add_patch(Rectangle((x, y1 - 1.2), tw, 1.2, fc=col, ec="none", zorder=2))
    T(x + tw / 2, y1 - 2.2, lab, fontsize=9.5, color=col, ha="center", va="top")
    T(x + tw / 2, y0 + 0.9, sub, fontsize=7.2, color=st.GRAY, ha="center", va="bottom", linespacing=1.1)
    cx0, cy0, cx1, cy1 = x + 1.0, y0 + 5.0, x + tw - 1.0, y1 - 5.2   # image area
    if i == 0:
        h = cy1 - cy0; w = h * 4 / 3; wsum = 2 * w + 0.5; ox = (cx1 - cx0 - wsum) / 2
        ax.imshow(A["ir"], cmap="gray", extent=(cx0 + ox, cx0 + ox + w, cy0, cy1), aspect="auto", zorder=3)
        ax.imshow(A["depth"], extent=(cx0 + ox + w + 0.5, cx0 + ox + 2 * w + 0.5, cy0, cy1), aspect="auto", zorder=3)
    elif i == 1:
        h = cy1 - cy0; w = h * 4 / 3; ox = (cx1 - cx0 - w) / 2
        ax.imshow(A["thermal"], extent=(cx0 + ox, cx0 + ox + w, cy0, cy1), aspect="auto", zorder=3)
    elif i == 2:
        k = A["skel"]; X, Z = k[:, 0], k[:, 2]; sc = (cy1 - cy0) / (Z.max() - Z.min()); cx = (cx0 + cx1) / 2
        for a, b in D["samples"]["h36m_bones"]:
            ax.plot([cx + (X[a] - X.mean()) * sc, cx + (X[b] - X.mean()) * sc], [cy0 + (Z[a] - Z.min()) * sc, cy0 + (Z[b] - Z.min()) * sc], color=st.TEAL, lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter(cx + (X - X.mean()) * sc, cy0 + (Z - Z.min()) * sc, s=16, color=st.TEAL, zorder=4, linewidths=0)
    elif i == 3:
        names = ["WTLA", "WTRA", "WTC(", "WTLL", "WTRL"]; rowh = (cy1 - cy0) / 5
        for j, nm in enumerate(names):
            v = A["imu"][nm]; v = (v - v.min()) / max(v.max() - v.min(), 1e-6)
            xx = np.linspace(cx0 + 0.5, cx1 - 0.5, len(v)); yy = cy1 - (j + 0.95) * rowh + v * rowh * 0.85
            ax.plot(xx, yy, color=st.TEAL_LIGHT, lw=1.2, zorder=3)
    else:
        R = A["radar"]; px = cx0 + 1 + (R[:, 0] - R[:, 0].min()) / (R[:, 0].max() - R[:, 0].min()) * (cx1 - cx0 - 2)
        py = cy0 + (R[:, 1] - R[:, 1].min()) / (R[:, 1].max() - R[:, 1].min()) * (cy1 - cy0)
        ax.scatter(px, py, s=3, color=st.PURPLE, alpha=0.5, zorder=3, linewidths=0)
    x += tw + gap
T(50, 104.6, "one training clip, jumping jacks, as each sensor sees it", fontsize=7.5, color=st.GRAY, ha="center", va="bottom")

# ------------------------------------------------------------------ the pipeline in four words
steps = [("FIVE SPECIALISTS", "one small model per sensor: 3 video, 1 skeleton + motion ensemble, 1 motion forest", st.INK),
         ("FUSION + CONTEXT", "combine the five votes, then use the natural order of a household routine", st.PURPLE),
         ("40 EVERYDAY ACTIVITIES", "wash face, pour a drink, sweep, jog, take medicine, ...", st.BLUE)]
ys_step = [85.0, 73.0, 64.5]
for i, ((big, small, col), yy) in enumerate(zip(steps, ys_step)):
    arrow((50, yy + 1.4), (50, yy - 1.0), color=st.GRAY, lw=1.4, ms=16)
    T(50, yy - 1.8, big, fontsize=15 if i < 2 else 17, color=col, ha="center", va="top")
    T(50, yy - 4.9, small, fontsize=7.6, color=st.GRAY, ha="center", va="top")
# specialist chips under the first step
chips = [("body", st.BLUE), ("hands", st.BLUE_LIGHT), ("heat", st.ORANGE), ("posture", st.TEAL), ("motion", st.TEAL_LIGHT)]
cw = 9.0; cx = 50 - (5 * cw + 4 * 1.0) / 2
for lab, col in chips:
    ax.add_patch(FancyBboxPatch((cx, 75.4), cw, 2.6, boxstyle="round,pad=0,rounding_size=1.3", fc=col, ec="none", zorder=3))
    T(cx + cw / 2, 76.7, lab, fontsize=7.5, color="white", ha="center", va="center"); cx += cw + 1.0

# ------------------------------------------------------------------ four facts
facts = [("96 MB", "one file,\nevery weight"), ("0.877", "private score,\n14th of 307 teams"), ("UNSEEN", "people at test time"), ("0 / 405", "predictions changed\nacross 5 GPUs")]
fw = 22.0; fx = 4.0; fy0, fy1 = 45.5, 56.0
for big, small in facts:
    ax.add_patch(FancyBboxPatch((fx, fy0), fw, fy1 - fy0, boxstyle="round,pad=0,rounding_size=0.8", fc=st.INK, ec="none", zorder=1))
    T(fx + fw / 2, fy1 - 1.4, big, fontsize=21 if len(big) <= 6 else 17, color="white", ha="center", va="top")
    T(fx + fw / 2, fy0 + 1.0, small, fontsize=7.2, color="#C9D2E0", ha="center", va="bottom", linespacing=1.15)
    fx += fw + 2.0

# ------------------------------------------------------------------ three findings
T(4, 42.6, "THREE THINGS WE MEASURED", fontsize=11, color=st.INK, ha="left", va="top")
px0, pw, ph, py0 = 4.0, 30.0, 30.5, 8.5
gapx = (W - 8 - 3 * pw) / 2
titles = [("01", "Different sensors rescue\ndifferent mistakes"), ("02", "More pixels are not always\nmore information"), ("03", "More model is not\nbetter generalisation")]
for i, (num, ttl) in enumerate(titles):
    x = px0 + i * (pw + gapx)
    ax.add_patch(FancyBboxPatch((x, py0), pw, ph, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=st.GRAY_LIGHT, lw=0.8, zorder=1))
    T(x + 1.6, py0 + ph - 1.2, num, fontsize=16, color=st.GRAY, ha="left", va="top")
    T(x + 8.0, py0 + ph - 1.6, ttl, fontsize=9.2, color=st.INK, ha="left", va="top", linespacing=1.15)

# 01: clips only one sensor gets right
x = px0; u = m["unique_rescues"]; order = [("skeleton", "posture"), ("thermal", "heat"), ("person", "body"), ("wrist", "hands"), ("imu", "motion")]
bx0, by0, bw, bh = x + 3.0, py0 + 8.5, pw - 6.0, 10.5; vmax = 80
for j, (k, lab) in enumerate(order):
    v = u[k]; xx = bx0 + j * (bw / 5); ww = bw / 5 - 1.0; hh = v / vmax * bh
    ax.add_patch(Rectangle((xx, by0), ww, hh, fc=st.MEMBER[k], ec="none", zorder=3))
    T(xx + ww / 2, by0 + hh + 0.4, str(v), fontsize=7.5, color=st.INK, ha="center", va="bottom")
    T(xx + ww / 2, by0 - 0.5, lab, fontsize=6.5, color=st.GRAY, ha="center", va="top")
T(x + pw / 2, py0 + ph - 7.0, "clips (of 2,700) that only ONE sensor gets right", fontsize=6.8, color=st.GRAY, ha="center", va="top")
T(x + pw / 2, py0 + 1.0, f"at least one sensor is right on {m['oracle_any_member']:,} clips;\nremove any one and the system loses 11 to 48", fontsize=6.6, color=st.INK, ha="center", va="bottom", linespacing=1.15)

# 02: body crop up, hand crop down
x = px0 + pw + gapx; gx0, gy0, gw, gh = x + 4.0, py0 + 8.5, pw - 8.0, 13.0
pres, pacc = r["person"]["res"], r["person"]["acc"]; wres, wacc = r["wrist"]["res"], r["wrist"]["acc"]
lo, hi = 0.700, 0.740
sx = lambda v: gx0 + (v - 224) / (448 - 224) * gw; sy = lambda v: gy0 + (v - lo) / (hi - lo) * gh
ax.plot([sx(v) for v in pres], [sy(v) for v in pacc], color=st.BLUE, lw=2.2, zorder=3); ax.scatter([sx(v) for v in pres], [sy(v) for v in pacc], s=22, color=st.BLUE, zorder=4)
ax.plot([sx(v) for v in wres], [sy(v) for v in wacc], color=st.BLUE_LIGHT, lw=2.2, zorder=3); ax.scatter([sx(v) for v in wres], [sy(v) for v in wacc], s=22, color=st.BLUE_LIGHT, zorder=4)
T(sx(384), sy(pacc[2]) + 1.0, "body crop  +1.9", fontsize=7.2, color=st.BLUE, ha="center", va="bottom")
T(sx(288) + 1.0, sy(wacc[1]) - 0.2, "hand crop  -1.4", fontsize=7.2, color=st.BLUE_LIGHT, ha="left", va="top")
for v in pres: T(sx(v), gy0 - 0.6, f"{v}", fontsize=6.5, color=st.GRAY, ha="center", va="top")
T(x + pw / 2, gy0 - 3.2, "input size (pixels)", fontsize=6.8, color=st.GRAY, ha="center", va="top")
ax.plot([gx0, gx0 + gw], [gy0, gy0], color=st.GRAY_LIGHT, lw=0.8, zorder=2)
T(x + pw / 2, py0 + 1.0, "the body fills the frame, so detail is real;\nthe hand is tiny, so extra pixels are interpolation", fontsize=6.6, color=st.INK, ha="center", va="bottom", linespacing=1.15)

# 03: what we tried
x = px0 + 2 * (pw + gapx)
rows = [("more capacity", False), ("more training tricks", False), ("more test-time tricks", False), ("more input resolution", True), ("a new sensor view", True), ("better fusion", True), ("recording structure", True)]
ry = py0 + ph - 8.6
for lab, ok in rows:
    T(x + 2.0, ry, lab, fontsize=7.6, color=st.INK, ha="left", va="center")
    mx = x + pw - 3.6; msz = 0.9   # drawn marks: Lato has no check/cross glyphs
    if ok: ax.plot([mx - msz, mx - 0.25 * msz, mx + msz], [ry, ry - msz * 0.9, ry + msz], color=st.GREEN, lw=2.0, solid_capstyle="round", zorder=5)
    else: ax.plot([mx - msz, mx + msz], [ry - msz, ry + msz], color=st.RED, lw=2.0, solid_capstyle="round", zorder=5); ax.plot([mx - msz, mx + msz], [ry + msz, ry - msz], color=st.RED, lw=2.0, solid_capstyle="round", zorder=5)
    ry -= 2.75
T(x + pw / 2, py0 + 1.0, "every lever measured against the same\nstatistical bar; the failures are the map", fontsize=6.6, color=st.INK, ha="center", va="bottom", linespacing=1.15)

# ------------------------------------------------------------------ footer
ax.add_patch(Rectangle((0, 0), W, 6.0, fc=st.INK, ec="none", zorder=1))
T(4, 3.0, "Team AGX  ·  Atharv, BITS Pilani K K Birla Goa Campus  ·  CUHK-X Challenge 2026, Small Model Track", fontsize=7.2, color="white", ha="left", va="center")
T(96, 3.0, "all numbers: docs/report/FIGURE_AUDIT.md", fontsize=6.5, color="#9DB7E0", ha="right", va="center")

OUT = HERE
fig.savefig(OUT / "poster.pdf", metadata={"Creator": "make_poster.py"})
fig.savefig(OUT / "poster.png", dpi=300)
fig.savefig(OUT / "poster_social.png", dpi=135)   # 1080 x 1350
print("written", sorted(p.name for p in OUT.glob("poster*")))
