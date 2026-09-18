#!/usr/bin/env python3
"""One-page public poster: page 1 of the technical report and the asset for the Most Popular vote.

The idea is one image: ONE PERSON, FIVE WAYS OF SEEING. A real training clip (jumping jacks, user2,
recording 5-1-1) sits in the centre as the depth camera sees it (a silhouette, no face: the public
message is that the system never needs to see who you are); around it the same instant through the
other lenses plus radar; below it the five specialists converge into one decision. Every number is
the measured value the report and its figure audit carry; nothing is drawn from invented data.

Real data used: depth frame 18 of 61 and its infrared frame (the arms-up apex), thermal frame 40 of 122
(the nearest arms-up frame; the two cameras are not frame-synchronised), the 17-joint skeleton of frame 18,
the five inertial units' |acceleration| over the clip, every radar detection of the clip (height vs time).
The small infrared thumbnail is contrast-stretched for print; the model reads the raw frame. The shipped fusion,
with user2 held out, predicts this clip correctly and so does every specialist (figdata.shipped_oof).

Two typefaces: Liberation Serif Bold (the paper's Times) for the title and the numbers, Liberation Sans
Bold (the paper's Helvetica) for labels. Black type on white; colour only where it carries meaning.

    python3 make_poster.py -> poster.pdf (8 x 10 in), poster.png (300 dpi), poster_social.png (1080 x 1350)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Circle, PathPatch, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
import figstyle as st
from generate_all_figures import D          # data tables (figures_src/*.json)

for f in Path("/usr/share/fonts/truetype/liberation").glob("Liberation*.ttf"):
    fm.fontManager.addfont(str(f))
SERIF, SANS = "Liberation Serif", "Liberation Sans"
BLACK = "#111111"
plt.rcParams.update({"font.family": SANS, "font.weight": "bold", "text.color": BLACK, "pdf.fonttype": 42, "figure.dpi": 150,
                     "mathtext.fontset": "custom", "mathtext.rm": f"{SANS}:bold", "mathtext.it": f"{SANS}:bold:italic", "mathtext.bf": f"{SANS}:bold"})
MIN = 7.5
COL = {"ir": st.BLUE, "thermal": st.ORANGE, "skeleton": st.TEAL, "imu": st.TEAL_LIGHT, "radar": st.PURPLE}

from clip_assets import load_clip, FRAME, TH_FRAME   # real data of the jumping-jacks clip (shared with the slides)

A = load_clip(); m = D["members"]; r = D["resolution"]

# ------------------------------------------------------------------ canvas and helpers
W, H = 100.0, 125.0
fig = plt.figure(figsize=(8, 10)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=0))


def T(x, y, s, size=MIN, color=BLACK, family=SANS, **k):
    return ax.text(x, y, s, fontsize=max(size, MIN), color=color, fontfamily=family, fontweight="bold", zorder=6, **k)


def rule(y, x0=4.0, x1=96.0, lw=1.0, color=BLACK, z=2):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=z)


def panel(x, y, w, h, fc="white"):
    ax.add_patch(Rectangle((x + 0.45, y - 0.45), w, h, fc=(0, 0, 0, 0.09), ec="none", zorder=1))   # soft shadow: depth without 3-D
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=BLACK, lw=0.7, zorder=2))


def lens_label(x, y, key, name):
    ax.add_patch(Rectangle((x, y - 0.7), 1.4, 1.4, fc=COL[key], ec="none", zorder=5))
    T(x + 2.1, y, name, 8.5, COL[key], ha="left", va="center")


def connector(p0, p1, color, lw=0.9):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=1.5, solid_capstyle="round")
    ax.add_patch(Circle(p0, 0.45, fc=color, ec="none", zorder=5))


def ribbon(p0, p1, color, lw=3.2, z=3, alpha=0.95):
    (x0, y0), (x1, y1) = p0, p1; ym = (y0 + y1) / 2
    path = MPath([p0, (x0, ym), (x1, ym), p1], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, alpha=alpha, zorder=z, capstyle="round"))


def mark(x, y, ok, s=0.75, lw=1.8):
    if ok: ax.plot([x - s, x - 0.25 * s, x + s], [y, y - 0.9 * s, y + s], color=st.GREEN, lw=lw, solid_capstyle="round", zorder=6)
    else:
        ax.plot([x - s, x + s], [y - s, y + s], color=st.RED, lw=lw, solid_capstyle="round", zorder=6)
        ax.plot([x - s, x + s], [y + s, y - s], color=st.RED, lw=lw, solid_capstyle="round", zorder=6)


# ================================================================== LEVEL 1: title (2 seconds)
T(4, 124.2, "FIVE SENSORS.", 32, BLACK, SERIF, va="top", ha="left")
T(4, 117.9, "ONE DECISION.", 32, st.BLUE, SERIF, va="top", ha="left")
T(96, 124.0, "Can a machine recognise everyday\nactivities without an ordinary camera?", 11, BLACK, SANS, va="top", ha="right", linespacing=1.25)
T(96, 118.9, "Multimodal activity recognition under a 100 MB constraint", 8, BLACK, SANS, va="top", ha="right")
rule(113.0, lw=2.0); rule(112.4, lw=0.5)

# ================================================================== LEVEL 2: one person, five ways of seeing (5 seconds)
PX0, PX1, PY0, PY1 = 42.0, 58.0, 70.0, 104.0            # the person: the depth frame (no face is needed, or shown)
crop = A["depth"][12:478, 50:270]; aspect = crop.shape[1] / crop.shape[0]
pw = (PY1 - PY0) * aspect; PX0 = 50 - pw / 2; PX1 = 50 + pw / 2
ax.add_patch(Rectangle((PX0 + 0.6, PY0 - 0.6), pw, PY1 - PY0, fc=(0, 0, 0, 0.12), ec="none", zorder=1))
ax.imshow(crop, extent=(PX0, PX1, PY0, PY1), aspect="auto", zorder=3, interpolation="bilinear")
ax.add_patch(Rectangle((PX0, PY0), pw, PY1 - PY0, fc="none", ec=BLACK, lw=0.9, zorder=4))
T(50, 68.8, "ONE HUMAN ACTION", 9, BLACK, SANS, ha="center", va="top")
T(50, 66.9, "jumping jacks, in depth: no face needed", 7.5, BLACK, SANS, ha="center", va="top")

LW, LX_L, LX_R, UY0, UY1, LY0, LY1 = 24.0, 10.0, 66.0, 86.0, 104.5, 65.5, 84.0
# --- upper left: infrared + depth (the same camera pair, raw)
panel(LX_L, UY0, LW, UY1 - UY0); lens_label(LX_L + 1.2, UY1 - 1.5, "ir", "IR + DEPTH")
ch = 13.0; cw = ch * 0.737; cx = LX_L + (LW - 2 * cw - 0.8) / 2; cy = UY0 + 3.0
ax.imshow(A["ir"][12:478, 50:393], cmap="gray", vmin=0, vmax=1, extent=(cx, cx + cw, cy, cy + ch), aspect="auto", zorder=3)
ax.imshow(A["depth"][12:478, 50:393], extent=(cx + cw + 0.8, cx + 2 * cw + 0.8, cy, cy + ch), aspect="auto", zorder=3)
T(LX_L + LW / 2, UY0 + 0.8, "infrared and depth, 640 x 480", 7.5, BLACK, SANS, ha="center", va="bottom")
connector((LX_L + LW, 95.5), (PX0, 95.5), COL["ir"])
# --- upper right: thermal
panel(LX_R, UY0, LW, UY1 - UY0); lens_label(LX_R + 1.2, UY1 - 1.5, "thermal", "THERMAL")
th = A["thermal"][:, 20:210]; cw = ch * th.shape[1] / th.shape[0]; cx = LX_R + (LW - cw) / 2
ax.imshow(th, extent=(cx, cx + cw, cy, cy + ch), aspect="auto", zorder=3)
T(LX_R + LW / 2, UY0 + 0.8, "heat of body and objects, 320 x 240", 7.5, BLACK, SANS, ha="center", va="bottom")
connector((LX_R, 95.5), (PX1, 95.5), COL["thermal"])
# --- lower left: skeleton
panel(LX_L, LY0, LW, LY1 - LY0, fc=st.BAND); lens_label(LX_L + 1.2, LY1 - 1.5, "skeleton", "SKELETON")
k = A["skel"]; X, Z = k[:, 0], k[:, 2]; sh = 13.4; sc = sh / (Z.max() - Z.min()); cxs, cys = LX_L + LW / 2, LY0 + 3.2
for a_, b_ in A["bones"]:
    ax.plot([cxs + (X[a_] - X.mean()) * sc, cxs + (X[b_] - X.mean()) * sc], [cys + (Z[a_] - Z.min()) * sc, cys + (Z[b_] - Z.min()) * sc], color=st.TEAL, lw=2.6, solid_capstyle="round", zorder=3)
ax.scatter(cxs + (X - X.mean()) * sc, cys + (Z - Z.min()) * sc, s=22, color=st.TEAL, zorder=4, linewidths=0.8, edgecolors="white")
T(LX_L + LW / 2, LY0 + 0.8, "17 joints in 3-D, same instant", 7.5, BLACK, SANS, ha="center", va="bottom")
connector((LX_L + LW, 75.0), (PX0, 75.0), COL["skeleton"])
# --- lower right: wrist and body inertial units (ridgeline of |acceleration|)
panel(LX_R, LY0, LW, LY1 - LY0, fc=st.BAND); lens_label(LX_R + 1.2, LY1 - 1.5, "imu", "WRIST IMU")
names = [("WTLA", "left wrist"), ("WTRA", "right wrist"), ("WTC(", "chest"), ("WTLL", "left ankle"), ("WTRL", "right ankle")]
gx0, gx1, gy0, gy1 = LX_R + 1.5, LX_R + LW - 9.2, LY0 + 3.0, LY1 - 3.2; rowh = (gy1 - gy0) / 5
for j, (nm, lab) in enumerate(names):
    v = A["imu"][nm]; v = (v - v.min()) / max(v.max() - v.min(), 1e-6); xx = np.linspace(gx0, gx1, len(v)); base = gy1 - (j + 1) * rowh
    yy = base + v * rowh * 1.35
    ax.fill_between(xx, base, yy, color=st.TEAL_LIGHT, alpha=0.35 if j < 2 else 0.18, lw=0, zorder=3 + (5 - j) * 0.01)
    ax.plot(xx, yy, color=st.TEAL_LIGHT if j < 2 else st.TEAL, lw=1.6 if j < 2 else 0.9, zorder=3.1 + (5 - j) * 0.01, solid_capstyle="round")
    T(gx1 + 0.8, base + 0.3, lab, 7.5, BLACK, SANS, ha="left", va="bottom")
T(LX_R + LW / 2, LY0 + 0.8, "|acceleration| over the clip", 7.5, BLACK, SANS, ha="center", va="bottom")
connector((LX_R, 75.0), (PX1, 75.0), COL["imu"])
# --- top centre: radar, a temporal trail (height of every detection against time)
RX0, RX1, RY0, RY1 = 36.0, 64.0, 105.0, 111.6
panel(RX0, RY0, RX1 - RX0, RY1 - RY0, fc=st.BAND); lens_label(RX0 + 1.2, RY1 - 1.3, "radar", "RADAR")
T(RX1 - 1.0, RY0 + 0.6, "every detection: height against time", 7.5, BLACK, SANS, ha="right", va="bottom")
R = A["radar"]; t = (R.frame - R.frame.min()) / (R.frame.max() - R.frame.min()); z = R.z.clip(-1.0, 1.3)
px = RX0 + 1.2 + t * (RX1 - RX0 - 2.4); pz = RY0 + 2.1 + (z - z.min()) / (z.max() - z.min()) * (RY1 - RY0 - 4.4)
ax.scatter(px, pz, s=2.5 + (R.snr - R.snr.min()) / (R.snr.max() - R.snr.min()) * 9, c=t, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("v", ["#C9B8E0", st.PURPLE]), zorder=3, linewidths=0, alpha=0.95)
connector((50, RY0), (50, PY1), COL["radar"])

# ================================================================== five specialists -> fusion -> context -> one decision
T(50, 64.7, "FIVE SPECIALISTS", 9.5, BLACK, SANS, ha="center", va="top")
spec = [("body", st.BLUE), ("hands", st.BLUE_LIGHT), ("heat", st.ORANGE), ("posture", st.TEAL), ("motion", st.TEAL_LIGHT)]
SX = [30, 40, 50, 60, 70]; SY = 59.7
for (lab, col), x in zip(spec, SX):
    ax.add_patch(Circle((x, SY), 1.15, fc=col, ec="white", lw=0.8, zorder=5))
    T(x, SY + 1.6, lab, 7.5, BLACK, SANS, ha="center", va="bottom")
FY = 52.4
for (lab, col), x in zip(spec, SX):
    ribbon((x, SY - 1.0), (50, FY + 1.6), col, lw=3.0)
ax.add_patch(Circle((50, FY), 1.9, fc="white", ec=st.INK, lw=1.8, zorder=5))
T(53.2, FY + 0.4, "FUSION", 9.5, st.INK, SANS, ha="left", va="center")
T(53.2, FY - 1.5, "five votes, one prior-corrected log-linear sum", 7.5, BLACK, SANS, ha="left", va="center")
CY = 47.4
ax.plot([50, 50], [FY - 1.9, CY + 1.9], color=st.INK, lw=4.0, zorder=3, solid_capstyle="butt")
ax.add_patch(Circle((50, CY), 1.9, fc="white", ec=st.INK, lw=1.8, zorder=5)); ax.add_patch(Circle((50, CY), 0.7, fc=st.PURPLE, ec="none", zorder=6))
T(53.2, CY + 0.4, "CONTEXT", 9.5, st.INK, SANS, ha="left", va="center")
T(53.2, CY - 1.5, "the order of the recording, from radar timestamps", 7.5, BLACK, SANS, ha="left", va="center")
ax.add_patch(FancyArrowPatch((50, CY - 1.9), (50, 43.9), arrowstyle="-|>", mutation_scale=18, color=st.INK, lw=4.0, zorder=3, shrinkA=0, shrinkB=0))
T(50, 43.1, "ONE DECISION", 17, st.INK, SERIF, ha="center", va="top")
T(50, 39.9, "jumping jacks: on this held-out clip all five specialists and the fusion say so", 7.5, BLACK, SANS, ha="center", va="top")

# ================================================================== the numbers
rule(37.6, lw=1.6); rule(28.2, lw=1.6)
nums = [(16, "40", 40, "ACTIVITIES"), (43, "96 MB", 30, "ONE FILE"), (68, "0.877", 26, "PRIVATE SCORE"), (88, "UNSEEN", 18, "PEOPLE AT TEST TIME")]
for x, big, size, small in nums:
    T(x, 36.6, big, size, BLACK, SERIF, ha="center", va="top")
    T(x, 29.4, small, 7.5, BLACK, SANS, ha="center", va="bottom")
T(50, 27.3, "14th of 307 teams  ·  0 / 405 predictions changed across 5 tested GPU generations (posteriors drift up to 2e-3)  ·  no network at inference", 7.5, BLACK, SANS, ha="center", va="top")

# ================================================================== LEVEL 3: three discoveries (10 seconds)
CX = [3.0, 35.0, 67.0]; CW = 30.0; TOPY = 24.2
heads = [("01", "DIFFERENT SENSORS\nRESCUE DIFFERENT MISTAKES"), ("02", "MORE PIXELS ARE NOT\nALWAYS MORE INFORMATION"), ("03", "MORE MODEL IS NOT\nBETTER GENERALISATION")]
for x, (num, ttl) in zip(CX, heads):
    rule(TOPY + 0.9, x0=x, x1=x + CW, lw=0.7)
    T(x, TOPY, num, 14, st.BLUE, SERIF, ha="left", va="top")
    T(x + 5.6, TOPY - 0.2, ttl, 8.2, BLACK, SANS, ha="left", va="top", linespacing=1.15)

# 01: squares of clips that only one specialist gets right (one square = 5 clips)
x = CX[0]; u = m["unique_rescues"]; order = [("skeleton", "posture"), ("thermal", "heat"), ("person", "body"), ("wrist", "hands"), ("imu", "motion")]
gx = x + 1.6; pitch = 0.95; sq = 0.78; groupw = 5.3; by = 11.2
for j, (key, lab) in enumerate(order):
    n = int(round(u[key] / 5)); x0 = gx + j * groupw
    for q in range(n):
        ax.add_patch(Rectangle((x0 + (q % 2) * pitch, by + (q // 2) * pitch), sq, sq, fc=st.MEMBER[key], ec="none", zorder=3))
    rows = (n + 1) // 2
    T(x0 + pitch - 0.1, by + rows * pitch + 0.2, str(u[key]), 8, BLACK, SANS, ha="center", va="bottom")
    T(x0 + pitch - 0.1, by - 0.5, lab, 7.5, BLACK, SANS, ha="center", va="top")
T(x + CW / 2, 7.2, "one square = 5 clips only this specialist gets\nright; 2,399 of 2,700 are right for at least one", 7.5, BLACK, SANS, ha="center", va="top", linespacing=1.15)

# 02: real pixels: a body patch at native scale, a wrist patch where one source pixel becomes nine input pixels
x = CX[1]; cell = 0.8; n = 9
body = A["ir"][114:123, 148:157]; wrist = np.kron(A["ir"][33:36, 130:133], np.ones((3, 3)))
for gi, (patch, lab, sub, blocks) in enumerate([(body, "BODY", "one source pixel\n= one input pixel", 1), (wrist, "WRIST", "one source pixel\n= nine input pixels", 3)]):
    gx0 = x + 3.4 + gi * 14.4; gy0 = 11.6
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1, extent=(gx0, gx0 + n * cell, gy0, gy0 + n * cell), aspect="auto", zorder=3, interpolation="nearest")
    for q in range(n + 1):
        lw_ = 0.9 if q % blocks == 0 else 0.3
        ax.plot([gx0 + q * cell] * 2, [gy0, gy0 + n * cell], color="white", lw=lw_, zorder=4); ax.plot([gx0, gx0 + n * cell], [gy0 + q * cell] * 2, color="white", lw=lw_, zorder=4)
    ax.add_patch(Rectangle((gx0, gy0), n * cell, n * cell, fc="none", ec=BLACK, lw=0.7, zorder=5))
    T(gx0 + n * cell / 2, gy0 + n * cell + 0.5, lab, 8, st.BLUE if gi == 0 else st.BLUE_LIGHT, SANS, ha="center", va="bottom")
    T(gx0 + n * cell / 2, gy0 - 0.5, sub, 7.5, BLACK, SANS, ha="center", va="top", linespacing=1.1)
T(x + CW / 2, 7.2, f"body crop {r['person']['acc'][0]:.3f} to {r['person']['acc'][2]:.3f}, 224 to 384 px\nwrist crop {100*(r['wrist']['acc'][1]-r['wrist']['acc'][0]):+.2f} points, 224 to 288 px", 7.5, BLACK, SANS, ha="center", va="top", linespacing=1.15)

# 03: one large model crossed out; five small specialists ticked; the levers, measured
x = CX[2]
bx, by0 = x + 1.2, 11.9
for q in range(5): ax.add_patch(FancyBboxPatch((bx, by0 + q * 1.45), 5.6, 1.15, boxstyle="round,pad=0,rounding_size=0.3", fc="#B8BEC6", ec="white", lw=0.5, zorder=3))
mark(bx + 2.8, by0 + 3.5, False, s=2.0, lw=2.6)
sx0 = x + 9.2
for q, (lab, col) in enumerate(spec):
    ax.add_patch(Circle((sx0 + q * 1.35, by0 + 6.7), 0.55, fc=col, ec="none", zorder=4)); ribbon((sx0 + q * 1.35, by0 + 6.15), (sx0 + 2.7, by0 + 2.4), col, lw=1.2, z=3)
ax.add_patch(Circle((sx0 + 2.7, by0 + 1.8), 0.8, fc=st.INK, ec="none", zorder=4)); mark(sx0 + 2.7 + 2.2, by0 + 1.8, True, s=0.8, lw=2.0)
T(x + 8.2, by0 - 0.5, "big model vs five small", 7.5, BLACK, SANS, ha="center", va="top")
lx = x + 17.2; ly = by0 + 7.3
for lab, ok in [("more capacity", False), ("training tricks", False), ("test-time tricks", False), ("new sensor view", True), ("better fusion", True), ("recording order", True)]:
    mark(lx, ly, ok, s=0.5, lw=1.5); T(lx + 1.3, ly, lab, 7.5, BLACK, SANS, ha="left", va="center"); ly -= 1.5
T(x + CW / 2, 7.2, "VideoMAE-B -1.8, Swin3D-T -9.8 points; every\nlever measured against one bar (paired folds, 2 SE)", 7.5, BLACK, SANS, ha="center", va="top", linespacing=1.15)

# ================================================================== footer
rule(2.2, lw=1.2)
T(4, 1.0, "Team AGX  ·  Atharv, BITS Pilani, K K Birla Goa Campus  ·  CUHK-X Challenge 2026, Small Model Track", 7.5, BLACK, SANS, ha="left", va="center")
T(96, 1.0, "technical report follows", 7.5, BLACK, SANS, ha="right", va="center")

OUT = HERE
fig.savefig(OUT / "poster.pdf", metadata={"Creator": "make_poster.py"})
fig.savefig(OUT / "poster.png", dpi=300)
fig.savefig(OUT / "poster_social.png", dpi=135)   # 1080 x 1350
print("written", sorted(p.name for p in OUT.glob("poster*")))
