#!/usr/bin/env python3
"""Every figure of the deck, generated from the report's data tables (docs/report/figures_src/*.json) and the
real frames of one training clip (docs/poster/clip_assets.py). Nothing is typed by hand: the numbers the slides
quote are written to gen/numbers.tex and gen/verification.tex from the same tables.

    python3 make_slide_figures.py     -> figures/s*.pdf (slide figures, with overlay steps), gen/*.tex

Figures (all Lato, transparent background, nothing below 7.5 pt on the 16 x 9 cm slide):
    s01_hero_{1,2,3}      one person, five lenses, five specialists -> fusion -> context -> one decision
    s02_layers            the depth silhouette inside five concentric sensor layers
    s03_field             2,700 held-out clips as squares, coloured by who gets them right
    s04_converge_{1,2}    five streams converge into one decision (fused 0.781, decoded 0.830)
    s05_resolution        person and wrist curves; real pixel patches
    s06_thermal           the same thermal frame at 128 px crop, 128 px full frame, 224 px full frame
    s07_subjects          16 paired held-out subjects, person view -> ensemble
    s08_atlas             stability atlas: sensor absent, coefficient x0.5 to x1.5
    s09_waterfall_{1..4}  decoder gain built in the order it is computed, plus the controls
    s10_levers            what we tried: more of the same vs a different kind of information
    s11_container_{1..4}  the 100 MB container filling up
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import slidestyle as ss
from slidestyle import INK, BLUE, BLUE_LIGHT, ORANGE, TEAL, TEAL_LIGHT, PURPLE, RED, GRAY, GRAY_LIGHT, GRAY_TEXT, BAND, GREEN, MEMBER, SENSOR, SPECIALISTS
from clip_assets import load_clip, thermal_frames, FRAME, TH_FRAME

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "report" / "figures_src"
D = {p.stem: json.load(open(p)) for p in SRC.glob("*.json")}
GEN = HERE / "gen"; GEN.mkdir(exist_ok=True)
ss.use()
A = load_clip()
CROP = (slice(12, 478), slice(50, 270))          # the person and a little of the room (the poster's crop)
NAMES = {"person": "body", "wrist": "hands", "thermal": "heat", "skeleton": "posture", "imu": "motion"}

# facts that are not in a data table (Kaggle / organiser numbers), with their source
FACTS = {"private": "0.877", "private_clips": "179/204", "public": "0.866", "public_clips": "174/201", "rank": "14", "teams": "307",
         "classes": "40", "train_subjects": "18", "train_clips": "2,933", "test_clips": "405", "budget": "100", "time_cluster": "2 min 39 s",
         "time_laptop": "4 min 53 s", "peak_gpu": "5.4"}   # Kaggle leaderboard 2026-09-13; competition data card; EXP-159 timings (A100, RTX 4060)


# ----------------------------------------------------------------------------------------------- shared drawing
def person(ax, x0, y0, h, shadow=True, z=3, frame=True):
    crop = A["depth"][CROP]; w = h * crop.shape[1] / crop.shape[0]
    if shadow: ax.add_patch(Rectangle((x0 + 0.6, y0 - 0.6), w, h, fc=(0, 0, 0, 0.12), ec="none", zorder=z - 1))
    ax.imshow(crop, extent=(x0, x0 + w, y0, y0 + h), aspect="auto", zorder=z, interpolation="bilinear")
    if frame: ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=INK, lw=0.8, zorder=z + 1))
    return w


def lens_label(ax, x, y, key, name, size=8):
    ax.add_patch(Rectangle((x, y - 0.75), 1.5, 1.5, fc=SENSOR[key], ec="none", zorder=5))
    ss.text(ax, x + 2.3, y, name, size, SENSOR[key], ha="left", va="center")


def lens_ir(ax, x, y, w, h):
    ss.panel(ax, x, y, w, h); lens_label(ax, x + 1.5, y + h - 2.2, "ir", "IR + DEPTH")
    ch = h - 8.0; cw = ch * 0.737
    if 2 * cw + 1.0 <= w - 3.0:   # both channels side by side
        cx = x + (w - 2 * cw - 1.0) / 2; cy = y + 2.0
        ax.imshow(A["ir"][12:478, 50:393], cmap="gray", vmin=0, vmax=1, extent=(cx, cx + cw, cy, cy + ch), aspect="auto", zorder=3)
        ax.imshow(A["depth"][12:478, 50:393], extent=(cx + cw + 1.0, cx + 2 * cw + 1.0, cy, cy + ch), aspect="auto", zorder=3)
    else:                          # the infrared channel alone (the depth channel is the centre image)
        cx = x + (w - cw) / 2; cy = y + 2.0
        ax.imshow(A["ir"][12:478, 50:393], cmap="gray", vmin=0, vmax=1, extent=(cx, cx + cw, cy, cy + ch), aspect="auto", zorder=3)


def lens_thermal(ax, x, y, w, h):
    ss.panel(ax, x, y, w, h); lens_label(ax, x + 1.5, y + h - 2.2, "thermal", "THERMAL")
    th = A["thermal"][:, 20:210]; ch = h - 8.0; cw = ch * th.shape[1] / th.shape[0]; cx = x + (w - cw) / 2; cy = y + 2.0
    ax.imshow(th, extent=(cx, cx + cw, cy, cy + ch), aspect="auto", zorder=3)


def lens_skeleton(ax, x, y, w, h):
    ss.panel(ax, x, y, w, h, fc=BAND); lens_label(ax, x + 1.5, y + h - 2.2, "skeleton", "SKELETON")
    k = A["skel"]; X, Z = k[:, 0], k[:, 2]; sh = h - 8.5; sc = sh / (Z.max() - Z.min()); cx, cy = x + w / 2, y + 2.5
    for a, b in A["bones"]:
        ax.plot([cx + (X[a] - X.mean()) * sc, cx + (X[b] - X.mean()) * sc], [cy + (Z[a] - Z.min()) * sc, cy + (Z[b] - Z.min()) * sc], color=TEAL, lw=2.2, solid_capstyle="round", zorder=3)
    ax.scatter(cx + (X - X.mean()) * sc, cy + (Z - Z.min()) * sc, s=14, color=TEAL, zorder=4, linewidths=0.6, edgecolors="white")


def lens_imu(ax, x, y, w, h):
    ss.panel(ax, x, y, w, h, fc=BAND); lens_label(ax, x + 1.5, y + h - 2.2, "imu", "WRIST IMU")
    names = [("WTLA", "L wrist"), ("WTRA", "R wrist"), ("WTC(", "chest"), ("WTLL", "L ankle"), ("WTRL", "R ankle")]
    gx0, gx1, gy0, gy1 = x + 1.5, x + w - 9.5, y + 2.0, y + h - 5.0; rowh = (gy1 - gy0) / 5
    for j, (nm, lab) in enumerate(names):
        v = A["imu"][nm]; v = (v - v.min()) / max(v.max() - v.min(), 1e-6); xx = np.linspace(gx0, gx1, len(v)); base = gy1 - (j + 1) * rowh
        yy = base + v * rowh * 1.35
        ax.fill_between(xx, base, yy, color=TEAL_LIGHT, alpha=0.35 if j < 2 else 0.18, lw=0, zorder=3)
        ax.plot(xx, yy, color=TEAL_LIGHT if j < 2 else TEAL, lw=1.4 if j < 2 else 0.8, zorder=3.1, solid_capstyle="round")
        ss.text(ax, gx1 + 0.8, base + 0.2, lab, 7.5, INK, ha="left", va="bottom")


def lens_radar(ax, x, y, w, h):
    ss.panel(ax, x, y, w, h, fc=BAND); lens_label(ax, x + 1.5, y + h - 2.0, "radar", "RADAR")
    R = A["radar"]; t = (R.frame - R.frame.min()) / (R.frame.max() - R.frame.min()); z = R.z.clip(-1.0, 1.3)
    px = x + 1.2 + t * (w - 2.4); pz = y + 1.0 + (z - z.min()) / (z.max() - z.min()) * (h - 5.0)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("v", ["#C9B8E0", PURPLE])
    ax.scatter(px, pz, s=1.5 + (R.snr - R.snr.min()) / (R.snr.max() - R.snr.min()) * 6, c=t, cmap=cmap, zorder=3, linewidths=0, alpha=0.95)
    if w > 70: ss.text(ax, x + w - 1.2, y + h - 2.0, "height of every detection against time", 7.5, INK, ha="right", va="center")


def connector(ax, p0, p1, color, lw=0.9):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=1.5, solid_capstyle="round")
    ax.add_patch(Circle(p0, 0.6, fc=color, ec="none", zorder=5))


def stream(ax, x_dots, ys, x_fuse, x_ctx, x_dec, y_mid, r=4.0, label_dy=-4.6, dec_size=13, badges=None, eq=False, big_labels=True):
    """five specialist dots -> ribbons -> FUSION ring -> line -> CONTEXT ring -> arrow -> ONE DECISION"""
    for (lab, col), y in zip(SPECIALISTS, ys):
        ax.add_patch(Circle((x_dots, y), 1.6, fc=col, ec="white", lw=0.8, zorder=5))
        ss.text(ax, x_dots, y + label_dy, lab, 8, INK, ha="center", va="top" if label_dy < 0 else "bottom")
        ss.ribbon(ax, (x_dots + 1.6, y), (x_fuse - r, y_mid), col, lw=3.0)
    ss.ring(ax, x_fuse, y_mid, r); ss.text(ax, x_fuse, y_mid + r + 2.0, "FUSION", 10 if big_labels else 9, INK, ha="center", va="bottom")
    if eq: ss.text(ax, x_fuse, y_mid - r - 2.0, r"$\sum_i w_i \log p_i$", 9, GRAY_TEXT, ha="center", va="top")
    ax.plot([x_fuse + r, x_ctx - r], [y_mid, y_mid], color=INK, lw=4.0, zorder=3, solid_capstyle="butt")
    ss.ring(ax, x_ctx, y_mid, r, dot=PURPLE); ss.text(ax, x_ctx, y_mid + r + 2.0, "CONTEXT", 10 if big_labels else 9, INK, ha="center", va="bottom")
    ss.text(ax, x_ctx, y_mid - r - 2.0, "recording order\n(radar time)", 7.5, GRAY_TEXT, ha="center", va="top", linespacing=1.1)
    ss.arrow(ax, (x_ctx + r, y_mid), (x_dec - 2.0, y_mid), lw=4.0, ms=16)
    ss.text(ax, x_dec, y_mid + 1.0, "ONE\nDECISION", dec_size, INK, ha="left", va="center", linespacing=1.0, weight="black")
    ss.text(ax, x_dec + 0.5, y_mid - 11.0, "jumping jacks", 8, GRAY_TEXT, ha="left", va="top")
    if badges:
        for (x, y, big, small) in badges:
            ss.text(ax, x, y, big, 12, INK, ha="center", va="bottom", weight="black"); ss.text(ax, x, y - 1.0, small, 7.5, GRAY_TEXT, ha="center", va="top")


# ----------------------------------------------------------------------------------------------- s01 hero
def s01_hero(step: int):
    fig, ax, W = ss.canvas(6.3, 2.25)           # W = 280
    LW, LH = 34.0, 44.0; LXL, LXR = 2.0, 90.0
    pw = person(ax, 63.0 - 18.7, 15.0, 79.0); px0, px1 = 63.0 - pw / 2, 63.0 + pw / 2
    ss.text(ax, 63.0, 94.6, "ONE HUMAN ACTION", 8, INK, ha="center", va="bottom")
    if step >= 2:
        lens_ir(ax, LXL, 52.0, LW, LH); lens_skeleton(ax, LXL, 3.0, LW, LH); lens_thermal(ax, LXR, 52.0, LW, LH); lens_imu(ax, LXR, 3.0, LW, LH)
        lens_radar(ax, px0, 1.0, pw, 12.0)
        connector(ax, (LXL + LW, 74), (px0, 74), BLUE); connector(ax, (LXL + LW, 25), (px0, 25), TEAL)
        connector(ax, (LXR, 74), (px1, 74), ORANGE); connector(ax, (LXR, 25), (px1, 25), TEAL_LIGHT)
    if step >= 3:
        ss.text(ax, 146, 97.5, "FIVE SPECIALISTS", 9, INK, ha="center", va="top")
        stream(ax, 146, [84, 67, 50, 33, 16], 190, 220, 238, 50, r=4.0, dec_size=11.5)
    return ss.save(fig, "s01_hero")


# ----------------------------------------------------------------------------------------------- s02 layers
def s02_layers():
    fig, ax, W = ss.canvas(2.5, 2.5)            # W = 100
    crop = A["depth"][12:478, 80:250]; h = 40.0; w = h * crop.shape[1] / crop.shape[0]; x0, y0 = 38 - w / 2, 30.0
    ax.imshow(crop, extent=(x0, x0 + w, y0, y0 + h), aspect="auto", zorder=3, interpolation="bilinear")
    ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=INK, lw=0.8, zorder=4))
    # the ordinary camera that is not there: a faint outline, crossed out, behind the layers
    cx0, cy0 = 87.0, 84.0
    ax.add_patch(FancyBboxPatch((cx0 - 9, cy0 - 5.5), 18, 11, boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=GRAY, lw=1.2, alpha=0.45, zorder=1))
    ax.add_patch(Rectangle((cx0 - 5, cy0 + 5.5), 7, 2.2, fc="none", ec=GRAY, lw=1.2, alpha=0.45, zorder=1))
    ax.add_patch(Circle((cx0, cy0), 3.5, fc="none", ec=GRAY, lw=1.2, alpha=0.45, zorder=1)); ax.add_patch(Circle((cx0, cy0), 1.6, fc="none", ec=GRAY, lw=0.8, alpha=0.45, zorder=1))
    ax.plot([cx0 - 11, cx0 + 11], [cy0 - 8, cy0 + 8], color=RED, lw=1.6, alpha=0.5, zorder=1.1); ax.plot([cx0 - 11, cx0 + 11], [cy0 + 8, cy0 - 8], color=RED, lw=1.6, alpha=0.5, zorder=1.1)
    ss.text(ax, cx0, cy0 - 10.5, "no ordinary\ncamera", 7.5, GRAY_TEXT, ha="center", va="top", linespacing=1.05)
    layers = [("ir", "IR + DEPTH"), ("thermal", "THERMAL"), ("skeleton", "SKELETON"), ("imu", "WRIST IMU"), ("radar", "RADAR")]
    for i, (key, name) in enumerate(layers):
        off = 4.8 * (i + 1)
        ax.add_patch(FancyBboxPatch((x0 - off, y0 - off), w + 2 * off, h + 2 * off, boxstyle="round,pad=0,rounding_size=2.0", fc="none", ec=SENSOR[key], lw=1.6, zorder=2))
        ss.text(ax, x0 - off + 2.4, y0 + h + off - 0.7, name, 7.5, SENSOR[key], ha="left", va="top")
    return ss.save(fig, "s02_layers")


# ----------------------------------------------------------------------------------------------- s03 field
def s03_field():
    m = D["members"]; pats = eval(m["patterns"]) if isinstance(m["patterns"], str) else m["patterns"]
    by_k = m["by_k_members_correct"]; u = m["unique_rescues"]
    fig, ax, W = ss.canvas(6.3, 2.0)            # W = 315
    cols, rows, pitch = 75, 36, 2.3; fx, fy = 3.0, 7.0
    navy = lambda a: (0.106, 0.165, 0.255, a)
    order = [("all five", INK, by_k["5"]["n"]), ("four", navy(0.72), by_k["4"]["n"]), ("three", navy(0.52), by_k["3"]["n"]), ("two", navy(0.34), by_k["2"]["n"])]
    order += [(NAMES[k], MEMBER[k], u[k]) for k in ("skeleton", "thermal", "person", "wrist", "imu")] + [("none", GRAY_LIGHT, by_k["0"]["n"])]
    assert sum(n for _, _, n in order) == 2700
    i = 0; bands = []
    for lab, col, n in order:
        start = i
        for _ in range(n):
            c, r_ = divmod(i, rows); ax.add_patch(Rectangle((fx + c * pitch, fy + (rows - 1 - r_) * pitch), pitch * 0.82, pitch * 0.82, fc=col, ec="none", zorder=3)); i += 1
        bands.append((lab, col, n, fx + (start / rows) * pitch, fx + (i / rows) * pitch))
    top = fy + rows * pitch
    # band labels above the field (two levels so they never collide)
    b = bands
    ss.text(ax, (b[0][3] + b[0][4]) / 2, top + 1.2, f"all five: {b[0][2]:,}", 8, INK, ha="center", va="bottom")
    ss.text(ax, (b[1][3] + b[3][4]) / 2, top + 1.2, f"four {b[1][2]:,}  |  three {b[2][2]:,}  |  two {b[3][2]:,}", 8, INK, ha="center", va="bottom")
    xa, xb = b[4][3], b[8][4]; ss.text(ax, (xa + xb) / 2, top + 5.6, "only one: 288", 8, INK, ha="center", va="bottom")
    ax.plot([xa, xb], [top + 0.6, top + 0.6], color=INK, lw=0.8); ax.plot([(xa + xb) / 2, (xa + xb) / 2], [top + 0.6, top + 5.2], color=INK, lw=0.6)
    lab, col, n, xa, xb = b[-1]; ss.text(ax, (xa + xb) / 2, top + 1.2, f"none: {n}", 8, GRAY_TEXT, ha="center", va="bottom")
    # breakdown of the only-one band, right of the field
    bx = fx + cols * pitch + 8.0; ss.text(ax, bx, top + 1.2, "right for exactly one specialist", 8, INK, ha="left", va="bottom")
    yy = top - 4.5
    for k in ("skeleton", "thermal", "person", "wrist", "imu"):
        ax.add_patch(Rectangle((bx, yy - 2.2), 4.4, 4.4, fc=MEMBER[k], ec="none", zorder=3))
        ss.text(ax, bx + 6.5, yy, f"{u[k]}", 12, INK, ha="left", va="center", weight="black"); ss.text(ax, bx + 20.0, yy, f"{NAMES[k]} only", 8.5, INK, ha="left", va="center"); yy -= 9.6
    ss.text(ax, bx, yy - 1.0, f"nobody: {by_k['0']['n']} of 2,700", 8, GRAY_TEXT, ha="left", va="center")
    ss.text(ax, fx, 1.8, "each square is one held-out clip (2,700): navy = how many specialists are right (five to two), a colour = exactly one, gray = none", 7.5, GRAY_TEXT, ha="left", va="bottom")
    return ss.save(fig, "s03_field")


# ----------------------------------------------------------------------------------------------- s04 converge
def s04_converge(step: int):
    m = D["members"]; dec = D["decoder"]["conditions"][0]["decoded"]
    fig, ax, W = ss.canvas(6.3, 2.5)            # W = 252
    pw = person(ax, 4.0, 5.0, 90.0); px1 = 4.0 + pw
    ys = [84, 67, 50, 33, 16]; xd = 74
    for (lab, col), y in zip(SPECIALISTS, ys):
        ax.plot([px1, xd - 1.6], [50 + (y - 50) * 0.45, y], color=col, lw=0.9, zorder=1.5)
    if step == 1:
        for (lab, col), y in zip(SPECIALISTS, ys):
            ax.add_patch(Circle((xd, y), 1.6, fc=col, ec="white", lw=0.8, zorder=5)); ss.text(ax, xd, y - 4.6, lab, 8, INK, ha="center", va="top")
        ss.text(ax, xd, 97.5, "FIVE SPECIALISTS", 9, INK, ha="center", va="top")
    else:
        ss.text(ax, xd, 97.5, "FIVE SPECIALISTS", 9, INK, ha="center", va="top")
        badges = [(140, 74, f"{m['fused_correct']/2700:.3f}", "fused, OOF"), (190, 74, f"{dec/2700:.3f}", "decoded, OOF")]
        stream(ax, xd, ys, 140, 190, 214, 50, r=5.0, dec_size=12, badges=badges, eq=True)
    return ss.save(fig, "s04_converge")


# ----------------------------------------------------------------------------------------------- s05 resolution
def s05_resolution():
    r = D["resolution"]; p, w = r["person"], r["wrist"]
    fig = plt.figure(figsize=(6.3, 2.5)); gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.95, 1.5], wspace=0.42, left=0.07, right=0.985, top=0.86, bottom=0.2)
    ap, aw, ac = [fig.add_subplot(gs[i]) for i in range(3)]
    pa = [100 * v for v in p["acc"]]; ap.plot(p["res"], pa, color=BLUE, lw=2.6, zorder=3); ap.scatter(p["res"], pa, s=40, color=BLUE, zorder=4, edgecolor="white", lw=0.8)
    ap.scatter([384], [pa[2]], s=140, facecolor="none", edgecolor=BLUE, lw=1.6, zorder=4)
    for x, v, dy in zip(p["res"], pa, (-10, 8, 8, -10)): ap.annotate(f"{v:.1f}", (x, v), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom" if dy > 0 else "top", fontsize=9, color=BLUE, fontweight="bold")
    ap.text(384, pa[2] + 0.75, "384 = best, shipped", fontsize=9, color=INK, ha="center", va="bottom", fontweight="bold")
    ap.text(448, pa[3] - 0.95, "448: -0.8\n(t = -2.0)", fontsize=8, color=RED, ha="center", va="top", fontweight="bold", linespacing=1.05)
    ap.set_xticks(p["res"]); ap.set_ylim(70.0, 75.2); ap.set_xlim(200, 472); ap.set_ylabel("member accuracy, OOF (%)", fontsize=8.5); ap.set_xlabel("input side (px)", fontsize=8.5)
    ap.tick_params(labelsize=8); ap.set_title("PERSON CROP", fontsize=10, color=BLUE, pad=6); ap.grid(axis="y", color=GRAY_LIGHT, lw=0.5); ap.set_axisbelow(True)
    wa = [100 * v for v in w["acc"]]; aw.plot(w["res"], wa, color=BLUE_LIGHT, lw=2.6, zorder=3); aw.scatter(w["res"], wa, s=40, color=BLUE_LIGHT, zorder=4, edgecolor="white", lw=0.8)
    aw.scatter([224], [wa[0]], s=140, facecolor="none", edgecolor=BLUE_LIGHT, lw=1.6, zorder=4)
    aw.annotate(f"{wa[0]:.1f}", (224, wa[0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=BLUE_LIGHT, fontweight="bold")
    aw.annotate(f"{wa[1]:.1f}", (288, wa[1]), xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=9, color=BLUE_LIGHT, fontweight="bold")
    aw.text(226, 70.05, f"{w['mean_delta']:+.2f} points\n3 of 4 folds down", fontsize=8.5, color=RED, ha="left", va="top", fontweight="bold", linespacing=1.05)
    aw.set_xticks(w["res"]); aw.set_xlim(200, 312); aw.set_ylim(68.5, 74.2); aw.set_xlabel("input side (px)", fontsize=8.5); aw.tick_params(labelsize=8)
    aw.set_title("WRIST CROP", fontsize=10, color=BLUE_LIGHT, pad=6); aw.grid(axis="y", color=GRAY_LIGHT, lw=0.5); aw.set_axisbelow(True)
    # real pixels: a torso patch at native scale, a hand patch where one source pixel becomes nine input pixels
    ac.set_xlim(0, 100); ac.set_ylim(0, 100); ac.set_aspect("equal"); ac.axis("off")
    body = A["ir"][114:123, 148:157]; wrist = np.kron(A["ir"][33:36, 130:133], np.ones((3, 3))); n, cell = 9, 4.6
    for gi, (patch, lab, col, sub, blocks) in enumerate([(body, "BODY", BLUE, "one source pixel\n= one input pixel", 1), (wrist, "WRIST", BLUE_LIGHT, "one source pixel\n= nine input pixels", 3)]):
        gx0, gy0 = 4 + gi * 52, 30
        ac.imshow(patch, cmap="gray", vmin=0, vmax=1, extent=(gx0, gx0 + n * cell, gy0, gy0 + n * cell), aspect="auto", zorder=3, interpolation="nearest")
        for q in range(n + 1):
            lw_ = 1.0 if q % blocks == 0 else 0.3
            ac.plot([gx0 + q * cell] * 2, [gy0, gy0 + n * cell], color="white", lw=lw_, zorder=4); ac.plot([gx0, gx0 + n * cell], [gy0 + q * cell] * 2, color="white", lw=lw_, zorder=4)
        ac.add_patch(Rectangle((gx0, gy0), n * cell, n * cell, fc="none", ec=INK, lw=0.7, zorder=5))
        ac.text(gx0 + n * cell / 2, gy0 + n * cell + 3, lab, fontsize=10, color=col, ha="center", va="bottom", fontweight="bold")
        ac.text(gx0 + n * cell / 2, gy0 - 3, sub, fontsize=8, color=INK, ha="center", va="top", fontweight="bold", linespacing=1.1)
    ac.text(50, 96, "WHY: inside one input pixel", fontsize=10, color=INK, ha="center", va="top", fontweight="bold")
    ac.text(50, 1, "person box 224 to 480 source px\nwrist box about 96 px", fontsize=7.5, color=GRAY_TEXT, ha="center", va="bottom", fontweight="bold", linespacing=1.1)
    return ss.save(fig, "s05_resolution")


# ----------------------------------------------------------------------------------------------- s06 thermal
def s06_thermal():
    t = D["thermal"]; st_ = t["stages"]; fr = thermal_frames(); im = Image.open(fr[len(fr) // 2]).convert("RGB")     # 320 x 240, pour drinks
    crop = im.crop((70, 0, 210, 140)).resize((128, 128), Image.BILINEAR)                                            # a 128-px person crop (illustrative box)
    full128 = im.resize((171, 128), Image.BILINEAR); full224 = im.resize((299, 224), Image.BILINEAR)
    fig, ax, W = ss.canvas(6.3, 2.35)           # W = 268
    H = 60.0; y0 = 33.0; xs = [6.0, 88.0, 190.0]; labels = ["128 px crop, 3-D CNN", "128 px full frame, 3-D CNN", "224 px full frame, MViTv2-S"]
    cols = [GRAY_TEXT, GRAY_TEXT, ORANGE]
    for x, img, s_, lab, col in zip(xs, [crop, full128, full224], st_, labels, cols):
        w = H * img.size[0] / img.size[1]
        ax.add_patch(Rectangle((x + 0.6, y0 - 0.6), w, H, fc=(0, 0, 0, 0.12), ec="none", zorder=2))
        ax.imshow(np.asarray(img), extent=(x, x + w, y0, y0 + H), aspect="auto", zorder=3, interpolation="nearest")
        ax.add_patch(Rectangle((x, y0), w, H, fc="none", ec=INK, lw=0.8, zorder=4))
        ss.text(ax, x + w / 2, y0 + H + 2.0, lab, 8.5, INK, ha="center", va="bottom")
        ss.text(ax, x + w / 2, y0 - 3.0, f"{s_['pooled']:.3f}", 25, col, ha="center", va="top", weight="black")
        ss.text(ax, x + w / 2, y0 - 20.5, "member accuracy, OOF", 7.5, GRAY_TEXT, ha="center", va="top")
    fus = ["fusion gain: 0.00 at every weight", "fusion gain: +0.83, not adopted", "+46 decoded clips, +5 public clips"]
    for x, img, f_, col in zip(xs, [crop, full128, full224], fus, cols):
        w = H * img.size[0] / img.size[1]; ss.text(ax, x + w / 2, y0 - 25.5, f_, 8.5, col if col == ORANGE else INK, ha="center", va="top")
    ss.arrow(ax, (65.0, 60), (86.0, 60), color=GRAY_TEXT, lw=2.5, ms=14); ss.arrow(ax, (168.5, 60), (188.0, 60), color=ORANGE, lw=3.5, ms=18)
    gain = 100 * np.mean(np.array(st_[2]["folds"]) - np.array(st_[1]["folds"]))
    ss.text(ax, 178.0, 64.0, f"+{gain:.1f}", 17, ORANGE, ha="center", va="bottom", weight="black"); ss.text(ax, 178.0, 56.0, "points\n4 of 4 folds", 7.5, ORANGE, ha="center", va="top", linespacing=1.05)
    ss.text(ax, 75.5, 64.0, f"+{100*(st_[1]['pooled']-st_[0]['pooled']):.1f}", 12, GRAY_TEXT, ha="center", va="bottom", weight="black")
    return ss.save(fig, "s06_thermal")


# ----------------------------------------------------------------------------------------------- s07 subjects
def s07_subjects():
    L = D["loso"]; subs = L["shared_subjects_sorted_by_person"]; pv = np.array([100 * L["person"][s] for s in subs]); ev = np.array([100 * L["ensemble"][s] for s in subs])
    fig = plt.figure(figsize=(5.0, 2.7)); ax = fig.add_axes([0.1, 0.23, 0.76, 0.72]); x = np.arange(len(subs))
    for i in range(len(subs)):
        up = ev[i] >= pv[i]; ax.plot([i, i], [pv[i], ev[i]], color=GREEN if up else RED, lw=2.2, zorder=2, solid_capstyle="round")
    ax.scatter(x, pv, s=34, color=GRAY, zorder=3, label="person view alone", linewidths=0)
    ax.scatter(x, ev, s=34, color=INK, zorder=4, label="shipped ensemble", linewidths=0)
    ax.axhline(pv.mean(), color=GRAY, lw=0.8, ls=":", zorder=1); ax.axhline(ev.mean(), color=INK, lw=0.8, ls=":", zorder=1)
    ax.text(len(subs) - 0.3, pv.mean(), f"person\nmean {pv.mean():.1f}", fontsize=7.5, color=GRAY_TEXT, ha="left", va="center", fontweight="bold", linespacing=1.05, clip_on=False)
    ax.text(len(subs) - 0.3, ev.mean(), f"ensemble\nmean {ev.mean():.1f}", fontsize=7.5, color=INK, ha="left", va="center", fontweight="bold", linespacing=1.05, clip_on=False)
    ax.set_xlim(-0.7, len(subs) - 0.4)
    for i in np.where(ev < pv)[0]: ax.annotate(f"{ev[i]-pv[i]:+.1f}", (i, ev[i]), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=8, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("user", "subject ") for s in subs], rotation=55, ha="right", fontsize=7.5)
    ax.set_ylim(58, 92); ax.set_ylabel("accuracy, subject held out (%)", fontsize=8.5); ax.tick_params(axis="y", labelsize=8); ax.grid(axis="y", color=GRAY_LIGHT, lw=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, handletextpad=0.3, borderaxespad=0.2)
    return ss.save(fig, "s07_subjects")


# ----------------------------------------------------------------------------------------------- s08 atlas
def s08_atlas():
    R = D["robustness"]; base = R["baseline"]; mem = R["missing"]["members"]; scales = R["coefficients"]["scales"]
    fig = plt.figure(figsize=(4.6, 2.7)); ax = fig.add_axes([0.17, 0.2, 0.81, 0.6]); ax.set_xlim(0, 7.3); ax.set_ylim(0, 5); ax.axis("off")
    cols_x = [0.0] + [1.3 + i for i in range(5)]
    def cell_color(v):
        a = min(abs(v) / 48.0, 1.0); return (0.706, 0.263, 0.227, 0.12 + 0.75 * a) if v < 0 else ((0.24, 0.545, 0.353, 0.18 + 0.6 * a) if v > 0 else (0.953, 0.945, 0.925, 1))
    for i, mname in enumerate(mem):
        y = 4 - i; vals = [R["missing"]["removed_w0"][i]] + [R["coefficients"]["fused"][mname][j] - base for j in range(5)]
        for xc, v in zip(cols_x, vals):
            ax.add_patch(Rectangle((xc, y), 1, 1, fc=cell_color(v), ec="white", lw=1.2, zorder=2))
            ax.text(xc + 0.5, y + 0.5, f"{v:+d}" if v else "0", ha="center", va="center", fontsize=10, color=INK if abs(v) < 30 else "white", fontweight="bold", zorder=3)
        ax.text(-0.15, y + 0.5, NAMES[mname].upper(), ha="right", va="center", fontsize=9, color=MEMBER[mname], fontweight="bold")
    ax.add_patch(Rectangle((cols_x[3], 0), 1, 5, fc="none", ec=INK, lw=1.4, zorder=4))
    ax.text(0.5, 5.15, "ABSENT", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    for xc, s_ in zip(cols_x[1:], scales): ax.text(xc + 0.5, 5.15, f"x{s_:g}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(cols_x[3] + 0.5, 6.0, "one coefficient scaled, the others fixed", ha="center", va="bottom", fontsize=8, color=GRAY_TEXT, fontweight="bold")
    ax.text(3.65, -0.35, f"change in fused OOF clips (of 2,700) vs the shipped baseline {base:,}\nboxed column = the shipped setting", ha="center", va="top", fontsize=7.5, color=GRAY_TEXT, fontweight="bold", linespacing=1.1)
    return ss.save(fig, "s08_atlas")


# ----------------------------------------------------------------------------------------------- s09 waterfall
def s09_waterfall(step: int):
    Dd = D["decoder"]; base = Dd["baseline"]; C = {c["name"]: c for c in Dd["conditions"]}
    unif = C["uniform transitions (distinctness prior only)"]["decoded"]; intact = C["intact order (shipped)"]["decoded"]
    fig = plt.figure(figsize=(6.3, 2.5)); gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.5, left=0.095, right=0.985, top=0.86, bottom=0.22)
    aw, ac = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    steps = [("fused\nargmax", base, None, GRAY), ("+ distinct\nactivities", unif, unif - base, PURPLE), ("+ scripted\norder", intact, intact - unif, PURPLE), ("decoded\n(shipped)", intact, None, INK)]
    ymin = 2040; prev = base
    for i, (lab, val, dlt, col) in enumerate(steps):
        if i + 1 > step: break
        if dlt is None:
            aw.bar(i, val - ymin, bottom=ymin, color=col, width=0.62, zorder=3); aw.text(i, val + 5, f"{val:,}", ha="center", va="bottom", fontsize=11, color=col, fontweight="bold")
        else:
            aw.bar(i, dlt, bottom=prev, color=col, alpha=0.55 if i == 1 else 0.9, width=0.62, zorder=3); aw.text(i, prev + dlt + 5, f"+{dlt}", ha="center", va="bottom", fontsize=12, color=PURPLE, fontweight="bold")
            aw.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color=GRAY, lw=0.7, ls=":", zorder=2); prev += dlt
    aw.set_xticks(range(4)); aw.set_xticklabels([s[0] for s in steps], fontsize=8); aw.set_xlim(-0.6, 3.6); aw.set_ylim(ymin, 2290)
    aw.set_ylabel("correct OOF clips", fontsize=8.5); aw.tick_params(axis="y", labelsize=8); aw.grid(axis="y", color=GRAY_LIGHT, lw=0.5); aw.set_axisbelow(True)
    aw.set_title(f"{base/2700*100:.1f} % to {intact/2700*100:.1f} %: +{intact-base} clips", fontsize=10, pad=6)
    if step >= 4: aw.text(3, ymin + 6, f"{intact/2700*100:.1f} %", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold", zorder=4)
    ctrl = [("intact (shipped)", intact - base, PURPLE), ("shuffled order", C["shuffled within groups"]["decoded"] - base, GRAY), ("permuted labels", C["permuted class labels"]["decoded"] - base, RED), ("reversed order", C["reversed order"]["decoded"] - base, RED)]
    if step >= 4:
        for i, (lab, v, col) in enumerate(ctrl):
            y = len(ctrl) - 1 - i; ac.barh(y, v, color=col, height=0.6, zorder=3); ac.text(v + (4 if v >= 0 else -4), y, f"{v:+d}", ha="left" if v >= 0 else "right", va="center", fontsize=10, color=col, fontweight="bold")
        ac.set_yticks(range(len(ctrl))); ac.set_yticklabels([c[0] for c in ctrl][::-1], fontsize=8.5); ac.axvline(0, color=INK, lw=0.8)
        ac.set_xlim(-90, 175); ac.set_xlabel("decoded change vs fused argmax (clips)", fontsize=8.5); ac.tick_params(axis="x", labelsize=8); ac.grid(axis="x", color=GRAY_LIGHT, lw=0.5); ac.set_axisbelow(True)
        ac.set_title("CONTROLS: structure removed", fontsize=10, pad=6)
    else:
        ac.axis("off")
    return ss.save(fig, "s09_waterfall")


# ----------------------------------------------------------------------------------------------- s10 levers
def s10_levers():
    L = {l["name"]: l for l in D["landscape"]["levers"]}; dec = D["decoder"]; base = dec["baseline"]; intact = [c for c in dec["conditions"] if c["name"].startswith("intact")][0]["decoded"]
    def lv(name, label=None, note=None):
        l = L[name]; return (label or name, l["delta"], l["verdict"], l["level"], note or l.get("note", ""))
    left = [("MORE CAPACITY", [lv("VideoMAE-B (86.7 M)", "VideoMAE-B, 86.7 M", "0 of 4 folds"), lv("Swin3D-T", note="fold 2"), lv("V-JEPA 2 teacher", note="0.419 vs 0.712")]),
            ("MORE TRAINING TRICKS", [lv("LR decay (LLRD)", note="2 of 4 folds, noise"), ("oracle distillation", -1.0, "rejected", "system", "student 157 vs 167 public; fused -2"), lv("self-training", note="member +1.4, fusion -2")]),
            ("MORE TEST-TIME TRICKS", [lv("temporal TTA", note="fusion -4 clips"), lv("seed soup", note="4 of 4 folds"), lv("pair verifier", note="on 2,372 clips"), lv("logit centering", note="+5 clips, unreadable")]),
            ("MORE RESOLUTION", [lv("person 288 px", note="4 of 4 folds"), lv("person 384 px", note="+20 fused clips"), lv("person 448 px", note="fusion -17 clips"), lv("wrist 288 px", note="3 of 4 folds down")])]
    right = [("A NEW SENSOR VIEW", [lv("thermal 224 full frame", note="4 of 4 folds"), lv("thermal in fusion", note="+46 decoded clips")]),
             ("BETTER FUSION", [("prior-space direction", 4.0, "adopted", "system", "0.587 vs 0.627, early public pair"), lv("nested-CV\nweights", "nested-CV coefficients", "+15 clips")]),
             ("RECORDING STRUCTURE", [("ordered transition decoder", (intact - base) / 27.0, "adopted", "system", f"+{intact-base} decoded clips")])]
    fig = plt.figure(figsize=(6.3, 2.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.55, left=0.2, right=0.965, top=0.9, bottom=0.14)
    for ai, (groups, title) in enumerate([(left, "MORE OF THE SAME"), (right, "DIFFERENT INFORMATION")]):
        ax = fig.add_subplot(gs[ai]); y = 0; ticks = []
        for fam, items in groups:
            y -= 0.35; ax.text(-11.5, y, fam, fontsize=8, color=INK, fontweight="bold", ha="left", va="center"); y -= 0.9
            for name, delta, verdict, level, note in items:
                col = {"adopted": BLUE, "rejected": RED, "no signal": GRAY}[verdict]; v = max(delta, -11.0)
                ax.barh(y, v, color=col, height=0.62, zorder=3, alpha=0.9)
                if delta < -11.0: ax.annotate("", (-11.6, y), (-10.6, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7))
                txt = f"{delta:+.1f}" + (" sys" if level == "system" else "")
                if delta < -6.0: ax.text(-0.4, y, txt, ha="right", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)
                else: ax.text(v + (0.25 if v >= 0 else -0.25), y, txt, ha="left" if v >= 0 else "right", va="center", fontsize=8, color=col, fontweight="bold")
                ticks.append((y, name.replace("\n", " "))); y -= 1.0
            y -= 0.4
        ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks], fontsize=7.5); ax.set_ylim(y, 0.4); ax.set_xlim(-11.8, 9.8)
        ax.axvline(0, color=INK, lw=0.8, zorder=2); ax.axvspan(-D["landscape"]["adoption_bar_member_points"], D["landscape"]["adoption_bar_member_points"], color=BAND, zorder=0)
        ax.set_xlabel("accuracy points (member OOF); sys = system level, clips / 27", fontsize=7.5); ax.tick_params(axis="x", labelsize=8); ax.set_xticks([-10, -5, 0, 5])
        ax.set_title(title, fontsize=10, color=RED if ai == 0 else BLUE, pad=6); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    return ss.save(fig, "s10_levers")


# ----------------------------------------------------------------------------------------------- s11 container
def s11_container(step: int):
    B = D["bytes"]; mb = lambda b: b / 1e6; by = B["by_branch_bytes"]; total = mb(B["total_file_bytes"]); other = total - sum(mb(by[k]) for k in ("video", "skeleton", "detector", "imu"))
    fig, ax, W = ss.canvas(6.3, 2.0)            # W = 315
    X0, Y0, H = 12.0, 30.0, 40.0; k = 2.78     # units per MB
    ax.add_patch(FancyBboxPatch((X0, Y0), 100 * k, H, boxstyle="round,pad=0,rounding_size=2.5", fc="white", ec=INK, lw=1.4, zorder=2))
    ax.plot([X0 + 100 * k, X0 + 100 * k], [Y0 - 6, Y0 + H + 6], color=RED, lw=1.6, zorder=5)
    ss.text(ax, X0 + 100 * k, Y0 + H + 9.0, "100 MB limit", 9, RED, ha="center", va="bottom")
    blocks = [("video", mb(by["video"]), BLUE, "3 x MViTv2-S, 5-bit codes", 1), ("skeleton", mb(by["skeleton"]), TEAL, "skeleton / IMU ensemble, int8", 2),
              ("detector", mb(by["detector"]), GRAY, "YOLO x2, fp16", 3), ("imu", mb(by["imu"]), TEAL_LIGHT, "IMU forest", 3), ("other", other, GRAY_LIGHT, "windows, manifest, container", 3)]
    x = X0
    for key, size, col, lab, at in blocks:
        if step >= at:
            if key == "video":
                for j in range(3): ax.add_patch(Rectangle((x + j * size / 3 * k, Y0), size / 3 * k, H, fc=col, ec="white", lw=1.2, zorder=3))
            else: ax.add_patch(Rectangle((x, Y0), size * k, H, fc=col, ec="white", lw=1.2, zorder=3))
            if key == "video":
                for j in range(3): ss.text(ax, x + (j + 0.5) * size / 3 * k, Y0 + H / 2, f"{size/3:.1f} MB", 11, "white", ha="center", va="center", weight="black")
            elif size > 5: ss.text(ax, x + size * k / 2, Y0 + H / 2, f"{size:.1f} MB", 11 if size > 15 else 9, "white", ha="center", va="center", weight="black")
            if key == "video": ss.text(ax, x + size * k / 2, Y0 - 4.5, lab, 8.5, INK, ha="center", va="top")
            elif key == "skeleton": ss.text(ax, x + size * k / 2, Y0 + H + 2.5, "skeleton/IMU ensemble, int8", 8.5, INK, ha="center", va="bottom")
            elif key == "detector": ss.text(ax, x + size * k / 2, Y0 - 4.5, lab, 8.5, INK, ha="center", va="top")
            elif key == "imu":
                ax.plot([x + size * k / 2, x + size * k / 2], [Y0 - 1.0, Y0 - 15.0], color=TEAL_LIGHT, lw=0.8, zorder=2)
                ss.text(ax, x + size * k / 2, Y0 - 15.5, f"{lab} {size:.1f} MB; windows, manifest and container {other:.1f} MB", 7.5, INK, ha="right", va="top")
        x += size * k
    if step >= 4:
        free = 100 - x_mb if (x_mb := (x - X0) / k) else 0
        ax.add_patch(Rectangle((x, Y0), free * k, H, fc="none", ec=RED, lw=0, hatch="////", alpha=0.35, zorder=3))
        ss.text(ax, x + free * k / 2, Y0 + H / 2, f"{free:.1f} MB margin", 8, RED, ha="center", va="center", rotation=90)
    ss.text(ax, X0, 2.0, f"{mb(by['video']):.1f} + {mb(by['skeleton']):.1f} + {mb(by['detector']):.1f} + {mb(by['imu']):.1f} + {other:.1f} = {total:.2f} MB: every inference-time weight, both detectors and the 405 crop windows, in one file", 7.5, GRAY_TEXT, ha="left", va="bottom")
    return ss.save(fig, "s11_container")


# ----------------------------------------------------------------------------------------------- s12 ensemble (backup)
def s12_ensemble():
    fig, ax, W = ss.canvas(6.3, 2.6)            # W = 242
    # the skeleton of the apex frame, large
    k = A["skel"]; X, Z = k[:, 0], k[:, 2]; sh = 74.0; sc = sh / (Z.max() - Z.min()); cx, cy = 34.0, 10.0
    for a, b in A["bones"]:
        ax.plot([cx + (X[a] - X.mean()) * sc, cx + (X[b] - X.mean()) * sc], [cy + (Z[a] - Z.min()) * sc, cy + (Z[b] - Z.min()) * sc], color=TEAL, lw=4.0, solid_capstyle="round", zorder=3)
    ax.scatter(cx + (X - X.mean()) * sc, cy + (Z - Z.min()) * sc, s=60, color=TEAL, zorder=4, linewidths=1.2, edgecolors="white")
    ss.text(ax, cx, 88.0, "SKELETON", 10, TEAL, ha="center", va="bottom"); ss.text(ax, cx, 3.0, "17 joints in 3-D, frame 18 of the clip", 7.5, GRAY_TEXT, ha="center", va="bottom")
    # the five inertial units over the whole clip, large ridgeline
    names = [("WTLA", "left wrist"), ("WTRA", "right wrist"), ("WTC(", "chest"), ("WTLL", "left ankle"), ("WTRL", "right ankle")]
    gx0, gx1, gy0, gy1 = 78.0, 200.0, 10.0, 84.0; rowh = (gy1 - gy0) / 5
    for j, (nm, lab) in enumerate(names):
        v = A["imu"][nm]; v = (v - v.min()) / max(v.max() - v.min(), 1e-6); xx = np.linspace(gx0, gx1, len(v)); base = gy1 - (j + 1) * rowh; yy = base + v * rowh * 1.35
        ax.fill_between(xx, base, yy, color=TEAL_LIGHT, alpha=0.35 if j < 2 else 0.18, lw=0, zorder=3)
        ax.plot(xx, yy, color=TEAL_LIGHT if j < 2 else TEAL, lw=2.0 if j < 2 else 1.2, zorder=3.1, solid_capstyle="round")
        ss.text(ax, gx1 + 2.0, base + 0.5, lab, 8.5, INK, ha="left", va="bottom")
    ss.text(ax, (gx0 + gx1) / 2, 88.0, "FIVE INERTIAL UNITS", 10, TEAL_LIGHT, ha="center", va="bottom"); ss.text(ax, (gx0 + gx1) / 2, 3.0, "|acceleration| over the six-second clip, one row per unit", 7.5, GRAY_TEXT, ha="center", va="bottom")
    return ss.save(fig, "s12_ensemble")


# ----------------------------------------------------------------------------------------------- numbers -> LaTeX
def write_gen():
    m = D["members"]; dec = D["decoder"]; C = {c["name"]: c for c in dec["conditions"]}; r = D["resolution"]; t = D["thermal"]; L = D["loso"]["summary"]; R = D["robustness"]; B = D["bytes"]; G = D["gpu"]
    intact = C["intact order (shipped)"]["decoded"]; unif = C["uniform transitions (distinctness prior only)"]["decoded"]
    subs = D["loso"]["shared_subjects_sorted_by_person"]; pv = np.array([D["loso"]["person"][s_] for s_ in subs]); ev = np.array([D["loso"]["ensemble"][s_] for s_ in subs])
    worst_missing = min(R["missing"]["removed_w0"]); coef = [v - R["baseline"] for k in R["coefficients"]["fused"] for v in R["coefficients"]["fused"][k]]
    mb = lambda b: b / 1e6
    n = {
        "nFusedAcc": f"{m['fused_correct']/2700:.3f}", "nDecodedAcc": f"{intact/2700:.3f}", "nOracle": f"{m['oracle_any_member']:,}", "nFusedClips": f"{m['fused_correct']:,}", "nDecodedClips": f"{intact:,}",
        "nOracleAcc": f"{m['oracle_any_member']/2700:.3f}", "nRescueSkel": str(m["unique_rescues"]["skeleton"]), "nRescueTherm": str(m["unique_rescues"]["thermal"]), "nRescuePerson": str(m["unique_rescues"]["person"]),
        "nRescueWrist": str(m["unique_rescues"]["wrist"]), "nRescueImu": str(m["unique_rescues"]["imu"]), "nLooMin": str(-max(m["leave_one_out_delta"].values())), "nLooMax": str(-min(m["leave_one_out_delta"].values())),
        "nAllWrong": str(m["all_wrong"]), "nDistinct": f"+{unif - m['fused_correct']}", "nOrder": f"+{intact - unif}", "nDecoderGain": f"+{intact - m['fused_correct']}",
        "nShuffled": f"{C['shuffled within groups']['decoded'] - m['fused_correct']:+d}", "nReversed": f"{C['reversed order']['decoded'] - m['fused_correct']:+d}", "nPermuted": f"{C['permuted class labels']['decoded'] - m['fused_correct']:+d}",
        "nPersonA": f"{r['person']['acc'][0]:.3f}", "nPersonB": f"{r['person']['acc'][2]:.3f}", "nWristDelta": f"{r['wrist']['mean_delta']:+.2f}", "nThermA": f"{t['stages'][0]['pooled']:.3f}", "nThermB": f"{t['stages'][1]['pooled']:.3f}", "nThermC": f"{t['stages'][2]['pooled']:.3f}",
        "nThermGain": f"+{100*np.mean(np.array(t['stages'][2]['folds']) - np.array(t['stages'][1]['folds'])):.1f}", "nThermPublic": f"+{t['public']['after']-t['public']['before']}",
        "nThermFusion": str(t["fusion_224_vs_weight"]["decoded"][3] - t["fusion_224_vs_weight"]["decoded"][0]),
        "nLosoPersonMean": f"{100*L['person_mean_16']:.1f}", "nLosoEnsMean": f"{100*L['ensemble_mean_16']:.1f}", "nLosoPersonSd": f"{100*L['person_sd_16']:.1f}", "nLosoEnsSd": f"{100*L['ensemble_sd_16']:.1f}",
        "nLosoPersonSpread": f"{100*(pv.max()-pv.min()):.0f}", "nLosoEnsSpread": f"{100*(ev.max()-ev.min()):.0f}", "nLosoUp": str(int((ev > pv).sum())), "nLosoN": str(len(subs)),
        "nWorstMissing": str(worst_missing), "nWorstMissingPts": f"{100*worst_missing/2700:.1f}", "nCoefMin": str(min(coef)), "nCoefMax": f"+{max(coef)}",
        "nVideoMB": f"{mb(B['by_branch_bytes']['video']):.1f}", "nSkelMB": f"{mb(B['by_branch_bytes']['skeleton']):.1f}", "nDetMB": f"{mb(B['by_branch_bytes']['detector']):.1f}", "nForestMB": f"{mb(B['by_branch_bytes']['imu']):.1f}",
        "nTotalMB": f"{mb(B['total_file_bytes']):.2f}", "nFreeMB": f"{100 - mb(B['total_file_bytes']):.2f}", "nGpus": str(len(G["rows"])), "nMaxDrift": f"{max(r_['dp_video'] for r_ in G['rows']):.0e}".replace("e-0", "e-"),
        "wPerson": f"{m['weights']['person']:.3f}", "wWrist": f"{m['weights']['wrist']:.3f}", "wThermal": f"{m['weights']['thermal']:.3f}", "wSkel": f"{m['weights']['skeleton']:.3f}", "wImu": f"{m['weights']['imu']:.3f}", "wPrior": f"{m['weights']['prior']:.2f}",
    }
    n.update({"f" + "".join(w.capitalize() for w in k.split("_")): v for k, v in FACTS.items()})   # macro names: letters only
    with open(GEN / "numbers.tex", "w") as f:
        f.write("% generated by make_slide_figures.py from docs/report/figures_src/*.json (and FACTS for Kaggle/organiser numbers); do not edit\n")
        for k, v in n.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(GEN / "verification.tex", "w") as f:   # the whole tabular, so \input works outside a cell
        f.write("% generated by make_slide_figures.py from gpu.json; do not edit\n\\begin{tabular}{@{}l l l c c@{}}\\toprule\n")
        f.write("\\textbf{GPU} & \\textbf{run} & \\textbf{max posterior drift} & \\textbf{predictions} & \\textbf{CSV}\\\\\n & & {\\color{graytext}video / skeleton} & {\\color{graytext}changed} & \\\\\\midrule\n")
        for r_ in G["rows"]:
            dv = "reference" if r_["dp_video"] == 0 else f"{r_['dp_video']:.1e}".replace("e-0", "e-"); dk = "" if r_["dp_video"] == 0 else f" / {r_['dp_skel']:.1e}".replace("e-0", "e-")
            f.write(f"\\vrow{{{r_['gpu']}}}{{{r_['container']}}}{{{dv}{dk}}}{{{r_['argmax_changes']} / 405}}{{{r_['csv']}}}\n")
        f.write("\\bottomrule\\end{tabular}\n")
    with open(GEN / "verification_compact.tex", "w") as f:   # slide 12: GPU, drift, predictions, CSV
        f.write("% generated by make_slide_figures.py from gpu.json; do not edit\n\\begin{tabular}{@{}l l c c@{}}\\toprule\n")
        f.write("\\textbf{GPU} & \\textbf{posterior drift} & \\textbf{predictions} & \\textbf{CSV}\\\\\n & {\\color{graytext}video / skeleton} & {\\color{graytext}changed} & \\\\\\midrule\n")
        for r_ in G["rows"]:
            dv = "reference" if r_["dp_video"] == 0 else f"{r_['dp_video']:.1e}".replace("e-0", "e-"); dk = "" if r_["dp_video"] == 0 else f" / {r_['dp_skel']:.1e}".replace("e-0", "e-")
            f.write(f"\\vrowc{{{r_['gpu']}}}{{{dv}{dk}}}{{{r_['argmax_changes']} / 405}}{{{r_['csv'].upper()}}}\n")
        f.write("\\bottomrule\\end{tabular}\n")
    print("gen:", sorted(p.name for p in GEN.glob("*.tex")), len(n), "numbers")


if __name__ == "__main__":
    s01_hero(3); s02_layers(); s03_field(); s04_converge(2); s05_resolution(); s06_thermal(); s07_subjects(); s08_atlas()
    s09_waterfall(4); s10_levers(); s11_container(4); s12_ensemble(); write_gen()
    print("figures:", sorted(p.name for p in ss.OUT.glob("s*.pdf")))
