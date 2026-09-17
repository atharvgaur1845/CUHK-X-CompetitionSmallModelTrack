#!/usr/bin/env python3
"""Generate every figure of the report (vector PDF, docs/report/figures/) from the data tables
built by figdata.py (docs/report/figures_src/*.json). Deterministic: no randomness anywhere.

    python3 generate_all_figures.py            # all figures
    python3 generate_all_figures.py fig02 fig05  # a subset
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon
from matplotlib.lines import Line2D
import figstyle as st
import figdata

st.use()
D = figdata.build_all()
MEM = figdata.MEMBERS
NICE = {"person": "person", "wrist": "wrist", "thermal": "thermal", "skeleton": "skeleton", "imu": "IMU forest"}


def pts(clips: float) -> float:
    return 100.0 * clips / 2700.0


# =====================================================================================
# FIG 1  graphical abstract
# =====================================================================================
def _load_clip_assets():
    from PIL import Image
    import pandas as pd
    s = D["samples"]["abstract_clip"]; B = Path(s["base"]); a, u, t = s["activity"], s["user"], s["trial"]
    ir = sorted(glob.glob(f"{B}/IR/{a}/{u}/{t}/*.png")); dp = sorted(glob.glob(f"{B}/Depth_Color/{a}/{u}/{t}/*.png")); th = sorted(glob.glob(f"{B}/Thermal/{a}/{u}/{t}/*.jpg"))
    sk = sorted(glob.glob(f"{B}/Skeleton/{a}/{u}/{t}/predictions/*.json"))
    mid = lambda L: L[len(L) // 2]
    out = {"ir": np.asarray(Image.open(mid(ir)).convert("L")), "depth": np.asarray(Image.open(mid(dp)).convert("RGB")),
           "thermal": np.asarray(Image.open(mid(th)).convert("RGB")), "n_ir": len(ir), "n_th": len(th)}
    out["skel"] = np.array(json.load(open(mid(sk)))[0]["keypoints"])
    # IMU: acceleration magnitude per device, both files
    dev = {}
    for f in glob.glob(f"{B}/IMU/{a}/{u}/{t}/*.csv"):
        df = pd.read_csv(f); name = df.iloc[:, 1].str[:4]
        acc = np.sqrt((df.iloc[:, 2:5].astype(float) ** 2).sum(1)).to_numpy()
        for dn in name.unique(): dev[dn] = acc[(name == dn).to_numpy()]
    out["imu"] = dev
    r = pd.read_csv(glob.glob(f"{B}/Radar/{a}/{u}/{t}/*.csv")[0]); out["radar"] = r[["x", "y", "v"]].to_numpy()
    return out


def fig01_abstract():
    A = _load_clip_assets(); m = D["members"]; by = D["bytes"]["by_branch_bytes"]
    fig = plt.figure(figsize=(7.16, 4.45)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.set_aspect("equal"); st.despine_all(ax)
    F0, F1 = st.sz(st.S_SMALL - 0.5), st.S_SMALL   # 7.5 and 8 pt
    # ---- sensor tiles (left column)
    XT, WT, HT = 0.8, 23.0, 10.6; ys = [49.6, 38.1, 26.6, 15.1, 3.6]
    tiles = [("IR + depth video", "640 x 480, 16 frames", st.BLUE), ("thermal video", "320 x 240, 16 frames", st.ORANGE), ("skeleton", "17 joints, 3-D", st.TEAL),
             ("five inertial units", "wrists, ankles, chest", st.TEAL_LIGHT), ("radar", "session key; scene repeated 3x", st.PURPLE)]
    for (lab, sub, col), y in zip(tiles, ys):
        ax.add_patch(FancyBboxPatch((XT, y), WT, HT, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=st.GRAY_LIGHT, lw=0.6, zorder=1))
        ax.add_patch(Rectangle((XT, y), 0.7, HT, fc=col, ec="none", zorder=2))
        ax.text(XT + 1.4, y + HT - 0.6, lab, fontsize=st.S_BODY, color=st.INK, va="top", zorder=3)
        ax.text(XT + 1.4, y + HT - 2.9, sub, fontsize=F0, color=st.GRAY_TEXT, va="top", zorder=3)
    CX0 = XT + 1.4; CY = lambda y: (y + 0.6, y + HT - 5.2)   # content box below the two text lines
    y0, y1 = CY(ys[0]); h = y1 - y0; w = h * 4 / 3
    ax.imshow(A["ir"], cmap="gray", extent=(CX0, CX0 + w, y0, y1), zorder=2, aspect="auto"); ax.imshow(A["depth"], extent=(CX0 + w + 0.5, CX0 + 2 * w + 0.5, y0, y1), zorder=2, aspect="auto")
    ax.text(CX0 + 2 * w + 1.3, (y0 + y1) / 2, "infrared,\ndepth", fontsize=F0, color=st.GRAY_TEXT, va="center")
    y0, y1 = CY(ys[1]); h = y1 - y0; w = h * 4 / 3
    ax.imshow(A["thermal"], extent=(CX0, CX0 + w, y0, y1), zorder=2, aspect="auto")
    ax.text(CX0 + w + 0.8, (y0 + y1) / 2, "heat of body\nand objects", fontsize=F0, color=st.GRAY_TEXT, va="center")
    y0, y1 = CY(ys[2]); k = A["skel"]; X, Z = k[:, 0], k[:, 2]; sc = (y1 - y0) / (Z.max() - Z.min()); cx = CX0 + 3.4
    for i, j in D["samples"]["h36m_bones"]:
        ax.plot([cx + (X[i] - X.mean()) * sc, cx + (X[j] - X.mean()) * sc], [y0 + (Z[i] - Z.min()) * sc, y0 + (Z[j] - Z.min()) * sc], color=st.TEAL, lw=1.4, zorder=3, solid_capstyle="round")
    ax.scatter(cx + (X - X.mean()) * sc, y0 + (Z - Z.min()) * sc, s=7, color=st.TEAL, zorder=4, linewidths=0)
    ax.text(cx + 4.6, (y0 + y1) / 2, "tracked joints,\nworld frame", fontsize=F0, color=st.GRAY_TEXT, va="center")
    y0, y1 = CY(ys[3]); names = ["WTLA", "WTRA", "WTC(", "WTLL", "WTRL"]; rowh = (y1 - y0) / 5
    for i, nm in enumerate(names):
        v = A["imu"][nm]; v = (v - v.min()) / max(v.max() - v.min(), 1e-6)
        xx = np.linspace(CX0, CX0 + 12.0, len(v)); yy = y1 - (i + 0.95) * rowh + v * rowh * 0.85
        ax.plot(xx, yy, color=st.TEAL_LIGHT, lw=0.8, zorder=3)
    ax.text(CX0 + 12.8, (y0 + y1) / 2, "|acceleration|,\n5 devices", fontsize=F0, color=st.GRAY_TEXT, va="center")
    y0, y1 = CY(ys[4]); R = A["radar"]; w = 6.5
    px = CX0 + (R[:, 0] - R[:, 0].min()) / (R[:, 0].max() - R[:, 0].min()) * w; py = y0 + (R[:, 1] - R[:, 1].min()) / (R[:, 1].max() - R[:, 1].min()) * (y1 - y0)
    ax.scatter(px, py, s=1.5, color=st.PURPLE, alpha=0.45, zorder=3, linewidths=0)
    tx = CX0 + w + 1.0; acts = ["jog", "squats", "jacks"]; bw = (XT + WT - 0.8 - tx) / 3 - 0.2
    for rep in range(3):
        for j, a in enumerate(acts):
            yy = y1 - 0.9 - rep * 1.6; ax.add_patch(Rectangle((tx + j * (bw + 0.2), yy - 0.65), bw, 1.3, fc=st.PURPLE, alpha=0.22 + 0.26 * j, ec="none", zorder=3))
            if rep == 0: ax.text(tx + j * (bw + 0.2) + bw / 2, yy, a, fontsize=F0, ha="center", va="center", color="white" if j == 2 else st.INK, zorder=4)
    # ---- specialist nodes
    XN, WN, HN = 30.0, 19.5, 7.2; acc = m["member_alone"]
    MB = {"person": by["video"] / 3e6, "wrist": by["video"] / 3e6, "thermal": by["video"] / 3e6, "skeleton": by["skeleton"] / 1e6, "imu": by["imu"] / 1e6}
    nodes = [("person", "person crop, 384 px", "MViTv2-S", 51.3), ("wrist", "wrist crop, 224 px", "MViTv2-S", 39.8), ("thermal", "thermal frame, 224 px", "MViTv2-S", 28.3),
             ("skeleton", "skeleton + IMU", "20-member ensemble", 16.8), ("imu", "IMU statistics", "ExtraTrees forest", 5.3)]
    ny = {}
    for key, t1, t2, y in nodes:
        col = st.MEMBER[key]; ny[key] = y + HN / 2
        ax.add_patch(FancyBboxPatch((XN, y), WN, HN, boxstyle="round,pad=0,rounding_size=0.5", fc="white", ec=col, lw=1.0, zorder=2))
        ax.text(XN + 1.0, y + HN - 0.6, t1, fontsize=st.S_BODY, color=col, va="top", zorder=3)
        ax.text(XN + 1.0, y + HN - 3.0, t2, fontsize=F0, color=st.GRAY_TEXT, va="top", zorder=3)
        ax.text(XN + 1.0, y + 0.5, f"{MB[key]:.1f} MB", fontsize=F0, color=st.GRAY_TEXT, va="bottom", zorder=3)
        ax.text(XN + WN - 1.0, y + 0.4, f"{acc[key]/2700:.3f}", fontsize=st.S_LABEL + 1, color=col, va="bottom", ha="right", zorder=3)
    ax.text(XN + WN, 60.4, "specialists (accuracy alone, OOF)", fontsize=F0, color=st.GRAY_TEXT, ha="right", va="top")
    ax.text(XT, 61.8, "five sensing streams (one training clip: jumping jacks)", fontsize=F0, color=st.GRAY_TEXT, ha="left", va="top")
    tile_mid = lambda i: (XT + WT, ys[i] + HT / 2)
    for i, key in [(0, "person"), (0, "wrist"), (1, "thermal"), (2, "skeleton"), (3, "skeleton"), (3, "imu")]:
        st.arrow(ax, tile_mid(i), (XN, ny[key]), color=st.MEMBER[key], lw=0.8, ms=6, coords="data")
    ax.text(XT + WT + 3.3, ys[0] + HT / 2 - 3.4, "YOLO11n\ncrops", fontsize=F0, color=st.BLUE, ha="center", va="top", linespacing=1.0)
    # ---- fusion
    XF, WF, HF, YF = 54.5, 15.5, 16.0, 23.0
    ax.add_patch(FancyBboxPatch((XF, YF), WF, HF, boxstyle="round,pad=0,rounding_size=0.6", fc=st.PAPER, ec=st.INK, lw=1.0, zorder=2))
    ax.text(XF + WF / 2, YF + HF - 0.9, "prior-corrected\nlog-linear fusion", fontsize=st.S_BODY, color=st.INK, ha="center", va="top", zorder=3, linespacing=1.1)
    ax.text(XF + WF / 2, YF + HF - 7.0, r"$\sum_i w_i \log p_i$", fontsize=st.S_LABEL + 2, color=st.INK, ha="center", va="top", zorder=3)
    ax.text(XF + WF / 2, YF + 0.7, "coefficients by\nnested CV", fontsize=F0, color=st.GRAY_TEXT, ha="center", va="bottom", zorder=3, linespacing=1.05)
    for key in MEM: st.arrow(ax, (XN + WN, ny[key]), (XF, YF + HF / 2 + (ny[key] - 32.5) * 0.26), color=st.MEMBER[key], lw=0.9, ms=6, coords="data")
    ax.text(XF + WF / 2, YF + HF + 1.0, f"fused {m['fused_correct']/2700:.3f}", fontsize=st.S_SMALL, color=st.GRAY_TEXT, ha="center", va="bottom")
    # ---- decoder
    XD, WD, HD, YD = 73.0, 14.0, 16.0, 23.0
    ax.add_patch(FancyBboxPatch((XD, YD), WD, HD, boxstyle="round,pad=0,rounding_size=0.6", fc="white", ec=st.PURPLE, lw=1.0, zorder=2))
    ax.text(XD + WD / 2, YD + HD - 0.9, "ordered\ntransition decoder", fontsize=st.S_BODY, color=st.PURPLE, ha="center", va="top", zorder=3, linespacing=1.1)
    ax.text(XD + WD / 2, YD + HD - 7.0, "activity-script\nprior, label-free,\nper recording", fontsize=F0, color=st.GRAY_TEXT, ha="center", va="top", zorder=3, linespacing=1.1)
    st.arrow(ax, (XF + WF, YF + HF / 2), (XD, YD + HD / 2), color=st.INK, lw=1.2, ms=9, coords="data")
    yb = 2.9; xm = XD + WD / 2
    ax.plot([XT + WT, XT + WT + 1.5, xm, xm], [ys[4] + HT / 2, yb, yb, YD - 0.3], color=st.PURPLE, lw=0.8, zorder=1, solid_joinstyle="round")
    st.arrow(ax, (xm, YD - 1.2), (xm, YD), color=st.PURPLE, lw=0.8, ms=6, coords="data")
    ax.text(xm - 1.0, yb + 0.4, "recording group, clip order", fontsize=F0, color=st.PURPLE, ha="right", va="bottom")
    ax.text(XD + WD / 2, YD + HD + 1.0, f"decoded {D['decoder']['conditions'][0]['decoded']/2700:.3f}", fontsize=st.S_SMALL, color=st.GRAY_TEXT, ha="center", va="bottom")
    # ---- output rail: four facts a reader keeps, everything else subordinate
    XO, WO = 88.0, 11.8
    st.arrow(ax, (XD + WD, YD + HD / 2), (XO, YD + HD / 2), color=st.INK, lw=1.2, ms=9, coords="data")
    ax.add_patch(FancyBboxPatch((XO, 3.6), WO, 57.0, boxstyle="round,pad=0,rounding_size=0.8", fc=st.INK, ec="none", zorder=2))
    rows = [("5", "sensors", st.S_HUGE + 4), ("40", "activities", st.S_HUGE + 4), ("96 MB", "one file", st.S_BIG + 1), ("0.877", "private score", st.S_BIG + 4)]
    yy = 59.2
    for big, small, sz_ in rows:
        ax.text(XO + WO / 2, yy, big, fontsize=sz_, color="white", ha="center", va="top", zorder=3)
        ax.text(XO + WO / 2, yy - sz_ * 0.16 - 0.8, small, fontsize=st.S_SMALL, color="#C9D2E0", ha="center", va="top", zorder=3)
        yy -= 13.6
    ax.text(XO + WO / 2, 5.0, "unseen people\n14th of 307 teams\n0/405 changed\non 5 GPUs", fontsize=F0, color="#C9D2E0", ha="center", va="bottom", zorder=3, linespacing=1.15)
    ax.text(XN, 0.2, "accuracies: pooled out-of-fold over 2,700 training clips of held-out subjects. private: Kaggle leaderboard, 204 clips of unseen subjects.",
            fontsize=F0, color=st.GRAY_TEXT, va="bottom")
    return st.save(fig, "fig01_abstract")


# =====================================================================================
# FIG 2  complementarity: which members are right on which clips
# =====================================================================================
def fig02_complementarity():
    m = D["members"]; pats = m["patterns"]
    keep = [p for p in pats if p["n"] >= 25]; other_n = sum(p["n"] for p in pats if p["n"] < 25); other_f = sum(p["fused_correct"] for p in pats if p["n"] < 25)
    fig = plt.figure(figsize=(7.16, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.6, 1.35], height_ratios=[2.2, 1.0], hspace=0.06, wspace=0.3, left=0.07, right=0.99, top=0.91, bottom=0.05)
    axb = fig.add_subplot(gs[0, 0]); axm = fig.add_subplot(gs[1, 0], sharex=axb); axl = fig.add_subplot(gs[:, 1])
    n = len(keep) + 1; x = np.arange(n)
    ns = [p["n"] for p in keep] + [other_n]; fs = [p["fused_correct"] for p in keep] + [other_f]
    axb.bar(x, ns, color=st.GRAY_LIGHT, width=0.72, zorder=2, label="clips with this pattern")
    axb.bar(x, fs, color=st.INK, width=0.72, zorder=3, label="of which the fusion is right")
    for xi, nn, ff in zip(x, ns, fs):
        axb.text(xi, nn + 8, f"{nn}", ha="center", va="bottom", fontsize=st.S_SMALL, color=st.INK)
        if nn - ff >= 20: axb.text(xi, ff + (nn - ff) / 2, f"{ff}", ha="center", va="center", fontsize=st.sz(st.S_SMALL - 1), color=st.INK)
    axb.set_ylim(0, 760); axb.set_ylabel("clips (of 2,700)"); st.hgrid(axb); axb.tick_params(axis="x", length=0, labelbottom=False)
    axb.legend(loc="upper right", ncol=1, fontsize=st.S_SMALL, handlelength=1.2, bbox_to_anchor=(1.0, 1.02))
    axb.set_title("Which members are right, and whether the fusion is", fontsize=st.S_TITLE)
    # membership matrix
    for j, key in enumerate(MEM):
        axm.axhspan(j - 0.5, j + 0.5, color=st.BAND if j % 2 == 0 else "white", zorder=0)
        for xi, p in enumerate(keep + [None]):
            if p is None: continue
            on = key in p["members"]
            axm.scatter(xi, j, s=26 if on else 9, color=st.MEMBER[key] if on else st.GRAY_LIGHT, zorder=3, linewidths=0)
        idx = [xi for xi, p in enumerate(keep) if key in p["members"]]
    for xi, p in enumerate(keep):
        ii = [MEM.index(k) for k in p["members"]]
        if len(ii) > 1: axm.plot([xi, xi], [min(ii), max(ii)], color=st.GRAY, lw=0.8, zorder=2)
    axm.text(n - 1, 2, "other\npatterns", ha="center", va="center", fontsize=st.sz(st.S_SMALL - 1), color=st.GRAY_TEXT)
    axm.set_yticks(range(5)); axm.set_yticklabels([NICE[k] for k in MEM]); axm.set_ylim(-0.6, 4.6); axm.invert_yaxis()
    for s in axm.spines.values(): s.set_visible(False)
    axm.tick_params(axis="both", length=0); axm.set_xticks([]); axm.set_xlim(-0.6, n - 0.4)
    # single-member patterns, listed once instead of five overlapping callouts
    singles = sorted([p for p in keep if len(p["members"]) == 1], key=lambda p: -p["n"])
    axb.text(0.22, 0.47, "clips only one member gets right:\n" + "   ".join(f"{NICE[p['members'][0]]} {p['n']}" for p in singles),
             transform=axb.transAxes, ha="left", va="top", fontsize=st.S_SMALL, color=st.INK, linespacing=1.3)
    axb.annotate(f"oracle over the five members {m['oracle_any_member']}/2,700 = {m['oracle_any_member']/2700:.3f}\nfusion {m['fused_correct']}/2,700 = {m['fused_correct']/2700:.3f}",
                 (0.99, 0.74), xycoords="axes fraction", ha="right", va="top", fontsize=st.S_SMALL, color=st.INK)
    st.panel_label(axb, "a", x=-0.06, y=1.02)
    # leave-one-out lollipops
    loo = m["leave_one_out_delta"]; order = sorted(MEM, key=lambda k: -loo[k]); yv = np.arange(len(order))
    for yi, k in zip(yv, order):
        axl.plot([0, loo[k]], [yi, yi], color=st.MEMBER[k], lw=2.4, solid_capstyle="round", zorder=2)
        axl.scatter(loo[k], yi, s=30, color=st.MEMBER[k], zorder=3)
        axl.text(min(loo[k] / 2, -9), yi + 0.22, f"{loo[k]:+d} ({pts(loo[k]):+.1f} pts)", ha="center", va="bottom", fontsize=st.S_SMALL, color=st.INK)
    axl.set_yticks(yv); axl.set_yticklabels([NICE[k] for k in order]); axl.set_xlim(-60, 3); axl.set_ylim(-0.7, 4.9); axl.axvline(0, color=st.INK, lw=0.6)
    axl.set_xticks([-40, -20, 0]); axl.set_xlabel("fused OOF clips lost when\nthe member is removed", fontsize=st.S_SMALL); axl.tick_params(axis="y", length=0)
    axl.set_title("Leave one member out", fontsize=st.S_TITLE); st.panel_label(axl, "b", x=-0.42, y=1.02)
    axl.spines["left"].set_visible(False)
    return st.save(fig, "fig02_complementarity")


# =====================================================================================
# FIG 3  resolution: the crop-size rule
# =====================================================================================
def fig03_resolution():
    r = D["resolution"]
    fig = plt.figure(figsize=(7.16, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.9, 2.0], wspace=0.12, left=0.07, right=0.995, top=0.88, bottom=0.17)
    ap = fig.add_subplot(gs[0]); aw = fig.add_subplot(gs[1], sharey=ap); asch = fig.add_subplot(gs[2])
    # (a) person
    res, acc = r["person"]["res"], [100 * a for a in r["person"]["acc"]]
    ap.plot(res, acc, "-", color=st.BLUE, lw=1.6, zorder=2); ap.scatter(res, acc, s=30, color=st.BLUE, zorder=3)
    ap.scatter([384], [acc[2]], s=90, facecolor="none", edgecolor=st.BLUE, lw=1.2, zorder=4)
    for x, a, dy in zip(res, acc, [8, 8, 9, 8]): ap.annotate(f"{a:.2f}", (x, a), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=st.S_SMALL, color=st.INK)
    ap.annotate("shipped", (384, acc[2]), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=st.sz(st.S_SMALL - 0.5), color=st.BLUE, fontweight="bold")
    ap.annotate("-0.81 member\n-17 fused (t = -2.0)", (448, acc[3]), xytext=(-4, -24), textcoords="offset points", ha="right", fontsize=st.sz(st.S_SMALL - 1.2), color=st.RED)
    ap.text(228, 72.7, "+1.23\n4/4 folds", fontsize=st.sz(st.S_SMALL - 1.2), color=st.GREEN, ha="left", va="bottom", linespacing=1.0)
    ap.set_xticks(res); ap.set_xticklabels([str(x) for x in res], fontsize=st.sz(st.S_SMALL - 0.5)); ap.set_ylim(70.0, 74.6); ap.set_xlim(200, 472)
    ap.set_ylabel("member accuracy, OOF (%)"); ap.set_title("person crop", color=st.BLUE, fontsize=st.S_LABEL); st.hgrid(ap); ap.set_xlabel("input side (px)", fontsize=st.S_SMALL)
    # (b) wrist
    wres, wacc = r["wrist"]["res"], [100 * a for a in r["wrist"]["acc"]]
    aw.plot(wres, wacc, "-", color=st.BLUE_LIGHT, lw=1.6, zorder=2); aw.scatter(wres, wacc, s=30, color=st.BLUE_LIGHT, zorder=3)
    aw.scatter([224], [wacc[0]], s=90, facecolor="none", edgecolor=st.BLUE_LIGHT, lw=1.2, zorder=4)
    aw.annotate(f"{wacc[0]:.2f}", (224, wacc[0]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=st.S_SMALL, color=st.INK)
    aw.annotate(f"{wacc[1]:.2f}", (288, wacc[1]), xytext=(2, 8), textcoords="offset points", ha="center", fontsize=st.S_SMALL, color=st.INK)
    aw.annotate("shipped", (224, wacc[0]), xytext=(9, -4), textcoords="offset points", ha="left", va="top", fontsize=st.sz(st.S_SMALL - 0.5), color=st.BLUE_LIGHT, fontweight="bold")
    fd = r["wrist"]["fold_delta_224_to_288"]
    aw.text(0.04, 0.04, "-1.36, 3/4 folds\nnegative; per fold\n" + ", ".join(f"{d:+.2f}" for d in fd), transform=aw.transAxes, fontsize=st.sz(st.S_SMALL - 1.4), color=st.RED, ha="left", va="bottom", linespacing=1.1)
    aw.text(0.5, 0.97, "thermal at 288:\na wash (320 x 240\nnative)", transform=aw.transAxes, fontsize=st.sz(st.S_SMALL - 1.4), color=st.ORANGE, ha="center", va="top", linespacing=1.1)
    aw.set_xticks([224, 288]); aw.set_xticklabels(["224", "288"], fontsize=st.sz(st.S_SMALL - 0.5)); aw.set_xlim(196, 316); aw.set_title("wrist crop", color=st.BLUE_LIGHT, fontsize=st.S_LABEL); st.hgrid(aw)
    aw.tick_params(axis="y", labelleft=False, length=0); aw.spines["left"].set_visible(False); aw.set_xlabel("input side (px)", fontsize=st.S_SMALL)
    # (c) schematic: what a pixel of input contains (equal aspect so the grids are squares)
    st.despine_all(asch); asch.set_xlim(0, 100); asch.set_ylim(0, 74); asch.set_aspect("equal")
    fx, fy, fw, fh = 1, 20, 30, 22.5  # the 640x480 IR frame, 4:3
    asch.add_patch(Rectangle((fx, fy), fw, fh, fc="#F5F5F3", ec=st.INK, lw=0.7))
    asch.text(fx, fy + fh + 13.5, "IR frame 640 x 480", fontsize=st.sz(st.S_SMALL - 1), ha="left", va="bottom", color=st.INK)
    asch.text(fx, fy + fh + 7.5, "person box 224 to 480 px", fontsize=st.sz(st.S_SMALL - 1.6), ha="left", va="bottom", color=st.BLUE)
    asch.text(fx, fy + fh + 1.5, "wrist box about 96 px", fontsize=st.sz(st.S_SMALL - 1.6), ha="left", va="bottom", color=st.BLUE_LIGHT)
    asch.add_patch(Rectangle((fx + 9, fy + 1.5), 12, 19.5, fc="none", ec=st.BLUE, lw=1.0))
    asch.add_patch(Rectangle((fx + 17.5, fy + 10), 3.0, 3.0, fc="none", ec=st.BLUE_LIGHT, lw=1.0))
    def grid(x0, y0, side, n, col, lw):
        asch.add_patch(Rectangle((x0, y0), side, side, fc="white", ec=col, lw=1.0, zorder=2))
        for i in range(1, n): asch.plot([x0 + i * side / n] * 2, [y0, y0 + side], color=col, lw=lw, alpha=0.5, zorder=3); asch.plot([x0, x0 + side], [y0 + i * side / n] * 2, color=col, lw=lw, alpha=0.5, zorder=3)
    grid(48, 31, 19, 16, st.BLUE, 0.3); grid(79, 31, 19, 4, st.BLUE_LIGHT, 0.5)
    asch.text(57.5, 53, "person at 384 px", fontsize=st.sz(st.S_SMALL - 0.8), color=st.BLUE, ha="center", va="bottom"); asch.text(88.5, 53, "wrist at 288 px", fontsize=st.sz(st.S_SMALL - 0.8), color=st.BLUE_LIGHT, ha="center", va="bottom")
    asch.text(57.5, 28.5, "one source pixel\nper input pixel:\nreal detail", fontsize=st.sz(st.S_SMALL - 1.6), color=st.INK, ha="center", va="top", linespacing=1.05)
    asch.text(88.5, 28.5, "nine input pixels\nper source pixel:\ninterpolation", fontsize=st.sz(st.S_SMALL - 1.6), color=st.RED, ha="center", va="top", linespacing=1.05)
    st.arrow(asch, (fx + 21, fy + 11), (48, 38), color=st.BLUE, lw=0.7, rad=0.0, ms=6, coords="data"); st.arrow(asch, (fx + 20.5, fy + 13), (79, 45), color=st.BLUE_LIGHT, lw=0.7, rad=-0.3, ms=6, coords="data")
    asch.set_title("why: what one input pixel contains", fontsize=st.S_LABEL, loc="left")
    st.panel_label(ap, "a", x=-0.25, y=1.02); st.panel_label(aw, "b", x=-0.02, y=1.02); st.panel_label(asch, "c", x=-0.01, y=1.02)
    return st.save(fig, "fig03_resolution")


# =====================================================================================
# FIG 4  thermal: the same sensor, a different reading
# =====================================================================================
def _thermal_insets():
    from PIL import Image
    s = D["samples"]["thermal_clip"]; B = Path(s["base"]); a, u, t = s["activity"], s["user"], s["trial"]
    th = sorted(glob.glob(f"{B}/Thermal/{a}/{u}/{t}/*.jpg")); im = Image.open(th[len(th) // 2]).convert("RGB")
    full = np.asarray(im.resize((224, 168)))
    # a person-centred square crop resampled to 128 (illustrative of the 128-px cropped input; window chosen on the frame centre)
    W, H = im.size; side = int(H * 0.9); cx = int(W * 0.55); box = (max(cx - side // 2, 0), 0, max(cx - side // 2, 0) + side, side)
    crop = np.asarray(im.crop(box).resize((128, 128)))
    return crop, full


def fig04_thermal():
    t = D["thermal"]; stages = t["stages"]; fw = t["fusion_224_vs_weight"]
    fig = plt.figure(figsize=(7.16, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 1.1], wspace=0.32, left=0.07, right=0.99, top=0.87, bottom=0.2)
    aa = fig.add_subplot(gs[0]); af = fig.add_subplot(gs[1]); ai = fig.add_subplot(gs[2])
    # (a) member accuracy ladder with per-fold points
    xs = [0, 1, 2]; cols = [st.GRAY, st.GRAY, st.ORANGE]
    aa.plot(xs, [100 * s_["pooled"] for s_ in stages], color=st.GRAY_LIGHT, lw=1.0, zorder=2)
    for x, sname, c in zip(xs, stages, cols):
        p = 100 * sname["pooled"]
        if "folds" in sname: aa.scatter([x + 0.12] * 4, [100 * f for f in sname["folds"]], s=9, color=c, alpha=0.55, zorder=3, linewidths=0)
        aa.scatter([x], [p], s=46, color=c, zorder=4, edgecolor="white", lw=0.6)
        if x < 2: aa.annotate(f"{p:.1f}", (x, p), xytext=(-7, -2), textcoords="offset points", ha="right", va="top", fontsize=st.S_BODY, fontweight="bold", color=st.tc(c))
        else: aa.annotate(f"{p:.1f}", (x, p), xytext=(-8, -9), textcoords="offset points", ha="right", va="top", fontsize=st.S_BODY, fontweight="bold", color=st.tc(c))
    aa.annotate("", (1.9, 100 * stages[2]["pooled"] + 0.4), (1.1, 100 * stages[1]["pooled"] + 0.4), arrowprops=dict(arrowstyle="-|>", color=st.ORANGE, lw=1.0, mutation_scale=8, connectionstyle="arc3,rad=-0.3"))
    aa.text(1.45, 100 * (stages[1]["pooled"] + stages[2]["pooled"]) / 2 + 3.6, "+5.6 points\n4/4 folds", fontsize=st.S_SMALL, color=st.ORANGE, ha="center", va="bottom", fontweight="bold", linespacing=1.0)
    aa.text(2.2, 100 * stages[2]["pooled"] + 3.4, "fold\nvalues", fontsize=st.sz(st.S_SMALL - 1.6), color=st.ORANGE, ha="left", va="bottom", alpha=0.8, linespacing=1.0)
    aa.set_xticks(xs); aa.set_xticklabels(["crop 128 px\n3-D CNN", "full frame 128\n3-D CNN", "full frame 224\nMViTv2-S"], fontsize=st.sz(st.S_SMALL - 0.5))
    aa.set_ylim(46.0, 69.0); aa.set_xlim(-0.5, 2.8); aa.set_ylabel("thermal member\naccuracy, OOF (%)"); st.hgrid(aa)
    aa.set_title("same camera, read at native resolution", fontsize=st.S_LABEL)
    aa.text(0, 47.6, "fusion gain:\n0.00\nat every weight", fontsize=st.sz(st.S_SMALL - 1.2), color=st.GRAY_TEXT, ha="center", va="bottom", linespacing=1.05)
    aa.text(1, 47.6, "fusion gain:\n+0.83\nnot adopted", fontsize=st.sz(st.S_SMALL - 1.2), color=st.GRAY_TEXT, ha="center", va="bottom", linespacing=1.05)
    aa.text(2, 47.6, "fusion gain:\n+46 clips\n+5 public", fontsize=st.sz(st.S_SMALL - 1.2), color=st.ORANGE, ha="center", va="bottom", fontweight="bold", linespacing=1.05)
    # (b) fusion gain vs weight at 224 (decoded OOF) with 128's flat zero
    w = fw["weights"]; d = [v - fw["decoded"][0] for v in fw["decoded"]]
    af.axhline(0, color=st.GRAY, lw=1.2, zorder=2); af.text(0.41, 1.5, "128 px: 0 at\nevery weight", fontsize=st.sz(st.S_SMALL - 1.2), color=st.GRAY_TEXT, ha="right", va="bottom")
    af.plot(w[1:], d[1:], "-", color=st.ORANGE, lw=1.5, zorder=3); af.scatter(w[1:], d[1:], s=18, color=st.ORANGE, zorder=4)
    af.scatter([0.20], [46], s=60, facecolor="none", edgecolor=st.ORANGE, lw=1.1, zorder=5); af.annotate("+46 at w = 0.20\n(shipped 0.17)", (0.20, 46), xytext=(4, -30), textcoords="offset points", ha="center", fontsize=st.sz(st.S_SMALL - 1.2), color=st.INK, linespacing=1.0)
    af.set_xlim(0.05, 0.44); af.set_ylim(-6, 58); af.set_xticks([0.1, 0.2, 0.3, 0.4]); af.set_xlabel("thermal coefficient"); af.set_ylabel("decoded OOF gain\n(clips of 2,700)"); st.hgrid(af)
    af.set_title("224 px: a plateau of gain", fontsize=st.S_LABEL)
    # (c) insets: what the member sees
    crop, full = _thermal_insets(); st.despine_all(ai); ai.set_xlim(0, 100); ai.set_ylim(0, 100)
    ai.imshow(crop, extent=(2, 40, 30, 68), aspect="auto"); ai.imshow(full, extent=(46, 98, 27, 66), aspect="auto")
    ai.text(21, 71, "crop 128 px", fontsize=st.sz(st.S_SMALL - 1), ha="center", va="bottom", color=st.GRAY_TEXT); ai.text(72, 71, "full frame 224 px", fontsize=st.sz(st.S_SMALL - 1), ha="center", va="bottom", color=st.ORANGE)
    ai.text(50, 10, "objects and scene stay in view;\n320 x 240 source, nothing upsampled", fontsize=st.sz(st.S_SMALL - 1.3), ha="center", va="center", color=st.INK, linespacing=1.1)
    ai.set_title("what the member sees", fontsize=st.S_LABEL)
    st.panel_label(aa, "a", x=-0.2, y=1.02); st.panel_label(af, "b", x=-0.42, y=1.02); st.panel_label(ai, "c", x=-0.02, y=1.02)
    return st.save(fig, "fig04_thermal")


# =====================================================================================
# FIG 5  subject generalisation: paired LOSO
# =====================================================================================
def fig05_subjects():
    L = D["loso"]; P, E = L["person"], L["ensemble"]; order = L["shared_subjects_sorted_by_person"]; S = L["summary"]
    fig = plt.figure(figsize=(7.16, 2.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.06, left=0.07, right=0.99, top=0.9, bottom=0.2)
    ax = fig.add_subplot(gs[0]); ad = fig.add_subplot(gs[1], sharey=ax)
    x = np.arange(len(order)); pv = np.array([100 * P[u] for u in order]); ev = np.array([100 * E[u] for u in order])
    for xi, p, e in zip(x, pv, ev):
        c = st.INK if e >= p else st.RED
        ax.plot([xi, xi], [p, e], color=c, lw=1.3, alpha=0.9, zorder=2)
    ax.scatter(x, pv, s=22, color=st.GRAY, zorder=3, label="person view alone", linewidths=0)
    ax.scatter(x, ev, s=26, color=st.INK, zorder=4, label="shipped ensemble", linewidths=0)
    ax.axhline(100 * S["person_mean_16"], color=st.GRAY, lw=0.7, ls=":", zorder=1); ax.axhline(100 * S["ensemble_mean_16"], color=st.INK, lw=0.7, ls=":", zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([u.replace("user", "subject ") for u in order], fontsize=st.sz(st.S_SMALL - 1), rotation=45, ha="right")
    ax.set_ylim(60, 94); ax.set_ylabel("accuracy, subject held out (%)"); st.hgrid(ax)
    ax.legend(loc="upper left", fontsize=st.S_SMALL, handletextpad=0.3, markerscale=1.1)
    n_up = int((ev > pv).sum()); ax.set_title(f"Fusion raises {n_up} of {len(order)} subjects, narrows the spread", fontsize=st.sz(st.S_LABEL - 0.5))
    ax.set_xlabel("held-out training subject, sorted by person-view accuracy", fontsize=st.S_SMALL)
    # worst subject callout
    i_min = int(np.argmin(ev)); ax.annotate(f"worst subject\n{ev[i_min]:.1f}", (x[i_min], ev[i_min]), xytext=(18, -18), textcoords="offset points", fontsize=st.sz(st.S_SMALL - 1), color=st.INK,
                                              arrowprops=dict(arrowstyle="-", color=st.GRAY, lw=0.6))
    i_neg = [i for i in range(len(order)) if ev[i] < pv[i]]
    for i in i_neg: ax.annotate(f"{ev[i]-pv[i]:+.1f}", (x[i], ev[i]), xytext=(0, -10), textcoords="offset points", ha="center", fontsize=st.sz(st.S_SMALL - 1.5), color=st.RED)
    # distribution strip: mean +- sd and min/max
    for j, (vals, c, lab) in enumerate([(pv, st.GRAY, "person"), (ev, st.INK, "ensemble")]):
        xx = j * 1.0
        ad.plot([xx, xx], [vals.min(), vals.max()], color=c, lw=0.8, zorder=1)
        ad.add_patch(Rectangle((xx - 0.22, vals.mean() - vals.std(ddof=1)), 0.44, 2 * vals.std(ddof=1), fc=c, alpha=0.25, ec="none", zorder=2))
        ad.plot([xx - 0.3, xx + 0.3], [vals.mean()] * 2, color=c, lw=1.6, zorder=3)
        ad.text(xx, vals.max() + 0.8, f"sd {vals.std(ddof=1):.1f}\nspread {vals.max()-vals.min():.0f}", ha="center", va="bottom", fontsize=st.sz(st.S_SMALL - 1.2), color=st.tc(c), linespacing=1.0)
        ad.text(xx, vals.min() - 0.9, lab, ha="center", va="top", fontsize=st.sz(st.S_SMALL - 1.0), color=st.tc(c))
    ad.set_xlim(-0.8, 1.8); ad.tick_params(axis="y", labelleft=False, length=0); ad.set_xticks([])
    for s_ in ad.spines.values(): s_.set_visible(False)
    ad.text(0.5, 0.0, "mean " + f"{100*S['person_mean_16']:.1f} to {100*S['ensemble_mean_16']:.1f}", transform=ad.transAxes, ha="center", va="bottom", fontsize=st.sz(st.S_SMALL - 1), color=st.INK, fontweight="bold")
    st.panel_label(ax, "a", x=-0.07, y=1.02); st.panel_label(ad, "b", x=-0.05, y=1.02)
    return st.save(fig, "fig05_subjects")


# =====================================================================================
# FIG 6  stability atlas
# =====================================================================================
def _cell_color(v, vmax=48):
    # diverging: red for loss, blue for gain, white at 0; linear in |v|
    import matplotlib.colors as mc
    t = min(abs(v) / vmax, 1.0)
    base = np.array(mc.to_rgb(st.RED if v < 0 else st.BLUE)); white = np.ones(3)
    return tuple(white * (1 - 0.85 * t) + base * 0.85 * t)


def fig06_robustness():
    R = D["robustness"]; base = R["baseline"]; C = R["coefficients"]
    cols = ["person", "wrist", "thermal", "skeleton", "imu", "prior"]; col_lab = ["person", "wrist", "thermal", "skeleton", "IMU forest", "class prior"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5)); fig.subplots_adjust(left=0.27, right=0.9, top=0.8, bottom=0.14)
    # rows: two "sensor absent" conditions, then five coefficient scales
    rows = [("absent, constant\nposterior", [R["missing"]["constant_posterior"][MEM.index(c)] if c != "prior" else None for c in cols]),
            ("absent, removed\nfrom fusion", [R["missing"]["removed_w0"][MEM.index(c)] if c != "prior" else None for c in cols])]
    for j, sc in enumerate(C["scales"]):
        rows.append((f"coefficient x{sc:g}", [C["fused"][c][j] - base for c in cols]))
    ypos = [0, 1, 2.5, 3.5, 4.5, 5.5, 6.5]
    for (lab, vals), y in zip(rows, ypos):
        for i, v in enumerate(vals):
            if v is None: ax.text(i + 0.5, y + 0.5, "n/a", ha="center", va="center", fontsize=st.sz(st.S_SMALL - 1), color=st.GRAY_TEXT); continue
            ax.add_patch(Rectangle((i, y), 1, 1, fc=_cell_color(v), ec="white", lw=1.5))
            ax.text(i + 0.5, y + 0.5, f"{v:+d}" if v else "0", ha="center", va="center", fontsize=st.S_BODY, color=st.INK)
    ax.add_patch(Rectangle((0, 4.5), 6, 1, fc="none", ec=st.INK, lw=1.0, zorder=5))
    ax.set_xlim(0, 6); ax.set_ylim(7.5, 0); ax.set_xticks(np.arange(6) + 0.5); ax.set_xticklabels(col_lab, fontsize=st.sz(st.S_SMALL - 0.5), rotation=25, ha="left"); ax.xaxis.tick_top()
    ax.set_yticks([y + 0.5 for y in ypos]); ax.set_yticklabels([r[0] for r in rows], fontsize=st.sz(st.S_SMALL - 0.5))
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0)
    ax.text(0, 2.08, "one coefficient scaled, the others fixed (shipped values boxed):", fontsize=st.sz(st.S_SMALL - 1), color=st.GRAY_TEXT, va="top")
    ax.text(0, 7.7, "worst case with a sensor absent: -48 clips = -1.8 points;\nany coefficient from x0.5 to x1.5: within -24 to +7 clips", fontsize=st.sz(st.S_SMALL - 1), color=st.INK, ha="left", va="top", linespacing=1.1)
    fig.text(0.02, 0.985, "Fused OOF change under perturbation (clips of 2,700)", ha="left", va="top", fontsize=st.S_TITLE, fontweight="bold")
    return st.save(fig, "fig06_robustness")


# =====================================================================================
# FIG 7  decoder decomposition
# =====================================================================================
def fig07_decoder():
    Dd = D["decoder"]; base = Dd["baseline"]; C = {c["name"]: c for c in Dd["conditions"]}
    intact, unif = C["intact order (shipped)"], C["uniform transitions (distinctness prior only)"]
    fig = plt.figure(figsize=(7.16, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.5, left=0.07, right=0.985, top=0.87, bottom=0.24)
    aw = fig.add_subplot(gs[0]); ac = fig.add_subplot(gs[1])
    # waterfall
    steps = [("pre-decoder\nfused\nargmax", base, None, st.GRAY), ("+ distinct\nactivities in\na recording", unif["decoded"], unif["decoded"] - base, st.PURPLE),
             ("+ scripted\norder\n(transitions)", intact["decoded"], intact["decoded"] - unif["decoded"], st.PURPLE), ("decoded\n(shipped)", intact["decoded"], None, st.INK)]
    x = np.arange(4); prev = base
    for i, (lab, val, dlt, col) in enumerate(steps):
        if dlt is None:
            aw.bar(i, val - 2000, bottom=2000, color=col, width=0.62, zorder=3)
            aw.text(i, val + 4, f"{val}", ha="center", va="bottom", fontsize=st.S_BODY, fontweight="bold", color=st.tc(col))
        else:
            aw.bar(i, dlt, bottom=prev, color=col, alpha=0.55 if i == 1 else 0.85, width=0.62, zorder=3)
            aw.text(i, prev + dlt + 4, f"+{dlt}", ha="center", va="bottom", fontsize=st.S_BODY, fontweight="bold", color=st.tc(col))
            aw.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color=st.GRAY, lw=0.6, ls=":", zorder=2)
        prev = val
    aw.set_ylim(2000, 2300); aw.set_xticks(x); aw.set_xticklabels([s_[0] for s_ in steps], fontsize=st.sz(st.S_SMALL - 0.5), linespacing=1.0); aw.set_ylabel("correct OOF clips (of 2,700)")
    st.hgrid(aw); aw.set_title("Where the decoder's +131 comes from", fontsize=st.S_TITLE)
    aw.text(0.02, 0.9, f"{pts(base):.1f} %  to  {pts(intact['decoded']):.1f} %  (+{pts(131):.1f} points)", transform=aw.transAxes, fontsize=st.S_SMALL, color=st.INK)
    # controls: rescues vs harms
    names = ["intact order (shipped)", "uniform transitions (distinctness prior only)", "shuffled within groups", "permuted class labels", "reversed order"]
    short = ["intact order", "distinctness prior only", "order shuffled\nwithin groups", "class labels permuted", "order reversed"]
    y = np.arange(len(names))[::-1]
    for yi, nm, sh in zip(y, names, short):
        c = C[nm]; net = c["decoded"] - base; col = st.PURPLE if net > 20 else (st.GRAY if net >= 0 else st.RED)
        ac.barh(yi, c["rescues"], color=st.PURPLE, alpha=0.75, height=0.58, zorder=3); ac.barh(yi, -c["harms"], color=st.RED, alpha=0.75, height=0.58, zorder=3)
        ac.text(c["rescues"] + 4, yi, f"net {net:+d}", va="center", ha="left", fontsize=st.S_BODY, fontweight="bold", color=st.tc(col))
        ac.text(-c["harms"] - 4, yi, f"{c['harms']}", va="center", ha="right", fontsize=st.S_SMALL, color=st.RED)
        ac.text(c["rescues"] - 4, yi, f"{c['rescues']}", va="center", ha="right", fontsize=st.S_SMALL, color="white")
    ac.axvline(0, color=st.INK, lw=0.7); ac.set_yticks(y); ac.set_yticklabels(short, fontsize=st.S_SMALL); ac.set_xlim(-185, 290)
    ac.set_xlabel("clips harmed (left) and rescued (right) by the decoder"); ac.tick_params(axis="y", length=0); ac.spines["left"].set_visible(False)
    ac.set_title("Controls: no order, no gain; wrong order, harm", fontsize=st.S_LABEL)
    st.panel_label(aw, "a", x=-0.12, y=1.02); st.panel_label(ac, "b", x=-0.42, y=1.02)
    return st.save(fig, "fig07_decoder")


# =====================================================================================
# FIG 8  experiment landscape
# =====================================================================================
def _spread(ys, gap):
    """Push label y-positions apart so consecutive labels are at least `gap` apart (1-D relaxation, order preserved)."""
    idx = np.argsort(ys); out = np.array(ys, dtype=float)
    for _ in range(200):
        moved = False
        for a, b in zip(idx[:-1], idx[1:]):
            d = out[b] - out[a]
            if d < gap:
                sh = (gap - d) / 2; out[a] -= sh; out[b] += sh; moved = True
        if not moved: break
    return out


def fig08_landscape():
    Lv = D["landscape"]["levers"]; lanes = ["backbone", "resolution", "input view", "training signal", "inference time", "fusion", "structure"]
    lane_lab = {"backbone": "backbone /\ncapacity", "resolution": "input\nresolution", "input view": "new input\nview", "training signal": "training\nsignal", "inference time": "inference-time\ntricks", "fusion": "fusion", "structure": "recording\nstructure"}
    fig, ax = plt.subplots(figsize=(7.16, 4.0)); fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.18)
    YMIN, YMAX = -9.6, 9.6; GAP = 0.9
    for i, ln in enumerate(lanes):
        if i % 2 == 0: ax.axvspan(i - 0.5, i + 0.5, color=st.BAND, zorder=0)
    ax.axhline(0, color=st.INK, lw=0.7, zorder=1); ax.axhline(1.64, color=st.GREEN, lw=0.8, ls="--", zorder=1)
    ax.text(4.5, 1.64 + 0.12, "adoption bar, member level: +1.64 (2 SE; seed sd 1.16)", ha="center", va="bottom", fontsize=st.sz(st.S_SMALL - 0.8), color=st.GREEN)
    for xi, ln in enumerate(lanes):
        items = [lv for lv in Lv if lv["lane"] == ln]
        mirror = False
        xm = xi + 0.3 if mirror else xi - 0.3; xl = xi + 0.17 if mirror else xi - 0.17
        yv = np.array([max(min(lv["delta"], YMAX - 0.4), YMIN + 0.4) for lv in items]); ly = _spread(yv, GAP); ly = np.clip(ly, YMIN + 0.35, YMAX - 0.35)
        for lv, v, yl in zip(items, yv, ly):
            col = {"adopted": st.BLUE, "rejected": st.RED, "no signal": st.GRAY}[lv["verdict"]]
            mk = "o" if lv["level"] == "member" else ("D" if lv["level"] == "system" else "s"); clipped = lv["delta"] < YMIN
            ax.scatter(xm, v, s=34 if mk != "D" else 26, marker=mk, color=col if lv["verdict"] != "no signal" else "white", edgecolor=col, lw=1.0, zorder=4)
            if clipped: ax.annotate("", (xm, YMIN + 0.02), (xm, v - 0.3), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, mutation_scale=7))
            if abs(yl - v) > 0.05: ax.plot([xm + (-0.05 if mirror else 0.05), xl - (0.01 if mirror else -0.01)], [v, yl], color=st.GRAY_LIGHT, lw=0.5, zorder=3)
            lab = lv["name"] + (f"  ({lv['delta']:+.1f})" if clipped else "")
            ax.text(xl, yl, lab, fontsize=st.sz(st.S_SMALL - 1.4), va="center", ha="right" if mirror else "left", color=st.INK, zorder=5)
    ax.set_xlim(-0.5, len(lanes) + 0.25); ax.set_ylim(YMIN, YMAX); ax.set_xticks(range(len(lanes))); ax.set_xticklabels([lane_lab[l] for l in lanes], fontsize=st.S_SMALL)
    ax.set_ylabel("measured change (accuracy points)"); ax.tick_params(axis="x", length=0); ax.spines["bottom"].set_visible(False)
    h = [Line2D([], [], marker="o", color=st.BLUE, ls="", label="adopted"), Line2D([], [], marker="o", color=st.RED, ls="", label="rejected"),
         Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=st.GRAY, ls="", label="no signal"),
         Line2D([], [], marker="o", color=st.INK, ls="", label="member accuracy (points)"), Line2D([], [], marker="D", color=st.INK, ls="", markersize=4, label="system: fused or decoded OOF (27 clips = 1 point)")]
    ax.legend(handles=h, loc="upper center", ncol=5, fontsize=st.sz(st.S_SMALL - 1.0), handletextpad=0.2, columnspacing=0.9, bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.set_title("Every lever we measured, at the level it was measured", fontsize=st.S_TITLE)
    return st.save(fig, "fig08_landscape")


# =====================================================================================
# FIG 9  budget architecture
# =====================================================================================
def fig09_bytes():
    B = D["bytes"]; by = {k: v / 1e6 for k, v in B["by_branch_bytes"].items()}; total = B["total_file_bytes"] / 1e6; other = B["other_bytes"] / 1e6
    fig, ax = plt.subplots(figsize=(3.5, 3.4)); fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.05)
    ax.set_xlim(0, 10); ax.set_ylim(0, 110); ax.set_xticks([]); ax.spines["bottom"].set_visible(False); ax.set_ylabel("MB in the package file"); ax.set_yticks([0, 25, 50, 75, 100])
    X, W = 0.6, 3.2
    blocks = [("person view", by["video"] / 3, st.BLUE, "MViTv2-S, 5-bit codes"), ("wrist view", by["video"] / 3, st.BLUE_LIGHT, "MViTv2-S, 5-bit codes"),
              ("thermal view", by["video"] / 3, st.ORANGE, "MViTv2-S, 5-bit codes, fold soup"), ("skeleton + IMU ensemble", by["skeleton"], st.TEAL, "20 members, int8"),
              ("YOLO11n + YOLO11n-pose", by["detector"], st.GRAY, "fp16 (int8 moved 150 of 405 windows)"), ("IMU forest 1.6 MB, windows + manifest 0.2 MB", by["imu"] + by["windows"] + other, st.TEAL_LIGHT, "")]
    ys_mid = []; y = 0
    for lab, h, col, note in blocks: ys_mid.append(y + h / 2); y += h
    ly_all = np.minimum(_spread(np.array(ys_mid), 10.5), 97.0); y = 0
    for (lab, h, col, note), ly in zip(blocks, ly_all):
        if lab.startswith("IMU forest"):
            ax.add_patch(Rectangle((X, y), W, by["imu"], fc=st.TEAL_LIGHT, ec="white", lw=0.8, zorder=2)); ax.add_patch(Rectangle((X, y + by["imu"]), W, h - by["imu"], fc=st.GRAY_LIGHT, ec="white", lw=0.8, zorder=2))
        else:
            ax.add_patch(Rectangle((X, y), W, h, fc=col, ec="white", lw=0.8, zorder=2))
            ax.text(X + W / 2, y + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=st.S_BODY, color="white", zorder=3)
        ax.plot([X + W + 0.1, X + W + 0.6], [y + h / 2, ly], color=col, lw=0.6, zorder=1)
        ax.text(X + W + 0.7, ly + (1.6 if note else 0), lab, ha="left", va="center", fontsize=st.sz(st.S_SMALL - 0.5), color=st.INK if note else col)
        if note: ax.text(X + W + 0.7, ly - 2.6, note, ha="left", va="center", fontsize=st.sz(st.S_SMALL - 1.4), color=st.GRAY_TEXT)
        y += h
    ax.add_patch(Rectangle((X, y), W, 100 - y, fc="none", ec=st.RED, lw=0, hatch="////", alpha=0.35, zorder=2)); ax.text(X + W / 2, y + (100 - y) / 2, f"{100-total:.1f} free", ha="center", va="center", fontsize=st.sz(st.S_SMALL - 1), color=st.RED, zorder=3)
    ax.add_patch(Rectangle((X, 0), W, 100, fc="none", ec=st.INK, lw=1.0, zorder=4))
    ax.axhline(100, color=st.RED, lw=1.2, zorder=5); ax.text(X + W + 0.7, 103.0, "100 MB rule: every inference-time\nweight in one file", ha="left", va="bottom", fontsize=st.sz(st.S_SMALL - 0.5), color=st.RED, linespacing=1.0)
    ax.text(X + W / 2, -1.5, f"{total:.2f} MB", ha="center", va="top", fontsize=st.S_LABEL, color=st.INK)
    ax.set_title("What 100 MB buys, and what it forbids", fontsize=st.S_TITLE)
    return st.save(fig, "fig09_bytes")


# =====================================================================================
# FIG 10  verification matrix
# =====================================================================================
def fig10_verification():
    G = D["gpu"]["rows"]; cols = ["max |dp|\nvideo", "max |dp|\nskeleton", "predictions\nchanged", "CSV\nbytes", "network", "run as"]
    fig, ax = plt.subplots(figsize=(3.5, 2.1)); fig.subplots_adjust(left=0.17, right=0.99, top=0.72, bottom=0.02)
    n, m_ = len(G), len(cols); ax.set_xlim(0, m_); ax.set_ylim(n, 0)
    for i, g in enumerate(G):
        ax.axhspan(i, i + 1, color=st.BAND if i % 2 == 0 else "white", zorder=0)
        vals = [("ref." if g["dp_video"] == 0 else f"{g['dp_video']:.1e}"), ("ref." if g["dp_skel"] == 0 else f"{g['dp_skel']:.1e}"), f"{g['argmax_changes']} / 405", "identical", "off", g["container"]]
        for j, v in enumerate(vals):
            c = st.INK; w = "normal"
            if j == 2: c = st.GREEN; w = "bold"
            if j == 3: c = st.GREEN
            if j in (0, 1) and v != "ref.": c = st.ORANGE if j == 0 else st.TEAL
            ax.text(j + 0.5, i + 0.5, v, ha="center", va="center", fontsize=st.S_SMALL + (0.5 if j == 2 else 0), color=c, fontweight=w, zorder=3)
    ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels([g["gpu"] for g in G], fontsize=st.S_BODY, fontweight="bold")
    ax.set_xticks(np.arange(m_) + 0.5); ax.set_xticklabels(cols, fontsize=st.sz(st.S_SMALL - 0.8)); ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.plot([2, 2], [0, n], color=st.GRAY_LIGHT, lw=0.8, zorder=1)
    ax.set_title("Drift exists; the predictions do not move", fontsize=st.S_TITLE, pad=26)
    return st.save(fig, "fig10_verification")


# =====================================================================================
# FIG 11  error structure: subject x confusion pair
# =====================================================================================
def fig11_errors():
    E = D["errors"]; subj = [s_["user"] for s_ in E["subjects"]]; pairs = E["top_pairs"][:10]
    ABBR = {"Read_documents": "Read docs", "Turn_pages": "Turn pages", "Sweep_the_floor": "Sweep", "Mop_the_floor": "Mop", "Use_a_mobile_phone": "Mobile phone", "Play_games": "Play games",
            "Drink_water": "Drink", "Take_medicine": "Take medicine", "Eat_food": "Eat", "Take_and_use_tableware": "Tableware", "Pour_drinks": "Pour drinks", "Check_the_time": "Check time",
            "Tap_the_keyboard": "Keyboard", "Write": "Write", "Stir_drinks": "Stir drinks", "Peel_fruits": "Peel fruits"}
    short = lambda n: ABBR.get(n.split("_", 1)[1], n.split("_", 1)[1].replace("_", " "))
    fig = plt.figure(figsize=(7.16, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.0], wspace=0.04, left=0.07, right=0.99, top=0.8, bottom=0.09)
    ax = fig.add_subplot(gs[0]); ab = fig.add_subplot(gs[1], sharey=ax)
    for i, u in enumerate(subj):
        ax.axhspan(i - 0.5, i + 0.5, color=st.BAND if i % 2 == 0 else "white", zorder=0)
        for j, p in enumerate(pairs):
            ab_, ba = p["per_subject"].get(u, [0, 0])
            if ab_: ax.scatter(j - 0.15, i, s=ab_ * 14, color=st.RED, alpha=0.85, zorder=3, linewidths=0)
            if ba: ax.scatter(j + 0.15, i, s=ba * 14, color=st.PURPLE, alpha=0.85, zorder=3, linewidths=0)
    ax.set_xlim(-0.6, len(pairs) - 0.4); ax.set_ylim(len(subj) - 0.5, -0.5)
    ax.set_yticks(range(len(subj))); ax.set_yticklabels([u.replace("user", "subject ") for u in subj], fontsize=st.S_SMALL)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([f"{short(p['name_a'])}\n{short(p['name_b'])}" for p in pairs], fontsize=st.sz(st.S_SMALL - 0.8), rotation=0, linespacing=1.0)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    h = [Line2D([], [], marker="o", color=st.RED, ls="", label="A predicted as B"), Line2D([], [], marker="o", color=st.PURPLE, ls="", label="B predicted as A"), Line2D([], [], marker="o", color=st.GRAY, ls="", markersize=np.sqrt(3 * 14), label="3 clips")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.01, -0.1), ncol=3, fontsize=st.sz(st.S_SMALL - 0.5), handletextpad=0.1, columnspacing=1.0)
    # per-subject share of errors in repeated cells
    for i, s_ in enumerate(E["subjects"]):
        fr = s_["errors_in_repeated_cells"] / max(s_["errors"], 1)
        ab.barh(i, fr, color=st.INK, alpha=0.8, height=0.6, zorder=3); ab.text(fr + 0.03, i, f"{s_['errors_in_repeated_cells']}/{s_['errors']}", va="center", fontsize=st.sz(st.S_SMALL - 1.6), color=st.INK)
    ab.set_xlim(0, 1.0); ab.set_xticks([]); ab.tick_params(axis="y", labelleft=False, length=0)
    for s_ in ab.spines.values(): s_.set_visible(False)
    ab.set_title("errors in a\nrepeated cell", fontsize=st.sz(st.S_SMALL - 0.5), loc="center", pad=6)
    fig.text(0.02, 0.985, f"Errors repeat inside a subject: {E['errors_in_repeated_cells']} of {E['errors']} sit in a (subject, true, predicted) cell seen 3+ times; {E['one_directional_pair_cells']} of {E['subject_pair_cells']} such cells are one-directional", fontsize=st.sz(st.S_LABEL - 0.5), fontweight="bold", va="top", color=st.INK)
    return st.save(fig, "fig11_errors")


ALL = {"fig01": fig01_abstract, "fig02": fig02_complementarity, "fig03": fig03_resolution, "fig04": fig04_thermal, "fig05": fig05_subjects,
       "fig06": fig06_robustness, "fig07": fig07_decoder, "fig08": fig08_landscape, "fig09": fig09_bytes, "fig10": fig10_verification, "fig11": fig11_errors}

if __name__ == "__main__":
    want = sys.argv[1:] or list(ALL)
    for k in want:
        print(k, "->", ALL[k]())
