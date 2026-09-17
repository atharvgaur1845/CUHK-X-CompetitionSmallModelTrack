"""Visual style for every figure of the report: one semantic palette, one typeface, one set of
line/marker/annotation rules. Import and call `use()` before drawing; every figure is saved with
`save(fig, name)` as vector PDF with embedded fonts.

Colour is semantic and fixed across the paper:
    video / person   BLUE      wrist (video)     BLUE_LIGHT
    skeleton / IMU   TEAL      IMU forest        TEAL_LIGHT
    thermal          ORANGE    decoder/structure PURPLE
    negative         RED       baseline/neutral  GRAY
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- palette
INK = "#1B2A41"          # primary text / deep navy
BLUE = "#2E5C9E"         # video, person view
BLUE_LIGHT = "#6F97D1"   # wrist view
TEAL = "#1F7F7A"         # skeleton / IMU ensemble
TEAL_LIGHT = "#5FB3A9"   # IMU forest
ORANGE = "#D9772B"       # thermal
PURPLE = "#6B4FA0"       # decoder / structure
RED = "#B4433A"          # negative / harm
GRAY = "#8A929B"         # baseline / reference
GRAY_LIGHT = "#D6DBE1"   # light rules
BAND = "#F3F1EC"         # warm highlight band
PAPER = "#FBFAF7"        # warm near-white (cards only; figure background stays white)
GREEN = "#3D8B5A"        # adopted / positive (used sparingly)

MEMBER = {"person": BLUE, "wrist": BLUE_LIGHT, "thermal": ORANGE, "skeleton": TEAL, "imu": TEAL_LIGHT}
MEMBER_LABEL = {"person": "person view", "wrist": "wrist view", "thermal": "thermal view",
                "skeleton": "skeleton/IMU ensemble", "imu": "IMU forest"}

# ---------------------------------------------------------------- typography
_FONT_DIRS = [Path("/usr/share/fonts/truetype/lato")]


def _register_fonts() -> str:
    found = False
    for d in _FONT_DIRS:
        if d.is_dir():
            for f in d.glob("Lato-*.ttf"):
                try:
                    fm.fontManager.addfont(str(f)); found = True
                except Exception:
                    pass
    return "Lato" if found else "DejaVu Sans"


FONT = _register_fonts()

# point sizes: body 7.5 is the floor for print; numerals bigger
S_BODY, S_SMALL, S_LABEL, S_TITLE, S_BIG, S_HUGE = 7.5, 6.5, 8.5, 9.5, 13, 20


def use() -> None:
    plt.rcParams.update({
        "font.family": FONT, "font.size": S_BODY,
        "axes.titlesize": S_TITLE, "axes.labelsize": S_BODY, "xtick.labelsize": S_SMALL, "ytick.labelsize": S_SMALL,
        "legend.fontsize": S_SMALL, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.4, "lines.markersize": 4.5, "axes.grid": False, "figure.dpi": 150,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 6,
    })


# ---------------------------------------------------------------- helpers
def panel_label(ax, text: str, x: float = -0.02, y: float = 1.04, size: float = S_LABEL + 1) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, fontweight="bold", ha="right", va="bottom", color=INK)


def despine_all(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


def hgrid(ax, color=GRAY_LIGHT, lw=0.5) -> None:
    ax.grid(axis="y", color=color, lw=lw, zorder=0); ax.set_axisbelow(True)


def card(ax, x, y, w, h, text_lines, fc=PAPER, ec=GRAY_LIGHT, sizes=None, colors=None, weights=None, pad=0.02, r=0.02):
    """Rounded card with stacked text lines (axes coordinates)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size={r}", fc=fc, ec=ec, lw=0.6,
                                transform=ax.transAxes, zorder=1))
    n = len(text_lines); sizes = sizes or [S_BODY] * n; colors = colors or [INK] * n; weights = weights or ["normal"] * n
    ys = [y + h - (i + 0.5) * h / n for i in range(n)]
    for t, yy, s, c, wgt in zip(text_lines, ys, sizes, colors, weights):
        ax.text(x + w / 2, yy, t, transform=ax.transAxes, ha="center", va="center", fontsize=s, color=c, fontweight=wgt, zorder=2)


def arrow(ax, p0, p1, color=GRAY, lw=1.0, rad=0.0, style="-|>", ms=7, z=3, coords="axes"):
    from matplotlib.patches import FancyArrowPatch
    tr = ax.transAxes if coords == "axes" else ax.transData
    a = FancyArrowPatch(p0, p1, transform=tr, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0)
    ax.add_patch(a); return a


def delta_label(ax, x, y, text, color=INK, dy=6, size=S_BODY, weight="bold", ha="center"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, dy), ha=ha, va="bottom", fontsize=size,
                color=color, fontweight=weight)


OUT = Path(__file__).resolve().parent / "figures"


def save(fig, name: str) -> Path:
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{name}.pdf"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, metadata={"Creator": "generate_all_figures.py"})
    plt.close(fig)
    return p
