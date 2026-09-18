"""Visual system of the deck, shared with the poster and the report figures: the same semantic palette
(figstyle), Lato (the deck's typeface, loaded by fontspec from the same TTF files), transparent figure
backgrounds so the slides' warm paper shows through, and a few drawing helpers (ribbons, marks, panels)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Circle, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report")); sys.path.insert(0, str(HERE.parent / "poster"))
import figstyle as st  # noqa: E402  (palette)

INK, BLUE, BLUE_LIGHT, ORANGE, TEAL, TEAL_LIGHT, PURPLE, RED, GRAY, GRAY_LIGHT, GRAY_TEXT, BAND, PAPER, GREEN = (
    st.INK, st.BLUE, st.BLUE_LIGHT, st.ORANGE, st.TEAL, st.TEAL_LIGHT, st.PURPLE, st.RED, st.GRAY, st.GRAY_LIGHT, st.GRAY_TEXT, st.BAND, st.PAPER, st.GREEN)
MEMBER = st.MEMBER                      # person BLUE, wrist BLUE_LIGHT, thermal ORANGE, skeleton TEAL, imu TEAL_LIGHT
SENSOR = {"ir": BLUE, "thermal": ORANGE, "skeleton": TEAL, "imu": TEAL_LIGHT, "radar": PURPLE}
SPECIALISTS = [("body", BLUE), ("hands", BLUE_LIGHT), ("heat", ORANGE), ("posture", TEAL), ("motion", TEAL_LIGHT)]

FONT_DIR = Path("/usr/share/fonts/truetype/lato")
for f in FONT_DIR.glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(f))
FONT = "Lato"
MIN = 7.5   # nothing in a slide figure is set below this (the slide is 16 x 9 cm; 7.5 pt projects large)


def use() -> None:
    plt.rcParams.update({
        "font.family": FONT, "font.weight": "bold", "font.size": 9, "text.color": INK, "axes.labelcolor": INK, "axes.labelweight": "bold",
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.edgecolor": "none",
        "pdf.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
        "mathtext.fontset": "custom", "mathtext.rm": f"{FONT}:bold", "mathtext.it": f"{FONT}:bold:italic", "mathtext.bf": f"{FONT}:bold",
    })


def sz(x: float) -> float:
    return max(float(x), MIN)


def canvas(w_in: float, h_in: float, units: float = 100.0):
    """A figure whose axes spans [0, units*w/h] x [0, units] in equal-aspect data coordinates."""
    fig = plt.figure(figsize=(w_in, h_in)); ax = fig.add_axes([0, 0, 1, 1])
    W = units * w_in / h_in; ax.set_xlim(0, W); ax.set_ylim(0, units); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax, W


def text(ax, x, y, s, size=9, color=INK, weight="bold", **k):
    return ax.text(x, y, s, fontsize=sz(size), color=color, fontweight=weight, zorder=6, **k)


def panel(ax, x, y, w, h, fc="white", ec=INK, lw=0.7, shadow=True, z=2):
    if shadow: ax.add_patch(Rectangle((x + 0.5, y - 0.5), w, h, fc=(0, 0, 0, 0.09), ec="none", zorder=z - 1))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))


def ribbon(ax, p0, p1, color, lw=3.0, z=3, alpha=0.95, horizontal=True):
    """Cubic ribbon from p0 to p1; horizontal ribbons bend in x, vertical ones in y."""
    (x0, y0), (x1, y1) = p0, p1
    if horizontal:
        xm = (x0 + x1) / 2; ctrl = [(xm, y0), (xm, y1)]
    else:
        ym = (y0 + y1) / 2; ctrl = [(x0, ym), (x1, ym)]
    path = MPath([p0, ctrl[0], ctrl[1], p1], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, alpha=alpha, zorder=z, capstyle="round"))


def mark(ax, x, y, ok, s=0.8, lw=1.8, z=6):
    if ok: ax.plot([x - s, x - 0.25 * s, x + s], [y, y - 0.9 * s, y + s], color=GREEN, lw=lw, solid_capstyle="round", zorder=z)
    else:
        ax.plot([x - s, x + s], [y - s, y + s], color=RED, lw=lw, solid_capstyle="round", zorder=z)
        ax.plot([x - s, x + s], [y + s, y - s], color=RED, lw=lw, solid_capstyle="round", zorder=z)


def arrow(ax, p0, p1, color=INK, lw=1.4, ms=12, z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def ring(ax, x, y, r, ec=INK, fc="white", lw=1.6, z=5, dot=None):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
    if dot: ax.add_patch(Circle((x, y), r * 0.38, fc=dot, ec="none", zorder=z + 1))


OUT = HERE / "figures"


def save(fig, name: str) -> Path:
    OUT.mkdir(exist_ok=True); p = OUT / f"{name}.pdf"
    fig.savefig(p, bbox_inches=None, pad_inches=0, metadata={"Creator": "make_slide_figures.py"}); plt.close(fig); return p
