"""Shared style + paths for all paper figures (spec §1)."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "outputs")

CM = 1 / 2.54
W = 7.50 * CM
SIZES = {  # height in cm (spec §0)
    "fig1_overview.pdf": 6.8, "fig1_emergence.pdf": 6.4, "fig2_external.pdf": 7.6,
    "fig3_mechanism.pdf": 5.6, "fig4_dissociation.pdf": 7.2, "fig6_filtering.pdf": 4.6,
}

TASKS = ["cartpole", "reacher", "pendulum"]
COL = {"cartpole": "#0072B2", "reacher": "#E69F00", "pendulum": "#009E73"}
MRK = {"cartpole": "o", "reacher": "^", "pendulum": "s"}
NULL = "#BBBBBB"
READ = "#CC79A7"   # internal readout
EXT = "#D55E00"    # external quality (E^state)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.linewidth": 0.5, "lines.linewidth": 0.8,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.transparent": False,
})


def fig(name):
    return plt.figure(figsize=(W, SIZES[name] * CM))


def save(f, name):
    f.savefig(os.path.join(OUT, name), format="pdf", bbox_inches=None, pad_inches=0.01)
    prev = os.environ.get("FIG_PREVIEW_DIR")
    if prev:  # optional raster preview for visual QA only; not a deliverable
        os.makedirs(prev, exist_ok=True)
        f.savefig(os.path.join(prev, name.replace(".pdf", ".png")), dpi=300)
    plt.close(f)
    print("wrote", name)


def letter(ax, s, x=-0.28, y=1.02):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def load(rel):
    with open(os.path.join(RES, rel)) as fh:
        return json.load(fh)


def check(label, got, want, tol):
    """Spec §2: assert must-match values; raise on mismatch (never adjust)."""
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'MISMATCH'}] {label}: data={got:.4f} paper={want} (tol {tol})")
    if not ok:
        raise AssertionError(f"must-match failed: {label}: data={got} paper={want}")
