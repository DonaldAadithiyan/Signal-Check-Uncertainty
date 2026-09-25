"""Fig 1: overview diagram (no data). Built with matplotlib patches (TikZ not installed).
Coordinates are in cm on a 7.5 x 6.8 cm canvas."""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from style import fig, save, READ, EXT, plt

NAME = "fig1_overview.pdf"
f = fig(NAME)
ax = f.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 7.5); ax.set_ylim(0, 6.8); ax.axis("off")
GLYPH = dict(fontfamily="DejaVu Sans")


def box(x0, y0, w, h, txt, ec="k", fc="white", tc="k", fs=7, lw=0.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                ec=ec, fc=fc, lw=lw, ls=ls))
    ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)


def arrow(p, q, color="k", ls="-", lw=0.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=color,
                                 lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def step(x, y, n):
    ax.add_patch(Circle((x, y), 0.15, fc="k", ec="k", lw=0))
    ax.text(x, y - 0.01, str(n), ha="center", va="center", fontsize=6.5, color="white",
            fontweight="bold")


GREY = "#888888"
# 1. world model
box(1.0, 5.95, 5.5, 0.65, r"Recurrent world model (RSSM)   $h_t$")
# 2. split
box(0.3, 4.75, 3.0, 0.62, r"probe readout $v^{\top} h_t$")
box(4.0, 4.75, 3.2, 0.62, "instantaneous signals:\nKL, Recon, EMARecon", ec=GREY, tc=GREY, fs=6.5)
arrow((2.6, 5.95), (1.8, 5.37)); arrow((4.9, 5.95), (5.6, 5.37), color=GREY)
# 3. emergence -> C_t
box(0.3, 3.45, 3.0, 0.62, "$C_t$: accumulated\npredictive difficulty")
arrow((1.8, 4.75), (1.8, 4.07))
step(2.12, 4.41, 1); ax.text(2.35, 4.41, "Emergence", va="center", fontsize=6.5)
# 4. external validity -> E^state
box(0.3, 2.15, 3.0, 0.62, "future imagination error\n" + r"$E^{\mathrm{state}}_{t,K}$",
    ec=EXT)
arrow((1.8, 3.45), (1.8, 2.77))
step(2.12, 3.19, 2); ax.text(2.35, 3.24, "External validity", va="center", fontsize=6.5)
ax.text(2.35, 2.98, "beyond KL/Recon/EMA", va="center", fontsize=6, color=GREY)
# baselines are the controls: dashed elbow into E^state
ax.plot([4.6, 4.6], [4.75, 2.46], color=GREY, lw=0.7, ls=(0, (3, 2)))
arrow((4.6, 2.46), (3.3, 2.46), color=GREY, ls=(0, (3, 2)))
# 5. mechanism side box
box(4.8, 2.15, 2.4, 1.75, "", ec="k")
step(5.08, 3.6, 3); ax.text(5.3, 3.6, "Mechanism", va="center", fontsize=7)
ax.text(4.95, 3.1, r"dense $v$: diffuse", va="center", fontsize=6.5)
ax.text(4.95, 2.6, "sparse atoms:\n#612, #156", va="center", fontsize=6.5)
# 6. intervention on dense v: fork to both outcomes
ax.plot([1.8, 1.8], [2.15, 1.55], color="k", lw=0.7)
ax.plot([1.8, 4.9], [1.55, 1.55], color="k", lw=0.7)
arrow((1.8, 1.55), (1.8, 1.35)); arrow((4.9, 1.55), (4.9, 1.35))
step(2.12, 1.82, 4); ax.text(2.35, 1.82, r"Intervention on dense $v$", va="center", fontsize=6.5)
box(0.3, 0.15, 3.3, 1.2, "", ec=READ, fc="#F7E4EF", lw=0.9)
ax.text(1.95, 1.0, "Internal readout", ha="center", va="center", fontsize=7)
ax.text(1.35, 0.5, "\u2713", ha="center", va="center", fontsize=11, color=READ, **GLYPH)
ax.text(1.65, 0.5, "strong", ha="left", va="center", fontsize=7)
box(3.9, 0.15, 3.3, 1.2, "", ec=EXT, fc="#FBE3D6", lw=0.9)
ax.text(5.55, 1.0, "External quality", ha="center", va="center", fontsize=7)
ax.text(4.75, 0.5, "\u2717", ha="center", va="center", fontsize=11, color=EXT, **GLYPH)
ax.text(5.0, 0.5, r"null (dense $v$)", ha="left", va="center", fontsize=7)
# sparse atoms -> external quality
arrow((6.85, 2.15), (6.85, 1.35), color=EXT)
ax.text(6.75, 1.82, "sparse atoms", ha="right", va="center", fontsize=6, color=EXT)
ax.text(6.95, 1.82, "\u2713", ha="left", va="center", fontsize=8, color=EXT, **GLYPH)
save(f, NAME)
