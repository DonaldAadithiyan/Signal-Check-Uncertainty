"""Fig 4 (was Fig 5): causal dissociation (hero figure).
Dense v: phase6_causal_steering/seed_extension/phase6_seed_extension_aggregate.json
         (z of steering slope vs 50-direction null; per-seed values, mean +/- sd).
Atoms:   phase3_sae/phase3b_causal_results.json (reacher #612) and
         phase3b_causal_results_atom156.json (cartpole #156); z vs 50 random directions.
Readout k=10 filled, k=0 hollow; E^state vermillion."""
import numpy as np
from matplotlib.patches import Circle, Rectangle
from style import fig, save, load, check, letter, TASKS, COL, MRK, READ, EXT, NULL, plt

NAME = "fig4_dissociation.pdf"
agg = load("phase6_causal_steering/seed_extension/phase6_seed_extension_aggregate.json")
a612 = load("phase3_sae/phase3b_causal_results.json")["reacher"]
a156 = load("phase3_sae/phase3b_causal_results_atom156.json")
assert a612["atom"] == 612 and a156["atom"] == 156

want = {"cartpole": (5, 12.87, 1.64, 12.15, 1.37, 0.22, 0.37),
        "reacher": (4, 13.34, 1.41, 5.88, 0.66, 0.32, 0.40),
        "pendulum": (4, 15.56, 2.96, 11.36, 0.59, 0.62, 0.80)}
for t in TASKS:
    g, w = agg[t], want[t]
    assert g["n_seeds"] == w[0]
    for key, i in [("probe_k0_z_mean", 1), ("probe_k0_z_std", 2), ("probe_k10_z_mean", 3),
                   ("probe_k10_z_std", 4), ("estate_z_mean", 5), ("estate_z_std", 6)]:
        check(f"{t} {key}", g[key], w[i], 0.006)
allread = np.concatenate([agg[t][k] for t in TASKS for k in ("probe_k0_z_values", "probe_k10_z_values")])
check("min readout z (k0 u k10)", allread.min(), 5.2, 0.05)
check("max readout z (k0 u k10)", allread.max(), 19.2, 0.05)
allE = np.concatenate([agg[t]["estate_z_values"] for t in TASKS])
check("max individual dense E^state z", allE.max(), 1.48, 0.005)
z612 = {k: a612["null_summary"][k]["z"] for k in ("dprobe_0", "dprobe_10", "d_e_state")}
z156 = {k: a156["null_summary"][k]["z"] for k in ("dprobe_0", "dprobe_10", "d_e_state")}
check("#612 E^state z", z612["d_e_state"], 2.67, 0.005)
assert abs(z612["dprobe_0"]) < 2 and abs(z612["dprobe_10"]) < 2
check("#156 readout z (k=0)", z156["dprobe_0"], -5.0, 0.05)
check("#156 E^state z", z156["d_e_state"], 10.3, 0.05)

f = fig(NAME)
# ------------------------------------------------------------ top schematic
axs = f.add_axes([0.0, 0.66, 1.0, 0.34])
axs.set_xlim(0, 10); axs.set_ylim(0, 3.3); axs.axis("off")
cols = [("Dense $v$", True, False, ""), ("Reacher #612", False, True, ""),
        ("Cartpole #156", True, True, r"$\downarrow$")]
xc = [3.05, 5.75, 8.45]
axs.text(0.25, 2.0, "Readout", color=READ, fontsize=7, va="center")
axs.text(0.25, 1.0, r"$E^{\mathrm{state}}$", color=EXT, fontsize=7, va="center")
for x, (lab, rd, ex, arrow) in zip(xc, cols):
    axs.add_patch(Rectangle((x - 1.1, 0.5), 2.2, 2.0, fill=False, lw=0.5, ec="#999999"))
    axs.text(x, 2.8, lab, ha="center", va="center", fontsize=7)
    for yy, on, c in [(2.0, rd, READ), (1.0, ex, EXT)]:
        axs.add_patch(Circle((x, yy), 0.3, facecolor=c if on else "white", edgecolor=c, lw=1.0))
    if arrow:
        axs.text(x + 0.45, 2.0, arrow, fontsize=8, va="center", color=READ)
axs.text(9.55, 0.05, "filled = effect beyond null; hollow = null", ha="right", va="bottom", fontsize=6)
letter(axs, "a", x=0.005, y=0.86)

# ------------------------------------------------------------ bottom dot plot
ax = f.add_axes([0.255, 0.10, 0.715, 0.52])
rows = [("cartpole, $v$", "cartpole"), ("reacher, $v$", "reacher"),
        ("pendulum, $v$", "pendulum"), ("reacher #612", None), ("cartpole #156", None)]
ys = np.arange(len(rows))[::-1].astype(float)
ax.axvspan(-2, 2, color="#EEEEEE", lw=0, zorder=0)
ax.text(0, ys[0] + 0.55, "null", ha="center", va="center", fontsize=6, color="#666666",
        bbox=dict(facecolor="#EEEEEE", lw=0, pad=0.5), zorder=1)
ax.axvline(0, color="#999999", lw=0.4, zorder=0)
dy = 0.17
for y, (lab, t) in zip(ys, rows):
    if t is not None:
        g = agg[t]
        k0, k10, es = (np.array(g[k]) for k in ("probe_k0_z_values", "probe_k10_z_values", "estate_z_values"))
        sds = (g["probe_k10_z_std"], g["probe_k0_z_std"], g["estate_z_std"])
        for (vals, yy, c, filled), sd in zip([(k10, y + dy, READ, True), (k0, y + dy, READ, False),
                                               (es, y - dy, EXT, True)], sds):
            ax.scatter(vals, np.full(len(vals), yy), s=2.5, color=c, lw=0, alpha=0.6, zorder=2)
            ax.errorbar(vals.mean(), yy, xerr=sd, fmt=MRK[t], ms=3.8,
                        mfc=c if filled else "white", mec=c, ecolor=c, elinewidth=0.8,
                        capsize=1.5, capthick=0.8, zorder=3)
    else:
        z = z612 if "612" in lab else z156
        mk = MRK["reacher"] if "612" in lab else MRK["cartpole"]
        ax.plot(z["dprobe_10"], y + dy, mk, ms=3.8, color=READ, zorder=3)
        ax.plot(z["dprobe_0"], y + dy, mk, ms=3.8, mfc="white", mec=READ, zorder=3)
        ax.plot(z["d_e_state"], y - dy, mk, ms=3.8, color=EXT, zorder=3)
ax.axhline(1.5, color="k", lw=0.4, ls=":")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows])
ax.set_ylim(-0.6, 4.8)
ax.set_xlim(-7, 21)
ax.set_xticks([-5, 0, 5, 10, 15, 20])
ax.set_xlabel(r"causal $z$ against 50-direction null", labelpad=1)
h = [plt.Line2D([], [], marker="o", ls="", color=READ, ms=3.5, label="readout, $k{=}10$"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=READ, ms=3.5, label="readout, $k{=}0$"),
     plt.Line2D([], [], marker="o", ls="", color=EXT, ms=3.5, label=r"$E^{\mathrm{state}}$")]
ax.legend(handles=h, loc="lower right", handletextpad=0.2, borderaxespad=0.1, fontsize=6)
letter(ax, "b", x=-0.345, y=1.0)
save(f, NAME)
