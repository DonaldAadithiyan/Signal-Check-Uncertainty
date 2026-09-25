"""Fig 3 (was Fig 4): mechanism.
Left:  cumulative share of v carried by the top-n SAE atoms, 3 tasks x 3 SAE seeds
       (phase3_results.json decomposition). Only the top-10 shares were saved, so n=1..10.
       The random-vector control was not saved in any results file and is not drawn.
Right: r(activation, KL) vs r(activation, E^state) on the HELD-OUT half for the atom
       selected in each SAE seed (addendum_2_1_2_2_2_3_results.json); reacher #612 and
       cartpole #156 highlighted. The full held-out candidate pool was not saved."""
import numpy as np
from style import fig, save, load, check, letter, TASKS, COL, MRK, NULL, plt

NAME = "fig3_mechanism.pdf"
dec = load("phase3_sae/phase3_results.json")["decomposition"]
ad = load("phase3_sae/addendum_2_1_2_2_2_3_results.json")

mism = []
for t in TASKS:
    for s, x in enumerate(dec[t]):
        t1, t4 = 100 * x["top1_share"], 100 * x["top4_share"]
        if not (0.3 <= round(t1, 1) <= 0.5):
            mism.append(f"{t} SAE seed {s}: top-1 share {t1:.2f}% (paper: 0.3-0.5%)")
        if not (0.9 <= round(t4, 1) <= 1.2):
            mism.append(f"{t} SAE seed {s}: top-4 share {t4:.2f}% (paper: 0.9-1.2%)")
for m in mism:
    print("  [MISMATCH]", m)

h612 = ad["reacher"]["per_seed"]["0"]["heldout"]
h156 = ad["cartpole"]["per_seed"]["0"]["heldout"]
assert h612["atom"] == 612 and h156["atom"] == 156
check("#612 r(act,KL) in 0.02-0.06", h612["r_kl"], 0.04, 0.02)
check("#612 r(act,E) in 0.21-0.29", h612["r_external"], 0.25, 0.04)
check("#156 r(act,KL)", h156["r_kl"], 0.499, 0.0005)
assert h156["r_external"] > 0

f = fig(NAME)
# ---- left -----------------------------------------------------------------
ax = f.add_axes([0.14, 0.19, 0.33, 0.72])
n = np.arange(1, 11)
ls = ["-", "--", ":"]
for t in TASKS:
    for s, x in enumerate(dec[t]):
        cum = 100 * np.cumsum(x["top10_shares"])
        ax.plot(n, cum, color=COL[t], ls=ls[s], lw=0.7, marker=MRK[t], ms=1.8,
                markevery=[0, 3, 9], label=t if s == 0 else None)
ax.set_xscale("log")
ax.set_xticks([1, 2, 4, 10])
ax.set_xticklabels(["1", "2", "4", "10"])
ax.set_xlabel("top-$n$ SAE atoms", labelpad=1)
ax.set_ylabel(r"share of $v$ (%)", labelpad=1)
ax.set_ylim(0, 3.2)
ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0.1)
ax.text(0.98, 0.03, "line style = SAE seed", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=6)
letter(ax, "a", x=-0.40, y=1.0)

# ---- right ----------------------------------------------------------------
ax = f.add_axes([0.64, 0.19, 0.33, 0.72])
for t in TASKS:
    for s in ("0", "1", "2"):
        hd = ad[t]["per_seed"][s]["heldout"]
        ax.scatter(hd["r_kl"], hd["r_external"], s=8, color=NULL, marker=MRK[t],
                   edgecolors="#777777", lw=0.4, zorder=2)
for hd, t, lab, off in [(h612, "reacher", "reacher #612", (0.08, -0.22)),
                        (h156, "cartpole", "cartpole #156", (-1.2, 0.25))]:
    ax.scatter(hd["r_kl"], hd["r_external"], s=22, color=COL[t], marker=MRK[t],
               edgecolors="k", lw=0.5, zorder=3)
    ax.annotate(lab, (hd["r_kl"], hd["r_external"]),
                (hd["r_kl"] + off[0], hd["r_external"] + off[1]), fontsize=6,
                arrowprops=dict(arrowstyle="-", lw=0.4, color="k"))
ax.axhline(0, color="k", lw=0.4)
ax.axvline(0, color="k", lw=0.4)
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_xlabel(r"$r$(activation, KL)", labelpad=1)
ax.set_ylabel(r"$r$(activation, $E^{\mathrm{state}}$)", labelpad=1)
ax.text(0.98, 0.02, "held-out half;\ngrey: atom selected\nper SAE seed", transform=ax.transAxes,
        fontsize=6, va="bottom", ha="right")
letter(ax, "b", x=-0.45, y=1.0)
save(f, NAME)
