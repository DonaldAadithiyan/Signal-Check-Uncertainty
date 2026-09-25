"""Fig 6: incremental dR^2 of the C_t analogue in a scalar Kalman filter, by regime.
Source: outputs/phase5_filtering_theory/phase5_results.json (mean/sd over 10 seeds;
per-seed values were not saved, so mean +/- sd is drawn)."""
import numpy as np
from style import fig, save, load, check, plt

NAME = "fig6_filtering.pdf"
d = load("phase5_filtering_theory/phase5_results.json")["results"]
regs = [("stationary", "stationary,\ncorrect"), ("misspecified", "misspecified"),
        ("nonstationary", "non-stationary")]
m = np.array([d[r]["incremental_r2_mean"] for r, _ in regs])
s = np.array([d[r]["incremental_r2_std"] for r, _ in regs])
n = [d[r]["n_seeds"] for r, _ in regs]
z = (m - m[0]) / s[0]

check("stationary mean", m[0], 0.0004, 5e-5); check("stationary sd", s[0], 0.0003, 5e-5)
check("misspec mean", m[1], 0.0046, 5e-5); check("misspec sd", s[1], 0.0018, 5e-5)
check("nonstat mean", m[2], 0.0825, 5e-5); check("nonstat sd", s[2], 0.0142, 5e-5)
check("misspec z", z[1], 13.3, 0.05); check("nonstat z", z[2], 257.6, 0.05)
assert n == [10, 10, 10]

f = fig(NAME)
ax = f.add_axes([0.17, 0.25, 0.80, 0.70])
x = np.arange(3)
shades = ["#999999", "#555555", "#000000"]
mk = ["o", "D", "s"]
for i in range(3):
    lo = max(m[i] - s[i], 1e-5)
    ax.errorbar(x[i], m[i], yerr=[[m[i] - lo], [s[i]]], fmt=mk[i], color=shades[i],
                ms=4, capsize=2, elinewidth=0.8, capthick=0.8)
    txt = f"{m[i]:+.4f}" + ("" if i == 0 else f"\n$z={z[i]:+.1f}$")
    ax.text(x[i] + 0.12, m[i], txt, fontsize=6, va="center", ha="left")
ax.set_yscale("log")
ax.set_ylim(1e-5, 0.3)
ax.set_xlim(-0.4, 2.8)
ax.set_xticks(x)
ax.set_xticklabels([l for _, l in regs])
ax.set_ylabel(r"incremental $\Delta R^2$ of $C_t$ analogue")
ax.text(0.01, 0.97, "mean $\\pm$ sd, 10 seeds per regime", transform=ax.transAxes,
        fontsize=6, va="top")
save(f, NAME)
