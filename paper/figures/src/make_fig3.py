"""Fig 2 (was Fig 3): external validation.
Row 1: r(C_t, E^state_K) vs K       <- phase1_results.json h1_by_horizon (site bootstrap CI)
Row 2: incremental dR^2 + 95% CI    <- phase1 addendum (K=10, trajectory bootstrap, 1000)
Row 3: real vs time-scrambled C_t   <- phase2_results.json scrambling r_ext (20 scrambles)"""
import numpy as np
from style import fig, save, load, check, letter, TASKS, COL, MRK, NULL, plt

NAME = "fig2_external.pdf"
p1 = load("phase1_external_validation/phase1_results.json")
ad = load("phase1_external_validation/addendum_1_1_1_2_results.json")["results"]
sc = load("phase2_rigor_controls/phase2_results.json")["scrambling"]
Ks = [1, 5, 10, 20]

want = {"cartpole": (0.0280, 0.0154, 0.0433), "reacher": (0.0182, 0.0083, 0.0309),
        "pendulum": (0.0019, 0.0001, 0.0051)}
for t in TASKS:
    a = ad[t]
    check(f"{t} dR2", a["incremental_r2_point"], want[t][0], 5e-5)
    check(f"{t} CI lo", a["incremental_r2_ci_lo"], want[t][1], 5e-5)
    check(f"{t} CI hi", a["incremental_r2_ci_hi"], want[t][2], 5e-5)
    assert p1[t]["K_headline"] == 10
for t, zw in zip(TASKS, [19.6, 21.5, -11.7]):
    check(f"{t} scramble z(E^state)", sc[t]["z_ext"], zw, 0.05)

f = fig(NAME)
# ---- row 1 --------------------------------------------------------------
ax1 = f.add_axes([0.17, 0.765, 0.80, 0.195])
for t in TASKS:
    h = p1[t]["h1_by_horizon"]
    r = [h[str(k)]["pearson_r_state"] for k in Ks]
    lo = [h[str(k)]["pearson_r_state_ci"][1] for k in Ks]
    hi = [h[str(k)]["pearson_r_state_ci"][2] for k in Ks]
    ax1.fill_between(Ks, lo, hi, color=COL[t], alpha=0.25, lw=0)
    ax1.plot(Ks, r, color=COL[t], marker=MRK[t], ms=3, label=t)
ax1.set_xticks(Ks)
ax1.set_xlabel("imagination horizon $K$", labelpad=1)
ax1.set_ylabel(r"$r(C_t, E^{\mathrm{state}}_{t,K})$")
ax1.set_ylim(0, 0.82)
ax1.set_yticks([0, 0.2, 0.4, 0.6])
ax1.legend(loc="upper right", ncol=3, handlelength=1.5, columnspacing=0.8,
           bbox_to_anchor=(1.0, 1.06))
letter(ax1, "a", x=-0.2, y=0.97)

# ---- row 2 --------------------------------------------------------------
ax2 = f.add_axes([0.17, 0.465, 0.46, 0.20])
y = np.arange(3)[::-1]
for yi, t in zip(y, TASKS):
    a = ad[t]
    pt, lo, hi = a["incremental_r2_point"], a["incremental_r2_ci_lo"], a["incremental_r2_ci_hi"]
    ax2.errorbar(pt, yi, xerr=[[pt - lo], [hi - pt]], fmt=MRK[t], color=COL[t], ms=3.5,
                 capsize=2, elinewidth=0.8, capthick=0.8)
ax2.axvline(0, color="k", lw=0.5)
ax2.set_yticks(y)
ax2.set_yticklabels(TASKS)
ax2.set_xlim(-0.004, 0.047)
ax2.set_ylim(-0.6, 2.6)
ax2.set_xlabel(r"$\Delta R^2$ from $C_t$ ($K{=}10$, 95% CI)", labelpad=1)
letter(ax2, "b", x=-0.37, y=0.97)
# inset: pendulum zoom so +0.0019 is visible
ax2i = f.add_axes([0.72, 0.465, 0.25, 0.17])
a = ad["pendulum"]
pt, lo, hi = a["incremental_r2_point"], a["incremental_r2_ci_lo"], a["incremental_r2_ci_hi"]
ax2i.errorbar(pt, 0, xerr=[[pt - lo], [hi - pt]], fmt=MRK["pendulum"], color=COL["pendulum"],
              ms=3.5, capsize=2, elinewidth=0.8, capthick=0.8)
ax2i.axvline(0, color="k", lw=0.5)
ax2i.set_xlim(-0.001, 0.006)
ax2i.set_xticks([0, 0.002, 0.004])
ax2i.set_xticklabels(["0", ".002", ".004"])
ax2i.set_yticks([])
ax2i.spines["left"].set_visible(False)
ax2i.set_ylim(-1, 1)
ax2i.set_title("pendulum (zoom)", fontsize=6, pad=1)

# ---- row 3 --------------------------------------------------------------
for i, t in enumerate(TASKS):
    ax = f.add_axes([0.17 + i * 0.285, 0.065, 0.22, 0.17])
    s = sc[t]
    null = np.array(s["r_ext_scrambled_all"])
    bp = ax.boxplot(null, positions=[0], widths=0.5, patch_artist=True, showfliers=False)
    for k in ("boxes",):
        for b in bp[k]:
            b.set(facecolor=NULL, edgecolor="#777777", lw=0.5)
    for k in ("whiskers", "caps", "medians"):
        for b in bp[k]:
            b.set(color="#777777", lw=0.5)
    rng = np.random.default_rng(0)
    ax.scatter(rng.uniform(-0.12, 0.12, len(null)), null, s=1.5, color="#777777", zorder=3)
    ax.scatter([1], [s["r_ext_real"]], marker=MRK[t], s=16, color=COL[t], zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["scr.", "real"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.05, 0.65)
    ax.set_yticks([0, 0.3, 0.6])
    if i == 0:
        ax.set_ylabel(r"$r(C_t, E^{\mathrm{state}})$")
        letter(ax, "c", x=-0.75, y=1.12)
    else:
        ax.set_yticklabels([])
    ax.text(0.5, 1.0, f"{t}\n$z={s['z_ext']:+.1f}$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6)
save(f, NAME)
