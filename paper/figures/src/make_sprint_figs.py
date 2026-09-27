"""Sprint figures, regenerated ONLY from outputs/sprint/results.json (the
corrected pipeline). Never mixes in old-pipeline results.

  fig1_emergence.pdf    (a) R^2(readout, KL / C_t / C_t^past), eval, 95% CI
                        (b) R^2(readout, C_t^past) across the gamma grid
  fig2_external.pdf     (a) Delta R^2 of C_t, C_t^past for E^state_10 beyond
                            KL+Rec+EMARec+EMAKL, 95% episode-bootstrap CI
                        (b) r(C_t^past, E^state_K) and r(KL, E^state_K) vs K
  fig4_dissociation.pdf z of the steering slope vs the 50-direction null:
                        readout k=0 (hollow), k=10 (filled), E^state (vermillion)
  fig5_robustness.pdf   (a) Delta R^2 per C_t variant (calibration gamma)
                        (b) Delta R^2 vs gamma for C_t and C_t^past

Usage: python make_sprint_figs.py [--smoke]
  --smoke reads outputs/sprint/smoke/results.json and writes to
  outputs/sprint/smoke/figures/ (never into paper/figures/).
"""
import argparse
import json
import os

import numpy as np

import style
from style import fig, letter, TASKS, COL, MRK, READ, EXT, NULL, plt, CM

style.SIZES["fig5_robustness.pdf"] = 7.6
VAR_LABEL = {"C": "$C_t$", "C_past": "$C^{past}_t$", "C_p60": "$C_t$ p60",
             "C_p60_past": "p60 past", "C_p70": "$C_t$ p70", "C_p70_past": "p70 past",
             "Ccont": "cont.", "Ccont_past": "cont. past"}  # x-tick labels


def save(f, name, out):
    os.makedirs(out, exist_ok=True)
    f.savefig(os.path.join(out, name), format="pdf", pad_inches=0.01)
    prev = os.environ.get("FIG_PREVIEW_DIR")
    if prev:
        os.makedirs(prev, exist_ok=True)
        f.savefig(os.path.join(prev, name.replace(".pdf", ".png")), dpi=300)
    plt.close(f)
    print("wrote", os.path.join(out, name))


def ci_err(d):
    return [[d["point"] - d["lo"]], [d["hi"] - d["point"]]]


def tasks_in(R, key):
    return [t for t in TASKS if t in R.get(key, {})]


def _models(p, t):
    return sorted(p.get(t, {}), key=lambda s: (s != "primary", s))


def fig1(R, out):
    """(a) readout vs C_t, cartpole primary (seeded sample); (b) R^2 of the readout
    on KL_t (hatched), C_t (solid), C_t^past (open), primary bar + seed dots;
    (c) share of v in the top-50 PCs vs the random-vector band; (d) ridge h->C_t
    R^2, real order vs scrambled (current flag fixed), z above."""
    p1 = R["p1"]
    ts = [t for t in TASKS if "primary" in p1.get(t, {})]
    f = fig("fig1_emergence.pdf")
    ax = [f.add_axes(r) for r in ([0.16, 0.60, 0.32, 0.31], [0.66, 0.60, 0.33, 0.31],
                                   [0.16, 0.12, 0.32, 0.31], [0.66, 0.12, 0.33, 0.31])]
    if "cartpole" in ts:
        d = p1["cartpole"]["primary"]
        ss = d["scatter_sample"]
        ax[0].scatter(ss["ct"], ss["readout"], s=0.6, color=COL["cartpole"], alpha=0.25, lw=0,
                      rasterized=True)
        ax[0].set_title(f"cartpole, $\\gamma$={d['variants']['C']['gamma']}, "
                        f"$R^2$={d['variants']['C']['r2']['point']:.2f}", fontsize=6, pad=2)
    ax[0].set_xlabel("$C_t$", labelpad=1)
    ax[0].set_ylabel("readout", labelpad=1)
    letter(ax[0], "a", x=-0.50)

    keys = [("r2_readout_kl", "////", True), ("C", None, True), ("C_past", None, False)]
    def val(m, k):
        return m[k]["point"] if k == "r2_readout_kl" else m["variants"][k]["r2"]["point"]
    for i, t in enumerate(ts):
        for j, (k, hatch, filled) in enumerate(keys):
            x = i + (j - 1) * 0.27
            prim = p1[t]["primary"]
            ax[1].bar(x, val(prim, k), 0.24, color=COL[t] if filled and hatch is None else "white",
                      edgecolor=COL[t], lw=0.6, hatch=hatch)
            seeds = [val(p1[t][s], k) for s in _models(p1, t) if s != "primary"]
            ax[1].plot([x] * len(seeds), seeds, ".", color="k", ms=1.5)
    ax[1].set_xticks(range(len(ts)), ts)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("$R^2$(readout, ·)", labelpad=1)
    from matplotlib.patches import Patch
    ax[1].legend([Patch(fc="white", ec="0.3", hatch="////", lw=0.5), Patch(fc="0.3", ec="0.3", lw=0.5),
                  Patch(fc="white", ec="0.3", lw=0.5)], ["KL$_t$", "$C_t$", "$C^{past}_t$"],
                 loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.22), handlelength=1, columnspacing=0.5)
    letter(ax[1], "b", x=-0.46)

    for i, t in enumerate(ts):
        g = p1[t]["primary"]["geometry"]["standardized_top50"]
        ax[2].add_patch(plt.Rectangle((i - 0.3, g["random_p5"]), 0.6, 2 * (g["random_mean"] - g["random_p5"]),
                                      color=NULL, alpha=0.5, lw=0))
        ax[2].plot([i - 0.3, i + 0.3], [g["random_mean"]] * 2, color="0.4", lw=0.6)
        fr = [p1[t][s]["geometry"]["standardized_top50"]["frac"] for s in _models(p1, t)]
        ax[2].plot([i] * len(fr), fr, MRK[t], color=COL[t], ms=2.5, mfc="white")
        ax[2].plot(i, g["frac"], MRK[t], color=COL[t], ms=3.5)
    ax[2].set_xticks(range(len(ts)), ts)
    ax[2].set_xlim(-0.6, len(ts) - 0.4)
    ax[2].set_ylim(0, None)
    ax[2].set_ylabel("top-50 PC share of $v$", labelpad=1)
    letter(ax[2], "c", x=-0.50)

    for i, t in enumerate(ts):
        rsc = p1[t]["primary"]["ridge_scramble"]["C"]
        ax[3].bar(i - 0.17, rsc["real"], 0.3, color=COL[t])
        ax[3].bar(i + 0.17, rsc["null_mean"], 0.3, color=NULL,
                  yerr=rsc["null_std"], error_kw=dict(lw=0.5, capsize=1))
        ax[3].text(i, max(rsc["real"], rsc["null_mean"]) + 0.04, f"$z$={rsc['z']:+.1f}",
                   ha="center", fontsize=5.5)
    ax[3].set_xticks(range(len(ts)), ts)
    ax[3].set_ylim(0, 1.1)
    ax[3].set_ylabel("$R^2$($h_t \\to C_t$)", labelpad=1)
    letter(ax[3], "d", x=-0.46)
    save(f, "fig1_emergence.pdf", out)


def fig2(R, out):
    """(a) r(C_t, E^state_K) vs K with 95% bands; (b) Delta R^2 of C_t (filled) and
    C_t^past (open) with CIs; (c) external Delta R^2 real vs scrambled, z above."""
    p2 = R["p2"]
    ts = tasks_in(R, "p2")
    f = fig("fig2_external.pdf")
    ax = [f.add_axes(r) for r in ([0.16, 0.58, 0.31, 0.34], [0.66, 0.58, 0.33, 0.34],
                                   [0.16, 0.13, 0.83, 0.27])]
    for t in ts:
        H = p2[t]["horizons"]
        Ks = sorted(H, key=int)
        x = [int(k) for k in Ks]
        ax[0].plot(x, [H[k]["C"]["point"] for k in Ks], marker=MRK[t], ms=2.5, color=COL[t], label=t)
        ax[0].fill_between(x, [H[k]["C"]["lo"] for k in Ks], [H[k]["C"]["hi"] for k in Ks],
                           color=COL[t], alpha=0.15, lw=0)
    ax[0].set_xticks([1, 5, 10, 20])
    ax[0].set_xlabel("horizon $K$", labelpad=1)
    ax[0].set_ylabel("$r$($C_t$, $E^{state}_K$)", labelpad=1)
    ax[0].axhline(0, color="0.5", lw=0.5)
    ax[0].legend(loc="best", handlelength=1, fontsize=5.5)
    letter(ax[0], "a", x=-0.50)
    for i, t in enumerate(ts):
        for j, (name, filled) in enumerate((("C", True), ("C_past", False))):
            d = p2[t]["table3"][name]["delta_r2"]
            ax[1].errorbar(i + (j - 0.5) * 0.3, d["point"], yerr=ci_err(d), fmt=MRK[t], ms=3, lw=0.7,
                           capsize=1.5, color=COL[t], mfc=COL[t] if filled else "white")
    ax[1].axhline(0, color="0.5", lw=0.5)
    ax[1].set_xticks(range(len(ts)), ts)
    ax[1].set_ylabel("$\\Delta R^2$, $E^{state}_{10}$", labelpad=1)
    ax[1].plot([], [], "o", color="0.3", ms=3, ls="", label="$C_t$")
    ax[1].plot([], [], "o", color="0.3", mfc="white", ms=3, ls="", label="$C^{past}_t$")
    ax[1].legend(loc="best", handlelength=0.8, fontsize=5.5)
    letter(ax[1], "b", x=-0.50)
    for i, t in enumerate(ts):
        for j, name in enumerate(("C", "C_past")):
            sc = p2[t]["scrambling_external"][name]
            x0 = i * 2.5 + j
            ax[2].bar(x0 - 0.2, sc["real"], 0.38, color=COL[t], alpha=1.0 if j == 0 else 0.5)
            ax[2].bar(x0 + 0.2, sc["null_mean"], 0.38, color=NULL, yerr=sc["null_std"],
                      error_kw=dict(lw=0.5, capsize=1))
            ax[2].text(x0, max(sc["real"], sc["null_mean"]), f"$z$={sc['z']:+.1f}", ha="center",
                       va="bottom", fontsize=5)
    ax[2].set_xticks([i * 2.5 + j for i in range(len(ts)) for j in (0, 1)],
                     [f"{t}\n{lab}" for t in ts for lab in ("$C_t$", "$C^{past}_t$")], fontsize=5.5)
    ax[2].axhline(0, color="0.5", lw=0.5)
    ax[2].set_ylabel("$\\Delta R^2$ real/scr.", labelpad=1)
    letter(ax[2], "c", x=-0.17)
    save(f, "fig2_external.pdf", out)


def fig4(R, out):
    p3 = R["p3"]
    ts = tasks_in(R, "p3")
    f = fig("fig4_dissociation.pdf")
    ax = f.add_axes([0.14, 0.14, 0.84, 0.72])
    for i, t in enumerate(ts):
        labels = sorted(p3[t], key=lambda s: (s != "primary", s))
        for s_i, lab in enumerate(labels):
            nz = p3[t][lab]["nulls"].get("null50")
            if nz is None:
                continue
            jit = (s_i - (len(labels) - 1) / 2) * 0.05
            ax.plot(i - 0.22 + jit, nz["slope_r0"]["z"], MRK[t], mfc="white", mec=READ, ms=3.5)
            ax.plot(i + jit, nz["slope_r10"]["z"], MRK[t], color=READ, ms=3.5)
            ax.plot(i + 0.22 + jit, nz["slope_es"]["z"], MRK[t], color=EXT, ms=3.5)
    ax.axhspan(-2, 2, color=NULL, alpha=0.35, lw=0)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xticks(range(len(ts)), ts)
    ax.set_xlim(-0.5, len(ts) - 0.5)
    ax.set_ylabel("steering slope, $z$ vs 50-direction null")
    ax.plot([], [], "o", mfc="white", mec=READ, ms=3.5, ls="", label="readout $k{=}0$")
    ax.plot([], [], "o", color=READ, ms=3.5, ls="", label="readout $k{=}10$")
    ax.plot([], [], "o", color=EXT, ms=3.5, ls="", label="$E^{state}_{10}$")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), handlelength=0.8)
    save(f, "fig4_dissociation.pdf", out)


def fig5(R, out):
    p2 = R["p2"]
    ts = tasks_in(R, "p2")
    f = fig("fig5_robustness.pdf")
    ax1 = f.add_axes([0.17, 0.58, 0.81, 0.30])
    ax2 = f.add_axes([0.17, 0.13, 0.81, 0.26])
    names = list(VAR_LABEL)
    for i, t in enumerate(ts):
        for j, n in enumerate(names):
            d = p2[t]["variants"][n]["delta_r2"]
            x = j + (i - 1) * 0.22
            ax1.errorbar(x, d["point"], yerr=ci_err(d), fmt=MRK[t], ms=2.5, lw=0.6, capsize=1,
                         color=COL[t], mfc=COL[t] if not n.endswith("past") else "white",
                         label=t if j == 0 else None)
    ax1.axhline(0, color="0.5", lw=0.5)
    ax1.set_xticks(range(len(names)), [VAR_LABEL[n] for n in names], fontsize=5.5,
                   rotation=30, ha="right")
    ax1.set_ylabel("$\\Delta R^2$ ($E^{state}_{10}$)")
    ax1.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.28), handlelength=0.8)
    letter(ax1, "a", x=-0.2)
    for t in ts:
        for n, ls in (("C", "-"), ("C_past", "--")):
            g = p2[t]["variants"][n]["grid"]
            gs = sorted(g, key=float)
            xs = [float(x) for x in gs]
            ax2.plot(xs, [g[x]["point"] for x in gs], ls=ls, marker=MRK[t], ms=2.2, color=COL[t],
                     mfc=COL[t] if ls == "-" else "white")
            ax2.fill_between(xs, [g[x]["lo"] for x in gs], [g[x]["hi"] for x in gs],
                             color=COL[t], alpha=0.12, lw=0)
    ax2.axhline(0, color="0.5", lw=0.5)
    ax2.set_xlabel("$\\gamma$ (fixed)")
    ax2.set_ylabel("$\\Delta R^2$")
    ax2.plot([], [], "-", color="0.3", label="$C_t$")
    ax2.plot([], [], "--", color="0.3", label="$C^{past}_t$")
    ax2.legend(loc="upper left", handlelength=1.5)
    letter(ax2, "b", x=-0.2)
    save(f, "fig5_robustness.pdf", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    base = os.path.join(style.REPO, "outputs", "sprint", "smoke" if a.smoke else "")
    with open(os.path.join(base, "results.json")) as fh:
        R = json.load(fh)
    out = os.path.join(base, "figures") if a.smoke else style.OUT
    if "p1" in R:
        fig1(R, out)
    if "p2" in R:
        fig2(R, out)
        fig5(R, out)
    if "p3" in R:
        fig4(R, out)


if __name__ == "__main__":
    main()
