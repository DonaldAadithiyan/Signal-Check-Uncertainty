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


def fig1(R, out):
    p1 = R["p1"]
    f = fig("fig1_emergence.pdf")
    ax1, ax2 = f.add_axes([0.13, 0.20, 0.38, 0.66]), f.add_axes([0.62, 0.20, 0.36, 0.66])
    keys = [("kl", "KL$_t$", "white"), ("C", "$C_t$", None), ("C_past", "$C^{past}_t$", None)]
    ts = [t for t in TASKS if t in p1 and "primary" in p1[t]]
    for i, t in enumerate(ts):
        d = p1[t]["primary"]
        vals = [d["r2_readout_kl"], d["variants"]["C"]["r2"], d["variants"]["C_past"]["r2"]]
        for j, (v, (_, _, fc)) in enumerate(zip(vals, keys)):
            x = i + (j - 1) * 0.26
            ax1.bar(x, v["point"], 0.22, color=COL[t] if fc is None else "white",
                    edgecolor=COL[t], lw=0.6, alpha=1.0 if j != 2 else 0.55,
                    hatch="////" if j == 2 else None)
            ax1.errorbar(x, v["point"], yerr=ci_err(v), color="k", lw=0.5, capsize=1)
    ax1.set_xticks(range(len(ts)), ts)
    ax1.set_ylabel("$R^2$(readout, statistic)")
    ax1.set_ylim(0, 1)
    from matplotlib.patches import Patch
    ax1.legend([Patch(fc="white", ec="k", lw=0.6), Patch(fc="0.4", ec="k", lw=0.6),
                Patch(fc="0.75", ec="k", lw=0.6, hatch="////")],
               ["KL$_t$", "$C_t$", "$C^{past}_t$"], loc="upper left", ncol=3,
               handlelength=1, columnspacing=0.6, bbox_to_anchor=(0, 1.13))
    letter(ax1, "a", x=-0.36)
    for t in ts:
        g = p1[t]["primary"]["variants"]["C_past"]["grid"]
        gs = sorted(g, key=float)
        ax2.plot([float(x) for x in gs], [g[x] for x in gs], marker=MRK[t], ms=2.5,
                 color=COL[t], label=t)
        ax2.axhline(p1[t]["primary"]["r2_readout_kl"]["point"], color=COL[t], ls=":", lw=0.6)
    ax2.set_xlabel("$\\gamma$")
    ax2.set_ylabel("$R^2$(readout, $C^{past}_t$)")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right", handlelength=1.2)
    letter(ax2, "b", x=-0.34)
    save(f, "fig1_emergence.pdf", out)


def fig2(R, out):
    p2 = R["p2"]
    ts = tasks_in(R, "p2")
    f = fig("fig2_external.pdf")
    ax1, ax2 = f.add_axes([0.14, 0.18, 0.36, 0.68]), f.add_axes([0.62, 0.18, 0.36, 0.68])
    for i, t in enumerate(ts):
        for j, (name, filled) in enumerate((("C", True), ("C_past", False))):
            d = p2[t]["table3"][name]["delta_r2"]
            x = i + (j - 0.5) * 0.3
            ax1.errorbar(x, d["point"], yerr=ci_err(d), fmt=MRK[t], ms=3.5, lw=0.7, capsize=1.5,
                         color=COL[t], mfc=COL[t] if filled else "white")
    ax1.axhline(0, color="0.5", lw=0.5)
    ax1.set_xticks(range(len(ts)), ts)
    ax1.set_ylabel("$\\Delta R^2$ for $E^{state}_{10}$")
    ax1.plot([], [], "o", color="k", ms=3, label="$C_t$")
    ax1.plot([], [], "o", color="k", mfc="white", ms=3, label="$C^{past}_t$")
    ax1.legend(loc="upper right", handlelength=0.8)
    letter(ax1, "a", x=-0.38)
    for t in ts:
        H = p2[t]["horizons"]
        Ks = sorted(H, key=int)
        ax2.plot([int(k) for k in Ks], [H[k]["C_past"]["point"] for k in Ks], marker=MRK[t],
                 ms=2.5, color=COL[t], label=t)
        ax2.plot([int(k) for k in Ks], [H[k]["kl"]["point"] for k in Ks], ls=":", color=COL[t], lw=0.7)
    ax2.set_xlabel("horizon $K$")
    ax2.set_ylabel("$r$(statistic, $E^{state}_K$)")
    ax2.set_xticks([1, 5, 10, 20])
    ax2.plot([], [], color="0.3", label="$C^{past}_t$")
    ax2.plot([], [], color="0.3", ls=":", label="KL$_t$")
    ax2.legend(loc="best", handlelength=1.2, fontsize=5.5)
    letter(ax2, "b", x=-0.34)
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
