"""Fig 1 (was Fig 2): emergence.
(a) probe score vs C_t, cartpole, gamma=0.95. Points regenerated deterministically from
    the saved outputs/data/training_states.npz with run_confusion_integral.py's own
    functions (that script printed R^2 but saved no points); asserted to reproduce
    R^2=0.798 (C_t) and 0.519 (KL alone).
(b) R^2 of readout explained by current KL vs C_t.  cartpole: primary model (from (a));
    reacher/pendulum: C_t = mean of 4 seeds (multiseed_env/aggregate.json, dots = seeds),
    KL-only = seed 0 only (second_env / third_env; not saved for other seeds).
(c) angle of v to top-50 PC subspace (three_env_comparison.json + multiseed_env).
(d) R^2(h_t -> C_t), real order vs 20 time-scrambles (phase2_results.json)."""
import os
import sys
import numpy as np
from style import fig, save, load, check, letter, TASKS, COL, MRK, NULL, HERE, REPO, plt

NAME = "fig1_emergence.pdf"

# ---------- (a) regenerate from saved training states --------------------------
cache = os.path.join(HERE, "_cache_fig2a.npz")
if not os.path.exists(cache):
    sys.path.insert(0, REPO)
    cwd = os.getcwd(); os.chdir(REPO)
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from src.config import XS_CONFIG
    from src.probe.linear_probe import binarise_by_median, train_probe
    from run_confusion_integral import compute_streak_and_integral, MAX_LAG
    tr = dict(np.load(XS_CONFIG["training_data_path"]))
    h, kl, tid = tr["h"], tr["kl"], tr["traj_id"]
    y = binarise_by_median(kl)
    high = (kl > np.median(kl)).astype(np.int32)
    tr_idx, te_idx = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=0)
    clf, scl = train_probe(h[tr_idx], y[tr_idx])
    probe = clf.predict_proba(scl.transform(h))[:, 1]
    _, integ = compute_streak_and_integral(kl, tid, high, [0.95], MAX_LAG)
    os.chdir(cwd)
    np.savez(cache, probe_te=probe[te_idx], ct_te=integ[te_idx, 0], kl_te=kl[te_idx])
c = np.load(cache)
p, ct, klte = c["probe_te"], c["ct_te"], c["kl_te"]


def r2(x, yv):
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    res = yv - A @ coef
    return 1 - res.var() / yv.var(), coef


r2_ct, coef = r2(ct.astype(float), p.astype(float))
r2_kl, _ = r2(klte.astype(float), p.astype(float))
check("cartpole R2(readout, C_t)", r2_ct, 0.80, 0.005)
check("cartpole R2(readout, KL)", r2_kl, 0.52, 0.005)

# ---------- (b)/(c) data -------------------------------------------------------
agg = load("multiseed_env/aggregate.json")
three = load("third_env/three_env_comparison.json")["table"]
key = {"cartpole": "cartpole-swingup", "reacher": "reacher-easy", "pendulum": "pendulum-swingup"}
ct_seeds = {"cartpole": [three[key["cartpole"]]["best_ct_r2"]],
            "reacher": agg["reacher"]["best_ct_r2"]["values"],
            "pendulum": agg["pendulum"]["best_ct_r2"]["values"]}
ct_bar = {"cartpole": r2_ct, "reacher": np.mean(ct_seeds["reacher"]),
          "pendulum": np.mean(ct_seeds["pendulum"])}
kl_bar = {"cartpole": r2_kl, "reacher": three[key["reacher"]]["r2_kl_baseline"],
          "pendulum": three[key["pendulum"]]["r2_kl_baseline"]}
check("reacher R2(C_t) 4-seed mean", ct_bar["reacher"], 0.26, 0.005)
check("pendulum R2(C_t) 4-seed mean", ct_bar["pendulum"], 0.86, 0.006)

ang = {"cartpole": [three[key["cartpole"]]["nullspace_angle"]],
       "reacher": agg["reacher"]["nullspace_angle"]["values"],
       "pendulum": agg["pendulum"]["nullspace_angle"]["values"]}
for t, w in zip(TASKS, [88.0, 89.4, 88.1]):
    check(f"{t} angle (primary/seed-0 model)", ang[t][0], w, 0.05)

scr = load("phase2_rigor_controls/phase2_results.json")["scrambling"]
for t in TASKS:
    assert scr[t]["z_ht"] >= 18.67, (t, scr[t]["z_ht"])
    print(f"  [OK ] {t} scramble z(h->C_t) = {scr[t]['z_ht']:.2f} (>= +18.7 after rounding)")

# ---------- draw ---------------------------------------------------------------
f = fig(NAME)
L, R, B, T = 0.15, 0.64, 0.12, 0.57
w, hh = 0.33, 0.33

# (a)
ax = f.add_axes([L, T, w, hh])
ax.hexbin(ct, p, gridsize=28, cmap="Blues", mincnt=1, bins="log", linewidths=0.1)
xs = np.linspace(ct.min(), ct.max(), 10)
ax.plot(xs, coef[0] * xs + coef[1], color="k", lw=0.8)
ax.text(0.04, 0.96, f"$R^2={r2_ct:.2f}$", transform=ax.transAxes, va="top", fontsize=7)
ax.set_xlabel(r"$C_t$ ($\gamma{=}0.95$)", labelpad=1)
ax.set_ylabel("probe score", labelpad=1)
ax.set_ylim(0, 1)
letter(ax, "a", x=-0.42, y=1.0)

# (b)
ax = f.add_axes([R, T, w, hh])
x = np.arange(3)
bw = 0.36
for i, t in enumerate(TASKS):
    ax.bar(x[i] - bw / 2, kl_bar[t], bw, color="white", edgecolor=COL[t], hatch="////", lw=0.6)
    ax.bar(x[i] + bw / 2, ct_bar[t], bw, color=COL[t], edgecolor=COL[t], lw=0.6)
    if len(ct_seeds[t]) > 1:
        ax.scatter(np.full(len(ct_seeds[t]), x[i] + bw / 2), ct_seeds[t], s=3, color="k",
                   marker=MRK[t], zorder=3, lw=0)
ax.set_xticks(x)
ax.set_xticklabels(["cart", "reach", "pend"])
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel(r"$R^2$ of readout", labelpad=1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="white", edgecolor="k", hatch="////", lw=0.5, label="KL$_t$"),
                   Patch(facecolor="#777777", label="$C_t$")],
          loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8, borderaxespad=0.1,
          bbox_to_anchor=(0, 1.02))
letter(ax, "b", x=-0.42, y=1.0)

# (c)
ax = f.add_axes([L, B, w, hh])
for i, t in enumerate(TASKS):
    v = ang[t]
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)) * (len(v) > 1), v,
               s=5, color=COL[t], marker=MRK[t], lw=0, alpha=0.7)
    ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color="k", lw=0.8)
ax.axhline(90, color="k", ls="--", lw=0.5)
ax.text(2.45, 90.1, "orthogonal", ha="right", va="bottom", fontsize=6)
ax.set_ylim(80, 91)
ax.set_xlim(-0.5, 2.5)
ax.set_xticks(range(3))
ax.set_xticklabels(["cart", "reach", "pend"])
ax.set_ylabel(r"angle to top-50 PCs ($^\circ$)", labelpad=1)
letter(ax, "c", x=-0.42, y=1.0)

# (d)
ax = f.add_axes([R, B, w, hh])
for i, t in enumerate(TASKS):
    null = np.array(scr[t]["r2_ht_scrambled_all"])
    bp = ax.boxplot(null, positions=[i - 0.17], widths=0.25, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor=NULL, edgecolor="#777777", lw=0.5)
    for k in ("whiskers", "caps", "medians"):
        for b in bp[k]:
            b.set(color="#777777", lw=0.5)
    ax.scatter([i + 0.17], [scr[t]["r2_ht_real"]], marker=MRK[t], s=12, color=COL[t], zorder=3)
    ax.text(i, 1.02, f"{scr[t]['z_ht']:+.1f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(3))
ax.set_xticklabels(["cart", "reach", "pend"])
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.5, 1.0])
ax.set_xlim(-0.5, 2.5)
ax.set_ylabel(r"$R^2(h_t \rightarrow C_t)$", labelpad=1)
ax.text(1.0, -0.2, "grey: scrambled; colour: real; top: $z$", transform=ax.transAxes,
        ha="right", va="top", fontsize=6)
letter(ax, "d", x=-0.42, y=1.0)
save(f, NAME)
