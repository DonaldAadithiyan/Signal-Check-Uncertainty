#!/usr/bin/env python3.11
"""
One figure for the paper: visualise the Set-C inversion mechanism (Task N).

Three scatter panels (cartpole, reacher, pendulum), each showing per-state
reconstruction error (x) vs the confusion integral C_t (y) for states within a
single representative KL bin (the middle decile), with a linear trend line.

The visual point: the trend tilts UP on cartpole (+) and DOWN on reacher and
pendulum (−) — the within-KL-bin recon↔C_t sign flip that makes Set C's recon-based
labelling agree with confusion on cartpole but run opposite to it (invert) on pendulum.

No new numbers or claims — this reproduces Task N's exact analysis (same collection
procedure, `run_task_n_mechanism.collect`) so the annotated correlations are pulled
from the data, not retyped, and saves the plot in the repo's figure style.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.probe.intervention import compute_ct
import run_task_n_mechanism as N

OUT = 'outputs/figures/setc_inversion_mechanism.png'
N_BINS = 10
MID_BIN = 5                      # middle KL decile (representative)
SETC = {'cartpole': 0.72, 'reacher': 0.62, 'pendulum': 0.32}   # Set C AUROC per env (Tasks C/D/J)
COLORS = {'cartpole': '#2a6f97', 'reacher': '#e07a1f', 'pendulum': '#9b2226'}


def collect_bin(env, spec):
    """Task N's exact procedure: fresh collection, C_t with env γ, middle KL decile."""
    model = N.load_model(spec['ck'], spec['obs_dim'], spec['act_dim'])
    d = N.collect(model, spec['factory'], spec['act_dim'], 20)   # seed 555 default (as Task N)
    ct = compute_ct(d['kl'], d['traj_id'], gamma=N.GAMMA[env])
    edges = np.percentile(d['kl'], np.linspace(0, 100, N_BINS + 1))
    kb = np.digitize(d['kl'], edges[1:-1])
    # mean within-bin correlation across all valid bins (Task N's headline statistic)
    per = []
    for b in range(N_BINS):
        idx = np.where(kb == b)[0]
        if len(idx) > 30 and np.std(ct[idx]) > 1e-9 and np.std(d['recon'][idx]) > 1e-9:
            per.append(pearsonr(d['recon'][idx], ct[idx])[0])
    mean_r = float(np.mean(per))
    # middle-bin points (what we plot)
    m = np.where(kb == MID_BIN)[0]
    recon_b, ct_b = d['recon'][m], ct[m]
    bin_r = float(pearsonr(recon_b, ct_b)[0])
    kl_lo, kl_hi = edges[MID_BIN], edges[MID_BIN + 1]
    return dict(recon=recon_b, ct=ct_b, bin_r=bin_r, mean_r=mean_r,
                n=len(m), kl_lo=float(kl_lo), kl_hi=float(kl_hi))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    data = {env: collect_bin(env, spec) for env, spec in N.ENVS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=False)
    fig.suptitle('Set-C inversion mechanism: within-KL-bin recon–$C_t$ correlation flips sign',
                 fontsize=9.5, fontweight='bold', y=1.02)

    order = ['cartpole', 'reacher', 'pendulum']
    for ax, env in zip(axes, order):
        dd = data[env]
        c = COLORS[env]
        # subsample for legibility if dense
        idx = np.arange(len(dd['recon']))
        if len(idx) > 600:
            idx = np.random.default_rng(0).choice(idx, 600, replace=False)
        x, y = dd['recon'][idx], dd['ct'][idx]
        ax.scatter(x, y, s=7, alpha=0.30, c=c, rasterized=True, edgecolors='none')
        # linear trend on the FULL bin (not just subsample)
        xf, yf = dd['recon'], dd['ct']
        coef = np.polyfit(xf, yf, 1)
        xs = np.linspace(np.percentile(xf, 1), np.percentile(xf, 99), 100)
        ax.plot(xs, np.polyval(coef, xs), color='black', lw=1.8)
        sign = '+' if dd['bin_r'] >= 0 else '−'
        ax.set_title(f"{env}  (Set C AUROC {SETC[env]:.2f})", fontsize=8.5, fontweight='bold')
        # annotation: this bin's r (matches the slope) + mean over all bins (Task N headline)
        ax.text(0.04, 0.96,
                f"bin $r$ = {dd['bin_r']:+.2f}\nmean-over-bins $r$ = {dd['mean_r']:+.2f}",
                transform=ax.transAxes, va='top', ha='left', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c, alpha=0.9))
        ax.set_xlabel('reconstruction error', fontsize=8)
        if env == 'cartpole':
            ax.set_ylabel('confusion integral $C_t$', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # shared caption line under the panels
    fig.text(0.5, -0.10,
             'Middle KL decile shown per environment (recon and $C_t$ matched on KL). The trend tilts '
             'up on cartpole and down on reacher/pendulum: within a KL bin, high-recon states are '
             'more-confused on cartpole but less-confused on pendulum, so the recon-based Set C label '
             'agrees with confusion on cartpole and inverts on pendulum.',
             ha='center', va='top', fontsize=6.6, wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f"Saved {OUT}")
    print("Per-env (middle bin / mean-over-bins):")
    for env in order:
        dd = data[env]
        print(f"  {env:<10} bin{MID_BIN} n={dd['n']} r={dd['bin_r']:+.3f} | mean-over-bins r={dd['mean_r']:+.3f}"
              f" | Set C {SETC[env]:.2f}")


if __name__ == '__main__':
    main()
