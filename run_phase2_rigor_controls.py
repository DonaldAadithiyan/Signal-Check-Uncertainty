#!/usr/bin/env python3.11
"""
Phase 2 — Cheap falsification and rigor controls (Reframed_Confusion_RSSM_Project,
Section 7). Both sub-experiments here are marked MANDATORY in the doc's priority
matrix (Section 17), to be completed before the expensive SAE/POMDP/scaling work.

Depends on Gate 1 having passed (outputs/phase1_external_validation/phase1_results.json,
run_phase1_external_validation.py) — it did, 3/3 tasks (see
outputs/deliverables/phase1_external_validation.md). This phase asks two sharper
falsification questions about WHY C_t works, not just whether it does.

§7.1 — Temporal scrambling
    C_t = sum_i gamma^i * 1[KL_{t-i} > median] is a function of an ORDERED recent
    history. Permuting the order of the same set of high/low-KL flags within a
    trajectory destroys temporal structure while preserving the exact same COUNT
    of recent high-KL steps in any given window (in expectation). If a scrambled
    C_t explains h_t and predicts external imagination failure just as well as the
    real C_t, the finding reduces to "h_t reflects the AMOUNT of recent high-KL
    activity" -- interesting, but not "h_t reflects the ORDER/RECENCY-WEIGHTED
    HISTORY of predictive difficulty," which is the stronger claim made throughout
    this project (closed-form fit at gamma~0.95, ~13-step memory).

    Test, per task, per trajectory:
      1. real C_t   = discounted count over the TRUE KL order.
      2. scrambled C_t = discounted count over a WITHIN-TRAJECTORY RANDOM PERMUTATION
         of the high/low-KL flag sequence (boundary-respecting: permute within each
         trajectory only, never across trajectories).
      3. Compare R^2(h_t -> C_t) [probe score / ridge fit] for real vs scrambled,
         and compare predictive correlation with the EXTERNAL Phase-1 target
         E^state_{t,K=10} for real vs scrambled, using the SAME held-out sites
         Phase 1 already collected trajectories for (re-collected here fresh with
         the same protocol/seed for a clean, self-contained repro).
      Report across N_SCRAMBLES independent permutations (mean +/- std), not one
      lucky/unlucky shuffle.

§7.2 — EMA reconstruction baseline, head-to-head
    Section 7.2 flags a specific reviewer risk: recurrence may be unnecessary if a
    simple SMOOTHED reconstruction-error statistic (no RSSM memory, no recurrent
    state) does the same job as C_t. alpha is tuned ONLY on a train/calibration
    split (never on the eval sites), exactly as in Phase 1. This experiment reports
    the comparison as its own clean, standalone result:
      - standalone correlation of EMARecon_t vs C_t with external E^state,
      - incremental R^2 of C_t over EMARecon_t ALONE (no other baselines) -- the
        single cleanest version of "does recurrence add anything over a smoothed
        scalar", and
      - incremental R^2 of EMARecon_t over C_t ALONE (the converse direction).

Runs on the 3 EXISTING frozen models. No retraining. CPU only.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from run_phase1_external_validation import (
    ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, bootstrap_pearson_ci, MIN_T, N_TRAJ, SEED,
)
from src.probe.linear_probe import binarise_by_median, train_probe

N_SCRAMBLES = 20
K_HEADLINE = 10
OUT_DIR = 'outputs/phase2_rigor_controls'
FIG_DIR = 'outputs/figures'


# ─── §7.1 temporal scrambling ─────────────────────────────────────────────────

def compute_ct_ordered(high_flags, gamma, max_lag=50):
    """C_t from an explicit 0/1 high-KL flag sequence (single trajectory, no
    boundary crossing since called per-trajectory)."""
    N = len(high_flags)
    ct = np.zeros(N, dtype=np.float64)
    for i in range(N):
        val = 0.0
        for lag in range(max_lag):
            j = i - lag
            if j < 0:
                break
            val += (gamma ** lag) * high_flags[j]
        ct[i] = val
    return ct


def scrambled_ct_for_traj(kl_traj, kl_median, gamma, rng):
    high = (kl_traj > kl_median).astype(np.float64)
    perm = rng.permutation(len(high))
    high_scrambled = high[perm]
    return compute_ct_ordered(high_scrambled, gamma)


def real_ct_for_traj(kl_traj, kl_median, gamma):
    high = (kl_traj > kl_median).astype(np.float64)
    return compute_ct_ordered(high, gamma)


def run_scrambling(task, spec, cfg, gamma):
    print(f"\n{'='*78}\n{task.upper()} — §7.1 TEMPORAL SCRAMBLING\n{'='*78}")
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    kl_median = float(np.median(tr['kl']))

    print(f"  collecting {N_TRAJ} held-out trajectories (shared protocol w/ Phase 1)...")
    trajs = collect_trajectories(model, spec, N_TRAJ, cfg, seed=SEED)

    # sites with headline horizon available
    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - K_HEADLINE - 1):
            sites.append((ti, t))
    rng_site = np.random.default_rng(SEED)
    if len(sites) > 4000:
        sel = rng_site.choice(len(sites), 4000, replace=False)
        sites = [sites[i] for i in sel]
    print(f"  {len(sites)} evaluation sites")

    # external target E^state_{t,K=10} at each site, computed once (independent of C_t)
    e_state_target = []
    real_ct_vals = []
    trajs_ct_real = [real_ct_for_traj(t['kl'], kl_median, gamma) for t in trajs]
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, _, _ = imagined_vs_real_obs(model, trj, t, K_HEADLINE, spec['domain'])
        if len(state_dist_full) < K_HEADLINE:
            e_state_target.append(np.nan)
        else:
            e_state_target.append(e_state(state_dist_full, K_HEADLINE))
        real_ct_vals.append(trajs_ct_real[ti][t])
    e_state_target = np.array(e_state_target)
    real_ct_vals = np.array(real_ct_vals)
    valid = ~np.isnan(e_state_target)
    e_state_target = e_state_target[valid]
    real_ct_vals = real_ct_vals[valid]
    valid_sites = [s for s, v in zip(sites, valid) if v]
    print(f"  {len(valid_sites)} sites with full horizon")

    # h_t at these sites (for the R^2(h_t -> C_t) explanation test)
    h_at_sites = np.array([trajs[ti]['h'][t] for (ti, t) in valid_sites])

    # real C_t: R^2 explaining h_t (ridge regression h_t -> C_t) and correlation with E^state
    scaler_h = StandardScaler()
    Xh = scaler_h.fit_transform(h_at_sites)
    ridge_real = Ridge(alpha=1.0).fit(Xh, real_ct_vals)
    r2_ht_real = ridge_real.score(Xh, real_ct_vals)
    r_ext_real, p_ext_real = pearsonr(real_ct_vals, e_state_target)

    # scrambled C_t: repeat N_SCRAMBLES times with independent permutations
    r2_ht_scrambled_list, r_ext_scrambled_list = [], []
    for s in range(N_SCRAMBLES):
        rng = np.random.default_rng(1000 + s)
        trajs_ct_scr = [scrambled_ct_for_traj(t['kl'], kl_median, gamma, rng) for t in trajs]
        scr_ct_vals = np.array([trajs_ct_scr[ti][t] for (ti, t) in valid_sites])

        ridge_scr = Ridge(alpha=1.0).fit(Xh, scr_ct_vals)
        r2_ht_scrambled_list.append(ridge_scr.score(Xh, scr_ct_vals))

        r_ext_scr, _ = pearsonr(scr_ct_vals, e_state_target)
        r_ext_scrambled_list.append(r_ext_scr)

    r2_ht_scrambled_mean = float(np.mean(r2_ht_scrambled_list))
    r2_ht_scrambled_std = float(np.std(r2_ht_scrambled_list))
    r_ext_scrambled_mean = float(np.mean(r_ext_scrambled_list))
    r_ext_scrambled_std = float(np.std(r_ext_scrambled_list))

    print(f"  R^2(h_t -> C_t):        real={r2_ht_real:.4f}   "
          f"scrambled={r2_ht_scrambled_mean:.4f} +/- {r2_ht_scrambled_std:.4f} "
          f"(n={N_SCRAMBLES} shuffles)")
    print(f"  r(C_t, E^state_K=10):   real={r_ext_real:+.4f} (p={p_ext_real:.2g})   "
          f"scrambled={r_ext_scrambled_mean:+.4f} +/- {r_ext_scrambled_std:.4f}")

    # verdict: does temporal order matter, beyond scrambled variability?
    z_ht = (r2_ht_real - r2_ht_scrambled_mean) / (r2_ht_scrambled_std + 1e-8)
    z_ext = (abs(r_ext_real) - abs(r_ext_scrambled_mean)) / (r_ext_scrambled_std + 1e-8)
    order_matters_ht = z_ht > 2.0
    order_matters_ext = z_ext > 2.0
    print(f"  z(real vs scrambled), R^2(h_t->C_t):   {z_ht:+.2f}  "
          f"-> temporal order {'MATTERS' if order_matters_ht else 'does NOT clearly matter'}")
    print(f"  z(real vs scrambled), r(C_t,E^state):  {z_ext:+.2f}  "
          f"-> temporal order {'MATTERS' if order_matters_ext else 'does NOT clearly matter'}")

    return dict(
        task=task, gamma=gamma, n_sites=len(valid_sites), n_scrambles=N_SCRAMBLES,
        r2_ht_real=r2_ht_real,
        r2_ht_scrambled_mean=r2_ht_scrambled_mean, r2_ht_scrambled_std=r2_ht_scrambled_std,
        r2_ht_scrambled_all=r2_ht_scrambled_list,
        r_ext_real=r_ext_real, p_ext_real=p_ext_real,
        r_ext_scrambled_mean=r_ext_scrambled_mean, r_ext_scrambled_std=r_ext_scrambled_std,
        r_ext_scrambled_all=r_ext_scrambled_list,
        z_ht=float(z_ht), z_ext=float(z_ext),
        order_matters_ht=bool(order_matters_ht), order_matters_ext=bool(order_matters_ext),
    )


# ─── §7.2 EMA head-to-head ────────────────────────────────────────────────────

def run_ema_headtohead(task, spec, cfg, gamma):
    print(f"\n{'='*78}\n{task.upper()} — §7.2 EMA RECONSTRUCTION HEAD-TO-HEAD\n{'='*78}")
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])
    print(f"  EMA alpha tuned on train split only: {ema_alpha}")

    trajs = collect_trajectories(model, spec, N_TRAJ, cfg, seed=SEED)
    kl_median = float(np.median(tr['kl']))
    for trj in trajs:
        trj['ct'] = real_ct_for_traj(trj['kl'], kl_median, gamma)
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - K_HEADLINE - 1):
            sites.append((ti, t))
    rng_site = np.random.default_rng(SEED)
    if len(sites) > 4000:
        sel = rng_site.choice(len(sites), 4000, replace=False)
        sites = [sites[i] for i in sel]

    ct_vals, ema_vals, targets = [], [], []
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, _, _ = imagined_vs_real_obs(model, trj, t, K_HEADLINE, spec['domain'])
        if len(state_dist_full) < K_HEADLINE:
            continue
        ct_vals.append(trj['ct'][t])
        ema_vals.append(trj['ema_recon'][t])
        targets.append(e_state(state_dist_full, K_HEADLINE))
    ct_vals, ema_vals, targets = map(lambda a: np.array(a, dtype=np.float64),
                                      (ct_vals, ema_vals, targets))
    print(f"  {len(ct_vals)} sites")

    r_ct, p_ct = pearsonr(ct_vals, targets)
    r_ema, p_ema = pearsonr(ema_vals, targets)
    print(f"  standalone r(C_t, E^state)       = {r_ct:+.4f} (p={p_ct:.2g})")
    print(f"  standalone r(EMARecon, E^state)  = {r_ema:+.4f} (p={p_ema:.2g})")

    # incremental R^2 of C_t over EMA alone, and vice versa
    X_ema = StandardScaler().fit_transform(ema_vals.reshape(-1, 1))
    X_both = StandardScaler().fit_transform(np.column_stack([ct_vals, ema_vals]))

    r2_ema_only = LinearRegression().fit(X_ema, targets).score(X_ema, targets)
    r2_both = LinearRegression().fit(X_both, targets).score(X_both, targets)
    incr_ct_over_ema = r2_both - r2_ema_only

    X_ct = StandardScaler().fit_transform(ct_vals.reshape(-1, 1))
    r2_ct_only = LinearRegression().fit(X_ct, targets).score(X_ct, targets)
    incr_ema_over_ct = r2_both - r2_ct_only

    print(f"  R^2(EMA alone)           = {r2_ema_only:.4f}")
    print(f"  R^2(C_t alone)           = {r2_ct_only:.4f}")
    print(f"  R^2(both)                = {r2_both:.4f}")
    print(f"  incremental R^2: C_t over EMA alone  = {incr_ct_over_ema:+.4f}")
    print(f"  incremental R^2: EMA over C_t alone  = {incr_ema_over_ct:+.4f}")

    recurrence_adds_value = incr_ct_over_ema > 0.002  # same practical threshold as Phase 1 Gate 1
    print(f"  -> recurrence (C_t) adds value over a simple smoothed scalar (EMA): "
          f"{'YES' if recurrence_adds_value else 'NO / negligible'}")

    return dict(
        task=task, ema_alpha=ema_alpha, n_sites=len(ct_vals),
        r_ct=r_ct, p_ct=p_ct, r_ema=r_ema, p_ema=p_ema,
        r2_ema_only=r2_ema_only, r2_ct_only=r2_ct_only, r2_both=r2_both,
        incremental_r2_ct_over_ema=incr_ct_over_ema,
        incremental_r2_ema_over_ct=incr_ema_over_ct,
        recurrence_adds_value=bool(recurrence_adds_value),
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    scrambling_results = {}
    ema_results = {}
    for task, spec in ENVS.items():
        gamma = spec['gamma_ct']
        scrambling_results[task] = run_scrambling(task, spec, cfg, gamma)
        ema_results[task] = run_ema_headtohead(task, spec, cfg, gamma)

    print(f"\n{'='*78}\nPHASE 2 SUMMARY\n{'='*78}")
    print("\n§7.1 Temporal scrambling:")
    for task, r in scrambling_results.items():
        print(f"  {task}: order matters for R^2(h_t->C_t)? {r['order_matters_ht']} "
              f"(z={r['z_ht']:+.2f}); for external prediction? {r['order_matters_ext']} "
              f"(z={r['z_ext']:+.2f})")
    print("\n§7.2 EMA head-to-head:")
    for task, r in ema_results.items():
        print(f"  {task}: incremental R^2 C_t-over-EMA={r['incremental_r2_ct_over_ema']:+.4f} "
              f"-> recurrence adds value: {r['recurrence_adds_value']}")

    all_results = dict(scrambling=scrambling_results, ema_headtohead=ema_results)
    out_path = os.path.join(OUT_DIR, 'phase2_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    # ── figure ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, task in enumerate(ENVS.keys()):
        r = scrambling_results[task]
        ax = axes[0, i]
        ax.hist(r['r2_ht_scrambled_all'], bins=10, alpha=0.6, color='gray', label='scrambled')
        ax.axvline(r['r2_ht_real'], color='red', lw=2, label='real order')
        ax.set_title(f'{task}: R^2(h_t -> C_t)\nz={r["z_ht"]:+.2f}')
        ax.set_xlabel('R^2'); ax.legend(fontsize=8)

        ax2 = axes[1, i]
        e = ema_results[task]
        bars = ['EMA alone', 'C_t alone', 'both']
        vals = [e['r2_ema_only'], e['r2_ct_only'], e['r2_both']]
        ax2.bar(bars, vals, color=['steelblue', 'darkorange', 'green'])
        ax2.set_title(f'{task}: R^2 vs E^state_K=10\nC_t-over-EMA incr={e["incremental_r2_ct_over_ema"]:+.4f}')
        ax2.set_ylabel('R^2')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'phase2_rigor_controls.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {fig_path}")


if __name__ == '__main__':
    main()
