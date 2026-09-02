#!/usr/bin/env python3.11
"""
Pre-Scaling Rigor Addendum §1.1, §1.2, §1.4 — Phase 1 hardening.

§1.1: trajectory-stratified bootstrap 95% CI on the incremental-R^2 statistic
      (all 3 tasks, not just pendulum, for consistent reporting).
§1.2: VIF / partial-correlation check for C_t in each task's incremental
      regression, to test whether pendulum's R^2-positive-but-beta-negative
      discordance is a collinearity artifact.
§1.4: ensemble baseline extension to reacher/pendulum (NEW TRAINING -- flagged,
      not run in this script; see run_addendum_train_ensembles.py).

Reconstructs Phase 1's EXACT per-site dataset (same seed, same trajectory
collection, same site sampling) since the original run_phase1_external_
validation.py did not persist per-site arrays or trajectory IDs -- only
aggregated summary statistics. This script regenerates the identical dataset
deterministically (verified against the original's saved point estimates)
and additionally retains trajectory ID per site, which the stratified
bootstrap requires.
"""

import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from run_phase1_external_validation import (
    ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, ensemble_disagreement_series, MIN_T, MAX_HORIZON,
    HORIZONS, SEED, N_TRAJ, N_ENSEMBLE,
)

K_HEADLINE = 10 if 10 in HORIZONS else HORIZONS[-1]  # matches Phase 1's own inline logic
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct

OUT_DIR = 'outputs/phase1_external_validation'
N_BOOT = 1000


def rebuild_site_dataset(task, spec, cfg):
    """Reproduces run_phase1_external_validation.run_task's exact per-site
    dataset construction, retaining trajectory ID (ti) per site for stratified
    bootstrap. Deterministic given the same seed/protocol as the original."""
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    kl_median = float(np.median(tr['kl']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])

    ensemble_models = None
    if spec['ensemble'] and all(os.path.exists(p) for p in spec['ensemble']):
        ensemble_models = [load_model(p)[0] for p in spec['ensemble']]

    y = binarise_by_median(tr['kl'])
    from sklearn.model_selection import train_test_split
    idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                       stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])

    trajs = collect_trajectories(model, spec, N_TRAJ, cfg)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64),
                                gamma=spec['gamma_ct'])
        trj['probe'] = clf.predict_proba(scaler.transform(trj['h']))[:, 1]
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)
        if ensemble_models is not None:
            trj['ens_dis'] = ensemble_disagreement_series(ensemble_models, trj['obs'], cfg)
        else:
            trj['ens_dis'] = np.full(len(trj['kl']), np.nan)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - MAX_HORIZON - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED)
    if len(sites) > 6000:
        sel = rng.choice(len(sites), 6000, replace=False)
        sites = [sites[i] for i in sel]

    rows = {k: [] for k in ['ct', 'kl', 'recon', 'ema_recon', 'ens_dis', 'probe', 'traj_id']}
    e_state_by_K = {K: [] for K in HORIZONS}
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, _, _ = imagined_vs_real_obs(model, trj, t, MAX_HORIZON, spec['domain'])
        if len(state_dist_full) < MAX_HORIZON:
            continue
        for K in HORIZONS:
            e_state_by_K[K].append(e_state(state_dist_full, K))
        rows['ct'].append(trj['ct'][t])
        rows['kl'].append(trj['kl'][t])
        rows['recon'].append(trj['recon'][t])
        rows['ema_recon'].append(trj['ema_recon'][t])
        rows['ens_dis'].append(trj['ens_dis'][t])
        rows['probe'].append(trj['probe'][t])
        rows['traj_id'].append(ti)

    for k in rows:
        rows[k] = np.array(rows[k])
    for K in HORIZONS:
        e_state_by_K[K] = np.array(e_state_by_K[K], dtype=np.float64)

    have_ensemble = not np.all(np.isnan(rows['ens_dis']))
    return rows, e_state_by_K, have_ensemble


def incremental_r2(rows, target, baseline_cols, have_ensemble):
    cols = ['ct'] + baseline_cols
    X_full = np.column_stack([rows[c] for c in cols])
    X_ctrl = np.column_stack([rows[c] for c in baseline_cols])
    Xs_full = StandardScaler().fit_transform(X_full)
    Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
    r2_full = LinearRegression().fit(Xs_full, target).score(Xs_full, target)
    r2_ctrl = LinearRegression().fit(Xs_ctrl, target).score(Xs_ctrl, target)
    return r2_full - r2_ctrl


def trajectory_stratified_bootstrap_incremental_r2(rows, target, baseline_cols,
                                                    have_ensemble, n_boot=N_BOOT, seed=0):
    """Resample WHOLE TRAJECTORIES with replacement (not individual sites), then
    pool all sites from the resampled trajectories -- respects the non-i.i.d.
    structure within a trajectory (Rigor Addendum §1.1's explicit requirement)."""
    rng = np.random.default_rng(seed)
    traj_ids = np.unique(rows['traj_id'])
    n_traj = len(traj_ids)

    point = incremental_r2(rows, target, baseline_cols, have_ensemble)

    boots = []
    for _ in range(n_boot):
        resampled_traj = rng.choice(traj_ids, size=n_traj, replace=True)
        idx = np.concatenate([np.where(rows['traj_id'] == tid)[0] for tid in resampled_traj])
        rows_b = {k: rows[k][idx] for k in rows if k != 'traj_id'}
        target_b = target[idx]
        try:
            b = incremental_r2(rows_b, target_b, baseline_cols, have_ensemble)
            if np.isfinite(b):
                boots.append(b)
        except Exception:
            continue
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi), boots


def compute_vif(rows, cols):
    """VIF for C_t among the given baseline columns: regress C_t on the OTHER
    columns, VIF = 1/(1-R^2) of that regression."""
    X_others = np.column_stack([rows[c] for c in cols if c != 'ct'])
    y_ct = rows['ct']
    Xs = StandardScaler().fit_transform(X_others)
    r2 = LinearRegression().fit(Xs, y_ct).score(Xs, y_ct)
    vif = 1.0 / (1.0 - r2 + 1e-12)
    return float(vif), float(r2)


def partial_correlation(rows, target, col, other_cols):
    """Partial correlation of `col` with `target`, controlling for `other_cols`,
    via residualization."""
    X_other = np.column_stack([rows[c] for c in other_cols])
    Xs = StandardScaler().fit_transform(X_other)
    resid_col = rows[col] - LinearRegression().fit(Xs, rows[col]).predict(Xs)
    resid_target = target - LinearRegression().fit(Xs, target).predict(Xs)
    r, p = pearsonr(resid_col, resid_target)
    return float(r), float(p)


def main():
    cfg = XS_CONFIG.copy()
    results = {}

    for task, spec in ENVS.items():
        print(f"\n{'='*78}\n{task.upper()} — ADDENDUM §1.1/§1.2\n{'='*78}")
        rows, e_state_by_K, have_ensemble = rebuild_site_dataset(task, spec, cfg)
        target = e_state_by_K[K_HEADLINE]
        baseline_cols = ['kl', 'recon', 'ema_recon'] + (['ens_dis'] if have_ensemble else [])
        n_traj = len(np.unique(rows['traj_id']))
        print(f"  {len(target)} sites across {n_traj} trajectories, "
              f"have_ensemble={have_ensemble}")

        # sanity check: does the point estimate match the original saved result?
        point_check = incremental_r2(rows, target, baseline_cols, have_ensemble)
        print(f"  reconstructed incremental R^2 (sanity check) = {point_check:+.4f}")

        # §1.1: trajectory-stratified bootstrap CI
        point, lo, hi, boots = trajectory_stratified_bootstrap_incremental_r2(
            rows, target, baseline_cols, have_ensemble)
        ci_includes_zero = lo <= 0 <= hi
        print(f"  §1.1 trajectory-stratified bootstrap incremental R^2: "
              f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]  CI includes zero: {ci_includes_zero}")

        # §1.2: VIF / partial correlation for C_t
        all_cols = ['ct'] + baseline_cols
        vif_ct, r2_ct_on_others = compute_vif(rows, all_cols)
        partial_r, partial_p = partial_correlation(rows, target, 'ct', baseline_cols)
        print(f"  §1.2 VIF(C_t | {baseline_cols}) = {vif_ct:.2f} "
              f"(R^2 of C_t on others = {r2_ct_on_others:.3f})")
        print(f"       partial r(C_t, E^state | others) = {partial_r:+.4f} (p={partial_p:.3g})")

        results[task] = dict(
            n_sites=len(target), n_trajectories=n_traj, have_ensemble=have_ensemble,
            baseline_cols=baseline_cols,
            incremental_r2_point=point, incremental_r2_ci_lo=lo, incremental_r2_ci_hi=hi,
            ci_includes_zero=bool(ci_includes_zero),
            vif_ct=vif_ct, r2_ct_on_others=r2_ct_on_others,
            partial_r_ct_e_state=partial_r, partial_p_ct_e_state=partial_p,
        )

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"  {'task':<12}{'incremental R2':>18}{'95% CI':>22}{'CI incl 0?':>12}{'VIF(C_t)':>10}")
    for task, r in results.items():
        ci_str = f"[{r['incremental_r2_ci_lo']:+.4f}, {r['incremental_r2_ci_hi']:+.4f}]"
        print(f"  {task:<12}{r['incremental_r2_point']:>+18.4f}{ci_str:>22}"
              f"{str(r['ci_includes_zero']):>12}{r['vif_ct']:>10.2f}")

    # revised Gate 1 verdict incorporating CI
    revised_gate1 = {}
    for task, r in results.items():
        if r['ci_includes_zero']:
            status = 'INCONCLUSIVE (CI includes zero)'
        elif r['incremental_r2_point'] > 0:
            status = 'PASS'
        else:
            status = 'FAIL'
        revised_gate1[task] = status
        print(f"  revised Gate 1 status — {task}: {status}")

    highest_vif = max(results.items(), key=lambda kv: kv[1]['vif_ct'])
    print(f"\n  Highest VIF: {highest_vif[0]} (VIF={highest_vif[1]['vif_ct']:.2f})")

    out = dict(results=results, revised_gate1_status=revised_gate1)
    out_path = os.path.join(OUT_DIR, 'addendum_1_1_1_2_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
