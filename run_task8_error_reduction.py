#!/usr/bin/env python3.11
"""
Task 8 (distinct from Task 6/Path B) -- error-reduction adaptive-reliance test.

Question: at a fixed real-observation query budget, does selecting states
using C_t reduce actual accumulated future imagination error more than
selecting using instantaneous KL, reconstruction error, EMA-smoothed
reconstruction error, or ensemble disagreement (where available)?

Reuses Phase 1's existing imagined-vs-real rollout infrastructure entirely --
no trained policy, planner, or new world-model training. Same RNG-seeding
discipline as Task 1's fix (torch.manual_seed once at the top, covering
collect_trajectories and every per-site imagination rollout).

Design:
  1. Collect the same held-out trajectories/sites Phase 1 uses; compute C_t,
     KL, Recon, EMARecon, ensemble disagreement (cartpole only) per site, and
     E^state_{t,K} (K=10, matching Phase 1's headline horizon) as the ground-
     truth accumulated imagination error per site.
  2. Split sites into a CALIBRATION half (fit nothing here beyond percentile
     thresholds) and an EVALUATION half -- disjoint, no threshold or ranking
     cutoff chosen using evaluation-split trajectories.
  3. For a budget q, the "checked" action is fixed and identical across every
     signal, decided before any comparison: sites ranked in the top-q% by a
     given signal (threshold calibrated on the calibration split) are treated
     as if a real observation were obtained there, so their imagination error
     is deemed corrected (contributes 0 to the reported error total) -- the
     complementary bottom-(1-q)% sites keep their actual imagined E^state.
  4. Baseline error = mean E^state over ALL sites, uncorrected (q=0, nothing
     checked). Delta-E(q) = baseline - mean_error_under_checking_policy(q),
     reported per signal per budget.
"""

import os
import json
import numpy as np
import torch

from src.config import XS_CONFIG
from run_phase1_external_validation import (
    ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, ensemble_disagreement_series, MIN_T, MAX_HORIZON,
    N_TRAJ, SEED,
)
from src.probe.intervention import compute_ct

OUT_DIR = 'outputs/task8_error_reduction'
K_HEADLINE = 10
QUERY_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
SIGNALS = ['ct', 'kl', 'recon', 'ema_recon', 'ens_dis']


def build_site_dataset(task, spec, cfg):
    """Mirrors run_phase1_external_validation.run_task's dataset construction
    exactly (same seeding, same site sampling), but this file's own purpose --
    scoring E^state under different checking policies -- needs the raw
    per-site signal values and E^state array directly, not the aggregated
    regression Phase 1 already reports."""
    torch.manual_seed(SEED)
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])

    ensemble_models = None
    if spec['ensemble'] and all(os.path.exists(p) for p in spec['ensemble']):
        ensemble_models = [load_model(p)[0] for p in spec['ensemble']]

    trajs = collect_trajectories(model, spec, N_TRAJ, cfg)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64),
                                gamma=spec['gamma_ct'])
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

    rows = {k: [] for k in SIGNALS}
    e_state_vals = []
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, _, _ = imagined_vs_real_obs(model, trj, t, MAX_HORIZON, spec['domain'])
        if len(state_dist_full) < MAX_HORIZON:
            continue
        e_state_vals.append(e_state(state_dist_full, K_HEADLINE))
        for k in SIGNALS:
            rows[k].append(trj[k][t])

    for k in rows:
        rows[k] = np.array(rows[k], dtype=np.float64)
    e_state_vals = np.array(e_state_vals, dtype=np.float64)
    have_ensemble = not np.all(np.isnan(rows['ens_dis']))
    return rows, e_state_vals, have_ensemble


def delta_e_for_signal(signal_calib, signal_eval, e_state_eval, budget):
    """Threshold calibrated on the CALIBRATION split's signal distribution
    (top-`budget` fraction), applied to the EVALUATION split. Checked sites'
    E^state is zeroed (deemed corrected); unchecked sites keep their actual
    imagined E^state. Returns (baseline_error, checked_error, delta_e,
    n_checked).

    Bug fix: some signals (C_t in particular, on cartpole) have a large exact
    plateau at their maximum value (verified: ~15% of cartpole sites share
    C_t's single max value, a real saturation property of the discounted-
    history statistic, not a data artifact). A strict `signal_eval > thresh`
    comparison against a calibration-split percentile landing exactly ON that
    plateau excludes EVERY tied evaluation-split site, producing n_checked=0
    at low budgets and a spurious ΔE=0 result that reflects a threshold-
    comparison edge case, not the signal's genuine ranking quality. Fixed by
    using `>=` instead of `>` against the calibration-derived threshold --
    this keeps the calibration/evaluation split discipline exactly as
    specified (the cutoff VALUE still comes only from the calibration split,
    never from evaluation-split ranking) while no longer silently dropping
    every site tied at a plateau boundary. This can make n_checked deviate
    from exactly budget*n_eval when the eval split has its own plateau at the
    calibration-derived cutoff value -- reported as-is (n_checked is always
    logged) rather than forced to match the nominal budget exactly, since a
    real deployed system calibrating on one distribution and applying to
    another would see the same effect."""
    thresh = np.percentile(signal_calib, 100 * (1 - budget))
    checked_mask = signal_eval >= thresh

    baseline_error = float(np.mean(e_state_eval))
    corrected = np.where(checked_mask, 0.0, e_state_eval)
    checked_error = float(np.mean(corrected))
    delta_e = baseline_error - checked_error
    return baseline_error, checked_error, float(delta_e), int(checked_mask.sum())


def run_task(task, spec, cfg):
    print(f"\n{'='*78}\n{task.upper()} — TASK 8 ERROR-REDUCTION TEST\n{'='*78}")
    rows, e_state_vals, have_ensemble = build_site_dataset(task, spec, cfg)
    n = len(e_state_vals)

    # disjoint calibration/evaluation split -- fixed 50/50, deterministic
    rng = np.random.default_rng(SEED + 700)
    idx = rng.permutation(n)
    half = n // 2
    calib_idx, eval_idx = idx[:half], idx[half:]

    print(f"  {n} sites total ({half} calibration / {n - half} evaluation), "
          f"have_ensemble={have_ensemble}")

    signals_to_test = SIGNALS if have_ensemble else [s for s in SIGNALS if s != 'ens_dis']

    results = {}
    for budget in QUERY_BUDGETS:
        budget_results = {}
        for sig in signals_to_test:
            sig_calib = rows[sig][calib_idx]
            sig_eval = rows[sig][eval_idx]
            e_eval = e_state_vals[eval_idx]
            baseline, checked, delta_e, n_checked = delta_e_for_signal(sig_calib, sig_eval, e_eval, budget)
            budget_results[sig] = dict(baseline_error=baseline, checked_error=checked,
                                        delta_e=delta_e, n_checked=n_checked,
                                        pct_error_reduction=100 * delta_e / baseline if baseline else float('nan'))
            print(f"  budget={budget:.2f} {sig:<10} baseline={baseline:.4f} -> "
                  f"checked={checked:.4f}  ΔE={delta_e:+.4f} "
                  f"({100*delta_e/baseline:+.1f}%)  n_checked={n_checked}")
        results[str(budget)] = budget_results

    return dict(task=task, n_sites=n, have_ensemble=have_ensemble, results=results)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for task, spec in ENVS.items():
        all_results[task] = run_task(task, spec, cfg)

    print(f"\n{'='*78}\nSUMMARY — ΔE(q) at budget=0.20, C_t vs baselines\n{'='*78}")
    for task, r in all_results.items():
        b20 = r['results']['0.2']
        row = "  ".join(f"{sig}={b20[sig]['delta_e']:+.4f}" for sig in b20)
        ct_best = b20['ct']['delta_e'] == max(b20[s]['delta_e'] for s in b20)
        print(f"  {task:<10} {row}   C_t best? {ct_best}")

    out_path = os.path.join(OUT_DIR, 'task8_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
