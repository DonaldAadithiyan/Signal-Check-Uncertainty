#!/usr/bin/env python3.11
"""
Task 11 -- Failure Onset and Persistence Prediction (final use-case attempt).

Additive-only, same standing as Tasks 9/10: does not touch the four locked
paper pillars. THIS IS THE LAST USE-CASE ATTEMPT for C_t -- if both
sub-experiments below fail their ship rules, the honest conclusion ("C_t is
an emergent, mechanistically real internal representation, but not yet a
demonstrably superior OPERATIONAL signal") is the final word on utility, and
no Task 12 is proposed regardless of what new ideas surface afterward. This
is stated up front, not just at the end, per the task spec's own instruction.

Infrastructure check (outputs/deliverables/task_11_infrastructure_check.md):
CONFIRMED cheap -- reuses run_phase1_external_validation's
imagined_vs_real_obs() per-step state_dist array entirely; only the target
variable (onset/persistence, derived from that same array against a
threshold) is new. No new rollout or intervention machinery.

DESIGN
------
For every eligible site (ti, t) in a normal (unshifted, on-policy) rollout:
  state_dist = imagined_vs_real_obs(...)[0]   # length up to MAX_HORIZON=20
  above = state_dist > TAU_GOOD[task]          # per-task recovery threshold

Sub-experiment A (persistence): P_tH = mean(above) -- fraction of the horizon
spent in "bad" (E^state above threshold) territory. Continuous in [0,1],
avoids L_t's censoring problem (L_t undefined when state_dist never recovers
within the horizon).

Sub-experiment B (onset): onset_within_H = above.any() -- binary, did a
failure (E^state > tau) begin at all within the horizon, whether or not
current error is already elevated. To specifically test ONSET (not just
"this site is already bad"), the eligible site pool for B is restricted to
sites where the CURRENT step is already good (state_dist_full computed from
t, so the "current" error proxy is KL_t/Recon_t at t, both required to be
below their own medians for a site to enter B's pool) -- otherwise onset
prediction collapses into persistence prediction for currently-bad sites.

MATCHED-CURRENT-ERROR COMPARISON (the critical step, per the task spec)
------------------------------------------------------------------------
Rather than binary group matching (Task 9's approach, appropriate for a
single sparse atom's activation), this uses REGRESSION RESIDUALIZATION,
appropriate for testing incremental contribution of a continuous C_t signal
in a regression already used throughout this project (identical construction
to Phase 3 Sec 9.7's incremental-R^2):
  1. On a CALIBRATION split (seed disjoint from evaluation), fit
     C_t ~ KL_t + Recon_t (linear regression).
  2. On the EVALUATION split, compute C_t_matched = C_t - predicted(KL_t, Recon_t)
     using the calibration-fit model (never refit on evaluation data).
  3. VERIFY BALANCE explicitly before any downstream comparison: report
     corr(C_t_matched, KL_t) and corr(C_t_matched, Recon_t) on the evaluation
     split -- both should be ~0 (that's what residualization guarantees
     in-sample on calibration data; verifying it holds out-of-sample on
     evaluation data is the actual check, since the model was fit elsewhere).
  4. Incremental R^2 (persistence) / incremental pseudo-R^2 via logistic
     deviance (onset): target ~ {KL_t, Recon_t, EMARecon_t} (baseline) vs.
     target ~ {KL_t, Recon_t, EMARecon_t, C_t_matched} (full) -- since
     C_t_matched is already orthogonalized against KL_t/Recon_t by
     construction, this incremental term is a direct, non-circular test of
     whether the historical-accumulation part of C_t (the part KL/Recon/EMA
     cannot express) carries any predictive signal at all.

Ship rule (independent per sub-experiment, per the task spec):
  C_t_matched's incremental contribution is statistically significant
  (bootstrap CI on incremental R^2 / incremental pseudo-R^2 excludes 0),
  reported PER TASK, not averaged.
"""

import os
import json
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from src.config import XS_CONFIG
from run_phase1_external_validation import (
    ENVS as P1_ENVS, load_model, collect_trajectories, imagined_vs_real_obs,
    fit_ema_alpha, ema_series, MIN_T, MAX_HORIZON, SEED,
)
from src.probe.intervention import compute_ct, bootstrap_ci

OUT_DIR = 'outputs/task11_onset_persistence'
CALIB_SEED = SEED + 11_000    # calibration split, disjoint from every other seed range in the project
EVAL_SEED = SEED + 11_500     # evaluation split, disjoint from calibration and from CALIB_SEED
N_TRAJ_CALIB = 60
N_TRAJ_EVAL = 60
N_BOOT = 1000

# tau_good: per-task recovery threshold, set as the 75th percentile of E^state
# (mean over horizon 1) on NORMAL (unshifted) calibration trajectories -- i.e.
# "good" means "typical," "bad" (above tau) means "notably worse than typical
# for this task," decided on calibration data only, never adjusted after
# seeing evaluation results.
TAU_PERCENTILE = 75

TASKS = {
    'cartpole': dict(gamma_ct=0.95),
    'reacher':  dict(gamma_ct=0.70),
    'pendulum': dict(gamma_ct=0.90),
}


def build_dataset(task, cfg, seed, n_traj, spec_override=None):
    """spec_override: optional dict overriding P1_ENVS[task]'s checkpoint/
    training_states paths (used by the Task 11 closing spec's multi-seed
    spot-check to reuse an independently-trained checkpoint without touching
    this function's default, already-verified behavior when unset)."""
    spec = spec_override if spec_override is not None else P1_ENVS[task]
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])
    kl_median = float(np.median(tr['kl']))
    recon_median = float(np.median(tr['recon']))

    torch.manual_seed(seed)
    trajs = collect_trajectories(model, spec, n_traj, cfg, seed=seed)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64), gamma=TASKS[task]['gamma_ct'])
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - MAX_HORIZON - 1):
            sites.append((ti, t))

    rows = []
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist, _, _ = imagined_vs_real_obs(model, trj, t, MAX_HORIZON, spec['domain'])
        if len(state_dist) < MAX_HORIZON:
            continue
        rows.append(dict(
            ti=ti, t=t, state_dist=state_dist,
            kl=float(trj['kl'][t]), recon=float(trj['recon'][t]),
            ema_recon=float(trj['ema_recon'][t]), ct=float(trj['ct'][t]),
            kl_is_low=bool(trj['kl'][t] <= kl_median),
            recon_is_low=bool(trj['recon'][t] <= recon_median),
        ))
    return rows, model


def compute_tau(calib_rows):
    e_state_h1 = np.array([r['state_dist'][0] for r in calib_rows])
    return float(np.percentile(e_state_h1, TAU_PERCENTILE))


def targets_from_rows(rows, tau):
    P_tH, onset_within_H, onset_eligible = [], [], []
    for r in rows:
        above = r['state_dist'] > tau
        P_tH.append(float(above.mean()))
        onset_within_H.append(bool(above.any()))
        onset_eligible.append(bool(r['kl_is_low'] and r['recon_is_low']))
    return (np.array(P_tH), np.array(onset_within_H, dtype=bool), np.array(onset_eligible, dtype=bool))


def fit_ct_residual_model(calib_rows):
    X = np.column_stack([[r['kl'] for r in calib_rows], [r['recon'] for r in calib_rows]])
    y = np.array([r['ct'] for r in calib_rows])
    scaler = StandardScaler().fit(X)
    reg = LinearRegression().fit(scaler.transform(X), y)
    return scaler, reg


def apply_ct_residual(rows, scaler, reg):
    X = np.column_stack([[r['kl'] for r in rows], [r['recon'] for r in rows]])
    pred = reg.predict(scaler.transform(X))
    ct = np.array([r['ct'] for r in rows])
    return ct - pred


def check_balance(ct_matched, kl, recon):
    r_kl = float(np.corrcoef(ct_matched, kl)[0, 1])
    r_recon = float(np.corrcoef(ct_matched, recon)[0, 1])
    return dict(corr_with_kl=r_kl, corr_with_recon=r_recon)


def incremental_r2_persistence(target, kl, recon, ema, ct_matched, n_boot=N_BOOT, seed=0):
    def r2_diff(idx):
        t, k, rc, e, c = target[idx], kl[idx], recon[idx], ema[idx], ct_matched[idx]
        if np.std(t) == 0:
            return np.nan
        X_base = np.column_stack([k, rc, e])
        X_full = np.column_stack([k, rc, e, c])
        Xs_base = StandardScaler().fit_transform(X_base)
        Xs_full = StandardScaler().fit_transform(X_full)
        r2_base = LinearRegression().fit(Xs_base, t).score(Xs_base, t)
        r2_full = LinearRegression().fit(Xs_full, t).score(Xs_full, t)
        return r2_full - r2_base

    n = len(target)
    point = r2_diff(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = r2_diff(idx)
        if np.isfinite(v):
            boots.append(v)
    if len(boots) < 20:
        return dict(point=float(point), lo=float('nan'), hi=float('nan'), n_boot_valid=len(boots))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi), n_boot_valid=len(boots))


def incremental_pseudo_r2_onset(target, kl, recon, ema, ct_matched, n_boot=N_BOOT, seed=0):
    """McFadden pseudo-R^2 (1 - loglik_full/loglik_null-style, here using the
    baseline-vs-full deviance the same way incremental R^2 is used for the
    continuous target) via logistic regression deviance."""
    def pseudo_r2_diff(idx):
        t, k, rc, e, c = target[idx], kl[idx], recon[idx], ema[idx], ct_matched[idx]
        if len(np.unique(t)) < 2:
            return np.nan
        X_base = np.column_stack([k, rc, e])
        X_full = np.column_stack([k, rc, e, c])
        Xs_base = StandardScaler().fit_transform(X_base)
        Xs_full = StandardScaler().fit_transform(X_full)
        try:
            clf_base = LogisticRegression(max_iter=1000).fit(Xs_base, t)
            clf_full = LogisticRegression(max_iter=1000).fit(Xs_full, t)
        except ValueError:
            return np.nan
        ll_base = -log_loss(t, clf_base.predict_proba(Xs_base)[:, 1], labels=[False, True]) * len(t)
        ll_full = -log_loss(t, clf_full.predict_proba(Xs_full)[:, 1], labels=[False, True]) * len(t)
        ll_null = -log_loss(t, np.full(len(t), t.mean()), labels=[False, True]) * len(t)
        if ll_null == 0:
            return np.nan
        pr2_base = 1 - ll_base / ll_null
        pr2_full = 1 - ll_full / ll_null
        return pr2_full - pr2_base

    n = len(target)
    point = pseudo_r2_diff(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = pseudo_r2_diff(idx)
        if np.isfinite(v):
            boots.append(v)
    if len(boots) < 20:
        return dict(point=float(point), lo=float('nan'), hi=float('nan'), n_boot_valid=len(boots))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi), n_boot_valid=len(boots))


def run_task(task, cfg):
    print(f"\n{'='*78}\n{task.upper()} — TASK 11 ONSET/PERSISTENCE\n{'='*78}")
    calib_rows, model = build_dataset(task, cfg, CALIB_SEED, N_TRAJ_CALIB)
    eval_rows, _ = build_dataset(task, cfg, EVAL_SEED, N_TRAJ_EVAL)
    print(f"  calibration sites: {len(calib_rows)}  evaluation sites: {len(eval_rows)}")

    tau = compute_tau(calib_rows)
    print(f"  tau_good ({TAU_PERCENTILE}th pctile of calib E^state_h1) = {tau:.4f}")

    scaler, reg = fit_ct_residual_model(calib_rows)
    ct_matched_eval = apply_ct_residual(eval_rows, scaler, reg)
    kl_eval = np.array([r['kl'] for r in eval_rows])
    recon_eval = np.array([r['recon'] for r in eval_rows])
    ema_eval = np.array([r['ema_recon'] for r in eval_rows])

    balance = check_balance(ct_matched_eval, kl_eval, recon_eval)
    print(f"  balance check (should be ~0): corr(C_t_matched, KL)={balance['corr_with_kl']:+.4f}  "
          f"corr(C_t_matched, Recon)={balance['corr_with_recon']:+.4f}")

    P_tH, onset_within_H, onset_eligible = targets_from_rows(eval_rows, tau)

    # ── Sub-experiment A: persistence ──
    incr_A = incremental_r2_persistence(P_tH, kl_eval, recon_eval, ema_eval, ct_matched_eval, seed=CALIB_SEED)
    ship_A = incr_A['point'] > 0 and incr_A['lo'] is not None and np.isfinite(incr_A['lo']) and incr_A['lo'] > 0
    print(f"  Sub-A (persistence P_tH) incremental R^2: {incr_A['point']:+.4f} "
          f"[{incr_A['lo']:+.4f}, {incr_A['hi']:+.4f}]  -> {'SHIP' if ship_A else 'NO-SHIP'}")

    # ── Sub-experiment B: onset (restricted to currently-good sites) ──
    elig_idx = np.where(onset_eligible)[0]
    n_elig = len(elig_idx)
    print(f"  Sub-B eligible sites (currently KL&Recon below median): {n_elig} / {len(eval_rows)}")
    if n_elig >= 50 and len(np.unique(onset_within_H[elig_idx])) >= 2:
        incr_B = incremental_pseudo_r2_onset(
            onset_within_H[elig_idx], kl_eval[elig_idx], recon_eval[elig_idx],
            ema_eval[elig_idx], ct_matched_eval[elig_idx], seed=CALIB_SEED + 1)
        ship_B = incr_B['point'] > 0 and incr_B['lo'] is not None and np.isfinite(incr_B['lo']) and incr_B['lo'] > 0
        print(f"  Sub-B (onset within horizon) incremental pseudo-R^2: {incr_B['point']:+.4f} "
              f"[{incr_B['lo']:+.4f}, {incr_B['hi']:+.4f}]  -> {'SHIP' if ship_B else 'NO-SHIP'}")
    else:
        incr_B = dict(point=None, lo=None, hi=None, n_boot_valid=0)
        ship_B = False
        print(f"  Sub-B: INSUFFICIENT DATA (n_eligible={n_elig}, or onset outcome not variable) -> NO-SHIP (underpowered)")

    return dict(
        task=task, tau_good=tau, balance=balance,
        n_calib=len(calib_rows), n_eval=len(eval_rows), n_eligible_onset=n_elig,
        sub_A_persistence=dict(incremental_r2=incr_A, ship=ship_A),
        sub_B_onset=dict(incremental_pseudo_r2=incr_B, ship=ship_B),
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for task in TASKS:
        all_results[task] = run_task(task, cfg)
        with open(os.path.join(OUT_DIR, f'task11_{task}_results.json'), 'w') as f:
            json.dump(all_results[task], f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, 'task11_all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*78}\nSHIP-RULE SUMMARY (per task, per sub-experiment)\n{'='*78}")
    for task in TASKS:
        r = all_results[task]
        print(f"  {task}: Sub-A={'SHIP' if r['sub_A_persistence']['ship'] else 'NO-SHIP'}  "
              f"Sub-B={'SHIP' if r['sub_B_onset']['ship'] else 'NO-SHIP'}")
    print(f"\nWrote results to {OUT_DIR}/")


if __name__ == '__main__':
    main()
