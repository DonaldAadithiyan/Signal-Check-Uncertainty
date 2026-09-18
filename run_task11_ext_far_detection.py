#!/usr/bin/env python3.11
"""
Task 11 Extension -- Fixed-False-Alarm-Rate Failure Detection vs. Random
Directions. Not "Task 12": same underlying experiment as Task 11 Sub-B
(onset), sharper framing. Additive, does not reopen the four locked pillars.

Reuses Task 11's exact setup (run_task11_onset_persistence.build_dataset,
same calibration/evaluation split discipline, same onset_within_H target)
entirely -- no new rollouts, no new labels.

WHY: every causal claim in this project (Phase 6, Phase 3's atoms) is
defended against a 50-random-direction empirical null. Task 11's predictive
claim never was -- it beat KL/Recon/EMA, but never "does h_t simply contain
enough information that almost any linear readout of it would look
predictive." This closes that gap with a matching defense.

CANONICAL NULL DIRECTIONS: no prior script in this project ever saved its 50
random directions to disk -- each phase (Phase 3, Phase 6, Task 9) generates
a fresh set from a different ad-hoc seed at call time, so there was no
literal existing artifact to reuse despite the project's general convention
of a 50-direction empirical null. This extension generates ONE canonical set
now (seed=2024+12000=14024, never used elsewhere in this project's seed
namespace), unit-norm in R^256 (h_t's dimensionality throughout the project),
and saves it to outputs/canonical_null/random_directions_50_dim256.npz --
the first such saved artifact, so future work can genuinely reuse it rather
than regenerating yet another ad-hoc set.

FIXED-FAR CALIBRATION (core methodological requirement): for every signal,
independently, threshold is calibrated ONLY on calibration-split sites where
onset_within_H is False (the "calm" population), to produce a target false-
alarm rate (5% headline, 1%/10% as secondary checks) -- fixed BEFORE any
evaluation-split number is computed. Never re-tuned using evaluation
outcomes.
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import XS_CONFIG
from run_task11_onset_persistence import (
    TASKS, CALIB_SEED, EVAL_SEED, N_TRAJ_CALIB, N_TRAJ_EVAL,
    build_dataset, compute_tau, targets_from_rows,
    fit_ct_residual_model, apply_ct_residual, check_balance, assert_balance_ok,
)

OUT_DIR = 'outputs/task11_ext_far_detection'
CANONICAL_DIR_PATH = 'outputs/canonical_null/random_directions_50_dim256.npz'
FAR_HEADLINE = 0.05
FAR_SECONDARY = [0.01, 0.10]
N_NULL = 50
N_BOOT = 1000


def load_canonical_directions():
    d = dict(np.load(CANONICAL_DIR_PATH))
    directions = d['directions']
    assert directions.shape == (N_NULL, 256), directions.shape
    return directions


def onset_step_from_rows(rows, tau):
    """Returns onset_within_H (bool) and onset_step (int, np.nan if no onset
    within horizon) -- the number of steps into the horizon the failure
    begins, used as the lead-time basis: a detector firing at the current
    step gets onset_step steps of advance warning among true positives."""
    onset_within_H, onset_step = [], []
    for r in rows:
        above = r['state_dist'] > tau
        fires = bool(above.any())
        onset_within_H.append(fires)
        onset_step.append(int(np.argmax(above)) if fires else np.nan)
    return np.array(onset_within_H, dtype=bool), np.array(onset_step, dtype=np.float64)


def calibrate_threshold_for_far(signal_calib, onset_calib, far):
    """Threshold calibrated on CALM calibration-split sites only (onset_calib
    == False), producing exactly `far` false alarms among them: the
    (1-far)-th percentile of the calm population's signal distribution.
    Never touches evaluation-split data."""
    calm_vals = signal_calib[~onset_calib]
    return float(np.percentile(calm_vals, 100 * (1 - far)))


def compute_metrics_at_threshold(signal_eval, onset_eval, onset_step_eval, threshold):
    fires = signal_eval >= threshold
    tp = fires & onset_eval
    fp = fires & ~onset_eval
    fn = ~fires & onset_eval
    n_pos = onset_eval.sum()
    n_neg = (~onset_eval).sum()

    tpr = float(tp.sum() / n_pos) if n_pos > 0 else float('nan')
    precision = float(tp.sum() / fires.sum()) if fires.sum() > 0 else float('nan')
    lead_times = onset_step_eval[tp]
    lead_time_mean = float(np.nanmean(lead_times)) if tp.sum() > 0 else float('nan')

    return dict(tpr=tpr, precision=precision, lead_time_mean=lead_time_mean,
                n_true_positive=int(tp.sum()), n_positive=int(n_pos), n_negative=int(n_neg))


def compute_threshold_free_metrics(signal_eval, onset_eval):
    if len(np.unique(onset_eval)) < 2:
        return dict(auroc=float('nan'), auprc=float('nan'))
    return dict(auroc=float(roc_auc_score(onset_eval, signal_eval)),
                auprc=float(average_precision_score(onset_eval, signal_eval)))


def evaluate_signal(signal_calib, signal_eval, onset_calib, onset_eval, onset_step_eval, far_list):
    out = dict(threshold_free=compute_threshold_free_metrics(signal_eval, onset_eval), fixed_far={})
    for far in far_list:
        thresh = calibrate_threshold_for_far(signal_calib, onset_calib, far)
        m = compute_metrics_at_threshold(signal_eval, onset_eval, onset_step_eval, thresh)
        m['threshold'] = thresh
        out['fixed_far'][far] = m
    return out


def z_and_percentile(target, null_vals):
    null_vals = np.asarray(null_vals, dtype=np.float64)
    null_vals = null_vals[np.isfinite(null_vals)]
    if len(null_vals) < 5 or not np.isfinite(target):
        return dict(null_mean=float('nan'), null_std=float('nan'), z=float('nan'), percentile=float('nan'))
    z = (target - null_vals.mean()) / (null_vals.std() + 1e-12)
    pct = float((null_vals < target).mean() * 100)
    return dict(null_mean=float(null_vals.mean()), null_std=float(null_vals.std()),
                z=float(z), percentile=pct)


def run_task(task, cfg, directions):
    print(f"\n{'='*78}\n{task.upper()} — TASK 11 EXTENSION: FIXED-FAR DETECTION\n{'='*78}")
    calib_rows, _ = build_dataset(task, cfg, CALIB_SEED, N_TRAJ_CALIB)
    eval_rows, _ = build_dataset(task, cfg, EVAL_SEED, N_TRAJ_EVAL)
    print(f"  calibration sites: {len(calib_rows)}  evaluation sites: {len(eval_rows)}")

    tau = compute_tau(calib_rows)
    onset_calib, _ = onset_step_from_rows(calib_rows, tau)
    onset_eval, onset_step_eval = onset_step_from_rows(eval_rows, tau)
    print(f"  onset rate: calib={onset_calib.mean():.3f}  eval={onset_eval.mean():.3f}")

    kl_c = np.array([r['kl'] for r in calib_rows]); recon_c = np.array([r['recon'] for r in calib_rows])
    ema_c = np.array([r['ema_recon'] for r in calib_rows]); ct_c = np.array([r['ct'] for r in calib_rows])
    kl_e = np.array([r['kl'] for r in eval_rows]); recon_e = np.array([r['recon'] for r in eval_rows])
    ema_e = np.array([r['ema_recon'] for r in eval_rows]); ct_e = np.array([r['ct'] for r in eval_rows])
    h_c = np.stack([r['h'] for r in calib_rows]); h_e = np.stack([r['h'] for r in eval_rows])

    baseline_calib = np.column_stack([kl_c, recon_c, ema_c])
    baseline_eval = np.column_stack([kl_e, recon_e, ema_e])

    # incremental residual of C_t beyond {KL, Recon, EMA} -- reuse Task 11's
    # exact residualization machinery (calib-fit, eval-applied, balance-
    # gated) so the incremental comparison isn't circular.
    scaler, reg = fit_ct_residual_model(calib_rows)
    ct_matched_calib = apply_ct_residual(calib_rows, scaler, reg)
    ct_matched_eval = apply_ct_residual(eval_rows, scaler, reg)
    balance = check_balance(ct_matched_eval, kl_e, recon_e)
    print(f"  balance check (should be ~0): corr(C_t_matched, KL)={balance['corr_with_kl']:+.4f}  "
          f"corr(C_t_matched, Recon)={balance['corr_with_recon']:+.4f}")
    assert_balance_ok(balance, f'{task} (Task 11 ext incremental C_t)')

    far_list = [FAR_HEADLINE] + FAR_SECONDARY

    # ── Standalone comparison (A) ──
    print("  --- Standalone (A) ---")
    standalone = {}
    standalone['C_t'] = evaluate_signal(ct_c, ct_e, onset_calib, onset_eval, onset_step_eval, far_list)
    standalone['KL'] = evaluate_signal(kl_c, kl_e, onset_calib, onset_eval, onset_step_eval, far_list)
    standalone['Recon'] = evaluate_signal(recon_c, recon_e, onset_calib, onset_eval, onset_step_eval, far_list)
    standalone['EMA'] = evaluate_signal(ema_c, ema_e, onset_calib, onset_eval, onset_step_eval, far_list)

    null_standalone = []
    for i in range(N_NULL):
        proj_c = h_c @ directions[i]
        proj_e = h_e @ directions[i]
        null_standalone.append(evaluate_signal(proj_c, proj_e, onset_calib, onset_eval, onset_step_eval, far_list))
        if (i + 1) % 25 == 0:
            print(f"    standalone null {i+1}/{N_NULL} done", flush=True)

    # ── Incremental comparison (B): signal ADDED to {KL, Recon, EMA} ──
    print("  --- Incremental (B) ---")
    # For C_t, the "incremental" contribution IS ct_matched (already
    # orthogonalized against KL/Recon by construction -- EMA is not part of
    # the residualization model but is a near-deterministic function of past
    # Recon, so this matches Task 11's own established incremental
    # convention exactly, unmodified here).
    incr_ct = evaluate_signal(ct_matched_calib, ct_matched_eval, onset_calib, onset_eval, onset_step_eval, far_list)

    # For each random direction, its "incremental" contribution is its own
    # residual beyond {KL, Recon, EMA} -- fit the SAME way (linear
    # regression on calib, applied to eval), so random directions and C_t
    # are held to an identical incremental-construction standard.
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    def residualize_against_baseline(signal_calib, signal_eval):
        scaler_b = StandardScaler().fit(baseline_calib)
        reg_b = LinearRegression().fit(scaler_b.transform(baseline_calib), signal_calib)
        pred_eval = reg_b.predict(scaler_b.transform(baseline_eval))
        pred_calib = reg_b.predict(scaler_b.transform(baseline_calib))
        return signal_calib - pred_calib, signal_eval - pred_eval

    null_incremental = []
    for i in range(N_NULL):
        proj_c = h_c @ directions[i]
        proj_e = h_e @ directions[i]
        resid_c, resid_e = residualize_against_baseline(proj_c, proj_e)
        null_incremental.append(evaluate_signal(resid_c, resid_e, onset_calib, onset_eval, onset_step_eval, far_list))
        if (i + 1) % 25 == 0:
            print(f"    incremental null {i+1}/{N_NULL} done", flush=True)

    # ── assemble null-comparison statistics for C_t, both standalone and incremental ──
    def build_comparison(target_result, null_results, label):
        comp = {}
        auroc_null = [n['threshold_free']['auroc'] for n in null_results]
        auprc_null = [n['threshold_free']['auprc'] for n in null_results]
        comp['auroc'] = dict(value=target_result['threshold_free']['auroc'],
                              **z_and_percentile(target_result['threshold_free']['auroc'], auroc_null))
        comp['auprc'] = dict(value=target_result['threshold_free']['auprc'],
                              **z_and_percentile(target_result['threshold_free']['auprc'], auprc_null))
        for far in far_list:
            for metric in ['tpr', 'precision', 'lead_time_mean']:
                null_vals = [n['fixed_far'][far][metric] for n in null_results]
                key = f'far{far}_{metric}'
                comp[key] = dict(value=target_result['fixed_far'][far][metric],
                                  **z_and_percentile(target_result['fixed_far'][far][metric], null_vals))
        print(f"  [{label}] AUROC={comp['auroc']['value']:.4f} (null={comp['auroc']['null_mean']:.4f}"
              f"±{comp['auroc']['null_std']:.4f}, z={comp['auroc']['z']:+.2f})  "
              f"AUPRC={comp['auprc']['value']:.4f} (z={comp['auprc']['z']:+.2f})")
        for far in far_list:
            tpr_c = comp[f'far{far}_tpr']
            print(f"    FAR={far}: TPR={tpr_c['value']:.4f} (null={tpr_c['null_mean']:.4f}±{tpr_c['null_std']:.4f}, z={tpr_c['z']:+.2f})")
        return comp

    comparison_standalone = build_comparison(standalone['C_t'], null_standalone, 'STANDALONE C_t vs random')
    comparison_incremental = build_comparison(incr_ct, null_incremental, 'INCREMENTAL C_t vs random')

    return dict(
        task=task, tau=tau, balance=balance,
        onset_rate_calib=float(onset_calib.mean()), onset_rate_eval=float(onset_eval.mean()),
        standalone_baselines={k: v for k, v in standalone.items() if k != 'C_t'},
        standalone_ct=standalone['C_t'],
        incremental_ct=incr_ct,
        comparison_standalone=comparison_standalone,
        comparison_incremental=comparison_incremental,
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    directions = load_canonical_directions()
    print(f"Loaded {len(directions)} canonical null directions from {CANONICAL_DIR_PATH}")

    all_results = {}
    for task in TASKS:
        all_results[task] = run_task(task, cfg, directions)
        with open(os.path.join(OUT_DIR, f'task11_ext_{task}_results.json'), 'w') as f:
            json.dump(all_results[task], f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, 'task11_ext_all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*78}\nSHIP-RULE SUMMARY\n{'='*78}")
    for task in TASKS:
        r = all_results[task]
        cs, ci = r['comparison_standalone'], r['comparison_incremental']
        ship_standalone = (cs['auroc']['z'] > 2 and cs['auprc']['z'] > 2 and
                            any(cs[f'far{FAR_HEADLINE}_{m}']['z'] > 2 for m in ['tpr', 'precision', 'lead_time_mean']))
        ship_incremental = (ci['auroc']['z'] > 2 and ci['auprc']['z'] > 2 and
                             any(ci[f'far{FAR_HEADLINE}_{m}']['z'] > 2 for m in ['tpr', 'precision', 'lead_time_mean']))
        print(f"  {task}: standalone {'SHIP' if ship_standalone else 'NO-SHIP'}  "
              f"incremental {'SHIP' if ship_incremental else 'NO-SHIP'}")

    print(f"\nWrote results to {OUT_DIR}/")


if __name__ == '__main__':
    main()
