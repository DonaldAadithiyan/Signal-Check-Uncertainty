#!/usr/bin/env python3.11
"""
Task 11 Closing Spec -- items 1 and 3 (item 2 is a pure code-path read,
answered directly, no script needed -- see outputs/deliverables/
task_11_closing_verification.md).

Item 1: confirmatory balance check at a NEW pre-committed N=100 (matching
Phase 1's own N_TRAJ=100 scale, decided now, before running), on the
ORIGINAL cartpole model (P1_ENVS['cartpole']), to confirm the good balance
seen at N=60 was not an artifact of that specific stopping point.

Item 3: multi-seed spot-check of Sub-A (persistence) on cartpole, reusing
outputs/multiseed/seed_1/{world_model.pt,training_states.npz} -- an
independently-trained checkpoint from this project's existing multi-seed
replication set. No new base-model training. Same N=60 protocol as the
original Task 11 run (this item tests seed replication, not N sensitivity --
that's item 1's job), same calibration/evaluation split discipline, same
balance verification before trusting the result.
"""

import os
import json
import numpy as np
import torch

from src.config import XS_CONFIG
from run_task11_onset_persistence import (
    TASKS, CALIB_SEED, EVAL_SEED, build_dataset, compute_tau,
    fit_ct_residual_model, apply_ct_residual, check_balance,
    targets_from_rows, incremental_r2_persistence,
)

OUT_DIR = 'outputs/task11_onset_persistence'
N_CONFIRM = 100   # item 1: pre-committed NEW value, above the N=60 already checked


def item1_confirm_balance_at_n100():
    print(f"\n{'='*78}\nITEM 1 — confirmatory balance check at N={N_CONFIRM} (cartpole, original seed_0 model)\n{'='*78}")
    cfg = XS_CONFIG.copy()
    calib_rows, _ = build_dataset('cartpole', cfg, CALIB_SEED, N_CONFIRM)
    eval_rows, _ = build_dataset('cartpole', cfg, EVAL_SEED, N_CONFIRM)
    print(f"  calibration sites: {len(calib_rows)}  evaluation sites: {len(eval_rows)}")

    scaler, reg = fit_ct_residual_model(calib_rows)
    ct_matched_eval = apply_ct_residual(eval_rows, scaler, reg)
    kl_eval = np.array([r['kl'] for r in eval_rows])
    recon_eval = np.array([r['recon'] for r in eval_rows])
    balance = check_balance(ct_matched_eval, kl_eval, recon_eval)
    print(f"  balance at N={N_CONFIRM}: corr(C_t_matched, KL)={balance['corr_with_kl']:+.4f}  "
          f"corr(C_t_matched, Recon)={balance['corr_with_recon']:+.4f}")

    result = dict(n=N_CONFIRM, n_calib_sites=len(calib_rows), n_eval_sites=len(eval_rows), balance=balance)
    with open(os.path.join(OUT_DIR, 'item1_n100_confirmation.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)
    return result


def item3_multiseed_spotcheck():
    print(f"\n{'='*78}\nITEM 3 — multi-seed spot-check, cartpole persistence, outputs/multiseed/seed_1\n{'='*78}")
    cfg = XS_CONFIG.copy()
    spec_seed1 = dict(TASKS['cartpole'])
    spec_seed1.update(
        env_cls='cartpole', domain='cartpole', task='swingup',
        checkpoint='outputs/multiseed/seed_1/world_model.pt',
        training_states='outputs/multiseed/seed_1/training_states.npz',
    )

    calib_rows, _ = build_dataset('cartpole', cfg, CALIB_SEED, 60, spec_override=spec_seed1)
    eval_rows, _ = build_dataset('cartpole', cfg, EVAL_SEED, 60, spec_override=spec_seed1)
    print(f"  calibration sites: {len(calib_rows)}  evaluation sites: {len(eval_rows)}")

    tau = compute_tau(calib_rows)
    print(f"  tau_good = {tau:.4f}")

    scaler, reg = fit_ct_residual_model(calib_rows)
    ct_matched_eval = apply_ct_residual(eval_rows, scaler, reg)
    kl_eval = np.array([r['kl'] for r in eval_rows])
    recon_eval = np.array([r['recon'] for r in eval_rows])
    ema_eval = np.array([r['ema_recon'] for r in eval_rows])

    balance = check_balance(ct_matched_eval, kl_eval, recon_eval)
    print(f"  balance check (should be ~0): corr(C_t_matched, KL)={balance['corr_with_kl']:+.4f}  "
          f"corr(C_t_matched, Recon)={balance['corr_with_recon']:+.4f}")

    P_tH, _, _ = targets_from_rows(eval_rows, tau)
    incr_A = incremental_r2_persistence(P_tH, kl_eval, recon_eval, ema_eval, ct_matched_eval, seed=CALIB_SEED)
    ship_A = incr_A['point'] > 0 and incr_A['lo'] is not None and np.isfinite(incr_A['lo']) and incr_A['lo'] > 0
    print(f"  Sub-A (persistence P_tH) incremental R^2, seed_1: {incr_A['point']:+.4f} "
          f"[{incr_A['lo']:+.4f}, {incr_A['hi']:+.4f}]  -> {'SHIP' if ship_A else 'NO-SHIP'}")

    result = dict(seed='multiseed/seed_1', n=60, tau=tau, balance=balance,
                  incremental_r2=incr_A, ship=ship_A)
    with open(os.path.join(OUT_DIR, 'item3_seed1_spotcheck.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r1 = item1_confirm_balance_at_n100()
    r3 = item3_multiseed_spotcheck()

    print(f"\n{'='*78}\nCLOSING-CHECK SUMMARY\n{'='*78}")
    print(f"  Item 1 (N={N_CONFIRM} balance): KL={r1['balance']['corr_with_kl']:+.4f}  Recon={r1['balance']['corr_with_recon']:+.4f}")
    print(f"  Item 3 (seed_1 persistence): incr_R2={r3['incremental_r2']['point']:+.4f} "
          f"[{r3['incremental_r2']['lo']:+.4f}, {r3['incremental_r2']['hi']:+.4f}]  -> {'SHIP' if r3['ship'] else 'NO-SHIP'}")


if __name__ == '__main__':
    main()
