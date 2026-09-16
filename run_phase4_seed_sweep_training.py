#!/usr/bin/env python3.11
"""
Task 3 (gap-closing spec) — Phase 4 seed sweep, training half.

Trains 4 ADDITIONAL full/partial seed pairs beyond the original seed=4242 pair
(run_phase4_pomdp_training.py), using new seeds. Identical protocol to the
original training script in every respect except the seed -- same architecture,
optimizer, training budget (100K env steps), masking scheme (velocity dims
[3,4] zeroed for PARTIAL, decoder always reconstructs the full 5-dim state).

This directly reuses train_condition() from the original script rather than
reimplementing it, so there is exactly one training code path for both the
original pair and the new pairs -- no risk of the sweep silently drifting from
the original protocol.

New seeds: 1717, 2929, 5151, 8383 (arbitrary, fixed here for reproducibility,
chosen independently of the original 4242 and of each other by construction --
no cherry-picking, first 4 seeds decided before any run).

Checkpoints/states land in outputs/phase4_pomdp/seed_sweep/{seed}_{condition}_*
so they never collide with or overwrite the original seed=4242 pair.
"""

import os
import sys
import time

from src.config import XS_CONFIG
from run_phase4_pomdp_training import train_condition

SWEEP_SEEDS = [1717, 2929, 5151, 8383]
OUT_DIR = 'outputs/phase4_pomdp/seed_sweep'


def train_condition_to_dir(cfg, condition, seed, out_dir):
    """Wraps train_condition but redirects its OUT_DIR so sweep runs don't
    collide with the original single-pair outputs. train_condition hardcodes
    module-level OUT_DIR from run_phase4_pomdp_training, so we monkeypatch it
    for the duration of the call -- simplest way to reuse the exact function
    body without forking it."""
    import run_phase4_pomdp_training as t
    orig_out_dir = t.OUT_DIR
    t.OUT_DIR = out_dir
    try:
        os.makedirs(out_dir, exist_ok=True)
        return train_condition(cfg, condition, seed=seed)
    finally:
        t.OUT_DIR = orig_out_dir


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    for seed in SWEEP_SEEDS:
        seed_dir = os.path.join(OUT_DIR, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        for condition in ['full', 'partial']:
            ckpt_path = os.path.join(seed_dir, f'{condition}_world_model.pt')
            states_path = os.path.join(seed_dir, f'{condition}_training_states.npz')
            if os.path.exists(ckpt_path) and os.path.exists(states_path):
                print(f"[seed_sweep] seed={seed} {condition}: already exists, skipping")
                continue
            t0 = time.time()
            print(f"\n[seed_sweep] === seed={seed} condition={condition} ===", flush=True)
            train_condition_to_dir(cfg, condition, seed, seed_dir)
            print(f"[seed_sweep] seed={seed} {condition} done in {(time.time()-t0)/60:.1f}m",
                  flush=True)

    print("\n[seed_sweep] all seed pairs trained. Run run_phase4_seed_sweep_analysis.py next.")


if __name__ == '__main__':
    main()
