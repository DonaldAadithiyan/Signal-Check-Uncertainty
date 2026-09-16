#!/usr/bin/env python3.11
"""
Task 3 — train ONE seed's full+partial pair. Thin CLI wrapper around
run_phase4_seed_sweep_training's train_condition_to_dir, so each seed pair can
be launched as an independent background process (parallelized across seeds,
per the gap-closing spec's "parallelized across independent CPU instances" --
here, independent OS processes on one machine's cores instead of separate
cloud VMs, since 4 pairs x ~87min x 2 conditions fits comfortably on 10 cores
run 2-at-a-time).

Usage: python3.11 run_phase4_seed_sweep_one.py <seed>
"""

import os
import sys
import time

from src.config import XS_CONFIG
from run_phase4_seed_sweep_training import train_condition_to_dir, OUT_DIR


def main():
    seed = int(sys.argv[1])
    cfg = XS_CONFIG.copy()
    seed_dir = os.path.join(OUT_DIR, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)

    for condition in ['full', 'partial']:
        ckpt_path = os.path.join(seed_dir, f'{condition}_world_model.pt')
        states_path = os.path.join(seed_dir, f'{condition}_training_states.npz')
        if os.path.exists(ckpt_path) and os.path.exists(states_path):
            print(f"[seed_sweep seed={seed}] {condition}: already exists, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"[seed_sweep seed={seed}] === condition={condition} start ===", flush=True)
        train_condition_to_dir(cfg, condition, seed, seed_dir)
        print(f"[seed_sweep seed={seed}] {condition} done in {(time.time()-t0)/60:.1f}m", flush=True)

    print(f"[seed_sweep seed={seed}] BOTH CONDITIONS COMPLETE", flush=True)


if __name__ == '__main__':
    main()
