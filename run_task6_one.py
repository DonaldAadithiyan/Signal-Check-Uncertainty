#!/usr/bin/env python3.11
"""Task 6 -- train ONE (task, seed) actor-critic pair. Thin CLI wrapper around
run_task6_actor_critic_training's train_task6, so each pair can be launched
as an independent background process, parallelized across cores (matching
the approach used for the Phase 4 seed sweep in the gap-closing spec).

Usage: python3.11 run_task6_one.py <task> <seed_idx>
"""
import os
import sys

from src.config import XS_CONFIG
from run_task6_actor_critic_training import train_task6, TASK_SEEDS, OUT_DIR


def main():
    task = sys.argv[1]
    seed_idx = int(sys.argv[2])
    ckpt_path = TASK_SEEDS[task]['checkpoints'][seed_idx]

    out_path = os.path.join(OUT_DIR, task, f'seed{seed_idx}', 'actor_critic.pt')
    if os.path.exists(out_path):
        print(f"[task6 {task}/seed{seed_idx}] already trained, skipping")
        return
    if not os.path.exists(ckpt_path):
        print(f"[task6 {task}/seed{seed_idx}] checkpoint {ckpt_path} not found")
        return

    cfg = XS_CONFIG.copy()
    seed = 1000 * (seed_idx + 1) + hash(task) % 1000
    train_task6(task, seed_idx, ckpt_path, cfg, seed=seed)


if __name__ == '__main__':
    main()
