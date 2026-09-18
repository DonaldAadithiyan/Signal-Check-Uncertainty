#!/usr/bin/env python3.11
"""
Final closing spec, Item 3 -- Phase 6 seed extension (feeds: Causal
dissociation pillar).

The hardened Phase 6 causal-steering results (§5.1-5.5: the decisive dense-
direction probe-readout effect, z=+7 to +16; the null external-effect
finding, |z|<1.1 everywhere) are single-seed per task. This script extends
run_addendum_phase6_hardening.run_task_hardening -- unmodified, reused as-is,
already RNG-bug-fixed and independently verified reproducible -- to the
project's standard multi-seed counts (5 cartpole, 4 reacher/pendulum),
reusing the EXISTING multiseed base-model checkpoints (outputs/multiseed/
seed_{0..4}, outputs/multiseed_env/{reacher,pendulum}_seed{1,2,3} + the
original main checkpoint as seed 0) -- no new training.

run_task_hardening(task, spec, cfg) only uses `spec['checkpoint']` and
`spec['training_states']` for model/data loading; P1_ENVS[task] is used only
for `domain`, which does not vary by seed. This means the function can be
called directly with a per-seed `spec` dict, with no modification needed.
"""

import os
import json
import numpy as np

from src.config import XS_CONFIG
from run_addendum_phase6_hardening import run_task_hardening

OUT_DIR = 'outputs/phase6_causal_steering/seed_extension'

TASK_SEEDS = {
    'cartpole': [
        dict(checkpoint='outputs/checkpoints/world_model.pt',
             training_states='outputs/data/training_states.npz'),
        dict(checkpoint='outputs/multiseed/seed_1/world_model.pt',
             training_states='outputs/multiseed/seed_1/training_states.npz'),
        dict(checkpoint='outputs/multiseed/seed_2/world_model.pt',
             training_states='outputs/multiseed/seed_2/training_states.npz'),
        dict(checkpoint='outputs/multiseed/seed_3/world_model.pt',
             training_states='outputs/multiseed/seed_3/training_states.npz'),
        dict(checkpoint='outputs/multiseed/seed_4/world_model.pt',
             training_states='outputs/multiseed/seed_4/training_states.npz'),
    ],
    'reacher': [
        dict(checkpoint='outputs/second_env/reacher_easy_world_model.pt',
             training_states='outputs/second_env/reacher_easy_training_states.npz'),
        dict(checkpoint='outputs/multiseed_env/reacher_seed1/model.pt',
             training_states='outputs/multiseed_env/reacher_seed1/states.npz'),
        dict(checkpoint='outputs/multiseed_env/reacher_seed2/model.pt',
             training_states='outputs/multiseed_env/reacher_seed2/states.npz'),
        dict(checkpoint='outputs/multiseed_env/reacher_seed3/model.pt',
             training_states='outputs/multiseed_env/reacher_seed3/states.npz'),
    ],
    'pendulum': [
        dict(checkpoint='outputs/third_env/pendulum_swingup_world_model.pt',
             training_states='outputs/third_env/pendulum_swingup_training_states.npz'),
        dict(checkpoint='outputs/multiseed_env/pendulum_seed1/model.pt',
             training_states='outputs/multiseed_env/pendulum_seed1/states.npz'),
        dict(checkpoint='outputs/multiseed_env/pendulum_seed2/model.pt',
             training_states='outputs/multiseed_env/pendulum_seed2/states.npz'),
        dict(checkpoint='outputs/multiseed_env/pendulum_seed3/model.pt',
             training_states='outputs/multiseed_env/pendulum_seed3/states.npz'),
    ],
}


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for task, specs in TASK_SEEDS.items():
        all_results[task] = []
        for seed_idx, spec in enumerate(specs):
            out_path = os.path.join(OUT_DIR, f'{task}_seed{seed_idx}.json')
            if os.path.exists(out_path):
                print(f"[phase6_seed_ext] {task}/seed{seed_idx}: already done, loading cached result")
                with open(out_path) as f:
                    all_results[task].append(json.load(f))
                continue
            if not (os.path.exists(spec['checkpoint']) and os.path.exists(spec['training_states'])):
                print(f"[phase6_seed_ext] {task}/seed{seed_idx}: files missing, skipping")
                continue
            print(f"\n{'='*78}\n{task.upper()} SEED {seed_idx}\n{'='*78}", flush=True)
            result = run_task_hardening(task, spec, cfg)
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2, default=float)
            all_results[task].append(result)

    # aggregate across seeds: mean +/- std for the headline z-scores
    print(f"\n{'='*78}\nAGGREGATE ACROSS SEEDS\n{'='*78}")
    agg = {}
    for task, results in all_results.items():
        if not results:
            continue
        probe_k0_z = [r['null_summary']['probe_k0']['z'] for r in results]
        probe_k10_z = [r['null_summary']['probe_k10']['z'] for r in results]
        estate_z = [r['null_summary']['e_state']['z'] for r in results]
        agg[task] = dict(
            n_seeds=len(results),
            probe_k0_z_mean=float(np.mean(probe_k0_z)), probe_k0_z_std=float(np.std(probe_k0_z)),
            probe_k10_z_mean=float(np.mean(probe_k10_z)), probe_k10_z_std=float(np.std(probe_k10_z)),
            estate_z_mean=float(np.mean(estate_z)), estate_z_std=float(np.std(estate_z)),
            probe_k0_z_values=probe_k0_z, probe_k10_z_values=probe_k10_z, estate_z_values=estate_z,
        )
        print(f"  {task} (n={len(results)} seeds): "
              f"probe_k0 z={agg[task]['probe_k0_z_mean']:+.2f}±{agg[task]['probe_k0_z_std']:.2f}  "
              f"probe_k10 z={agg[task]['probe_k10_z_mean']:+.2f}±{agg[task]['probe_k10_z_std']:.2f}  "
              f"e_state z={agg[task]['estate_z_mean']:+.2f}±{agg[task]['estate_z_std']:.2f}")

    out_path = os.path.join(OUT_DIR, 'phase6_seed_extension_aggregate.json')
    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
