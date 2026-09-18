#!/usr/bin/env python3.11
"""
Task 9 -- Step 1: natural blind-spot characterization.

Uses the fresh held-out data already collected and grouped by Step 0
(outputs/task9_blind_spot/fresh_data_{reacher,cartpole}.npz +
step0_sample_size_results.json's grouping rule) -- no new trajectory
collection, no new atom selection.

For the four groups (low atom/low C_t; high atom/high C_t; HIGH ATOM/LOW
C_t = the blind-spot candidate; low atom/high C_t = matched control),
reports E^state's distribution in each group. Target finding: the candidate
group (atom fires, C_t stays low/normal) shows elevated E^state relative to
the matched control (C_t high but atom silent) and relative to the
low/low baseline -- i.e. the atom flags real difficulty the confusion
signal itself misses.
"""

import os
import json
import numpy as np
from scipy.stats import mannwhitneyu

from run_task9_step0_sample_size import sample_size_check, ATOM_612, ATOM_156

OUT_DIR = 'outputs/task9_blind_spot'


def characterize(task, atom_idx, data, check):
    atom = data['atom_activation']
    ct = data['ct']
    estate = data['e_state']

    fire_rate = check['atom_fire_rate']
    if fire_rate < 0.50:
        high_atom = atom > 0.0
        low_atom = atom <= 0.0
    else:
        thresh = check['atom_activation_threshold']
        high_atom = atom >= thresh
        low_atom = atom < thresh

    ct_thresh = check['ct_threshold']
    low_ct = ct <= ct_thresh
    high_ct = ct > ct_thresh

    groups = dict(
        low_atom_low_ct=low_atom & low_ct,
        high_atom_high_ct=high_atom & high_ct,
        candidate_high_atom_low_ct=high_atom & low_ct,
        matched_control_low_atom_high_ct=low_atom & high_ct,
    )

    group_stats = {}
    for name, mask in groups.items():
        vals = estate[mask]
        group_stats[name] = dict(
            n=int(mask.sum()),
            e_state_mean=float(vals.mean()) if len(vals) else float('nan'),
            e_state_std=float(vals.std()) if len(vals) else float('nan'),
            e_state_median=float(np.median(vals)) if len(vals) else float('nan'),
        )

    # target comparison: candidate vs matched control (both isolate the
    # atom's marginal contribution at matched C_t status -- candidate has
    # C_t LOW same as the low/low baseline, but fires; control has C_t HIGH
    # but the atom silent). The critical test for "the atom flags something
    # C_t itself misses" is candidate (atom high, C_t low) vs low/low
    # baseline (atom low, C_t low) -- SAME C_t level, atom status differs.
    candidate_vals = estate[groups['candidate_high_atom_low_ct']]
    baseline_low_vals = estate[groups['low_atom_low_ct']]
    control_vals = estate[groups['matched_control_low_atom_high_ct']]

    def safe_mwu(a, b):
        if len(a) < 5 or len(b) < 5:
            return dict(u=None, p=None)
        u, p = mannwhitneyu(a, b, alternative='greater')
        return dict(u=float(u), p=float(p))

    comparisons = dict(
        candidate_vs_low_low_baseline=dict(
            delta_mean=float(candidate_vals.mean() - baseline_low_vals.mean()) if len(candidate_vals) and len(baseline_low_vals) else None,
            mwu_greater=safe_mwu(candidate_vals, baseline_low_vals),
        ),
        candidate_vs_matched_control=dict(
            delta_mean=float(candidate_vals.mean() - control_vals.mean()) if len(candidate_vals) and len(control_vals) else None,
            mwu_greater=safe_mwu(candidate_vals, control_vals),
        ),
    )

    return dict(task=task, atom=atom_idx, group_stats=group_stats, comparisons=comparisons)


def main():
    results = {}
    with open(os.path.join(OUT_DIR, 'step0_sample_size_results.json')) as f:
        step0 = json.load(f)

    for task, atom_idx, fname in [('reacher', ATOM_612, 'fresh_data_reacher.npz'),
                                    ('cartpole', ATOM_156, 'fresh_data_cartpole.npz')]:
        key = f'{task}_{atom_idx}'
        data = dict(np.load(os.path.join(OUT_DIR, fname)))
        check = step0[key]
        print(f"\n{'='*78}\n{task.upper()} #{atom_idx} — STEP 1 CHARACTERIZATION\n{'='*78}")
        result = characterize(task, atom_idx, data, check)
        for name, stats in result['group_stats'].items():
            print(f"  {name:<35} n={stats['n']:>5}  E^state mean={stats['e_state_mean']:.4f}  "
                  f"median={stats['e_state_median']:.4f}  std={stats['e_state_std']:.4f}")
        print(f"  candidate vs low/low baseline: Δmean={result['comparisons']['candidate_vs_low_low_baseline']['delta_mean']:+.4f}  "
              f"MWU p={result['comparisons']['candidate_vs_low_low_baseline']['mwu_greater']['p']}")
        print(f"  candidate vs matched control:  Δmean={result['comparisons']['candidate_vs_matched_control']['delta_mean']:+.4f}  "
              f"MWU p={result['comparisons']['candidate_vs_matched_control']['mwu_greater']['p']}")
        results[key] = result

    out_path = os.path.join(OUT_DIR, 'step1_characterization_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
