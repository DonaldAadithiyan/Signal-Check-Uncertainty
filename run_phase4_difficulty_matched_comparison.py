#!/usr/bin/env python3.11
"""
RNG-bug audit spec, section 3, carried-forward item #2: the difficulty-matched
FULL-vs-PARTIAL comparison Task 3's seed sweep identified as needed but did
not itself resolve.

Task 3's own difficulty-bin check (run_phase4_seed_sweep_analysis.matched_bin_
check) binned each CONDITION's own sites into its own KL terciles separately --
useful for showing that causal effect scales with KL WITHIN a condition, but
it does not answer the actual confound question: at the SAME absolute KL
level, is PARTIAL's causal effect still larger than FULL's, or does the
raw aggregate gap disappear once both conditions are compared on equal
difficulty footing?

Method (same technique as the original workshop paper's Set C construction,
per the spec): pool FULL and PARTIAL's evaluation sites together, bin by KL
QUANTILE across the POOLED distribution (so bin edges are shared, not
computed separately per condition), then compare the causal ablation
z-score and confusion AUROC between conditions WITHIN each shared bin. This
directly tests whether the FULL-vs-PARTIAL gap survives once both conditions
draw from a KL-matched sub-population, rather than each other's own already-
different marginal KL distribution.

Reuses run_phase4_seed_sweep_analysis's exact trajectory-collection, probe-
fitting, and multi-step causal-test machinery -- only the site binning /
comparison logic is new.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import compute_ct, probe_direction, random_matched_direction
from run_phase4_seed_sweep_analysis import (
    load_condition, collect_trajectories, imagined_vs_real_obs_pomdp, e_state,
    best_gamma, multi_step_causal_test, ALL_SEEDS, N_TRAJ, MIN_T, HORIZONS,
    K_HEADLINE, N_NULL,
)

OUT_DIR = 'outputs/phase4_pomdp/seed_sweep'
N_QUANTILE_BINS = 3          # tercile bins, matched to the within-condition check
N_SITES_PER_CONDITION = 4000  # matches analyze_seed_condition's cap
N_NULL_MATCHED = 20           # reduced null for cost, matching the within-condition check's own tradeoff


def prepare_condition(seed, condition):
    """Fits the probe/gamma/direction and collects sites exactly as
    analyze_seed_condition does, but returns the raw per-site KL/causal
    inputs needed for cross-condition binning instead of aggregating."""
    cfg = XS_CONFIG.copy()
    model, tr, ckpt = load_condition(seed, condition)

    y = binarise_by_median(tr['kl'])
    idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                       stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])

    probe_score_all = clf.predict_proba(scaler.transform(tr['h']))[:, 1]
    gamma, r2_ct = best_gamma(tr['kl'], tr['traj_id'], probe_score_all)
    v = probe_direction(clf, scaler)

    trajs = collect_trajectories(model, condition, N_TRAJ, cfg, seed=seed)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64), gamma=gamma)
        trj['probe'] = clf.predict_proba(scaler.transform(trj['h']))[:, 1]

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - max(HORIZONS) - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(seed)
    if len(sites) > N_SITES_PER_CONDITION:
        sel = rng.choice(len(sites), N_SITES_PER_CONDITION, replace=False)
        sites = [sites[i] for i in sel]

    site_kl = np.array([trajs[ti]['kl'][t] for (ti, t) in sites])
    return dict(model=model, clf=clf, scaler=scaler, v=v, trajs=trajs, sites=sites,
                site_kl=site_kl, cfg=cfg)


def matched_comparison_for_seed(seed):
    full = prepare_condition(seed, 'full')
    partial = prepare_condition(seed, 'partial')

    # shared bin edges from the POOLED kl distribution across both conditions
    pooled_kl = np.concatenate([full['site_kl'], partial['site_kl']])
    edges = np.percentile(pooled_kl, np.linspace(0, 100, N_QUANTILE_BINS + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    result = dict(seed=seed, bins=[])
    for b in range(N_QUANTILE_BINS):
        lo, hi = edges[b], edges[b + 1]
        bin_result = dict(bin=b, lo=float(lo), hi=float(hi))
        for cond_name, cond in [('full', full), ('partial', partial)]:
            mask = (cond['site_kl'] > lo) & (cond['site_kl'] <= hi)
            bin_sites = [cond['sites'][i] for i in np.where(mask)[0]]
            n = len(bin_sites)
            bin_result[f'{cond_name}_n'] = n
            bin_result[f'{cond_name}_mean_kl'] = float(cond['site_kl'][mask].mean()) if n else float('nan')
            if n < 20:
                bin_result[f'{cond_name}_causal_z'] = float('nan')
                bin_result[f'{cond_name}_auroc_proxy'] = float('nan')
                continue
            rng_seed = seed + 80000 + b + (0 if cond_name == 'full' else 1000)
            rng_null = np.random.default_rng(rng_seed)
            eff = multi_step_causal_test(cond['model'], cond['clf'], cond['scaler'],
                                          cond['trajs'], bin_sites, cond['v'], lookahead=[0])
            null_effs = []
            for _ in range(N_NULL_MATCHED):
                vr = random_matched_direction(rng_null, cond['v'].shape[0])
                null_effs.append(multi_step_causal_test(
                    cond['model'], cond['clf'], cond['scaler'], cond['trajs'],
                    bin_sites, vr, lookahead=[0])[0])
            z = (eff[0] - np.mean(null_effs)) / (np.std(null_effs) + 1e-12)
            bin_result[f'{cond_name}_causal_z'] = float(z)
            # confusion-discrimination proxy within bin: correlation of probe score with KL rank
            probe_scores = np.array([cond['trajs'][ti]['probe'][t] for (ti, t) in bin_sites])
            kl_vals = cond['site_kl'][mask]
            r_probe_kl = pearsonr(probe_scores, kl_vals)[0] if len(set(kl_vals)) > 1 else float('nan')
            bin_result[f'{cond_name}_r_probe_kl'] = float(r_probe_kl)
        result['bins'].append(bin_result)
    return result


def main():
    all_results = []
    for seed in ALL_SEEDS:
        print(f"\n{'='*78}\nSEED {seed} — DIFFICULTY-MATCHED FULL vs PARTIAL\n{'='*78}")
        r = matched_comparison_for_seed(seed)
        for b in r['bins']:
            print(f"  bin {b['bin']} (KL {b['lo']:.1f}-{b['hi']:.1f}): "
                  f"FULL n={b['full_n']} mean_kl={b['full_mean_kl']:.1f} z={b['full_causal_z']:+.2f}  |  "
                  f"PARTIAL n={b['partial_n']} mean_kl={b['partial_mean_kl']:.1f} z={b['partial_causal_z']:+.2f}")
        all_results.append(r)

    # aggregate per-bin z-scores across seeds
    print(f"\n{'='*78}\nAGGREGATE ACROSS {len(all_results)} SEEDS\n{'='*78}")
    agg = {}
    for b_idx in range(N_QUANTILE_BINS):
        full_zs = [r['bins'][b_idx]['full_causal_z'] for r in all_results
                   if np.isfinite(r['bins'][b_idx]['full_causal_z'])]
        partial_zs = [r['bins'][b_idx]['partial_causal_z'] for r in all_results
                      if np.isfinite(r['bins'][b_idx]['partial_causal_z'])]
        full_mean, full_std = float(np.mean(full_zs)), float(np.std(full_zs))
        partial_mean, partial_std = float(np.mean(partial_zs)), float(np.std(partial_zs))
        stronger = abs(partial_mean) > abs(full_mean)
        print(f"  bin {b_idx}: FULL z={full_mean:+.2f}±{full_std:.2f} (n={len(full_zs)} seeds)  "
              f"PARTIAL z={partial_mean:+.2f}±{partial_std:.2f} (n={len(partial_zs)} seeds)  "
              f"partial_stronger={stronger}")
        agg[f'bin_{b_idx}'] = dict(full_mean=full_mean, full_std=full_std,
                                    partial_mean=partial_mean, partial_std=partial_std,
                                    partial_stronger=stronger)

    out = dict(per_seed=all_results, aggregate=agg)
    out_path = os.path.join(OUT_DIR, 'phase4_difficulty_matched_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
