#!/usr/bin/env python3.11
"""
Task J — Three-environment side-by-side comparison.

Reads the per-environment result JSONs produced by run_second_env.py (reacher,
pendulum) plus the cartpole pilot numbers, extends the existing two-column
cartpole/reacher table to three columns, and explicitly characterizes the pattern
across all three: does confound-cleanliness (within-task AUROC's distance from 0.5)
move together with closed-form C_t R² and best-γ, ordered by some identifiable
environment property (obs/act dim, episode length, KL scale)?

Resolves the n=2 ambiguity from Task D: with three points we can say whether reacher
was the outlier, cartpole was, or the closed-form parameters simply don't generalize
while the direction + null-space geometry do.
"""

import os
import json
import numpy as np

REACHER  = 'outputs/second_env/reacher_easy_results.json'
PENDULUM = 'outputs/third_env/pendulum_swingup_results.json'

# cartpole pilot headline numbers (n=1 pilot / 5-seed where noted), from DEV_LOG / Task C
CART = dict(env='cartpole_swingup', obs_dim=5, act_dim=1,
            auroc_id=0.9019, auroc_a=0.8632, auroc_b=0.8464, auroc_c=0.7227,
            auroc_within_task=0.506, best_gamma=0.95, best_ct_r2=0.798,
            nullspace_angle=88.0, frac_in_top10=0.09, n_states=100000)


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def main():
    reacher = load(REACHER)
    pendulum = load(PENDULUM)
    envs = [('cartpole-swingup', CART)]
    if reacher:
        envs.append(('reacher-easy', reacher))
    if pendulum:
        envs.append(('pendulum-swingup', pendulum))

    names = [n for n, _ in envs]
    print("=" * (34 + 20 * len(envs)))
    print("TASK J — THREE-ENVIRONMENT GENERALISATION (side-by-side)")
    print("=" * (34 + 20 * len(envs)))
    print(f"\n  {'Metric':<32}" + "".join(f"{n:>20}" for n in names))
    print(f"  {'-'*32}" + "".join(f"{'-'*20}" for _ in names))

    def row(label, key, fmt='{:.4f}'):
        cells = ""
        for _, d in envs:
            v = d.get(key)
            cells += f"{(fmt.format(v) if v is not None else 'n/a'):>20}"
        print(f"  {label:<32}{cells}")

    row('obs_dim', 'obs_dim', '{:d}')
    row('act_dim', 'act_dim', '{:d}')
    row('Probe A held-out AUROC', 'auroc_id')
    row('Probe A Set A AUROC', 'auroc_a')
    row('Probe A Set B AUROC', 'auroc_b')
    row('Probe A Set C AUROC (headline)', 'auroc_c')
    # Set C CI where available
    ci_cells = ""
    for _, d in envs:
        ci = d.get('ci_setc')
        ci_cells += f"{('[%.3f,%.3f]'%(ci[1],ci[2]) if ci else '—'):>20}"
    print(f"  {'  Set C 95% CI':<32}{ci_cells}")
    row('Within-task confound AUROC', 'auroc_within_task')
    row('C_t best γ', 'best_gamma', '{:.2f}')
    row('C_t best R²', 'best_ct_r2')
    row('Null-space angle (°)', 'nullspace_angle', '{:.1f}')
    row('Frac probe dir in top-10 PC', 'frac_in_top10')

    # ── cross-environment pattern ──
    print("\n  Cross-environment pattern:")
    rows = [(n, d) for n, d in envs]
    setc = np.array([d['auroc_c'] for _, d in rows])
    within = np.array([d['auroc_within_task'] for _, d in rows])
    within_dist = np.abs(within - 0.5)      # confound "uncleanliness": distance from chance
    ct_r2 = np.array([d['best_ct_r2'] for _, d in rows])
    gamma = np.array([d['best_gamma'] for _, d in rows])
    angle = np.array([d['nullspace_angle'] for _, d in rows])
    frac = np.array([d['frac_in_top10'] for _, d in rows])

    print(f"    {'env':<20}{'SetC':>8}{'within':>9}{'|within-0.5|':>13}{'C_t R²':>9}{'γ':>6}{'angle':>8}{'frac':>8}")
    for i, (n, _) in enumerate(rows):
        print(f"    {n:<20}{setc[i]:>8.3f}{within[i]:>9.3f}{within_dist[i]:>13.3f}"
              f"{ct_r2[i]:>9.3f}{gamma[i]:>6.2f}{angle[i]:>8.1f}{frac[i]:>8.4f}")

    # what generalizes vs what doesn't
    geometry_general = bool(np.all(angle > 80) and np.all(frac < 0.2))
    signal_general = bool(np.all(setc > 0.55))
    ct_r2_stable = bool(ct_r2.max() - ct_r2.min() < 0.2)
    gamma_stable = bool(gamma.max() - gamma.min() < 1e-6)

    print(f"\n  Null-space geometry generalizes (angle>80°, frac<0.2 on all {len(rows)}): {geometry_general}")
    print(f"  Confusion signal present on all (Set C > 0.55): {signal_general}")
    print(f"  Closed-form C_t R² stable across envs (span<0.2): {ct_r2_stable} "
          f"(span {ct_r2.max()-ct_r2.min():.3f})")
    print(f"  Best γ identical across envs: {gamma_stable} (values {sorted(set(gamma.tolist()))})")

    # does confound-cleanliness track C_t R²? (both are 'how well the closed form works')
    if len(rows) >= 3:
        from scipy.stats import pearsonr
        r_clean_r2, _ = pearsonr(-within_dist, ct_r2)   # cleaner confound (small dist) ~ higher R²?
        print(f"\n  r(confound-cleanliness, C_t R²) across 3 envs = {r_clean_r2:+.3f} "
              f"(do they move together?)")
        # order envs by C_t R² and see if within-cleanliness tracks
        order = np.argsort(-ct_r2)
        print(f"  Ordered by C_t R² (high→low): " +
              " > ".join(f"{rows[i][0].split('-')[0]}({ct_r2[i]:.2f},clean={within_dist[i]:.2f})"
                         for i in order))

    # ── verdict on the n=2 ambiguity ──
    print("\n" + "-" * 74)
    print("  RESOLUTION OF THE n=2 AMBIGUITY:")
    if signal_general and geometry_general and not ct_r2_stable:
        # figure out whether pendulum sided with cartpole or reacher on cleanliness
        if pendulum:
            pend_r2 = pendulum['best_ct_r2']
            closer_to_cart = abs(pend_r2 - CART['best_ct_r2']) < abs(pend_r2 - (reacher['best_ct_r2'] if reacher else 0))
            sided = 'cartpole (clean/high-R²)' if closer_to_cart else 'reacher (noisier/low-R²)'
            print(f"    The third environment (pendulum) sides with {sided}.")
        print("    ROBUST CLAIM: the confusion DIRECTION and the near-null-space GEOMETRY generalize")
        print("    across all three environments, but the closed-form C_t parameters (γ, R²) do NOT —")
        print("    they are environment-dependent. The paper should claim generality for the")
        print("    direction/geometry and explicitly scope the closed form to cartpole.")
    elif signal_general and geometry_general and ct_r2_stable:
        print("    ROBUST CLAIM: signal, geometry, AND closed-form C_t all generalize across three")
        print("    environments — reacher's weaker C_t was the outlier, and the result is more general")
        print("    than Task D alone suggested.")
    else:
        print("    PARTIAL: see per-env table; the generality claim is bounded by whichever metrics")
        print("    fail to hold across all three. Reported honestly.")

    out = {'envs': names, 'table': {n: d for n, d in envs},
           'geometry_general': geometry_general, 'signal_general': signal_general,
           'ct_r2_stable': ct_r2_stable, 'gamma_stable': gamma_stable}
    os.makedirs('outputs/third_env', exist_ok=True)
    with open('outputs/third_env/three_env_comparison.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  Saved: outputs/third_env/three_env_comparison.json")


if __name__ == '__main__':
    main()
