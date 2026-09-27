#!/usr/bin/env python3
"""Build outputs/sprint/RESULTS.md from outputs/sprint/results.json only.

Section 1 (top) = the W1 gate and every automatically detected result that
weakens a paper claim. The checks are fixed here, before any real data:
  - R^2(readout, C_t^past) not clearly above R^2(readout, KL_t)
    (95% CI of the paired difference includes or lies below 0)
  - partial R^2 of C_t^past beyond the current flag has CI including 0
  - external Delta R^2 (C_t or C_t^past) CI includes 0 or is negative
  - C_t loses a head-to-head (rival beyond C_t > C_t beyond rival)
  - scrambling z > -2 ... i.e. real statistic not above the scramble null (z < 2)
  - probe direction's top-k projection fraction NOT below the random 5th pct
  - steering readout z (k=0 or k=10) below 2 vs any null
  - steering E^state |z| >= 2 (the paper claims NO external causal effect)
  - PC-scrubbed or MLP probe AUROC differs from linear by more than 0.03
Usage: python build_sprint_report.py [--smoke]
"""
import argparse
import json
import os

TASKS = ['cartpole', 'reacher', 'pendulum']


def f(x, d=3):
    return 'n/a' if x is None else f'{x:.{d}f}'


def ci(c, d=3):
    return f"{c['point']:+.{d}f} [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}]"


def flags(R):
    out = []
    for t, models in R.get('p1', {}).items():
        for lab, d in models.items():
            v = d['variants']['C_past']
            if v['r2_minus_r2_kl']['lo'] <= 0:
                out.append(f"P1 {t}/{lab}: R²(readout, C^past) − R²(readout, KL) = {ci(v['r2_minus_r2_kl'])} "
                           f"— C^past NOT clearly above current KL.")
            if v['partial_r2_beyond_flag']['lo'] <= 0:
                out.append(f"P1 {t}/{lab}: partial R² of C^past beyond 1[KL_t>median] = "
                           f"{ci(v['partial_r2_beyond_flag'])} (CI includes 0).")
            for name, s in d['scrambling_readout'].items():
                if s['z'] < 2:
                    out.append(f"P1 {t}/{lab}: readout scrambling z for {name} = {s['z']:+.1f} (< 2).")
            for k, g in d['geometry'].items():
                if g['frac'] >= g['random_p5']:
                    out.append(f"P1 {t}/{lab}: geometry {k}: frac {g['frac']:.4f} not below random 5th pct "
                               f"{g['random_p5']:.4f} (random mean {g['random_mean']:.4f}).")
    for t, d in R.get('p2', {}).items():
        for name in ('C', 'C_past'):
            c = d['table3'][name]['delta_r2']
            if c['lo'] <= 0:
                out.append(f"P2 {t}: external ΔR² for {name} = {ci(c, 4)} (CI includes 0 or negative).")
            if d['scrambling_external'][name]['z'] < 2:
                out.append(f"P2 {t}: external scrambling z for {name} = "
                           f"{d['scrambling_external'][name]['z']:+.1f} (< 2).")
        for k, h in d['head_to_head'].items():
            if h['rival_beyond_ct']['point'] > h['ct_beyond_rival']['point']:
                out.append(f"P2 {t}: head-to-head {k}: rival adds {ci(h['rival_beyond_ct'], 4)} beyond C, "
                           f"C adds {ci(h['ct_beyond_rival'], 4)} beyond rival — rival wins.")
    for t, models in R.get('p3', {}).items():
        for lab, d in models.items():
            for nn, s in d['nulls'].items():
                for key in ('slope_r0', 'slope_r10'):
                    if s[key]['z'] < 2:
                        out.append(f"P3 {t}/{lab} {nn}: readout {key} z = {s[key]['z']:+.1f} (< 2).")
                if abs(s['slope_es']['z']) >= 2:
                    out.append(f"P3 {t}/{lab} {nn}: E^state steering z = {s['slope_es']['z']:+.1f} "
                               f"(|z| ≥ 2 — contradicts 'no external causal effect').")
    for t, d in R.get('p4', {}).items():
        base = d['linear']['eval_auroc']
        for k in ('pc_scrubbed', 'mlp'):
            if abs(d[k]['eval_auroc'] - base) > 0.03:
                out.append(f"P4 {t}: {k} AUROC {d[k]['eval_auroc']:.3f} vs linear {base:.3f} (|Δ| > 0.03).")
    return out


def rng_str(vals, d=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 'n/a'
    return f'{min(vals):.{d}f}–{max(vals):.{d}f}' if len(vals) > 1 else f'{vals[0]:.{d}f}'


def paper_tables(R, w):
    """Tables in icaart_main.tex layout (tab:crosstask, tab:external, tab:steer, tab:dissoc)."""
    p1, p2, p3 = R.get('p1', {}), R.get('p2', {}), R.get('p3', {})
    w('## 2a. Paper tables (icaart_main.tex layout) — drop-in values\n')
    if p1:
        w('**Table 2 (tab:crosstask).** Primary model; brackets = range over all models of that task '
          '(primary + replicates). The angle row is replaced by the projection fraction (W3).\n')
        w('| | ' + ' | '.join(TASKS) + ' |')
        w('|---|---|---|---|')
        def row(label, fn, d=2):
            cells = []
            for t in TASKS:
                ms = p1.get(t, {})
                if 'primary' not in ms:
                    cells.append('n/a'); continue
                vals = [fn(m) for m in ms.values()]
                cells.append(f"{fn(ms['primary']):.{d}f} [{rng_str(vals, d)}] (n={len(vals)})")
            w(f'| {label} | ' + ' | '.join(cells) + ' |')
        row('frac of v in top-50 PCs (standardised)', lambda m: m['geometry']['standardized_top50']['frac'], 3)
        row('  random-vector mean for comparison', lambda m: m['geometry']['standardized_top50']['random_mean'], 3)
        row('R²(readout, C_t)', lambda m: m['variants']['C']['r2']['point'])
        row('R²(readout, C_t^past)', lambda m: m['variants']['C_past']['r2']['point'])
        row('R²(readout, KL_t)', lambda m: m['r2_readout_kl']['point'])
        row('probe AUROC (clean / held-out)', lambda m: m['probe_eval_auroc'], 3)
        row('probe AUROC (σ=0.1 noise)', lambda m: m['table2']['auroc_noisy'], 3)
        row('KL-matched AUROC (pooled)', lambda m: m['table2']['kl_matched_pooled']['auroc'])
        row('KL-matched AUROC (clean only)', lambda m: m['table2']['kl_matched_clean']['auroc'])
        row('within-bin r(Rec, C_t)', lambda m: m['table2']['within_bin_r_rec_ct']['mean'])
        row('ridge h→C_t scramble z (Fig. 1d)', lambda m: m['ridge_scramble']['C']['z'], 1)
        row('ridge h→C_t^past scramble z', lambda m: m['ridge_scramble']['C_past']['z'], 1)
        row('γ (calibration) for C_t', lambda m: m['variants']['C']['gamma'])
        w('')
    if p2:
        w('**Table 3 (tab:external).** ΔR² for E^state_10 beyond KL, Rec, EMARec, EMAKL (no ensemble term); '
          '95% CI from 1,000 episode-bootstrap resamples; 100 evaluation episodes.\n')
        w('| task | sites | ΔR² C_t | ΔR² C_t^past | ΔR² C_t, paper controls (KL, Rec, EMARec) |')
        w('|---|---|---|---|---|')
        for t in TASKS:
            d = p2.get(t)
            if d:
                T3 = d['table3']
                w(f"| {t} | {d['n_sites']} | {ci(T3['C']['delta_r2'], 4)} | {ci(T3['C_past']['delta_r2'], 4)} | "
                  f"{ci(T3['C']['delta_r2_paper_controls'], 4)} |")
        w('\n**§5.3 text (head-to-head, C_t and EMARec only):** ' + '; '.join(
            f"{t} {p2[t]['head_to_head']['C_vs_emarec']['ct_beyond_rival']['point']:+.3f}" for t in TASKS if t in p2)
          + '. **External scramble z (Fig. 2c):** ' + '; '.join(
            f"{t} C {p2[t]['scrambling_external']['C']['z']:+.1f} / C^past {p2[t]['scrambling_external']['C_past']['z']:+.1f}"
            for t in TASKS if t in p2) + '.\n')
    if p3:
        import numpy as np
        w('**Table 5 (tab:steer).** z of the steering slope vs the 50-direction null, mean ± sd over models.\n')
        w('| task | n | readout k=0 | readout k=10 | E^state |')
        w('|---|---|---|---|---|')
        allr, alle = [], []
        for t in TASKS:
            ms = [m['nulls']['null50'] for m in p3.get(t, {}).values() if 'null50' in m['nulls']]
            if not ms:
                continue
            cols = [np.array([m[k]['z'] for m in ms]) for k in ('slope_r0', 'slope_r10', 'slope_es')]
            allr += list(cols[0]) + list(cols[1]); alle += list(cols[2])
            w(f'| {t} | {len(ms)} | ' + ' | '.join(f'{c.mean():+.2f} ± {c.std(ddof=1) if len(c) > 1 else 0:.2f}'
                                                  for c in cols) + ' |')
        if allr:
            w(f'\nReadout z range (k=0 ∪ k=10): {min(allr):+.1f} to {max(allr):+.1f}; '
              f'max |E^state z|: {max(abs(x) for x in alle):.2f}.\n')
        w('**Table 6 (tab:dissoc).** Dense row from the values above. Atom rows (#612, #156) come only from '
          'a P5 rerun; without it they are removed.\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    base = 'outputs/sprint/smoke' if a.smoke else 'outputs/sprint'
    with open(os.path.join(base, 'results.json')) as fh:
        R = json.load(fh)
    L = []
    w = L.append
    w('# Sprint results (corrected pipeline)\n')
    if a.smoke:
        w('> **SMOKE RUN — randomly initialised models. These numbers are NOT results.**\n')
    w('Generated by `build_sprint_report.py` from `results.json`. Every number here comes from '
      '`src/rerun/pipeline.py` (correct action convention, calibration thresholds, episode-level '
      'splits, explicit seeding). No old-pipeline number appears in this file.\n')

    w('## 1. W1 gate and results that weaken paper claims\n')
    g = R.get('w1_gate')
    if g:
        w('| task | r(KL correct, buggy) | KL mean correct / buggy | Rec mean correct / buggy | '
          'AUROC correct / buggy | pass |')
        w('|---|---|---|---|---|---|')
        for t in TASKS:
            if t in g:
                d = g[t]
                w(f"| {t} | {d['kl_pearson_r']:.3f} | {d['correct']['kl_mean']:.2f} / {d['buggy']['kl_mean']:.2f} | "
                  f"{d['correct']['rec_mean']:.4f} / {d['buggy']['rec_mean']:.4f} | "
                  f"{d['correct']['probe_auroc']:.3f} / {d['buggy']['probe_auroc']:.3f} | "
                  f"{'yes' if d['passes'] else '**NO**'} |")
        w(f"\n**Gate: {'PROCEED (Sept 29)' if g['_gate']['proceed_sept29'] else 'STOP → Oct 22'}** "
          f"(r ≥ {g['_gate']['r_min']}, |ΔAUROC| ≤ {g['_gate']['dauc_max']} on all tasks).\n")
    fl = flags(R)
    w('### Weakening results (auto-detected)\n')
    w('\n'.join(f'- {x}' for x in fl) if fl else '- none detected by the fixed checks')
    w('')

    paper_tables(R, w)
    if 'p1' in R:
        w('## 2b. Full tables\n')
        w('### Emergence (evaluation episodes; readout = probe probability)\n')
        w('| task | model | probe AUROC | R² KL_t | R² C_t (γ) | R² C^past (γ) | R²(C^past) − R²(KL) | '
          'partial R² C^past beyond flag | scramble z (C / C^past) |')
        w('|---|---|---|---|---|---|---|---|---|')
        for t in TASKS:
            for lab, d in sorted(R['p1'].get(t, {}).items(), key=lambda kv: (kv[0] != 'primary', kv[0])):
                v = d['variants']
                w(f"| {t} | {lab} | {d['probe_eval_auroc']:.3f} | {ci(d['r2_readout_kl'])} | "
                  f"{ci(v['C']['r2'])} ({v['C']['gamma']}) | {ci(v['C_past']['r2'])} ({v['C_past']['gamma']}) | "
                  f"{ci(v['C_past']['r2_minus_r2_kl'])} | {ci(v['C_past']['partial_r2_beyond_flag'])} | "
                  f"{d['scrambling_readout']['C']['z']:+.1f} / {d['scrambling_readout']['C_past']['z']:+.1f} |")
        w('\n### C_t variants — R²(readout, variant), calibration γ (primary models)\n')
        names = list(next(iter(R['p1'].values()))['primary']['variants'])
        w('| task | ' + ' | '.join(names) + ' |')
        w('|---|' + '---|' * len(names))
        for t in TASKS:
            d = R['p1'].get(t, {}).get('primary')
            if d:
                w(f'| {t} | ' + ' | '.join(f"{d['variants'][n]['r2']['point']:.3f} (γ={d['variants'][n]['gamma']})"
                                           for n in names) + ' |')
        w('\n### Geometry (probe-fit states): projection fraction ‖proj_U w‖² vs 1,000 random unit vectors\n')
        w('| task | model | space/top-k | frac | subspace angle | random mean | random 5th pct | percentile of w |')
        w('|---|---|---|---|---|---|---|---|')
        for t in TASKS:
            for lab, d in sorted(R['p1'].get(t, {}).items(), key=lambda kv: (kv[0] != 'primary', kv[0])):
                for k, gg in d['geometry'].items():
                    w(f"| {t} | {lab} | {k} | {gg['frac']:.4f} | {gg['subspace_angle_deg']:.1f}° | "
                      f"{gg['random_mean']:.4f} | {gg['random_p5']:.4f} | {gg['percentile_vs_random']:.1f} |")

    if 'p2' in R:
        w('\n### External validity (Table 3 layout): target E^state_10, controls KL, Rec, EMARec, EMAKL\n')
        w('| task | n sites / episodes | α EMARec / EMAKL | R² controls | ΔR² C_t | ΔR² C^past | '
          'ΔR² C_t (paper controls, no EMAKL) | cross-fit ΔR² C_t / C^past | scramble z C / C^past |')
        w('|---|---|---|---|---|---|---|---|---|')
        for t in TASKS:
            d = R['p2'].get(t)
            if d:
                T3 = d['table3']
                w(f"| {t} | {d['n_sites']} / {d['n_episodes']} | {d['alpha_emarec']} / {d['alpha_emakl']} | "
                  f"{ci(d['r2_controls'])} | {ci(T3['C']['delta_r2'], 4)} | {ci(T3['C_past']['delta_r2'], 4)} | "
                  f"{ci(T3['C']['delta_r2_paper_controls'], 4)} | {d['crossfit_delta_r2']['C']:+.4f} / "
                  f"{d['crossfit_delta_r2']['C_past']:+.4f} | {d['scrambling_external']['C']['z']:+.1f} / "
                  f"{d['scrambling_external']['C_past']['z']:+.1f} |")
        w('\n### Head-to-heads (ΔR² for E^state_10, each direction)\n')
        w('| task | comparison | statistic beyond rival | rival beyond statistic |')
        w('|---|---|---|---|')
        for t in TASKS:
            for k, h in R['p2'].get(t, {}).get('head_to_head', {}).items():
                w(f"| {t} | {k} | {ci(h['ct_beyond_rival'], 4)} | {ci(h['rival_beyond_ct'], 4)} |")
        w('\n### Horizon correlations r(statistic, E^state_K)\n')
        w('| task | K | C_t | C^past | KL_t |')
        w('|---|---|---|---|---|')
        for t in TASKS:
            for K, h in R['p2'].get(t, {}).get('horizons', {}).items():
                w(f"| {t} | {K} | {ci(h['C'])} | {ci(h['C_past'])} | {ci(h['kl'])} |")
        w('\n### ΔR² across variants (calibration γ) — fig5a\n')
        names = list(next(iter(R['p2'].values()))['variants'])
        w('| task | ' + ' | '.join(names) + ' |')
        w('|---|' + '---|' * len(names))
        for t in TASKS:
            d = R['p2'].get(t)
            if d:
                w(f'| {t} | ' + ' | '.join(ci(d['variants'][n]['delta_r2'], 4) for n in names) + ' |')

    if 'p3' in R:
        w('\n### Dense-direction causal dissociation (steering slope and ablation, z vs null)\n')
        w('| task | model | null | slope readout k=0 | slope readout k=10 | slope E^state | '
          'ablate readout k=0 | ablate readout k=10 | ablate E^state |')
        w('|---|---|---|---|---|---|---|---|---|')
        for t in TASKS:
            for lab, d in R['p3'].get(t, {}).items():
                for nn, s in d['nulls'].items():
                    w(f"| {t} | {lab} | {nn} (n={s['slope_r0']['n']}) | " + ' | '.join(
                        f"{s[k]['z']:+.1f}" for k in ('slope_r0', 'slope_r10', 'slope_es',
                                                      'ablate_r0', 'ablate_r10', 'ablate_es')) + ' |')

    if 'p4' in R:
        w('\n### Probe controls (evaluation AUROC; R²(readout, C^past))\n')
        w('| task | linear | PC-scrubbed (top-50 removed) | MLP (64) |')
        w('|---|---|---|---|')
        for t in TASKS:
            d = R['p4'].get(t)
            if d:
                w(f"| {t} | " + ' | '.join(f"{d[k]['eval_auroc']:.3f}; {d[k]['r2_c_past']['point']:.3f}"
                                           for k in ('linear', 'pc_scrubbed', 'mlp')) + ' |')

    if 'p6' in R:
        w('\n### KL autocorrelation (evaluation episodes) vs external ΔR² (prediction stated in advance: '
          'more temporal structure ↔ larger ΔR²)\n')
        w('| task | ACF KL lag1 / 5 / 10 | ACF flag lag1 / 5 / 10 | ΔR² C_t |')
        w('|---|---|---|---|')
        for t in TASKS:
            d = R['p6'].get(t)
            if d:
                a1, af = d['acf_kl'], d['acf_flag']
                dr = R.get('p2', {}).get(t, {}).get('table3', {}).get('C', {}).get('delta_r2', {}).get('point')
                w(f"| {t} | {a1['1']:.3f} / {a1['5']:.3f} / {a1['10']:.3f} | "
                  f"{af['1']:.3f} / {af['5']:.3f} / {af['10']:.3f} | {f(dr, 4)} |")

    mp = 'paper/SPRINT_PAPER_MAP.md'
    if os.path.exists(mp):
        w('\n' + open(mp).read())
    with open(os.path.join(base, 'RESULTS.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('wrote', os.path.join(base, 'RESULTS.md'), f'({len(fl)} weakening flags)')


if __name__ == '__main__':
    main()
