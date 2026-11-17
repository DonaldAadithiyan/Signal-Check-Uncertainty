#!/usr/bin/env python3.11
"""Writes outputs/camera_ready/CV_delta_r2.md from CV_delta_r2.json (no hand-typed values)."""
import json

D = json.load(open('outputs/camera_ready/CV_delta_r2.json'))
LAB = {
    'a_T4_C_beyond_controls': '(a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ens. on cartpole)',
    'b_T1_readout_prob_beyond_controls': '(b) Task 1: readout (probability) beyond the same controls',
    'b_T1_readout_logit_beyond_controls': '(b) Task 1: readout (logit) beyond the same controls',
    'c_T2_C_beyond_all_others_prob': '(c) Task 2: C_t beyond all others (6 standard + readout prob.)',
    'c_T2_C_beyond_all_others_logit': '(c) Task 2: C_t beyond all others (6 standard + readout logit)',
    'c_T2_C_beyond_6_standard': '(c) Task 2: C_t beyond the 6 standard signals',
    'c_T2_readout_prob_beyond_all_others': '(c) Task 2: readout (prob.) beyond all others (6 standard + C_t)',
    'c_T2_readout_logit_beyond_all_others': '(c) Task 2: readout (logit) beyond all others (6 standard + C_t)',
    'c_T2_readout_prob_beyond_6_standard': '(c) Task 2: readout (prob.) beyond the 6 standard signals',
    'c_T2_readout_logit_beyond_6_standard': '(c) Task 2: readout (logit) beyond the 6 standard signals',
}
f = lambda d: '%+.5f [%+.5f, %+.5f]' % (d['point'], d['lo'], d['hi'])
L = ['# Out-of-sample ΔR² (grouped 5-fold CV by trajectory), camera-ready, final rule', '',
     'Script: `run_cv_delta_r2.py`. Raw: `CV_delta_r2.json`.', '',
     '**Protocol:**',
     '- Data: the same 46,700 evaluation sites per task (100 episodes), target E_{t,10}.',
     '- Folds: the 100 episodes are split into 5 folds of 20 by a seeded permutation (seed 55). The same folds are used for every regression.',
     '- Fitting: OLS with intercept, fit on 4 folds, predicting the held-out fold.',
     '- ΔR²_cv = R²_cv(base + signal) − R²_cv(base), computed on the pooled out-of-fold predictions.',
     '- CI: trajectory bootstrap, 1,000 resamples (seed 7), over the fixed out-of-fold predictions. ΔR²_cv can be negative.',
     '- In-sample ΔR² (same columns, GramSet bootstrap) is shown alongside.',
     '- Definitions follow the final rule: refit probe; readout = probability, logit also shown; published γ and α; Section-6 C_t; Task 2\'s EMA-KL α. For Task 4 (walker, cheetah), γ and α come from the paper\'s procedure applied on probe-fit episodes.', '',
     '**Pre-registered reading:** a signal "adds out-of-sample predictive value" on a task only if its ΔR²_cv CI excludes zero (✓).', '']
for task in [t for t in ('cartpole', 'reacher', 'pendulum', 'walker', 'cheetah') if t in D]:
    m = D[task]['_meta']
    extra = f", γ = {m['gamma']}, EMARec α = {m['alpha_emarec']}" if 'gamma' in m else ''
    L += [f'## {task} ({m["n_sites"]} sites, {m["n_episodes"]} episodes{extra})', '',
          '| Row | in-sample ΔR² [95% CI] | out-of-sample ΔR²_cv [95% CI] | R²_cv base → full | adds OOS value |', '|---|---|---|---|---|']
    for k, lab in LAB.items():
        if k not in D[task]:
            continue
        r = D[task][k]
        L.append(f"| {lab} | {f(r['in_sample'])} | {f(r['cv'])} | {r['cv']['r2cv_base']:.4f} → {r['cv']['r2cv_full']:.4f} | "
                 f"{'✓' if r['cv']['adds_oos_value'] else '✗'} |")
    L.append('')
L += ['## Summary', '', '| Row | ' + ' | '.join(t for t in ('cartpole', 'reacher', 'pendulum', 'walker', 'cheetah') if t in D) + ' |',
      '|---|' + '---|' * sum(t in D for t in ('cartpole', 'reacher', 'pendulum', 'walker', 'cheetah'))]
for k, lab in LAB.items():
    cells = []
    for t in ('cartpole', 'reacher', 'pendulum', 'walker', 'cheetah'):
        if t not in D:
            continue
        cells.append(('✓' if D[t][k]['cv']['adds_oos_value'] else '✗') if k in D[t] else '—')
    L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
open('outputs/camera_ready/CV_delta_r2.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[-13:]))
