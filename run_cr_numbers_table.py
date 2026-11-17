#!/usr/bin/env python3.11
"""Rows of camera_ready_numbers.md from camera_ready_numbers.json (refit rule; no
hand-typed values). Also writes the rows + change flags into the JSON ('table').
Flag rules (relative to the published value): sign change; CI vs 0 crossing;
|z| crossing 2; AUROC crossing 0.5; a published qualitative claim no longer holding."""
import json

import numpy as np

PATH = 'outputs/camera_ready/camera_ready_numbers.json'
N = json.load(open(PATH))
E, X, PO, CZ, B = N['emergence'], N['external'], N['pomdp'], N['causal'], N['sae_retrained']
TASKS = ['cartpole', 'reacher', 'pendulum']
P13 = {'cartpole': ['primary', 'seed_1', 'seed_2', 'seed_3', 'seed_4'],
       'reacher': ['primary', 'reacher_seed1', 'reacher_seed2', 'reacher_seed3'],
       'pendulum': ['primary', 'pendulum_seed1', 'pendulum_seed2', 'pendulum_seed3']}
SEEDS = (4242, 1717, 2929, 5151, 8383)
rows = []


def ci(d, f='{:+.4f}'):
    return f'{f.format(d["point"])} [{f.format(d["lo"])}, {f.format(d["hi"])}]'


def rng(xs, f='{:.2f}'):
    return f'{f.format(min(xs))}–{f.format(max(xs))}'


def ms(xs):
    return f'{np.mean(xs):+.1f} ± {np.std(xs, ddof=1):.1f}'


def add(sec, q, pub, fin, note='', flag=''):
    rows.append(dict(section=sec, quantity=q, published=pub, final=fin, note=note, flag=flag))


ep = E['cartpole']['primary']
add('5.1', 'cartpole probe AUROC, clean', '0.872', f"{ep['auroc_clean']:.3f}")
add('5.1', 'cartpole probe AUROC, σ=0.1 noise', '0.807', f"{ep['auroc_noisy']:.3f}")
k = ep['kl_matched_pooled']
add('5.1', 'KL-matched groups: KL (nats), low- vs high-Rec', '22.9 vs 23.9', f"{k['kl_mean_low_rec']:.1f} vs {k['kl_mean_high_rec']:.1f}")
add('5.1', 'KL-matched groups: Rec, low vs high', '0.052 vs 0.471', f"{k['rec_mean_low_rec']:.3f} vs {k['rec_mean_high_rec']:.3f}")
add('5.1', 'cartpole KL-matched AUROC (clean+noisy / clean only)', '0.71 / 0.71',
    f"{k['auroc']:.2f} / {ep['kl_matched_clean']['auroc']:.2f}")
add('5.1', 'cartpole within-bin r(Rec, C_t)', '+0.39', f"{ep['within_bin_r_rec_C']:+.2f}")
for t in TASKS:
    e = E[t]['primary']
    d = e['r2_readout_C']['point'] - e['r2_readout_kl']['point']
    dl = e['logit']['r2_readout_C']['point'] - e['logit']['r2_readout_kl']['point']
    add('5.2/Fig1b', f'{t} R²(readout, C_t), primary', {'cartpole': '0.80'}.get(t, '—'), ci(e['r2_readout_C'], '{:.3f}'),
        f"logit {e['logit']['r2_readout_C']['point']:.3f}")
    pub_ahead = {'cartpole': 'C_t clearly ahead', 'reacher': 'about the same', 'pendulum': 'C_t clearly ahead'}[t]
    nb = sum(E[t][l]['r2_readout_C']['point'] > E[t][l]['r2_readout_kl']['point'] for l in P13[t])
    nbl = sum(E[t][l]['logit']['r2_readout_C']['point'] > E[t][l]['logit']['r2_readout_kl']['point'] for l in P13[t])
    flag = ''
    if t != 'reacher' and nb < len(P13[t]):
        flag = f'C_t not ahead on {len(P13[t]) - nb}/{len(P13[t])} models'
    add('5.2/Fig1b', f'{t} R²(readout, KL_t), primary; C_t − KL', {'cartpole': '0.52'}.get(t, '—') + f'; {pub_ahead}',
        ci(e['r2_readout_kl'], '{:.3f}') + f'; {d:+.3f} (C_t ahead on {nb}/{len(P13[t])} models)',
        f"logit: R²(KL) {e['logit']['r2_readout_kl']['point']:.3f}, C_t − KL {dl:+.3f}, C_t ahead on {nbl}/{len(P13[t])} models",
        flag)
rs = N['ridge_scramble']
for t, pz, ps in (('cartpole', 18.7, '0.69'), ('reacher', 39.0, '0.14'), ('pendulum', 51.4, '0.45')):
    r = rs[f'corrected/{t}']
    add('5.2/Fig1d', f'{t} scramble z, ridge h→C_t (scrambled R²)', f'{pz:+.1f} ({ps})',
        f"{r['z']:+.1f} ({r['scrambled_mean']:.2f}; real {r['real']:.2f})", flag='|z| < 2' if r['z'] < 2 else '')
tg = CZ['taskg_cartpole']
add('5.2', 'cartpole v-ablation, readout z (Task-G sites)', 'z ≈ −22, 100th pct',
    f"{tg['ablate_r0']['z']:+.1f} (pct {tg['ablate_r0']['percentile']:.0f})",
    f"logit z {tg['ablate_p0']['z']:+.1f}. Probability null DEGENERATE: random-direction ablations swing the saturated "
    f"probability (null sd {tg['ablate_r0']['null_std']:.2f}, v's change {tg['ablate_r0']['value']:+.2f}); the logit null is not",
    flag='|z| < 2 (probability; null degenerate)' if abs(tg['ablate_r0']['z']) < 2 else '')
for t, pub in (('cartpole', '0.5%'), ('reacher', '0.2%'), ('pendulum', '1.1–1.5%')):
    vals = [E[t][l]['geometry']['standardized_top10']['frac'] * 100 for l in P13[t]]
    g = E[t]['primary']['geometry']['standardized_top10']
    above = sum(x > g['random_mean'] * 100 for x in vals)
    add('5.3/T3', f'{t} share of ‖v‖² in top-10 PCs: primary (range over models); random mean / 5th pct',
        pub + '; random 3.9%', f"{g['frac'] * 100:.2f}% ({rng(vals)}%); random {g['random_mean'] * 100:.1f}% / {g['random_p5'] * 100:.1f}%",
        flag=f'above random mean on {above}/{len(vals)} models' if above else '')
g50 = E['cartpole']['primary']['geometry']['standardized_top50']
add('5.3', 'cartpole share in top-50 PCs; random', '9%; 19.5%', f"{g50['frac'] * 100:.1f}%; {g50['random_mean'] * 100:.1f}%")
add('8.3', 'cartpole share of v outside top-50; random', '91%; ≈80%', f"{(1 - g50['frac']) * 100:.0f}%; {(1 - g50['random_mean']) * 100:.0f}%")
for t in TASKS:
    a = E[t]['primary']['angle_top50']
    add('Fig1c (old stat.)', f'{t} mean angle of v to top-50 PCs', '≈88°', f"{a['mean_angle_deg']:.1f}° (random {a['random_mean_angle_deg']:.1f}°)")
for t, pub in (('cartpole', '0.80'), ('reacher', '0.26 (mean of 4)'), ('pendulum', '0.86 (mean of 4)')):
    v = [E[t][l]['r2_readout_C']['point'] for l in P13[t]]
    vl = [E[t][l]['logit']['r2_readout_C']['point'] for l in P13[t]]
    add('T3', f'{t} R²(readout, C_t): primary; mean (range) over models', pub,
        f"{E[t]['primary']['r2_readout_C']['point']:.2f}; {np.mean(v):.2f} ({rng(v)})", f'logit mean {np.mean(vl):.2f} ({rng(vl)})')
for t, pub, side in (('cartpole', '0.71', 1), ('reacher', '0.57–0.75', 1), ('pendulum', '0.32–0.50', -1)):
    v = [E[t][l]['kl_matched_pooled']['auroc'] for l in P13[t]]
    wrong = sum(np.sign(x - 0.5) != side for x in v)
    add('T3', f'{t} KL-matched AUROC: primary (range over models)', pub, f"{E[t]['primary']['kl_matched_pooled']['auroc']:.2f} ({rng(v)})",
        flag=f'other side of 0.5 on {wrong}/{len(v)} models' if wrong else '')
for t, pub in (('cartpole', 0.39), ('reacher', -0.09), ('pendulum', -0.12)):
    w = E[t]['primary']['within_bin_r_rec_C']
    add('T3', f'{t} within-bin r(Rec, C_t), primary', f'{pub:+.2f}', f'{w:+.2f}', flag='sign flip' if np.sign(w) != np.sign(pub) else '')
pe = E['pendulum']['primary']
add('5.4', 'pendulum r(KL_t, C_t)', '0.66', ci(pe['r_kl_C'], '{:.2f}'))
add('5.4', 'pendulum AUROC with groups relabelled by C_t', '0.53', f"{pe['relabel_by_C_auroc']:.2f}")
add('4.2', 'γ / EMARec α', '0.95, 0.70, 0.90 / 0.02, 0.5, 0.02', 'unchanged (fixed constants)')
for t, pub in (('cartpole', (0.0280, 0.0154, 0.0433)), ('reacher', (0.0182, 0.0083, 0.0309)), ('pendulum', (0.0019, 0.0001, 0.0051))):
    x = X[t]
    d = x['delta_r2_C']
    add('T4/Fig2b', f'{t} ΔR² of C_t, K=10', f'{pub[0]:+.4f} [{pub[1]:+.4f}, {pub[2]:+.4f}]', ci(d),
        f"{x['n_sites']} sites / {x['n_episodes']} episodes; controls {', '.join(x['controls'])}",
        flag='CI includes 0' if d['lo'] <= 0 else '')
    add('6.2', f'{t} baseline R² of controls', {'cartpole': '≈2/3', 'reacher': '≈1/8', 'pendulum': '≈2/5'}[t], ci(x['r2_controls'], '{:.3f}'))
    h = x['horizons']
    decays = h['20']['point'] < h['1']['point'] - 0.02
    pubtxt = {'cartpole': 'all > 0, ≈0.6, flat', 'reacher': 'all > 0, decays', 'pendulum': 'all > 0, decays; ≈0.17 at K=10'}[t]
    fl = 'CI includes 0' if any(h[K]['lo'] <= 0 for K in h) else ''
    if t == 'reacher' and not decays:
        fl = 'no longer decays (flat)'
    add('6.1/Fig2a', f'{t} r(C_t, E_K), K=1/5/10/20', pubtxt, ' / '.join(f"{h[K]['point']:+.2f}" for K in ('1', '5', '10', '20')),
        'K=10 CI [{:+.2f}, {:+.2f}]'.format(h['10']['lo'], h['10']['hi']), fl)
    hh = x['head_to_head']['C_beyond_emarec']
    add('6.3', f'{t} ΔR² adding C_t to EMARec alone', {'cartpole': '+0.177', 'reacher': '+0.049', 'pendulum': '+0.013'}[t], ci(hh),
        f"reverse (EMARec beyond C_t) {x['head_to_head']['emarec_beyond_C']['point']:+.4f}", 'CI includes 0' if hh['lo'] <= 0 else '')
    sz = x['scramble_section6_def']['z']
    pz = {'cartpole': 19.6, 'reacher': 21.5, 'pendulum': -11.7}[t]
    add('6.3/Fig2c', f'{t} external scramble z (Section-6 C_t, same sites as Fig 2a)', f'{pz:+.1f}', f'{sz:+.1f}',
        f"real r {x['scramble_section6_def']['real']:+.3f} vs scrambled {x['scramble_section6_def']['scrambled_mean']:+.3f}",
        'SIGN FLIP' if np.sign(sz) != np.sign(pz) else '')
for t, pub in (('cartpole', '162/6000; C_t 7.5 vs 8.9'), ('reacher', '91/6000; C_t 1.2 vs 1.7')):
    r = X[t]['reward_inflation']
    add('10.3', f'{t} reward-inflating sites; C_t drift vs other', pub, f"{r['n_drift']}/{r['n_sites']}; {r['C_drift']:.2f} vs {r['C_other']:.2f}",
        flag='' if r['C_drift'] < r['C_other'] else 'C_t not lower')
for key, pub, lab, claim in (('auroc', ('0.871±0.028', '0.951±0.013'), 'probe AUROC', 'up'),
                             ('r2_readout_C', ('0.761±0.035', '0.824±0.013'), 'R²(readout, C_t)', 'up'),
                             ('delta_r2_C', ('0.062±0.027', '0.009±0.003'), 'ΔR² of C_t for E (beyond KL, Rec)', 'down')):
    get = (lambda d: d[key]) if key == 'auroc' else (lambda d: d[key]['point'])
    fv = [get(PO[f'full_{s}']) for s in SEEDS]
    pv = [get(PO[f'partial_{s}']) for s in SEEDS]
    n = sum(p > f for f, p in zip(fv, pv)) if claim == 'up' else sum(p < f for f, p in zip(fv, pv))
    note = ''
    if key == 'r2_readout_C':
        lf = [PO[f'full_{s}']['logit']['r2_readout_C']['point'] for s in SEEDS]
        lp = [PO[f'partial_{s}']['logit']['r2_readout_C']['point'] for s in SEEDS]
        note = f"logit {np.mean(lf):.3f} / {np.mean(lp):.3f}; partial > full on {sum(b > a for a, b in zip(lf, lp))}/5"
    add('T5', f'partial obs, {lab}: full / partial (5 pairs)', f'{pub[0]} / {pub[1]}',
        f"{np.mean(fv):.3f}±{np.std(fv, ddof=1):.3f} / {np.mean(pv):.3f}±{np.std(pv, ddof=1):.3f}; "
        f"{'partial > full' if claim == 'up' else 'partial < full'} on {n}/5", note,
        f'"every pair" fails ({n}/5)' if (claim == 'up' and n < 5) else (f'direction holds on {n}/5' if n < 5 else ''))
allr, alll, alle = [], [], []
for t, pub in (('cartpole', '+12.9±1.6 / +12.2±1.4 / +0.2±0.4'), ('reacher', '+13.3±1.4 / +5.9±0.7 / +0.3±0.4'),
               ('pendulum', '+15.6±3.0 / +11.4±0.6 / +0.6±0.8')):
    z = lambda s: [CZ[t][l][s]['z'] for l in P13[t]]
    r0, r10, l0, l10, es = z('slope_r0'), z('slope_r10'), z('slope_p0'), z('slope_p10'), z('slope_es')
    allr += r0 + r10; alll += l0 + l10; alle += es
    fl = []
    if min(r0) < 2: fl.append(f'k=0 readout z < 2 on {sum(x < 2 for x in r0)}/{len(r0)}')
    if min(r10) < 2: fl.append(f'k=10 readout z < 2 on {sum(x < 2 for x in r10)}/{len(r10)}')
    if max(abs(x) for x in es) >= 2: fl.append('|E z| ≥ 2')
    add('T6', f'{t} steering z: readout k=0 / k=10 / E (n={len(r0)})', pub, f'{ms(r0)} / {ms(r10)} / {ms(es)}',
        f"k=10 per model: {', '.join(f'{x:+.1f}' for x in r10)}; logit k=0 / k=10: {ms(l0)} / {ms(l10)} (k=10 per model: "
        f"{', '.join(f'{x:+.1f}' for x in l10)})", '; '.join(fl))
add('8.1', 'readout z range over 13 models (k=0 and k=10)', '+5.2..+19.2', f'{min(allr):+.1f}..{max(allr):+.1f}',
    f'logit {min(alll):+.1f}..{max(alll):+.1f}', 'minimum < 2' if min(allr) < 2 else '')
add('8.1', 'max |E z| over 13 models', '1.48', f'{max(abs(z) for z in alle):.2f}', flag='≥ 2' if max(abs(z) for z in alle) >= 2 else '')
mz = max(abs(CZ[f'{t}_noisy']['slope_es']['z']) for t in TASKS)
add('8.2', 'steering E null under σ=0.1 noise, max |z| (3 primary models)', '1.43', f'{mz:.2f}', flag='≥ 2' if mz >= 2 else '')
for c, pub in (('full', '−0.37±0.36'), ('partial', '+0.13±0.22')):
    z = [CZ['pomdp'][f'{c}_{s}']['ablate_es']['z'] for s in SEEDS]
    add('8.2', f'v-ablation E z, {c} observation (5 models)', pub, f'{np.mean(z):+.2f}±{np.std(z, ddof=1):.2f}',
        f"max |z| {max(abs(x) for x in z):.2f}", '|z| ≥ 2 on a model' if max(abs(x) for x in z) >= 2 else '')
sh = {k: [r['shares'][k] * 100 for t in TASKS for r in B[t]['runs']] for k in ('top1', 'top4', 'top10')}
add('T7', 'share of v: top-1 / top-4 / top-10 atoms (3 tasks × 3 SAEs)', '0.3–0.5% / 0.9–1.2% / ≤5%',
    f"{rng(sh['top1'])}% / {rng(sh['top4'])}% / {rng(sh['top10'])}%")
rc = [r['random_shares']['recon_cos']['mean'] for t in TASKS for r in B[t]['runs']]
add('9.1', 'random-vector reconstruction cosine', '≈0.97', f'{np.mean(rc):.3f}')
pubT8 = {'cartpole': ('#156', '+0.499', 'positive', 10.3, -5.0), 'reacher': ('#612', '0.02–0.06', '+0.21–0.29', 2.67, None),
         'pendulum': ('#310', '—', '—', 0.0, 0.0)}
for t in TASKS:
    a, rk, re_, ze, zr = pubT8[t]
    for r in B[t]['runs']:
        fl = []
        e_sig, r_sig = r['ablate_E']['z'] > 2, abs(r['ablate_readout_prob']['z']) > 2
        if t == 'pendulum' and e_sig: fl.append('E effect now clears z > 2')
        if t != 'pendulum' and not e_sig: fl.append('E effect < 2')
        if t == 'cartpole' and not r_sig: fl.append('readout effect no longer clears |z| > 2')
        if t == 'reacher' and abs(r['heldout_r_kl']) > 0.1: fl.append('not KL-independent')
        add('T8/T9', f"{t} SAE run {r['sae_run']}: atom #{r['atom']} — r(act,KL) / r(act,E); ablation z E / readout; cos(atom, v)",
            f'{a}: {rk} / {re_}; {ze:+.2f} / ' + ('n.s.' if zr is None else f'{zr:+.1f}') if r['sae_run'] == 0 else '(new run)',
            f"{r['heldout_r_kl']:+.3f} / {r['heldout_r_E']:+.3f}; {r['ablate_E']['z']:+.2f} / {r['ablate_readout_prob']['z']:+.2f}; {r['cos_atom_v']:+.3f}",
            f"readout logit z {r['ablate_readout_logit']['z']:+.2f}; probability null sd {B[t]['null_readout_prob_sd']:.3f}",
            '; '.join(fl))
print('| § | Quantity | Published | Final | Note | Flag |\n|---|---|---|---|---|---|')
for r in rows:
    print('| ' + ' | '.join(str(r[k]).replace('|', '/') for k in ('section', 'quantity', 'published', 'final', 'note', 'flag')) + ' |')
N['table'] = rows
json.dump(N, open(PATH, 'w'), indent=1)
