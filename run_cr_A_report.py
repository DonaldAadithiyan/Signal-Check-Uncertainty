#!/usr/bin/env python3.11
"""Camera-ready Part A, step 5: side-by-side table (published vs corrected) and the
pre-registered stop-rule evaluation. Reads outputs/camera_ready/A_raw/*.json; writes
outputs/camera_ready/A_consistency.{json,md}.
"""
import glob
import json
import os

import numpy as np

RAW = 'outputs/camera_ready/A_raw'
OUT = 'outputs/camera_ready'
TASKS = ['cartpole', 'reacher', 'pendulum']


def load(p):
    with open(p) as f:
        return json.load(f)


def em(task, label='primary'):
    return load(f'{RAW}/emergence_{task}_{label}.json')


def labels(task):
    return sorted(os.path.basename(p)[len(f'emergence_{task}_'):-5]
                  for p in glob.glob(f'{RAW}/emergence_{task}_*.json'))


def wm_labels(task):
    return [l for l in labels(task)]


def ci(d):
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def rng_str(xs, fmt='{:.2f}'):
    xs = [x for x in xs if x is not None]
    return (fmt.format(min(xs)) + '–' + fmt.format(max(xs))) if len(xs) > 1 else fmt.format(xs[0])


def msd(xs, fmt='{:+.1f}'):
    return f"{fmt.format(np.mean(xs))} ± {np.std(xs, ddof=1):.1f}" if len(xs) > 1 else fmt.format(xs[0])


rows = []      # (id, section, quantity, published, corrected, diff, flag, note)


def add(i, sec, q, pub, cor, diff='', flag='', note=''):
    rows.append(dict(id=i, section=sec, quantity=q, published=pub, corrected=cor, diff=diff, flag=flag, note=note))


def num_diff(a, b, fmt='{:+.3f}'):
    return fmt.format(b - a)


def main():
    E = {t: em(t) for t in TASKS}
    X = {t: E[t]['external'] for t in TASKS}
    M = {t: {l: load(f'{RAW}/emergence_{t}_{l}.json')['emergence'] for l in wm_labels(t)} for t in TASKS}
    stop = []

    # ── Sec. 5.1 ──
    e = E['cartpole']['emergence']
    add('5.1a', '5.1', 'cartpole probe AUROC, clean held-out', '0.872', f"{e['auroc_clean']:.3f}",
        num_diff(0.872, e['auroc_clean']), note='evaluation episodes (paper: 10,000 fresh held-out states)')
    add('5.1b', '5.1', 'cartpole probe AUROC, sigma=0.1 noise', '0.807', f"{e['auroc_noisy']:.3f}",
        num_diff(0.807, e['auroc_noisy']))
    km = e['kl_matched_pooled']
    add('5.1c', '5.1', 'KL-matched groups: KL low-Rec vs high-Rec (nats)', '22.9 vs 23.9',
        f"{km['kl_mean_low_rec']:.1f} vs {km['kl_mean_high_rec']:.1f}")
    add('5.1d', '5.1', 'KL-matched groups: Rec low vs high', '0.052 vs 0.471',
        f"{km['rec_mean_low_rec']:.3f} vs {km['rec_mean_high_rec']:.3f}")
    add('5.1e', '5.1', 'cartpole KL-matched AUROC (pooled clean+noisy)', '0.71', f"{km['auroc']:.3f}",
        num_diff(0.71, km['auroc']), 'SIDE OF 0.5 CHANGES' if (km['auroc'] - 0.5) * (0.71 - 0.5) < 0 else '')
    add('5.1f', '5.1', 'cartpole KL-matched AUROC (clean only)', '0.71', f"{e['kl_matched_clean']['auroc']:.3f}",
        num_diff(0.71, e['kl_matched_clean']['auroc']))
    # ── Sec. 5.2 ──
    for t in TASKS:
        e = E[t]['emergence']
        add(f'5.2a-{t}', '5.2/Fig1b', f'{t} R²(readout, C_t) primary (gamma_cal={e["gamma_cal"]})',
            {'cartpole': '0.80', 'reacher': '—', 'pendulum': '—'}[t], ci(e['r2_readout_C']),
            note=f"at paper gamma: {e['r2_readout_C_paper_gamma']:.3f}")
        add(f'5.2a2-{t}', '5.2/Fig1b', f'{t} R²(projection v·h, C_t) primary (gamma_cal_proj={e["gamma_cal_proj"]})',
            'not published (paper used probability)', ci(e['r2_proj_C']),
            note=f"at gamma_cal(prob): {e['r2_proj_C']['point']:.3f}; R²(proj, KL) {e['r2_proj_kl']['point']:.3f}")
        add(f'5.2b-{t}', '5.2/Fig1b', f'{t} R²(readout, KL_t) primary',
            {'cartpole': '0.52', 'reacher': '—', 'pendulum': '—'}[t], ci(e['r2_readout_kl']))
        ahead = e['r2_readout_C']['point'] - e['r2_readout_kl']['point']
        pub_q = {'cartpole': 'C_t clearly ahead', 'pendulum': 'C_t clearly ahead', 'reacher': 'about the same'}[t]
        add(f'5.2c-{t}', '5.2/Fig1b', f'{t} R²(ro,C) − R²(ro,KL), primary', pub_q, f'{ahead:+.3f}',
            flag=('CHANGES IN KIND' if t != 'reacher' and ahead <= 0 else ''))
        if t != 'reacher' and ahead <= 0:
            stop.append(f'{t}: C_t no longer ahead of KL in explaining the readout')
        rs = e['ridge_scramble']
        add(f'5.2d-{t}', '5.2/Fig1d', f'{t} scramble z (ridge h→C_t)',
            {'cartpole': '+18.7', 'reacher': '+39.0', 'pendulum': '+51.4'}[t], f"{rs['z']:+.1f}",
            flag='SIGN/SIG CHANGE' if rs['z'] < 2 else '')
        if rs['z'] < 2:
            stop.append(f'{t}: real-order C_t no longer fit better than scrambled (z={rs["z"]:+.1f})')
        add(f'5.2e-{t}', '5.2', f'{t} scrambled R²(h→C_t)', {'cartpole': '0.69', 'reacher': '0.14', 'pendulum': '0.45'}[t],
            f"{rs['scrambled_mean']:.2f} (real {rs['real']:.2f})")
    tg = f'{RAW}/causal_taskg_cartpole_primary.json'
    if os.path.exists(tg):
        G = load(tg)['null50']
        g, gp_ = G['ablate_r0'], G['ablate_p0']
        add('5.2f', '5.2', 'cartpole v-ablation readout change at k=0, z vs 50-dir null (Task-G sites)', 'z≈−22, 100th pct',
            f"prob z={g['z']:+.1f} (pct {g['percentile']:.0f}); proj z={gp_['z']:+.1f} (pct {gp_['percentile']:.0f})",
            flag='SIG CHANGE' if g['z'] > -2 else '')
    # ── Sec. 5.3 / Table 3 ──
    for t in TASKS:
        g = E[t]['emergence']['geometry']['raw_top10']
        g50 = E[t]['emergence']['geometry']['raw_top50']
        pub10 = {'cartpole': '0.5%', 'reacher': '0.2%', 'pendulum': '1.1–1.5%'}[t]
        vals = [M[t][l]['geometry']['raw_top10']['frac'] * 100 for l in M[t]]
        add(f'5.3a-{t}', '5.3/T3', f'{t} share of ‖v‖² in top-10 PCs (primary; range over models)', pub10,
            f"{g['frac'] * 100:.2f}% ({rng_str(vals)}%); random {g['random_mean'] * 100:.1f}%, pct {g['percentile_vs_random']:.0f}",
            flag='ABOVE RANDOM MEAN' if g['frac'] > g['random_mean'] else '')
        add(f'5.3b-{t}', '5.3', f'{t} share in top-50 PCs', {'cartpole': '9%', 'reacher': '—', 'pendulum': '—'}[t],
            f"{g50['frac'] * 100:.1f}%; random {g50['random_mean'] * 100:.1f}%, pct {g50['percentile_vs_random']:.0f}")
        a = E[t]['emergence']['angle_top50']
        add(f'5.3c-{t}', 'Fig1c', f'{t} mean angle to top-50 PCs (old statistic)', '≈88–89°',
            f"{a['mean_angle_deg']:.1f}° (random {a['random_mean_angle_deg']:.1f}°)")
    for t in TASKS:
        r2s = [M[t][l]['r2_readout_C']['point'] for l in M[t]]
        pub = {'cartpole': '0.80', 'reacher': '0.26 (mean of 4)', 'pendulum': '0.86 (mean of 4)'}[t]
        cor = (f"{M[t]['primary']['r2_readout_C']['point']:.2f} (primary); mean {np.mean(r2s):.2f}, range {rng_str(r2s)}")
        add(f'T3a-{t}', 'T3', f'{t} R²(readout prob, C_t)', pub, cor)
        r2p = [M[t][l]['r2_proj_C']['point'] for l in M[t]]
        add(f'T3a2-{t}', 'T3', f'{t} R²(projection v·h, C_t)', 'not published',
            f"{M[t]['primary']['r2_proj_C']['point']:.2f} (primary); mean {np.mean(r2p):.2f}, range {rng_str(r2p)}")
        kms = [M[t][l]['kl_matched_pooled']['auroc'] for l in M[t]]
        pubk = {'cartpole': '0.71', 'reacher': '0.57–0.75', 'pendulum': '0.32–0.50'}[t]
        pub_side = {'cartpole': 1, 'reacher': 1, 'pendulum': -1}[t]
        side_change = any(np.sign(k - 0.5) != pub_side for k in kms) if t != 'cartpole' else \
            np.sign(M[t]['primary']['kl_matched_pooled']['auroc'] - 0.5) != pub_side
        add(f'T3b-{t}', 'T3', f'{t} KL-matched AUROC (primary; range over models)', pubk,
            f"{M[t]['primary']['kl_matched_pooled']['auroc']:.2f} ({rng_str(kms)})",
            flag='RANGE/SIDE OF 0.5 CHANGES' if side_change else '')
        wb = M[t]['primary']['within_bin_r_rec_C']['mean']
        pubw = {'cartpole': +0.39, 'reacher': -0.09, 'pendulum': -0.12}[t]
        add(f'T3c-{t}', 'T3', f'{t} within-bin r(Rec, C_t), primary', f'{pubw:+.2f}', f'{wb:+.2f}',
            num_diff(pubw, wb, '{:+.2f}'), 'SIGN FLIP' if np.sign(wb) != np.sign(pubw) else '')
    p = E['pendulum']['emergence']
    add('5.4a', '5.4', 'pendulum r(KL_t, C_t)', '0.66', f"{p['r_kl_C']['point']:.2f}")
    add('5.4b', '5.4', 'pendulum AUROC, groups relabelled by C_t', '0.53', f"{p['relabel_by_C_auroc']:.2f}")
    # ── Sec. 4.2 parameters ──
    for t in TASKS:
        add(f'4.2-{t}', '4.2', f'{t} gamma / EMARec alpha',
            {'cartpole': '0.95 / 0.02', 'reacher': '0.70 / 0.5', 'pendulum': '0.90 / 0.02'}[t],
            f"{E[t]['emergence']['gamma_cal']} / {X[t]['alpha_emarec_paper_rule']}",
            note='both selected on calibration episodes (paper: on the reported split / training log)')
    # ── Sec. 6 ──
    pubT4 = {'cartpole': (0.0280, 0.0154, 0.0433), 'reacher': (0.0182, 0.0083, 0.0309), 'pendulum': (0.0019, 0.0001, 0.0051)}
    for t in TASKS:
        for nm, lab in (('C_paperdef', 'paper definition: per-trajectory median, gamma=paper'),
                        ('C_cal', 'calibration median, gamma_cal')):
            d = X[t][f'delta_r2_{nm}']
            pp = pubT4[t]
            fl = ''
            if np.sign(d['point']) != np.sign(pp[0]):
                fl = 'SIGN FLIP'
            elif d['lo'] <= 0 <= d['hi']:
                fl = 'CI NOW INCLUDES 0'
            if fl:
                stop.append(f'Table 4 {t} ({nm}): {fl} ({ci(d)})')
            add(f'T4-{t}-{nm}', 'T4', f'{t} ΔR² of C_t (K=10), {lab}',
                f'{pp[0]:+.4f} [{pp[1]:+.4f}, {pp[2]:+.4f}]', ci(d), num_diff(pp[0], d['point'], '{:+.4f}'), fl,
                note=f"{X[t]['n_sites']} sites / {X[t]['n_episodes']} eps; controls {X[t]['controls']}")
        add(f'6.2b-{t}', '6.2', f'{t} baseline R² of controls', {'cartpole': '≈2/3', 'reacher': '≈1/8', 'pendulum': '≈2/5'}[t],
            f"{X[t]['r2_baseline']['point']:.3f}")
        hz = X[t]['horizons']
        rs = {K: hz[K]['C_paperdef']['point'] for K in hz}
        add(f'6.1-{t}', '6.1/Fig2a', f'{t} r(C_t, E_K), K=1/5/10/20 (paper def.)',
            {'cartpole': 'all >0, ≈0.6, flat', 'reacher': 'all >0, decays', 'pendulum': 'all >0, decays (≈0.17 at K=10)'}[t],
            ' / '.join(f'{rs[K]:+.2f}' for K in rs) + '  (cal: ' + ' / '.join(f"{hz[K]['C_cal']['point']:+.2f}" for K in hz) + ')',
            flag='NON-POSITIVE r' if any(hz[K]['C_paperdef']['lo'] <= 0 for K in hz) else '')
        h2h = X[t]['head_to_head_emarec']
        pubh = {'cartpole': 0.177, 'reacher': 0.049, 'pendulum': 0.013}[t]
        add(f'6.3a-{t}', '6.3', f'{t} ΔR² adding C_t to EMARec alone', f'{pubh:+.3f}',
            f"{ci(h2h['C_paperdef'])} (cal {h2h['C_cal']['point']:+.3f})",
            flag='CI INCLUDES 0' if h2h['C_paperdef']['lo'] <= 0 else '')
        sx = X[t]['scramble_external']['C_paperdef']
        pubz = {'cartpole': 19.6, 'reacher': 21.5, 'pendulum': -11.7}[t]
        fl = 'SIGN FLIP' if np.sign(sx['z']) != np.sign(pubz) else ('|z|<2' if abs(sx['z']) < 2 else '')
        if fl:
            stop.append(f'Fig 2c {t}: external scramble z {pubz:+.1f} -> {sx["z"]:+.1f} ({fl})')
        add(f'6.3b-{t}', '6.3/Fig2c', f'{t} external scramble z (r(C_t,E10) real vs scrambled)', f'{pubz:+.1f}',
            f"{sx['z']:+.1f} (cal {X[t]['scramble_external']['C_cal']['z']:+.1f})", flag=fl)
    # ── Sec. 7 / Table 5 ──
    pairs = []
    for s in (4242, 1717, 2929, 5151, 8383):
        try:
            f_ = load(f'{RAW}/emergence_cartpole_pomdp_full_{s}.json')
            p_ = load(f'{RAW}/emergence_cartpole_pomdp_partial_{s}.json')
            pairs.append((s, f_, p_))
        except FileNotFoundError:
            pass
    if pairs:
        def col(k, cond):
            out = []
            for _, f_, p_ in pairs:
                d = f_ if cond == 'full' else p_
                if k == 'auroc':
                    out.append(d['emergence']['auroc_clean'])
                elif k == 'r2':
                    out.append(d['emergence']['r2_readout_C']['point'])
                elif k == 'r2proj':
                    out.append(d['emergence']['r2_proj_C']['point'])
                else:
                    out.append(d['external']['delta_r2_C_cal']['point'])
            return out
        for k, pub in (('auroc', ('0.871±0.028', '0.951±0.013')), ('r2', ('0.761±0.035', '0.824±0.013')),
                       ('dr2', ('0.062±0.027', '0.009±0.003')), ('r2proj', ('not published', 'not published'))):
            fv, pv = col(k, 'full'), col(k, 'partial')
            fmt = lambda x: f"{np.mean(x):.3f}±{np.std(x, ddof=1):.3f}" if len(x) > 1 else f'{x[0]:.3f}'
            n_str = sum(p > f for f, p in zip(fv, pv))
            add(f'T5-{k}', 'T5', f'partial-obs {k}: full / partial (n={len(fv)} pairs)', f'{pub[0]} / {pub[1]}',
                f'{fmt(fv)} / {fmt(pv)}; partial>full on {n_str}/{len(fv)} pairs')
            if k in ('auroc', 'r2', 'r2proj') and n_str < len(fv):
                stop.append(f'Table 5 {k}: partial > full on only {n_str}/{len(fv)} pairs (paper: every pair)')
                rows[-1]['flag'] = 'NOT EVERY PAIR'
            if k == 'dr2' and np.mean(pv) >= np.mean(fv):
                stop.append('Table 5: incremental value not smaller under partial observation')
                rows[-1]['flag'] = 'DIRECTION CHANGES'
    # ── Table 6 / Sec. 8 ──
    allz_r, allz_p, allz_e = [], [], []
    for t in TASKS:
        st = [load(p) for p in sorted(glob.glob(f'{RAW}/causal_steer_{t}_*.json'))]
        if not st:
            continue
        z = lambda k: [s_['null50'][k]['z'] for s_ in st]
        r0, r10, p0, p10, ez = z('slope_r0'), z('slope_r10'), z('slope_p0'), z('slope_p10'), z('slope_es')
        allz_r += r0 + r10
        allz_p += p0 + p10
        allz_e += ez
        pub = {'cartpole': '+12.9±1.6 / +12.2±1.4 / +0.2±0.4', 'reacher': '+13.3±1.4 / +5.9±0.7 / +0.3±0.4',
               'pendulum': '+15.6±3.0 / +11.4±0.6 / +0.6±0.8'}[t]
        for defn, a0, a10 in (('probability', r0, r10), ('projection', p0, p10)):
            fl = []
            if min(a0) < 2:
                fl.append(f'k=0 z<2 on {sum(x < 2 for x in a0)}/{len(a0)} models')
            if min(a10) < 2:
                fl.append(f'k=10 z<2 on {sum(x < 2 for x in a10)}/{len(a10)} models')
            if max(abs(x) for x in ez) >= 2:
                fl.append(f'|E z|>=2 on {sum(abs(x) >= 2 for x in ez)}/{len(ez)} models')
            add(f'T6-{t}-{defn}', 'T6', f'{t} steering z, readout={defn}: k=0 / k=10 / E (n={len(st)} models)',
                pub if defn == 'probability' else 'not published',
                f'{msd(a0)} / {msd(a10)} / {msd(ez)}; per model k=10: ' + ', '.join(f'{x:+.1f}' for x in a10),
                flag='; '.join(fl))
            if fl and defn == 'probability':
                stop.append(f'Table 6 {t} ({defn}): ' + '; '.join(fl))
    if allz_r:
        add('8.1a', '8.1', 'range of readout z over models (k=0 and k=10), probability', '+5.2..+19.2',
            f'{min(allz_r):+.1f}..{max(allz_r):+.1f}', flag='MIN < 2' if min(allz_r) < 2 else '')
        add('8.1a2', '8.1', 'range of readout z over models (k=0 and k=10), projection', 'not published',
            f'{min(allz_p):+.1f}..{max(allz_p):+.1f}')
        add('8.1b', '8.1', 'max |E z| over models', '1.48', f'{max(abs(z_) for z_ in allz_e):.2f}',
            flag='>=2' if max(abs(z_) for z_ in allz_e) >= 2 else '')
    nz = [load(p) for p in sorted(glob.glob(f'{RAW}/causal_noisy_*.json'))]
    if nz:
        mz = max(abs(s['null50']['slope_es']['z']) for s in nz)
        add('8.2b', '8.2', f'steering E null under sigma=0.1 noise, max |z| (n={len(nz)} primary models)', '1.43', f'{mz:.2f}',
            flag='|z|>=2' if mz >= 2 else '')
        if mz >= 2:
            stop.append(f'Sec 8.2: noisy steering E |z| = {mz:.2f}')
    pm = [load(p) for p in sorted(glob.glob(f'{RAW}/causal_pomdp_*.json'))]
    if pm:
        for cond, pub in (('full', '−0.37±0.36'), ('partial', '+0.13±0.22')):
            zs = [s['null50']['ablate_es']['z'] for s in pm if s['label'].startswith(cond)]
            if zs:
                add(f'8.2a-{cond}', '8.2', f'v-ablation E z, {cond} obs (n={len(zs)} models)', pub,
                    f"{np.mean(zs):+.2f}±{np.std(zs, ddof=1) if len(zs) > 1 else 0:.2f}",
                    flag='|z|>=2 on some model' if max(abs(z) for z in zs) >= 2 else '')
    for t in TASKS:
        g = E[t]['emergence']['geometry']['raw_top50']
        if t == 'cartpole':
            add('8.3', '8.3', 'cartpole share of v outside top-50 PCs; random', '91%; ≈80%',
                f"{(1 - g['frac']) * 100:.0f}%; {(1 - g['random_mean']) * 100:.0f}%")
    # ── Tables 7-9 (SAE) ──
    for var in ('published', 'retrained'):
        p = f'{RAW}/sae_{var}.json'
        if not os.path.exists(p):
            continue
        S = load(p)['tasks']
        t1 = [d['top1'] * 100 for t in S for d in S[t]['decomposition']]
        t4 = [d['top4'] * 100 for t in S for d in S[t]['decomposition']]
        t10 = [d['top10'] * 100 for t in S for d in S[t]['decomposition']]
        rc = [d['recon_cos']['mean'] for t in S for d in S[t]['random_decomposition']]
        add(f'T7-{var}', 'T7', f'[{var} SAE] share of v: top-1 / top-4 / top-10', '0.3–0.5% / 0.9–1.2% / ≤5%',
            f'{rng_str(t1)}% / {rng_str(t4)}% / {rng_str(t10)}%; random-vector recon cos {np.mean(rc):.2f}')
        for t in TASKS:
            a = S[t]['ablation']
            z_e, z_r, z_p = a['null50']['ablate_es']['z'], a['null50']['ablate_r0']['z'], a['null50']['ablate_p0']['z']
            sel = [x['atom'] for x in S[t]['selection']]
            pub = {'cartpole': '#156: E z +10.3, readout z −5.0', 'reacher': '#612: E z +2.67, readout n.s.',
                   'pendulum': '#310: |z|<1 both'}[t]
            fl = ''
            if t == 'cartpole' and not (z_e > 2 and z_r < -2):
                fl = 'NO LONGER CLEARS NULL (E and/or readout)'
            if t == 'reacher' and not (z_e > 2 and abs(z_r) < 2):
                fl = 'PATTERN CHANGES'
            if t == 'pendulum' and (abs(z_e) >= 2 or abs(z_r) >= 2):
                fl = 'NOW CLEARS NULL'
            add(f'T8-{var}-{t}', 'T8/T9', f'[{var} SAE] {t} atom #{a["atom"]} ablation z: E / readout k=0 prob / proj',
                pub, f'{z_e:+.2f} / {z_r:+.2f} / {z_p:+.2f}  (cos(atom,v)={a["cos_atom_v"]:+.3f}; reselected per run: {sel})', flag=fl)
            if fl and t != 'pendulum':
                stop.append(f'[{var} SAE] {t} atom #{a["atom"]}: {fl} (E z {z_e:+.2f}, readout z {z_r:+.2f})')
            if var == 'published':
                pa = S[t]['published_atom']
                add(f'T8c-{t}', 'T8', f'{t} published atom #{pa["atom"]} r(act,KL) / r(act,E), held-out half',
                    {'cartpole': '+0.499 / positive', 'reacher': '0.02–0.06 / +0.21–0.29', 'pendulum': '— / —'}[t],
                    (f"{pa['heldout']['r_kl']:+.3f} / {pa['heldout']['r_E']:+.3f}" if pa['heldout']['r_kl'] is not None
                     else 'atom inactive') + f"; reselected by paper rule: {pa['selected_again_run0']}")
            rk = [x['heldout']['r_kl'] for x in S[t]['selection'] if x['heldout']['r_kl'] is not None]
            re = [x['heldout']['r_E'] for x in S[t]['selection'] if x['heldout']['r_E'] is not None]
            if t == 'reacher':
                add(f'T8d-{var}', 'T8/9.2', f'[{var} SAE] reacher atom reselected per run: r(KL) / r(E) held-out',
                    '0.02–0.06 / +0.21–0.29', f'{rng_str(rk, "{:+.2f}")} / {rng_str(re, "{:+.2f}")}')
    # ── Sec. 10.3 ──
    for t in ('cartpole', 'reacher'):
        ri = X[t]['reward_inflation']
        pub = {'cartpole': '162/6000; C_t 7.5 vs 8.9', 'reacher': '91/6000; C_t 1.2 vs 1.7'}[t]
        lower = ri['C_paperdef_mean_drift'] is not None and ri['C_paperdef_mean_drift'] < ri['C_paperdef_mean_other']
        add(f'10.3-{t}', '10.3', f'{t} reward-inflating sites; C_t drift vs other (paper def.)', pub,
            f"{ri['n_drift']}/{ri['n_sites']}; C_t {ri['C_paperdef_mean_drift']:.2f} vs {ri['C_paperdef_mean_other']:.2f}",
            flag='' if lower else 'C_t NOT LOWER')
    add('10.2', '10.2', 'Kalman simulation (no world model involved)', 'as published', 'unaffected — not recomputed',
        note='pure simulation; does not use any logged state')

    res = dict(rows=rows, stop_rule_triggers=stop, stop=bool(stop))
    with open(f'{OUT}/A_consistency.json', 'w') as f:
        json.dump(res, f, indent=1)
    md = ['# Part A — side-by-side table (auto-generated by run_cr_A_report.py)', '',
          '| id | section | quantity | published | corrected | diff | flag | note |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append('| ' + ' | '.join(str(r[k]).replace('|', '/') for k in
                                    ('id', 'section', 'quantity', 'published', 'corrected', 'diff', 'flag', 'note')) + ' |')
    md += ['', '## Stop-rule triggers', ''] + ([f'- {s}' for s in stop] or ['- none'])
    with open(f'{OUT}/A_table_autogen.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'{len(rows)} rows; stop-rule triggers: {len(stop)}')
    for s in stop:
        print('  STOP:', s)


if __name__ == '__main__':
    main()
