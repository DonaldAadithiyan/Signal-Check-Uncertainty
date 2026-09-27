#!/usr/bin/env python3
"""Sprint P1 (emergence + C_t robustness), P2 (external validity, Table 3) and
P6 (KL autocorrelation) on the corrected pipeline (src/rerun/pipeline.py).

Protocol (fixed before running):
  * States: frozen final models, correct action convention, episode splits
    probe (100) / calibration (50) / evaluation (100) with fixed seeds.
  * Every threshold (median, 60th, 70th percentile of KL), the continuous
    variant's standardisation, gamma, and the EMA alphas come from the
    calibration episodes. Every reported number is on the evaluation episodes.
  * Probe: logistic regression on probe-fit h_t, label 1[KL_t > calibration median].
    Readout = probe probability.
  * gamma rule (all variants): maximise R^2(readout, variant) on calibration
    over GAMMA_GRID. The full grid is also reported on evaluation.
  * EMARec / EMAKL alpha: maximise R^2(E^state_10, EMA) on calibration.
  * CIs: 1,000 episode-level bootstrap resamples (seed pipeline.BOOT_SEED).
  * Scrambling: 20 scrambles; the current-step flag is held fixed and only the
    past flags are permuted within each episode.

Usage: python run_sprint_p1p2.py [--smoke] [--tasks ...] [--primary-only]
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.rerun import pipeline as P
from src.rerun.stats import GramSet, pearson_ci, acf_within

N_SCRAMBLE = 20
SCRAMBLE_SEED = 20_20
KMAX = max(P.HORIZONS)


# ─── states (cached arrays) ──────────────────────────────────────────────────

def states_for(task, label, path, smoke, need_estate):
    d = os.path.join(P.out_dir(smoke), 'arrays')
    os.makedirs(d, exist_ok=True)
    model = None
    out = {}
    for split in ('probe', 'calibration', 'evaluation'):
        f = os.path.join(d, f'{task}_{label}_{split}.npz')
        want_es = need_estate and split != 'probe'
        if os.path.exists(f):
            arr = dict(np.load(f))
            if not want_es or 'estate_10' in arr:
                out[split] = arr
                continue
        if model is None:
            model = P.get_model(task, path, smoke)
        ep = P.episodes_cached(task, split)
        st = P.log_states(model, ep, P.TORCH_SEEDS[split])
        if want_es:
            for K, v in P.estate_all_sites(model, ep, st, P.TORCH_SEEDS[split]).items():
                st[f'estate_{K}'] = v
        st['traj_id'] = np.repeat(ep['seeds'][:, None], st['kl'].shape[1], 1)
        st['t'] = np.tile(np.arange(st['kl'].shape[1]), (st['kl'].shape[0], 1))
        st['obs'], st['act'] = ep['obs'], ep['act']
        np.savez(f, **st)
        out[split] = st
    return out, model


# ─── variants ────────────────────────────────────────────────────────────────

def calib_params(kl_cal):
    v = kl_cal[:, P.MIN_T:]
    return dict(median=float(np.median(v)), p60=float(np.percentile(v, 60)),
                p70=float(np.percentile(v, 70)), mu=float(v.mean()), sd=float(v.std()))


VARIANTS = ['C', 'C_past', 'C_p60', 'C_p60_past', 'C_p70', 'C_p70_past', 'Ccont', 'Ccont_past']


def variant(name, kl, cp, gamma):
    past = name.endswith('_past')
    base = name[:-5] if past else name
    if base == 'Ccont':
        return P.compute_ct_continuous(kl, cp['mu'], cp['sd'], gamma, include_current=not past)
    thr = {'C': cp['median'], 'C_p60': cp['p60'], 'C_p70': cp['p70']}[base]
    return P.compute_ct(kl, thr, gamma, include_current=not past)


def scrambled(kl, thresh, gamma, rng, include_current):
    flags = (kl > thresh).astype(np.float64)
    perm = np.stack([row[rng.permutation(len(row))] for row in flags])
    d = P.discounted(perm, gamma)
    past = np.zeros_like(d)
    past[:, 1:] = gamma * d[:, :-1]
    return flags + past if include_current else past


def pick_gamma(readout_cal, kl_cal, cp, name, mask):
    scores = {}
    for g in P.GAMMA_GRID:
        gs = GramSet(dict(ro=readout_cal, x=variant(name, kl_cal, cp, g)), mask)
        scores[g] = gs.r2('ro', ['x'])
    return max(scores, key=scores.get), scores


def pick_alpha(series_cal, target_cal, mask):
    scores = {}
    for a in P.EMA_ALPHA_GRID:
        gs = GramSet(dict(y=target_cal, x=P.ema(series_cal, a)), mask)
        scores[a] = gs.r2('y', ['x'])
    return max(scores, key=scores.get), scores


# ─── P1 ──────────────────────────────────────────────────────────────────────

def run_p1(task, label, S):
    m_emg_cal = P.site_mask(S['calibration']['kl'].shape)
    m_emg = P.site_mask(S['evaluation']['kl'].shape)
    cp = calib_params(S['calibration']['kl'])
    hp = S['probe']['h'][:, P.MIN_T:].reshape(-1, S['probe']['h'].shape[-1])
    yp = (S['probe']['kl'][:, P.MIN_T:].ravel() > cp['median']).astype(int)
    clf, sc = P.fit_probe(hp, yp)
    ro_cal = P.probe_scores(clf, sc, S['calibration']['h'])
    ev = S['evaluation']
    ro = P.probe_scores(clf, sc, ev['h'])
    kl = ev['kl'].astype(np.float64)
    flag = (kl > cp['median']).astype(np.float64)
    auroc = float(roc_auc_score(flag[m_emg].astype(int), ro[m_emg]))

    gammas, cols = {}, dict(ro=ro, kl=kl, flag=flag)
    for name in VARIANTS:
        g, _ = pick_gamma(ro_cal, S['calibration']['kl'], cp, name, m_emg_cal)
        gammas[name] = g
        cols[name] = variant(name, kl, cp, g)
        for gg in P.GAMMA_GRID:
            cols[f'{name}@{gg}'] = variant(name, kl, cp, gg)
    gs = GramSet(cols, m_emg)
    W = gs.boot_weights()

    out = dict(label=label, calib=cp, probe_eval_auroc=auroc, gamma_calibration=gammas,
               n_eval_sites=int(m_emg.sum()),
               r2_readout_kl=gs.r2_ci('ro', ['kl'], W),
               r2_readout_flag=gs.r2_ci('ro', ['flag'], W), variants={})
    for name in VARIANTS:
        v = dict(gamma=gammas[name], r2=gs.r2_ci('ro', [name], W),
                 r2_minus_r2_kl=gs.stat_ci(lambda w, n=name: gs.r2('ro', [n], w) - gs.r2('ro', ['kl'], w), W),
                 grid={str(gg): gs.r2('ro', [f'{name}@{gg}']) for gg in P.GAMMA_GRID})
        if name.endswith('_past'):
            v['partial_r2_beyond_flag'] = gs.partial_r2_ci('ro', [name], ['flag'], W)
            v['incremental_r2_beyond_flag'] = gs.delta_r2_ci('ro', [name], ['flag'], W)
            v['partial_r2_beyond_kl_and_flag'] = gs.partial_r2_ci('ro', [name], ['kl', 'flag'], W)
        out['variants'][name] = v

    # temporal scrambling, current flag fixed
    rng = np.random.default_rng(SCRAMBLE_SEED)
    scr = {}
    for name, inc in (('C', True), ('C_past', False)):
        g = gammas[name]
        real = gs.r2('ro', [name])
        null = []
        for _ in range(N_SCRAMBLE):
            x = scrambled(kl, cp['median'], g, rng, inc)
            null.append(GramSet(dict(ro=ro, x=x), m_emg).r2('ro', ['x']))
        null = np.array(null)
        scr[name] = dict(real=real, null_mean=float(null.mean()), null_std=float(null.std()),
                         z=float((real - null.mean()) / (null.std() + 1e-12)), n=N_SCRAMBLE)
    out['scrambling_readout'] = scr
    out['geometry'] = P.geometry(hp, clf, sc)
    return out, (clf, sc, cp, gammas, ro, ro_cal)


# ─── P2 (+P6) ────────────────────────────────────────────────────────────────

def run_p2(task, S, p1_state):
    clf, sc, cp, gammas, ro, ro_cal = p1_state
    cal, ev = S['calibration'], S['evaluation']
    m_ext_cal = P.site_mask(cal['kl'].shape, KMAX)
    m_ext = P.site_mask(ev['kl'].shape, KMAX)
    a_rec, a_rec_scores = pick_alpha(cal['rec'], cal['estate_10'], m_ext_cal)
    a_kl, a_kl_scores = pick_alpha(cal['kl'], cal['estate_10'], m_ext_cal)

    def build(st, ro_):
        kl = st['kl'].astype(np.float64)
        c = dict(kl=kl, rec=st['rec'].astype(np.float64), emarec=P.ema(st['rec'], a_rec),
                 emakl=P.ema(st['kl'], a_kl), readout=ro_)
        for K in P.HORIZONS:
            c[f'E{K}'] = st[f'estate_{K}']
        for name in VARIANTS:
            c[name] = variant(name, kl, cp, gammas[name])
            for gg in P.GAMMA_GRID:
                c[f'{name}@{gg}'] = variant(name, kl, cp, gg)
        return c

    cols = build(ev, ro)
    gs = GramSet(cols, m_ext)
    W = gs.boot_weights()
    ctrl = ['kl', 'rec', 'emarec', 'emakl']
    ctrl_paper = ['kl', 'rec', 'emarec']
    out = dict(alpha_emarec=a_rec, alpha_emakl=a_kl, alpha_scores_emarec=a_rec_scores,
               alpha_scores_emakl=a_kl_scores, n_sites=int(m_ext.sum()),
               n_episodes=int(m_ext.shape[0]), controls=ctrl,
               r2_controls=gs.r2_ci('E10', ctrl, W), table3={}, head_to_head={},
               variants={}, horizons={})
    for name in ('C', 'C_past'):
        out['table3'][name] = dict(
            gamma=gammas[name],
            delta_r2=gs.delta_r2_ci('E10', [name], ctrl, W),
            delta_r2_paper_controls=gs.delta_r2_ci('E10', [name], ctrl_paper, W),
            beta_sign=np.sign(np.linalg.lstsq(
                np.column_stack([np.ones(m_ext.sum())] + [cols[c][m_ext] for c in [name] + ctrl]),
                cols['E10'][m_ext], rcond=None)[0][1]).item())
        for rival in ('emarec', 'emakl'):
            out['head_to_head'][f'{name}_vs_{rival}'] = dict(
                ct_beyond_rival=gs.delta_r2_ci('E10', [name], [rival], W),
                rival_beyond_ct=gs.delta_r2_ci('E10', [rival], [name], W))
    out['readout_delta_r2'] = gs.delta_r2_ci('E10', ['readout'], ctrl, W)
    for name in VARIANTS:
        out['variants'][name] = dict(
            gamma=gammas[name], delta_r2=gs.delta_r2_ci('E10', [name], ctrl, W),
            grid={str(gg): gs.delta_r2_ci('E10', [f'{name}@{gg}'], ctrl, W) for gg in P.GAMMA_GRID})

    # cross-fit: coefficients from calibration, R^2 on evaluation (out of sample)
    ccal = build(cal, ro_cal)
    def xfit(xs):
        Xc = np.column_stack([np.ones(m_ext_cal.sum())] + [ccal[c][m_ext_cal] for c in xs])
        beta = np.linalg.lstsq(Xc, ccal['E10'][m_ext_cal], rcond=None)[0]
        Xe = np.column_stack([np.ones(m_ext.sum())] + [cols[c][m_ext] for c in xs])
        y = cols['E10'][m_ext]
        return 1 - ((y - Xe @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out['crossfit_delta_r2'] = {n: float(xfit([n] + ctrl) - xfit(ctrl)) for n in ('C', 'C_past')}

    for K in P.HORIZONS:
        out['horizons'][K] = {n: pearson_ci(gs, n, f'E{K}', W) for n in ('C', 'C_past', 'kl')}

    # scrambling of the external delta-R^2, current flag fixed
    rng = np.random.default_rng(SCRAMBLE_SEED + 1)
    kl = ev['kl'].astype(np.float64)
    scr = {}
    for name, inc in (('C', True), ('C_past', False)):
        real = out['table3'][name]['delta_r2']['point']
        null = []
        for _ in range(N_SCRAMBLE):
            x = scrambled(kl, cp['median'], gammas[name], rng, inc)
            g2 = GramSet(dict(E10=cols['E10'], x=x, **{c: cols[c] for c in ctrl}), m_ext)
            null.append(g2.r2('E10', ['x'] + ctrl) - g2.r2('E10', ctrl))
        null = np.array(null)
        scr[name] = dict(real=real, null_mean=float(null.mean()), null_std=float(null.std()),
                         z=float((real - null.mean()) / (null.std() + 1e-12)), n=N_SCRAMBLE)
    out['scrambling_external'] = scr

    # P6: autocorrelation on evaluation episodes
    m_emg = P.site_mask(ev['kl'].shape)
    lags = range(1, 11)
    p6 = dict(acf_kl=acf_within(kl, m_emg, lags),
              acf_flag=acf_within((kl > cp['median']).astype(float), m_emg, lags))
    return out, p6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tasks', nargs='+', default=list(P.TASKS))
    ap.add_argument('--primary-only', action='store_true')
    a = ap.parse_args()
    P.seed_everything(0)
    torch.set_num_threads(os.cpu_count() or 1)
    res_path = os.path.join(P.out_dir(a.smoke), 'results.json')
    for task in a.tasks:
        models = P.model_paths(task)
        if a.primary_only or a.smoke:
            models = models[:1]
        for label, path in models:
            if not a.smoke and not os.path.exists(path):
                print(f'MISSING {task} {label}: {path}', flush=True)
                continue
            primary = label == 'primary'
            S, _ = states_for(task, label, path, a.smoke, need_estate=primary)
            p1, p1_state = run_p1(task, label, S)
            P.update_results(res_path, ['p1', task, label], p1)
            v = p1['variants']
            print(f"[P1] {task}/{label}: AUROC {p1['probe_eval_auroc']:.3f}  "
                  f"R2 ro~KL {p1['r2_readout_kl']['point']:.3f}  ro~C {v['C']['r2']['point']:.3f} "
                  f"(g={v['C']['gamma']})  ro~C_past {v['C_past']['r2']['point']:.3f} "
                  f"(g={v['C_past']['gamma']})  partial|flag {v['C_past']['partial_r2_beyond_flag']['point']:.3f}",
                  flush=True)
            if primary:
                p2, p6 = run_p2(task, S, p1_state)
                P.update_results(res_path, ['p2', task], p2)
                P.update_results(res_path, ['p6', task], p6)
                t3 = p2['table3']
                print(f"[P2] {task}: dR2 C {t3['C']['delta_r2']['point']:+.4f} "
                      f"[{t3['C']['delta_r2']['lo']:+.4f},{t3['C']['delta_r2']['hi']:+.4f}]  "
                      f"C_past {t3['C_past']['delta_r2']['point']:+.4f} "
                      f"[{t3['C_past']['delta_r2']['lo']:+.4f},{t3['C_past']['delta_r2']['hi']:+.4f}]",
                      flush=True)


if __name__ == '__main__':
    main()
