#!/usr/bin/env python3.11
"""Camera-ready Tasks 1-3 under the final rule:
corrected logging; probe v refit on corrected probe-fit episodes (paper recipe, label
1[KL_t > calibration median]); published gamma (0.95/0.70/0.90) and EMARec alpha
(0.02/0.5/0.02); readout = probe probability sigma(v^T h_t + b), logit stored; canonical
50-direction null; trajectory bootstrap (1000, seed 7). Primary model per task (the paper
reports per-model Delta R^2 only for primaries). Evaluation episodes 30000-30099.

Task 1  readout -> E_{t,10}: r, R^2, Delta R^2 beyond Table 4 controls ({KL, Rec, EMARec}
        + ensemble disagreement on cartpole); C_t (Section-6 definition) side by side.
        Null: each canonical direction u replaces v in the same readout,
        logit_u = ||w|| u.h + b_u with b_u matching v's mean logit on probe-fit states
        (same scale and centring), probability = sigmoid(logit_u).
Task 2  signals KL, Rec, EMARec, EMA-KL (alpha maximising R^2 with E10 on probe-fit
        episodes), prior entropy, posterior entropy, C_t, readout: standalone r and R^2;
        Delta R^2 of C_t and of the readout beyond all other signals together.
Task 3  selective trust: rank sites by decision-time signals, trust the lowest fraction c;
        risk = mean E10 of trusted sites, c in {0.1..1.0}; AURC = mean risk over the 10
        coverages. Readout ranked by its logit (order identical to the probability, no
        saturation ties); ties broken by a seeded random permutation and counted.
        C_t for Task 3 = sum_{i>=1} gamma^i 1[KL_{t-i} > calibration median] (decision-time:
        a per-trajectory median would use future steps). Null directions are oriented
        like v (sign of corr(u.h, KL) on probe-fit episodes).
"""
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr

from src.camera_ready import core as C
from src.rerun import pipeline as P
from src.rerun.stats import GramSet, pearson_ci
from run_cr_A_analysis import ct_per_traj_median

R = C.ROOT
GAMMA = {'cartpole': 0.95, 'reacher': 0.70, 'pendulum': 0.90}
ALPHA = {'cartpole': 0.02, 'reacher': 0.5, 'pendulum': 0.02}
TASKS = ['cartpole', 'reacher', 'pendulum']
COVER = np.round(np.arange(0.1, 1.01, 0.1), 2)
M = C.MIN_T
N_RAND = 100
SAT = 0.01


def refit(task):
    pr, cal = C.load_states(task, 'primary', 'probe'), C.load_states(task, 'primary', 'calibration')
    thr = float(np.median(cal['kl'][:, M:]))
    hp = pr['h'][:, M:].reshape(-1, 256)
    clf, sc = P.fit_probe(hp, (pr['kl'][:, M:].ravel() > thr).astype(int))
    w = (clf.coef_[0] / sc.scale_).astype(np.float64)
    b0 = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())   # logit = w.h + b0
    return clf, sc, w, b0, thr, hp, pr


def logit_of(h, w, b):
    return h.astype(np.float64) @ w + b


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def null_readouts(U, w, b0, hp, h):
    """Readout along each null direction with v's scale and centring."""
    nw = np.linalg.norm(w)
    mean_v = float((hp @ w).mean() + b0)
    out = []
    for u in U:
        bu = mean_v - nw * float((hp @ u).mean())
        out.append(nw * (h.astype(np.float64) @ u) + bu)
    return out


def sat_frac(p):
    return float(((p < SAT) | (p > 1 - SAT)).mean())


def zpct(val, null):
    n = np.asarray(null)
    return dict(value=float(val), z=float((val - n.mean()) / (n.std() + 1e-12)),
                percentile=float((n < val).mean() * 100), null_mean=float(n.mean()), null_sd=float(n.std()),
                null_max=float(n.max()))


# ─── Task 1 ──────────────────────────────────────────────────────────────────

def task1(task, ctx, U, ens):
    ev, w, b0, hp = ctx['ev'], ctx['w'], ctx['b0'], ctx['hp']
    mask = P.site_mask(ev['kl'].shape, C.KMAX)
    kl = ev['kl'].astype(np.float64)
    lg = logit_of(ev['h'], w, b0)
    cols = dict(kl=kl, rec=ev['rec'].astype(np.float64), emarec=P.ema(ev['rec'], ALPHA[task]),
                C=ct_per_traj_median(kl, GAMMA[task]), prob=sig(lg), logit=lg, E10=ev['estate_10'])
    ctrl = ['kl', 'rec', 'emarec']
    if ens is not None:
        torch.manual_seed(4242)
        cols['ens'] = C.ensemble_disagreement(ens, C.episodes(task, 'evaluation')['obs'])
        ctrl.append('ens')
    gs = GramSet(cols, mask)
    W = gs.boot_weights()

    def stats(name):
        r = pearson_ci(gs, name, 'E10', W)
        return dict(r=r, r2=float(r['point'] ** 2), delta_r2=gs.delta_r2_ci('E10', [name], ctrl, W))
    out = dict(n_sites=int(mask.sum()), n_episodes=int(mask.shape[0]), controls=ctrl,
               r2_controls=gs.r2_ci('E10', ctrl, W), C_t=stats('C'),
               readout_prob=stats('prob'), readout_logit=stats('logit'),
               saturation_v=sat_frac(cols['prob'][mask]))
    # null
    E = cols['E10'][mask]
    Xc = np.column_stack([np.ones(mask.sum())] + [cols[c][mask] for c in ctrl])
    beta = np.linalg.lstsq(Xc, E, rcond=None)[0]
    res = E - Xc @ beta
    sst = ((E - E.mean()) ** 2).sum()
    r2c = 1 - (res ** 2).sum() / sst

    def null_stat(x):
        if np.std(x) < 1e-12:
            return dict(r=0.0, r2=0.0, dr2=0.0)
        X = np.column_stack([Xc, x])
        bb = np.linalg.lstsq(X, E, rcond=None)[0]
        r = pearsonr(x, E)[0]
        return dict(r=float(r), r2=float(r * r), dr2=float(1 - ((E - X @ bb) ** 2).sum() / sst - r2c))
    nl = null_readouts(U, w, b0, hp, ev['h'])
    np_, nlg, sats = [], [], []
    for x in nl:
        x = x[mask]
        p = sig(x)
        sats.append(sat_frac(p))
        np_.append(null_stat(p)); nlg.append(null_stat(x))
    out['null'] = {}
    for name, nul in (('prob', np_), ('logit', nlg)):
        s = out[f'readout_{name}']
        out['null'][name] = dict(abs_r=zpct(abs(s['r']['point']), [abs(n['r']) for n in nul]),
                                 r2=zpct(s['r2'], [n['r2'] for n in nul]),
                                 delta_r2=zpct(s['delta_r2']['point'], [n['dr2'] for n in nul]))
    out['null']['prob_saturation'] = dict(median_frac_sites_saturated=float(np.median(sats)),
                                          n_dirs_over_90pct_saturated=int(sum(s > 0.9 for s in sats)),
                                          v_frac_sites_saturated=out['saturation_v'])
    out['null']['prob_degenerate'] = bool(np.median(sats) > 0.5)
    out['C_t_vs_null'] = dict(r2_logit_null=zpct(out['C_t']['r2'], [n['r2'] for n in nlg]),
                              delta_r2_logit_null=zpct(out['C_t']['delta_r2']['point'], [n['dr2'] for n in nlg]))
    return out


# ─── Task 2 ──────────────────────────────────────────────────────────────────

def task2(task, ctx):
    ev, pr, w, b0 = ctx['ev'], ctx['pr'], ctx['w'], ctx['b0']
    pe = np.load(os.path.join(R, 'task_data_corrected', f'{task}_probe_estate.npz'))
    mp = P.site_mask(pr['kl'].shape, C.KMAX)
    sc_ = {}
    for a in P.EMA_ALPHA_GRID:
        sc_[a] = GramSet(dict(y=pe['estate_10'], x=P.ema(pr['kl'], a)), mp).r2('y', ['x'])
    a_kl = max(sc_, key=sc_.get)
    mask = P.site_mask(ev['kl'].shape, C.KMAX)
    kl = ev['kl'].astype(np.float64)
    lg = logit_of(ev['h'], w, b0)
    cols = dict(KL=kl, Rec=ev['rec'].astype(np.float64), EMARec=P.ema(ev['rec'], ALPHA[task]),
                EMAKL=P.ema(ev['kl'], a_kl), prior_entropy=ev['ent_prior'].astype(np.float64),
                posterior_entropy=ev['ent_post'].astype(np.float64), C_t=ct_per_traj_median(kl, GAMMA[task]),
                readout_prob=sig(lg), readout_logit=lg, E10=ev['estate_10'])
    gs = GramSet(cols, mask)
    W = gs.boot_weights()
    std = ['KL', 'Rec', 'EMARec', 'EMAKL', 'prior_entropy', 'posterior_entropy']
    out = dict(alpha_emakl=a_kl, alpha_emakl_scores={str(k): v for k, v in sc_.items()}, n_sites=int(mask.sum()),
               standalone={}, delta_r2={}, r2_full_baseline=gs.r2_ci('E10', std, W))
    for s in std + ['C_t', 'readout_prob', 'readout_logit']:
        r = pearson_ci(gs, s, 'E10', W)
        out['standalone'][s] = dict(r=r, r2=float(r['point'] ** 2))
    out['delta_r2']['C_t_beyond_6_standard'] = gs.delta_r2_ci('E10', ['C_t'], std, W)
    out['delta_r2']['C_t_beyond_all_others_prob'] = gs.delta_r2_ci('E10', ['C_t'], std + ['readout_prob'], W)
    out['delta_r2']['C_t_beyond_all_others_logit'] = gs.delta_r2_ci('E10', ['C_t'], std + ['readout_logit'], W)
    for rd in ('prob', 'logit'):
        out['delta_r2'][f'readout_{rd}_beyond_6_standard'] = gs.delta_r2_ci('E10', [f'readout_{rd}'], std, W)
        out['delta_r2'][f'readout_{rd}_beyond_all_others'] = gs.delta_r2_ci('E10', [f'readout_{rd}'], std + ['C_t'], W)
    return out


# ─── Task 3 ──────────────────────────────────────────────────────────────────

def lagged(x):
    y = np.full_like(x, np.nan, dtype=np.float64)
    y[:, 1:] = x[:, :-1]
    return y


def ranks_with_ties(score, rng):
    """Order sites by ascending score; ties broken by a seeded random permutation."""
    perm = rng.permutation(len(score))
    order = perm[np.argsort(score[perm], kind='stable')]
    n_tied = int(len(score) - len(np.unique(score)))
    return order, n_tied


def curve(order, E):
    """risk (mean E10 of trusted sites) at each coverage for the given site order."""
    n = len(order)
    return np.array([E[order[:max(1, int(round(c * n)))]].mean() for c in COVER])


def task3(task, ctx, U):
    ev, pr, w, b0, thr = ctx['ev'], ctx['pr'], ctx['w'], ctx['b0'], ctx['thr']
    mask = P.site_mask(ev['kl'].shape, C.KMAX)
    e_idx, t_idx = np.nonzero(mask)
    E = ev['estate_10'][mask].astype(np.float64)
    kl = ev['kl'].astype(np.float64)
    rec = ev['rec'].astype(np.float64)
    pe = np.load(os.path.join(R, 'task_data_corrected', f'{task}_probe_estate.npz'))
    mp = P.site_mask(pr['kl'].shape, C.KMAX)
    sc_ = {a: GramSet(dict(y=pe['estate_10'], x=P.ema(pr['kl'], a)), mp).r2('y', ['x']) for a in P.EMA_ALPHA_GRID}
    a_kl = max(sc_, key=sc_.get)
    lg = logit_of(ev['h'], w, b0)
    sig_v = {
        'readout': lg[mask],                                                  # ranked by logit
        'C_t_past': P.compute_ct(kl, thr, GAMMA[task], include_current=False)[mask],
        'KL_tm1': lagged(kl)[mask],
        'Rec_tm1': lagged(rec)[mask],
        'EMARec_tm1': lagged(P.ema(rec, ALPHA[task]))[mask],
        'EMAKL_tm1': lagged(P.ema(kl, a_kl))[mask],
        'prior_entropy_t': ev['ent_prior'][mask].astype(np.float64),
        'oracle': E,
    }
    prob_ties = int(mask.sum() - len(np.unique(sig(lg[mask]))))
    rng = np.random.default_rng(303)
    orders, ties, curves = {}, {}, {}
    for k, s in sig_v.items():
        if not np.all(np.isfinite(s)):
            raise FloatingPointError(f'non-finite decision signal {k}')
        orders[k], ties[k] = ranks_with_ties(s, rng)
        curves[k] = curve(orders[k], E)
    rr = np.random.default_rng(404)
    rand_curves = np.array([curve(rr.permutation(len(E)), E) for _ in range(N_RAND)])
    curves['random'] = rand_curves.mean(0)
    # null directions, oriented like v (toward high KL on probe-fit episodes)
    hp_flat, klp = ctx['hp'], pr['kl'][:, M:].ravel()
    null_aurc = []
    for u in U:
        s_fit = hp_flat @ u
        sgn = np.sign(np.corrcoef(s_fit, klp)[0, 1]) or 1.0
        s = sgn * (ev['h'][mask].astype(np.float64) @ u)
        o, _ = ranks_with_ties(s, rng)
        null_aurc.append(curve(o, E).mean())
    aurc = {k: float(c.mean()) for k, c in curves.items()}
    baselines = ['KL_tm1', 'Rec_tm1', 'EMARec_tm1', 'EMAKL_tm1', 'prior_entropy_t', 'random']
    best = min(baselines, key=lambda k: aurc[k])
    # trajectory bootstrap of AURC and differences (rank order fixed globally; resample episodes)
    ep_of_site = e_idx
    ne = mask.shape[0]
    brng = np.random.default_rng(P.BOOT_SEED)
    pos = {k: np.empty(len(E), np.int64) for k in orders}
    for k in orders:
        pos[k][orders[k]] = np.arange(len(E))
    keys = ['readout', 'C_t_past'] + [b for b in baselines if b != 'random'] + ['oracle']
    boots = {k: [] for k in keys + ['random']}
    for _ in range(P.N_BOOT):
        epw = np.bincount(brng.integers(0, ne, ne), minlength=ne)
        wts = epw[ep_of_site].astype(np.float64)
        for k in keys:
            o = orders[k]
            ww, ee = wts[o], E[o]
            cw, ce = np.cumsum(ww), np.cumsum(ww * ee)
            tot = cw[-1]
            ix = [min(int(np.searchsorted(cw, c * tot - 1e-9)), len(cw) - 1) for c in COVER]
            risks = [ce[i] / cw[i] for i in ix]
            boots[k].append(np.mean(risks))
        boots['random'].append(float((wts * E).sum() / wts.sum()))   # random ranking: expected risk = mean E at every c
    for k in boots:
        boots[k] = np.array(boots[k])

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = dict(n_sites=int(len(E)), n_episodes=int(ne), alpha_emakl=a_kl, coverages=COVER.tolist(),
               aurc={k: dict(point=aurc[k], ci=ci(boots[k]) if k in boots else None) for k in aurc},
               risk={k: dict(c05=float(c[4]), c08=float(c[7]), curve=c.tolist()) for k, c in curves.items()},
               ties=dict(**{k: ties[k] for k in ties}, readout_if_ranked_by_probability=prob_ties),
               best_baseline=best, null_aurc=null_aurc)
    for cand in ('readout', 'C_t_past'):
        diff = boots[cand] - boots[best]
        d = dict(point=aurc[cand] - aurc[best], ci=ci(diff))
        nz = np.array(null_aurc)
        z = float((nz.mean() - aurc[cand]) / (nz.std() + 1e-12))       # positive = better (lower AURC) than null
        passes = d['point'] < 0 and d['ci'][1] < 0 and z > 2
        out[cand] = dict(diff_vs_best_baseline=d, null_z=z, null_mean=float(nz.mean()), null_sd=float(nz.std()),
                         null_percentile_better=float((nz > aurc[cand]).mean() * 100), passes=bool(passes))
    return out


def main():
    P.seed_everything(0)
    torch.set_num_threads(6)
    U = np.load('outputs/canonical_null/random_directions_50_dim256.npz')['directions'].astype(np.float64)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ens = [C.load_any(f'outputs/checkpoints/ensemble_seed{i}.pt', 'wm') for i in range(3)]
    res = {1: {}, 2: {}, 3: {}}
    for task in TASKS:
        clf, sc, w, b0, thr, hp, pr = refit(task)
        ev = C.load_states(task, 'primary', 'evaluation')
        ctx = dict(ev=ev, pr=pr, w=w, b0=b0, thr=thr, hp=hp)
        # sanity: our logit equals the classifier's decision function
        chk = clf.decision_function(sc.transform(ev['h'][0, :50].astype(np.float64)))
        if not np.allclose(chk, logit_of(ev['h'][0, :50], w, b0), atol=1e-6):
            raise AssertionError('logit reconstruction mismatch')
        res[1][task] = task1(task, ctx, U, ens if task == 'cartpole' else None)
        res[2][task] = task2(task, ctx)
        res[3][task] = task3(task, ctx, U)
        a, b, c = res[1][task], res[2][task], res[3][task]
        print(f"[T1] {task}: prob dR2 {a['readout_prob']['delta_r2']['point']:+.4f} null z {a['null']['prob']['delta_r2']['z']:+.2f} | "
              f"logit dR2 {a['readout_logit']['delta_r2']['point']:+.4f} null z {a['null']['logit']['delta_r2']['z']:+.2f} | "
              f"C_t {a['C_t']['delta_r2']['point']:+.4f} | prob null degenerate {a['null']['prob_degenerate']}", flush=True)
        print(f"[T2] {task}: C_t beyond all {b['delta_r2']['C_t_beyond_all_others_prob']['point']:+.4f} "
              f"[{b['delta_r2']['C_t_beyond_all_others_prob']['lo']:+.4f},{b['delta_r2']['C_t_beyond_all_others_prob']['hi']:+.4f}] "
              f"readout(prob) beyond all {b['delta_r2']['readout_prob_beyond_all_others']['point']:+.4f} "
              f"[{b['delta_r2']['readout_prob_beyond_all_others']['lo']:+.4f},{b['delta_r2']['readout_prob_beyond_all_others']['hi']:+.4f}]", flush=True)
        print(f"[T3] {task}: AURC readout {c['aurc']['readout']['point']:.4f} C_t {c['aurc']['C_t_past']['point']:.4f} "
              f"best baseline {c['best_baseline']} {c['aurc'][c['best_baseline']]['point']:.4f} oracle {c['aurc']['oracle']['point']:.4f} "
              f"| readout null z {c['readout']['null_z']:+.2f} pass {c['readout']['passes']} | C_t pass {c['C_t_past']['passes']}", flush=True)
        with open(os.path.join(R, 'T123_raw.json'), 'w') as f:
            json.dump(res, f, indent=1, default=P._json_default)


if __name__ == '__main__':
    main()
