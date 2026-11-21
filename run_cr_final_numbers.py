#!/usr/bin/env python3.11
"""Final numbers with the fixed choices
  * corrected logging (o_t with a_{t-1}); evaluation episodes 30000-30099
  * everything the paper learned from the (mis-paired) training log is refit on the
    corrected probe-fit episodes (seeds 10000-10099) with the paper's recipe:
    probe v (StandardScaler + logistic regression C=1, lbfgs, random_state=0;
    label 1[KL_t > calibration median]); PCA for the geometry statistics; SAEs
  * readout = probe probability sigma(v^T h_t + b) (as in every published number);
    the logit v^T h_t + b is stored alongside every readout number
  * C_t with the published gamma (0.95/0.70/0.90; partial-obs cartpole models: 0.95);
    Section-6 C_t (Table 4, Fig. 2a-c, Sec. 6.3, 10.3): per-trajectory median (paper
    definition; Fig. 2c uses it on the same sites as Fig. 2a); Section-5 / Fig. 1d
    global threshold: the paper used the training-log median (old pairing), replaced
    by the calibration-episode median (seeds 20000-20049)
  * EMARec alpha fixed at 0.02/0.5/0.02; controls {KL, Rec, EMARec} (+ ensemble
    disagreement on cartpole) as in Table 4
  * canonical 50-direction null with common random numbers (from the paper-probe
    causal runs in A_raw_paperprobe/ and B_sae_corrected.json)
  * trajectory bootstrap, 1000 resamples, seed 7
The logit is affine in the projection v.h, so every intervention z on the logit equals
the projection z recorded by the causal runs (slope_p*/ablate_p*).
Writes outputs/camera_ready/camera_ready_numbers.json and fig_data.json.
"""
import json
import os

os.environ['CR_PROBE'] = 'frozen'
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from src.camera_ready import core as C
from src.rerun import pipeline as P
from src.rerun.stats import GramSet, pearson_ci
from run_sprint_p1p2 import kl_matched_contrast, within_bin_r
from run_cr_A_analysis import contrast_by, mean_angle_top50, ct_per_traj_median

R = C.ROOT
PP = os.path.join(R, 'A_raw')          # refit-probe (frozen-model probe-fit episodes) causal runs
GAMMA = {'cartpole': 0.95, 'reacher': 0.70, 'pendulum': 0.90, 'cartpole_pomdp': 0.95}
ALPHA = {'cartpole': 0.02, 'reacher': 0.5, 'pendulum': 0.02}
TASKS = ['cartpole', 'reacher', 'pendulum']
PAPER13 = {'cartpole': ['primary', 'seed_1', 'seed_2', 'seed_3', 'seed_4'],
           'reacher': ['primary', 'reacher_seed1', 'reacher_seed2', 'reacher_seed3'],
           'pendulum': ['primary', 'pendulum_seed1', 'pendulum_seed2', 'pendulum_seed3']}
N_SCR, SCR_SEED = 20, 2027
M = C.MIN_T


def load(p):
    with open(p) as f:
        return json.load(f)


def refit_probe(task, label):
    pr, cal = C.load_states(task, label, 'probe'), C.load_states(task, label, 'calibration')
    thr = float(np.median(cal['kl'][:, M:]))
    hp = pr['h'][:, M:].reshape(-1, pr['h'].shape[-1])
    y = (pr['kl'][:, M:].ravel() > thr).astype(int)
    if not (0.2 < y.mean() < 0.8):
        raise AssertionError(f'label balance {y.mean():.3f} ({task}/{label})')
    clf, sc = P.fit_probe(hp, y)
    v = P.probe_direction_raw(clf, sc).astype(np.float64)
    return clf, sc, v, hp, thr


def readouts(clf, sc, h):
    flat = h.reshape(-1, h.shape[-1])
    lg = clf.decision_function(sc.transform(flat)).reshape(h.shape[:-1]).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-lg)), lg


def emergence(task, label, noisy=True):
    clf, sc, v, hfit, thr = refit_probe(task, label)
    ev = C.load_states(task, label, 'evaluation')
    g = GAMMA[task]
    mask = P.site_mask(ev['kl'].shape)
    kl = ev['kl'].astype(np.float64)
    prob, logit = readouts(clf, sc, ev['h'])
    Cg = P.compute_ct(kl, thr, g)
    gs = GramSet(dict(p=prob, lg=logit, kl=kl, C=Cg), mask)
    W = gs.boot_weights()
    out = dict(threshold=thr, gamma=g,
               auroc_clean=float(roc_auc_score((kl[mask] > thr).astype(int), prob[mask])),
               r2_readout_C=gs.r2_ci('p', ['C'], W), r2_readout_kl=gs.r2_ci('p', ['kl'], W),
               logit=dict(r2_readout_C=gs.r2_ci('lg', ['C'], W), r2_readout_kl=gs.r2_ci('lg', ['kl'], W)),
               saturation=dict(frac_prob_below_0p01=float((prob[mask] < 0.01).mean()),
                               frac_prob_above_0p99=float((prob[mask] > 0.99).mean())),
               r_kl_C=pearson_ci(gs, 'kl', 'C', W),
               geometry=P.geometry(hfit, clf, sc), angle_top50=mean_angle_top50(v, sc.transform(hfit)))
    proj = prob
    f = lambda a: a[mask]
    if noisy:
        nz = C.load_states(task, label, 'evaluation_noisy')
        pn, _ = readouts(clf, sc, nz['h'])
        out['auroc_noisy'] = float(roc_auc_score((nz['kl'][mask] > thr).astype(int), pn[mask]))
        out['kl_matched_pooled'] = kl_matched_contrast(np.r_[f(kl), f(nz['kl'])], np.r_[f(ev['rec']), f(nz['rec'])],
                                                       np.r_[f(proj), f(pn)])
        out['kl_matched_clean'] = kl_matched_contrast(f(kl), f(ev['rec']), f(proj))
        out['within_bin_r_rec_C'] = within_bin_r(f(kl), f(ev['rec']), f(Cg))['mean']
        out['relabel_by_C_auroc'] = contrast_by(f(kl), f(Cg), f(proj))
    return out, (clf, sc, v, prob, Cg, thr)


def scramble_r(kl, thr_fn, g, E, mask, rng):
    real_flags = kl > thr_fn(kl)
    real = pearsonr(P.discounted(real_flags.astype(float), g)[mask], E[mask])[0]
    null = []
    for _ in range(N_SCR):
        fl = np.stack([r[rng.permutation(len(r))] for r in real_flags.astype(float)])
        null.append(pearsonr(P.discounted(fl, g)[mask], E[mask])[0])
    null = np.array(null)
    return dict(real=float(real), scrambled_mean=float(null.mean()), scrambled_sd=float(null.std()),
                z=float((real - null.mean()) / null.std()), n=N_SCR)


def external(task, emg_state, ens=None):
    _, _, v, proj, Cg, thr = emg_state
    ev = C.load_states(task, 'primary', 'evaluation')
    mask = P.site_mask(ev['kl'].shape, C.KMAX)
    kl = ev['kl'].astype(np.float64)
    g = GAMMA[task]
    cols = dict(kl=kl, rec=ev['rec'].astype(np.float64), emarec=P.ema(ev['rec'], ALPHA[task]),
                C=ct_per_traj_median(kl, g), proj=proj)
    for K in C.HORIZONS:
        cols[f'E{K}'] = ev[f'estate_{K}']
    ctrl = ['kl', 'rec', 'emarec']
    if ens is not None:
        torch.manual_seed(4242)
        cols['ens'] = C.ensemble_disagreement(ens, C.episodes(task, 'evaluation')['obs'])
        ctrl.append('ens')
    gs = GramSet(cols, mask)
    W = gs.boot_weights()
    out = dict(n_sites=int(mask.sum()), n_episodes=int(mask.shape[0]), controls=ctrl, alpha=ALPHA[task],
               r2_controls=gs.r2_ci('E10', ctrl, W),
               delta_r2_C=gs.delta_r2_ci('E10', ['C'], ctrl, W),
               horizons={K: pearson_ci(gs, 'C', f'E{K}', W) for K in C.HORIZONS},
               head_to_head=dict(C_beyond_emarec=gs.delta_r2_ci('E10', ['C'], ['emarec'], W),
                                 emarec_beyond_C=gs.delta_r2_ci('E10', ['emarec'], ['C'], W)))
    if 'ens' in ctrl:
        out['delta_r2_C_without_ens'] = gs.delta_r2_ci('E10', ['C'], ['kl', 'rec', 'emarec'], W)
    rng = np.random.default_rng(SCR_SEED + 1)
    E = cols['E10']
    out['scramble_section6_def'] = scramble_r(kl, lambda k: np.median(k[:, M:], axis=1, keepdims=True), g, E, mask, rng)
    gap = (ev['imag_rew_10'] - ev['real_rew_10'])[mask]
    rs, ks, cc = cols['rec'][mask], kl[mask], cols['C'][mask]
    drift = (gap > np.percentile(gap, 75)) & (rs <= np.percentile(rs, 50)) & (ks <= np.percentile(ks, 50))
    out['reward_inflation'] = dict(n_drift=int(drift.sum()), n_sites=int(mask.sum()),
                                   C_drift=float(cc[drift].mean()) if drift.any() else None,
                                   C_other=float(cc[~drift].mean()),
                                   imag_reward_degenerate=bool(np.nanstd(ev['imag_rew_10'][mask]) < 1e-9))
    return out


def pomdp(label):
    clf, sc, v, _, thr = refit_probe('cartpole_pomdp', label)
    ev = C.load_states('cartpole_pomdp', label, 'evaluation')
    kl = ev['kl'].astype(np.float64)
    prob, logit = readouts(clf, sc, ev['h'])
    m = P.site_mask(kl.shape)
    gs = GramSet(dict(p=prob, lg=logit, C=P.compute_ct(kl, thr, 0.95)), m)
    me = P.site_mask(kl.shape, C.KMAX)
    gx = GramSet(dict(E10=ev['estate_10'], C=ct_per_traj_median(kl, 0.95), kl=kl, rec=ev['rec'].astype(np.float64)), me)
    W = gs.boot_weights()
    return dict(auroc=float(roc_auc_score((kl[m] > thr).astype(int), prob[m])),
                r2_readout_C=gs.r2_ci('p', ['C'], W), logit=dict(r2_readout_C=gs.r2_ci('lg', ['C'], W)),
                delta_r2_C=gx.delta_r2_ci('E10', ['C'], ['kl', 'rec'], gx.boot_weights()))


def main():
    P.seed_everything(0)
    res = dict(fixed_choices=__doc__, emergence={}, external={}, pomdp={}, causal={})
    fig = dict()
    ens = [C.load_any(f'outputs/checkpoints/ensemble_seed{i}.pt', 'wm') for i in range(3)]
    for task in TASKS:
        res['emergence'][task] = {}
        for label in PAPER13[task]:
            e, st = emergence(task, label)
            res['emergence'][task][label] = e
            print(f"[C] {task}/{label}: AUROC {e['auroc_clean']:.3f} R2(prob,C) {e['r2_readout_C']['point']:.3f} "
                  f"R2(prob,KL) {e['r2_readout_kl']['point']:.3f} | logit {e['logit']['r2_readout_C']['point']:.3f} / "
                  f"{e['logit']['r2_readout_kl']['point']:.3f}", flush=True)
            if label == 'primary':
                res['external'][task] = x = external(task, st, ens if task == 'cartpole' else None)
                print(f"    dR2 {x['delta_r2_C']['point']:+.4f} [{x['delta_r2_C']['lo']:+.4f},{x['delta_r2_C']['hi']:+.4f}] "
                      f"scr(sec6) z {x['scramble_section6_def']['z']:+.1f}",
                      flush=True)
                if task == 'cartpole':
                    proj, Cg = st[3], st[4]
                    mask = P.site_mask(proj.shape)
                    ee, tt = np.nonzero(mask)
                    pick = np.random.default_rng(99).choice(len(ee), 3000, replace=False)
                    fig['scatter'] = dict(C=Cg[ee[pick], tt[pick]].round(4).tolist(),
                                          readout_prob=proj[ee[pick], tt[pick]].round(5).tolist())
    for s in C.POMDP_SEEDS:
        for cond in ('full', 'partial'):
            res['pomdp'][f'{cond}_{s}'] = pomdp(f'{cond}_{s}')
    # causal results (paper probe, corrected logging, canonical null, CRN)
    for task in TASKS:
        res['causal'][task] = {l: load(os.path.join(PP, f'causal_steer_{task}_{l}.json'))['null50'] for l in PAPER13[task]}
        res['causal'][f'{task}_noisy'] = load(os.path.join(PP, f'causal_noisy_{task}_primary.json'))['null50']
    res['causal']['taskg_cartpole'] = load(os.path.join(PP, 'causal_taskg_cartpole_primary.json'))['null50']
    res['causal']['pomdp'] = {f'{c}_{s}': load(os.path.join(PP, f'causal_pomdp_cartpole_pomdp_{c}_{s}.json'))['null50']
                              for s in C.POMDP_SEEDS for c in ('full', 'partial')}
    res['ridge_scramble'] = load(os.path.join(R, 'A_ridge_scramble_paper_gamma.json'))
    res['sae_retrained'] = load(os.path.join(R, 'B_sae_corrected_refit.json'))
    with open(os.path.join(R, 'camera_ready_numbers.json'), 'w') as f:
        json.dump(res, f, indent=1, default=P._json_default)
    with open(os.path.join(R, 'fig_data.json'), 'w') as f:
        json.dump(fig, f)
    print('wrote camera_ready_numbers.json')


if __name__ == '__main__':
    main()
