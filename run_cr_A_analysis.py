#!/usr/bin/env python3.11
"""Camera-ready Part A, step 2: recompute the paper's correlational numbers on
corrected states (outputs/camera_ready/states).

Definitions follow the accepted paper; where the paper's own procedure violates the
handoff's rules (step-level splits, gamma tuned on the reported fit, per-trajectory
median that includes future steps), BOTH are reported:
  *_paperdef  : paper's definition applied to corrected states
  *_cal       : calibration-split threshold / gamma (handoff Rules 2-3)
Probe: logistic regression (C=1, lbfgs, standardised) on probe-fit episodes, label
1[KL_t > calibration median]. Readout = probe probability (as in the paper's code);
the logit is also reported. All reported numbers are on evaluation episodes.

Usage: python run_cr_A_analysis.py [--tasks ...]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from src.camera_ready import core as C
from src.rerun import pipeline as P
from src.rerun.stats import GramSet, pearson_ci
from run_sprint_p1p2 import (kl_matched_contrast, within_bin_r, ridge_fit, r2, calib_params,
                             N_KL_BINS, CONTRAST_FRAC)

OUT = os.environ.get('CR_RAW_DIR', os.path.join(C.ROOT, 'A_raw'))
PAPER_GAMMA = {'cartpole': 0.95, 'reacher': 0.70, 'pendulum': 0.90, 'cartpole_pomdp': None}
N_SCR = 20
SCR_SEED = 2027
EMA_PAPER_GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
M = C.MIN_T


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=P._json_default)


def ct_cal(kl, thr, g, inc=True):
    return P.compute_ct(kl, thr, g, include_current=inc)


def ct_per_traj_median(kl, g):
    """Paper Sec. 6 definition: median per evaluation trajectory (includes future steps)."""
    thr = np.median(kl[:, M:], axis=1, keepdims=True)
    return P.discounted((kl > thr).astype(np.float64), g)


def scramble_all(kl, thr, g, rng):
    """Paper scramble: high-KL flags permuted within each trajectory (all flags,
    including the current one), then re-accumulated."""
    flags = (kl > thr).astype(np.float64)
    return P.discounted(np.stack([r[rng.permutation(len(r))] for r in flags]), g)


def pick_gamma(ro, kl, thr, mask):
    sc = {g: GramSet(dict(ro=ro, x=ct_cal(kl, thr, g)), mask).r2('ro', ['x']) for g in P.GAMMA_GRID}
    return max(sc, key=sc.get), sc


def ema_alpha_paper(rec, kl, mask):
    """Paper/Phase-1 criterion: alpha maximising |corr(EMA(Rec), KL)| on the tuning split."""
    best, sc = None, {}
    for a in EMA_PAPER_GRID:
        e = P.ema(rec, a)
        sc[a] = abs(pearsonr(e[mask], kl[mask])[0])
    best = max(sc, key=sc.get)
    return best, sc


def mean_angle_top50(v, h):
    from sklearn.decomposition import PCA
    pcs = PCA(n_components=50, random_state=0).fit(h - h.mean(0)).components_
    cos = np.abs(pcs @ v) / np.linalg.norm(v)
    rng = np.random.default_rng(11)
    R = rng.standard_normal((1000, len(v)))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rand = np.degrees(np.arccos(np.abs(R @ pcs.T))).mean(1)
    return dict(mean_angle_deg=float(np.degrees(np.arccos(cos)).mean()),
                random_mean_angle_deg=float(rand.mean()))


def contrast_by(kl, key, score):
    """KL-matched contrast with groups defined by `key` (bottom/top 30% within KL bins)."""
    edges = np.percentile(kl, np.linspace(0, 100, N_KL_BINS + 1))
    b = np.clip(np.searchsorted(edges, kl, side='right') - 1, 0, N_KL_BINS - 1)
    lab = np.full(len(kl), -1)
    for i in range(N_KL_BINS):
        idx = np.where(b == i)[0]
        lo, hi = np.percentile(key[idx], [100 * CONTRAST_FRAC, 100 * (1 - CONTRAST_FRAC)])
        lab[idx[key[idx] <= lo]] = 0
        lab[idx[key[idx] >= hi]] = 1
    s = lab >= 0
    return float(roc_auc_score(lab[s], score[s]))


# ─── emergence (all models) ──────────────────────────────────────────────────

def emergence(task, label, kind):
    S = {s: C.load_states(task, label, s) for s in
         (('probe', 'calibration', 'evaluation') + (('evaluation_noisy',) if kind == 'wm' else ()))}
    cp = calib_params(S['calibration']['kl'])
    D = S['probe']['h'].shape[-1]
    hp = S['probe']['h'][:, M:].reshape(-1, D)
    yp = (S['probe']['kl'][:, M:].ravel() > cp['median']).astype(int)
    if not (0.2 < yp.mean() < 0.8):
        raise AssertionError(f'probe label balance {yp.mean():.3f} out of range ({task}/{label})')
    if C.PROBE_MODE == 'paper':
        clf, sc, v, hp = C.paper_probe(task, label)   # geometry then uses the training-log fit states (as the paper)
    else:
        clf, sc = P.fit_probe(hp, yp)
        v = P.probe_direction_raw(clf, sc).astype(np.float64)
    ev, cal = S['evaluation'], S['calibration']
    mask, mask_cal = P.site_mask(ev['kl'].shape), P.site_mask(cal['kl'].shape)
    ro, ro_cal = P.probe_scores(clf, sc, ev['h']), P.probe_scores(clf, sc, cal['h'])
    logit = np.log(ro / (1 - ro + 1e-12) + 1e-12)
    proj = ev['h'].astype(np.float64) @ v
    proj_cal = cal['h'].astype(np.float64) @ v
    kl = ev['kl'].astype(np.float64)
    out = dict(task=task, label=label, calib=cp, probe_mode=C.PROBE_MODE, n_eval_sites=int(mask.sum()),
               probe_label_rate=float(yp.mean()))
    out['auroc_clean'] = float(roc_auc_score((kl[mask] > cp['median']).astype(int), ro[mask]))
    g_cal, g_scores = pick_gamma(ro_cal, cal['kl'], cp['median'], mask_cal)
    out['gamma_cal'], out['gamma_cal_scores'] = g_cal, g_scores
    gp = PAPER_GAMMA[task]
    cols = dict(ro=ro, logit=logit, proj=proj, kl=kl, C=ct_cal(kl, cp['median'], g_cal))
    if gp is not None:
        cols['C_pg'] = ct_cal(kl, cp['median'], gp)
    gs = GramSet(cols, mask)
    W = gs.boot_weights()
    out['r2_readout_C'] = gs.r2_ci('ro', ['C'], W)
    out['r2_logit_C'] = gs.r2('logit', ['C'])
    out['r2_proj_C'] = gs.r2_ci('proj', ['C'], W)
    out['r2_proj_kl'] = gs.r2_ci('proj', ['kl'], W)
    g_proj, _ = pick_gamma(proj_cal, cal['kl'], cp['median'], mask_cal)
    out['gamma_cal_proj'] = g_proj
    out['r2_proj_C_gamma_proj'] = GramSet(dict(p=proj, x=ct_cal(kl, cp['median'], g_proj)), mask).r2('p', ['x'])
    out['r2_readout_C_paper_gamma'] = gs.r2('ro', ['C_pg']) if gp is not None else None
    out['r2_proj_C_paper_gamma'] = gs.r2('proj', ['C_pg']) if gp is not None else None
    out['r2_readout_kl'] = gs.r2_ci('ro', ['kl'], W)
    out['r_kl_C'] = pearson_ci(gs, 'kl', 'C', W)
    out['gamma_grid_r2'] = {str(g): GramSet(dict(ro=ro, x=ct_cal(kl, cp['median'], g)), mask).r2('ro', ['x'])
                            for g in P.GAMMA_GRID}
    out['geometry'] = P.geometry(hp, clf, sc)
    out['angle_top50'] = mean_angle_top50(v, hp)
    if kind == 'wm':
        nz = S['evaluation_noisy']
        ro_nz = P.probe_scores(clf, sc, nz['h'])
        out['auroc_noisy'] = float(roc_auc_score((nz['kl'][mask] > cp['median']).astype(int), ro_nz[mask]))
        f = lambda a: a[mask]
        out['kl_matched_pooled'] = kl_matched_contrast(np.r_[f(kl), f(nz['kl'])], np.r_[f(ev['rec']), f(nz['rec'])],
                                                       np.r_[f(ro), f(ro_nz)])
        out['kl_matched_clean'] = kl_matched_contrast(f(kl), f(ev['rec']), f(ro))
        out['within_bin_r_rec_C'] = within_bin_r(f(kl), f(ev['rec']), f(cols['C']))
        out['relabel_by_C_auroc'] = contrast_by(f(kl), f(cols['C']), f(ro))
    # ridge scramble (Fig. 1d): h -> C_t, fit on probe-fit episodes, R^2 on evaluation
    he = ev['h'][:, M:].reshape(-1, D)
    hp = S['probe']['h'][:, M:].reshape(-1, D)      # ridge h->C_t always fit on probe-fit episodes
    rng = np.random.default_rng(SCR_SEED)
    tp = ct_cal(S['probe']['kl'], cp['median'], g_cal)[:, M:].ravel()
    te = cols['C'][:, M:].ravel()
    real = r2(te, ridge_fit(hp, tp)(he))
    null = []
    for _ in range(N_SCR):
        sp = scramble_all(S['probe']['kl'], cp['median'], g_cal, rng)[:, M:].ravel()
        se = scramble_all(ev['kl'], cp['median'], g_cal, rng)[:, M:].ravel()
        null.append(r2(se, ridge_fit(hp, sp)(he)))
    null = np.array(null)
    out['ridge_scramble'] = dict(real=real, scrambled_mean=float(null.mean()), scrambled_sd=float(null.std()),
                                 z=float((real - null.mean()) / (null.std() + 1e-12)), n=N_SCR)
    return out, (clf, sc, cp, g_cal, ro, ro_cal)


# ─── external (Sec. 6, Table 4, Fig. 2, Sec. 10.3, Table 5 dR2) ───────────────

def external(task, label, kind, st, ens_models=None, controls=('kl', 'rec', 'emarec')):
    clf, sc, cp, g_cal, ro, ro_cal = st
    ev, cal = C.load_states(task, label, 'evaluation'), C.load_states(task, label, 'calibration')
    mask = P.site_mask(ev['kl'].shape, C.KMAX)
    mask_cal = P.site_mask(cal['kl'].shape, C.KMAX)
    kl = ev['kl'].astype(np.float64)
    a_paper, a_scores = ema_alpha_paper(cal['rec'].astype(np.float64), cal['kl'].astype(np.float64), mask_cal)
    gp = PAPER_GAMMA[task] or g_cal
    cols = dict(kl=kl, rec=ev['rec'].astype(np.float64), emarec=P.ema(ev['rec'], a_paper),
                C_cal=ct_cal(kl, cp['median'], g_cal), C_paperdef=ct_per_traj_median(kl, gp),
                C_cal_pg=ct_cal(kl, cp['median'], gp), readout=ro)
    # Fig. 2c / Phase-2 definition: one global threshold (calibration median here; the
    # paper used the training-log median) and the paper's gamma
    cols['C_glob_pg'] = cols['C_cal_pg']
    for K in C.HORIZONS:
        cols[f'E{K}'] = ev[f'estate_{K}']
    ctrl = list(controls)
    if ens_models is not None:
        ep = C.episodes(task, 'evaluation')
        import torch
        torch.manual_seed(4242)
        cols['ens'] = C.ensemble_disagreement(ens_models, ep['obs'])
        ctrl = ctrl + ['ens']
    gs = GramSet(cols, mask)
    W = gs.boot_weights()
    out = dict(alpha_emarec_paper_rule=a_paper, alpha_scores=a_scores, gamma_cal=g_cal, gamma_paper=gp,
               n_sites=int(mask.sum()), n_episodes=int(mask.shape[0]), controls=ctrl,
               r2_baseline=gs.r2_ci('E10', ctrl, W))
    for name in ('C_paperdef', 'C_cal', 'C_cal_pg'):
        out[f'delta_r2_{name}'] = gs.delta_r2_ci('E10', [name], ctrl, W)
    if 'ens' in ctrl:
        out['delta_r2_C_paperdef_no_ens'] = gs.delta_r2_ci('E10', ['C_paperdef'], [c for c in ctrl if c != 'ens'], W)
        out['delta_r2_C_cal_no_ens'] = gs.delta_r2_ci('E10', ['C_cal'], [c for c in ctrl if c != 'ens'], W)
    out['horizons'] = {K: {n: pearson_ci(gs, n, f'E{K}', W) for n in ('C_paperdef', 'C_cal')} for K in C.HORIZONS}
    out['head_to_head_emarec'] = {n: gs.delta_r2_ci('E10', [n], ['emarec'], W) for n in ('C_paperdef', 'C_cal')}
    # Fig. 2c: r(C_t, E10) real vs time-scrambled
    rng = np.random.default_rng(SCR_SEED + 1)
    scr = {}
    for name, thr in (('C_paperdef', None), ('C_cal', cp['median']), ('C_glob_pg', cp['median'])):
        g = g_cal if name == 'C_cal' else gp
        real = pearsonr(cols[name][mask], cols['E10'][mask])[0]
        null = []
        for _ in range(N_SCR):
            th = np.median(kl[:, M:], axis=1, keepdims=True) if thr is None else thr
            x = scramble_all(kl, th, g, rng)
            null.append(pearsonr(x[mask], cols['E10'][mask])[0])
        null = np.array(null)
        scr[name] = dict(real=float(real), scrambled_mean=float(null.mean()), scrambled_sd=float(null.std()),
                         z=float((real - null.mean()) / (null.std() + 1e-12)), n=N_SCR)
    out['scramble_external'] = scr
    # Sec. 10.3 reward-inflating sites (paper / Phase-1 Sec. 6.4 definition)
    gap = (ev['imag_rew_10'] - ev['real_rew_10'])[mask]
    rec_s, kl_s = cols['rec'][mask], kl[mask]
    drift = (gap > np.percentile(gap, 75)) & (rec_s <= np.percentile(rec_s, 50)) & (kl_s <= np.percentile(kl_s, 50))
    out['reward_inflation'] = dict(
        n_drift=int(drift.sum()), n_sites=int(mask.sum()), frac=float(drift.mean()),
        **{f'{n}_mean_drift': float(cols[n][mask][drift].mean()) if drift.any() else None
           for n in ('C_paperdef', 'C_cal')},
        **{f'{n}_mean_other': float(cols[n][mask][~drift].mean()) for n in ('C_paperdef', 'C_cal')},
        imag_rew_degenerate=bool(np.nanstd(ev['imag_rew_10'][mask]) < 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=['cartpole', 'reacher', 'pendulum', 'cartpole_pomdp'])
    ap.add_argument('--labels', nargs='+')
    a = ap.parse_args()
    P.seed_everything(0)
    reg = C.registry()
    for task in a.tasks:
        ens = None
        if task == 'cartpole':
            ens = [C.load_any(f'outputs/checkpoints/ensemble_seed{i}.pt', 'wm') for i in range(3)]
        for label, path, kind in reg[task]:
            if a.labels and label not in a.labels:
                continue
            f = os.path.join(OUT, f'emergence_{task}_{label}.json')
            if os.path.exists(f):
                continue
            em, st = emergence(task, label, kind)
            if kind == 'pomdp':
                ext = external(task, label, kind, st, controls=('kl', 'rec'))
            else:
                ext = external(task, label, kind, st, ens_models=ens if label == 'primary' else None)
            jdump(dict(emergence=em, external=ext), f)
            print(f"[A] {task}/{label}: AUROC {em['auroc_clean']:.3f} R2(ro,C) {em['r2_readout_C']['point']:.3f} "
                  f"(g={em['gamma_cal']}) R2(ro,KL) {em['r2_readout_kl']['point']:.3f} | "
                  f"dR2 paperdef {ext['delta_r2_C_paperdef']['point']:+.4f} "
                  f"[{ext['delta_r2_C_paperdef']['lo']:+.4f},{ext['delta_r2_C_paperdef']['hi']:+.4f}] "
                  f"cal {ext['delta_r2_C_cal']['point']:+.4f} "
                  f"[{ext['delta_r2_C_cal']['lo']:+.4f},{ext['delta_r2_C_cal']['hi']:+.4f}]", flush=True)


if __name__ == '__main__':
    main()
