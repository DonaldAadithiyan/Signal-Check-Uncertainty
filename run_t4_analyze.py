#!/usr/bin/env python3.11
"""Camera-ready Task 4 analysis (walker-walk, cheetah-run) under the final rule.

Models: outputs/t4_generality/{walker_walk,cheetah_run}/model.pt (run_t4_train.py).
Corrected logging on fixed episode splits (same seeds as the other tasks):
probe-fit 10000-10099, calibration 20000-20049, evaluation 30000-30099.
Paper procedures, with training-log steps refit on probe-fit episodes (the rule):
  * probe: StandardScaler + logistic regression (C=1, lbfgs, random_state=0),
    label 1[KL_t > calibration median]; readout = probability, logit stored
  * gamma: the paper's selection rule -- the value in the paper's grid
    {0.70, 0.80, 0.90, 0.95, 0.99} maximising R^2(readout, C_t) -- applied on probe-fit
    episodes (C_t with the calibration-median threshold)
  * EMARec alpha: the paper's rule -- alpha in {0.02, 0.05, 0.1, 0.2, 0.3, 0.5} maximising
    |corr(EMA(Rec), KL)| -- applied on probe-fit episodes
Core pipeline only: probe AUROC; R^2(readout, C_t) and R^2(readout, KL_t) (Table 3/5.2);
Delta R^2 of C_t (Section-6 definition, per-trajectory median, selected gamma) beyond
{KL, Rec, EMARec} for E_{t,10} (Table 4 protocol); steering z (Table 6 protocol: readout
k=0, k=10 and E vs the canonical 50-direction null, 500 sites, CRN).
"""
import json
import os

os.environ['CR_PROBE'] = 'frozen'
os.environ.setdefault('CR_RAW_DIR', 'outputs/t4_generality/raw')
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from src.rerun import pipeline as P

NEW = {'walker': dict(domain='walker', task='walk', primary='outputs/t4_generality/walker_walk/model.pt',
                      seeds_glob='outputs/t4_generality/none/*.pt'),
       'cheetah': dict(domain='cheetah', task='run', primary='outputs/t4_generality/cheetah_run/model.pt',
                       seeds_glob='outputs/t4_generality/none/*.pt')}
P.TASKS.update(NEW)

from src.camera_ready import core as C  # noqa: E402
from src.rerun.stats import GramSet, pearson_ci  # noqa: E402
from run_cr_A_analysis import ct_per_traj_median  # noqa: E402
import run_cr_A_log as LOG  # noqa: E402
import run_cr_A_causal as RC  # noqa: E402

PAPER_GAMMAS = (0.70, 0.80, 0.90, 0.95, 0.99)
PAPER_ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
OUT = 'outputs/camera_ready/T4_generality.json'
M = C.MIN_T


def analyse(task):
    path = NEW[task]['primary']
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    LOG.log_model(task, 'primary', path, 'wm', splits=('probe', 'calibration', 'evaluation'))
    pr, cal, ev = (C.load_states(task, 'primary', s) for s in ('probe', 'calibration', 'evaluation'))
    thr = float(np.median(cal['kl'][:, M:]))
    hp = pr['h'][:, M:].reshape(-1, 256)
    yp = (pr['kl'][:, M:].ravel() > thr).astype(int)
    if not (0.2 < yp.mean() < 0.8):
        raise AssertionError(f'{task}: label balance {yp.mean():.3f}')
    clf, sc = P.fit_probe(hp, yp)

    def ro(h):
        lg = clf.decision_function(sc.transform(h.reshape(-1, 256).astype(np.float64))).reshape(h.shape[:-1])
        return 1 / (1 + np.exp(-np.clip(lg, -700, 700))), lg
    # gamma: paper rule on probe-fit episodes
    pp, _ = ro(pr['h'])
    mp = P.site_mask(pr['kl'].shape)
    gsc = {g: GramSet(dict(p=pp, c=P.compute_ct(pr['kl'], thr, g)), mp).r2('p', ['c']) for g in PAPER_GAMMAS}
    gamma = max(gsc, key=gsc.get)
    # EMARec alpha: paper rule on probe-fit episodes
    asc = {a: abs(pearsonr(P.ema(pr['rec'], a)[mp], pr['kl'][mp])[0]) for a in PAPER_ALPHAS}
    alpha = max(asc, key=asc.get)
    # emergence on evaluation episodes
    kl = ev['kl'].astype(np.float64)
    prob, lg = ro(ev['h'])
    m = P.site_mask(kl.shape)
    gs = GramSet(dict(p=prob, lg=lg, kl=kl, C=P.compute_ct(kl, thr, gamma)), m)
    W = gs.boot_weights()
    out = dict(gamma=gamma, gamma_scores={str(k): v for k, v in gsc.items()}, alpha_emarec=alpha,
               alpha_scores={str(k): v for k, v in asc.items()}, threshold=thr,
               probe_auroc=float(roc_auc_score((kl[m] > thr).astype(int), prob[m])),
               r2_readout_C=gs.r2_ci('p', ['C'], W), r2_readout_kl=gs.r2_ci('p', ['kl'], W),
               logit=dict(r2_readout_C=gs.r2_ci('lg', ['C'], W), r2_readout_kl=gs.r2_ci('lg', ['kl'], W)),
               saturation=float(((prob[m] < 0.01) | (prob[m] > 0.99)).mean()))
    # Table 4 protocol
    me = P.site_mask(kl.shape, C.KMAX)
    gx = GramSet(dict(E10=ev['estate_10'], C=ct_per_traj_median(kl, gamma), kl=kl, rec=ev['rec'].astype(np.float64),
                      emarec=P.ema(ev['rec'], alpha)), me)
    Wx = gx.boot_weights()
    out.update(n_sites=int(me.sum()), n_episodes=int(me.shape[0]), r2_controls=gx.r2_ci('E10', ['kl', 'rec', 'emarec'], Wx),
               delta_r2_C=gx.delta_r2_ci('E10', ['C'], ['kl', 'rec', 'emarec'], Wx),
               r_C_E10=pearson_ci(gx, 'C', 'E10', Wx))
    # Table 6 protocol (steering; written to CR_RAW_DIR)
    RC.run(task, 'primary', path, 'wm', 'steer')
    st = json.load(open(os.path.join(RC.OUT, f'causal_steer_{task}_primary.json')))['null50']
    out['steering_z'] = {k: st[k]['z'] for k in ('slope_r0', 'slope_r10', 'slope_p0', 'slope_p10', 'slope_es')}
    return out


def main():
    P.seed_everything(0)
    torch.set_num_threads(6)
    os.makedirs(os.environ['CR_RAW_DIR'], exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for task in NEW:
        res[task] = analyse(task)
        o = res[task]
        print(f"[T4] {task}: gamma {o['gamma']} AUROC {o['probe_auroc']:.3f} R2(ro,C) {o['r2_readout_C']['point']:.3f} "
              f"R2(ro,KL) {o['r2_readout_kl']['point']:.3f} dR2 {o['delta_r2_C']['point']:+.4f} "
              f"[{o['delta_r2_C']['lo']:+.4f},{o['delta_r2_C']['hi']:+.4f}] steer z k0 {o['steering_z']['slope_r0']:+.1f} "
              f"k10 {o['steering_z']['slope_r10']:+.1f} E {o['steering_z']['slope_es']:+.2f}", flush=True)
        json.dump(res, open(OUT, 'w'), indent=1, default=P._json_default)


if __name__ == '__main__':
    main()
