#!/usr/bin/env python3.11
"""Out-of-sample Delta R^2 (camera-ready, ICAART 2027 #388), final rule.

Sites: the same 46,700 evaluation sites per task (100 episodes, t in [12, 479]) with E_{t,10}.
Grouped 5-fold CV by trajectory: the 100 evaluation episodes are permuted with a fixed
seed (FOLD_SEED) and split into 5 folds of 20 episodes; the same folds for every
regression and task. For each fold, OLS (with intercept) is fit on the other 4 folds and
predicts the held-out fold. R^2_cv = 1 - SSE/SST over the pooled out-of-fold predictions
(SST about the pooled mean of E). Delta R^2_cv = R^2_cv(base + signal) - R^2_cv(base).
CI: trajectory bootstrap (1,000 resamples, seed 7) over the FIXED out-of-fold predictions
(episodes resampled with replacement; SSE and SST recomputed with the resampled weights).
Delta R^2_cv can be negative. The in-sample Delta R^2 (same columns, GramSet bootstrap)
is reported side by side.

Rows: (a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ensemble disagreement on cartpole);
(b) Task 1: readout (probability, logit) beyond the same controls; (c) Task 2: C_t and the
readout beyond all other signals and beyond the six standard signals; Task 4 Table-4 row
for walker / cheetah (--t4).
Pre-registered reading: a signal adds out-of-sample predictive value on a task only if its
Delta R^2_cv CI excludes zero.
"""
import argparse
import json
import os

import numpy as np
import torch

from src.camera_ready import core as C
from src.rerun import pipeline as P
from src.rerun.stats import GramSet
from run_cr_A_analysis import ct_per_traj_median

FOLD_SEED = 55
K_FOLDS = 5
OUT = 'outputs/camera_ready/CV_delta_r2.json'


def folds_for(n_ep):
    perm = np.random.default_rng(FOLD_SEED).permutation(n_ep)
    fold = np.empty(n_ep, int)
    for k, chunk in enumerate(np.array_split(perm, K_FOLDS)):
        fold[chunk] = k
    return fold


def oof_pred(X, y, site_fold):
    X1 = np.column_stack([np.ones(len(y)), X])
    pred = np.empty_like(y)
    for k in range(K_FOLDS):
        te = site_fold == k
        beta = np.linalg.lstsq(X1[~te], y[~te], rcond=None)[0]
        pred[te] = X1[te] @ beta
    return pred


def r2w(y, pred, w):
    mu = (w * y).sum() / w.sum()
    return 1 - (w * (y - pred) ** 2).sum() / (w * (y - mu) ** 2).sum()


def cv_row(cols, mask, base, add, n_ep, ep_of_site, site_fold, W):
    y = cols['E10'][mask].astype(np.float64)
    Xb = np.column_stack([cols[c][mask] for c in base])
    Xf = np.column_stack([cols[c][mask] for c in base + add])
    pb, pf = oof_pred(Xb, y, site_fold), oof_pred(Xf, y, site_fold)
    one = np.ones_like(y)
    point = r2w(y, pf, one) - r2w(y, pb, one)
    boots = np.array([r2w(y, pf, w[ep_of_site]) - r2w(y, pb, w[ep_of_site]) for w in W])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    gs = GramSet({c: cols[c] for c in set(base + add + ['E10'])}, mask)
    ins = gs.delta_r2_ci('E10', add, base, gs.boot_weights())
    return dict(cv=dict(point=float(point), lo=float(lo), hi=float(hi), r2cv_base=float(r2w(y, pb, one)),
                        r2cv_full=float(r2w(y, pf, one)), adds_oos_value=bool(lo > 0)),
                in_sample=ins, base=base, add=add)


def setup(ev):
    mask = P.site_mask(ev['kl'].shape, C.KMAX)
    n_ep = mask.shape[0]
    ep_of_site, _ = np.nonzero(mask)
    fold = folds_for(n_ep)
    W = np.random.default_rng(P.BOOT_SEED)
    Wt = np.stack([np.bincount(W.integers(0, n_ep, n_ep), minlength=n_ep) for _ in range(P.N_BOOT)]).astype(float)
    return mask, n_ep, ep_of_site, fold[ep_of_site], Wt


def main_tasks():
    from run_t123 import refit, logit_of, sig, GAMMA, ALPHA
    raw = json.load(open('outputs/camera_ready/T123_raw.json'))
    ens = [C.load_any(f'outputs/checkpoints/ensemble_seed{i}.pt', 'wm') for i in range(3)]
    res = {}
    for task in ('cartpole', 'reacher', 'pendulum'):
        clf, sc, w, b0, thr, hp, pr = refit(task)
        ev = C.load_states(task, 'primary', 'evaluation')
        kl = ev['kl'].astype(np.float64)
        lg = logit_of(ev['h'], w, b0)
        cols = dict(E10=ev['estate_10'], kl=kl, rec=ev['rec'].astype(np.float64), emarec=P.ema(ev['rec'], ALPHA[task]),
                    C=ct_per_traj_median(kl, GAMMA[task]), prob=sig(lg), logit=lg,
                    emakl=P.ema(ev['kl'], raw['2'][task]['alpha_emakl']),
                    ent_prior=ev['ent_prior'].astype(np.float64), ent_post=ev['ent_post'].astype(np.float64))
        ctrl = ['kl', 'rec', 'emarec']
        if task == 'cartpole':
            torch.manual_seed(4242)
            cols['ens'] = C.ensemble_disagreement(ens, C.episodes(task, 'evaluation')['obs'])
            ctrl = ctrl + ['ens']
        std6 = ['kl', 'rec', 'emarec', 'emakl', 'ent_prior', 'ent_post']
        mask, n_ep, eps, sf, Wt = setup(ev)
        rows = {
            'a_T4_C_beyond_controls': (ctrl, ['C']),
            'b_T1_readout_prob_beyond_controls': (ctrl, ['prob']),
            'b_T1_readout_logit_beyond_controls': (ctrl, ['logit']),
            'c_T2_C_beyond_all_others_prob': (std6 + ['prob'], ['C']),
            'c_T2_C_beyond_all_others_logit': (std6 + ['logit'], ['C']),
            'c_T2_C_beyond_6_standard': (std6, ['C']),
            'c_T2_readout_prob_beyond_all_others': (std6 + ['C'], ['prob']),
            'c_T2_readout_logit_beyond_all_others': (std6 + ['C'], ['logit']),
            'c_T2_readout_prob_beyond_6_standard': (std6, ['prob']),
            'c_T2_readout_logit_beyond_6_standard': (std6, ['logit']),
        }
        res[task] = {k: cv_row(cols, mask, b, a, n_ep, eps, sf, Wt) for k, (b, a) in rows.items()}
        res[task]['_meta'] = dict(n_sites=int(mask.sum()), n_episodes=n_ep, fold_seed=FOLD_SEED, k=K_FOLDS)
        for k in rows:
            r = res[task][k]
            print(f"[CV] {task} {k}: in-sample {r['in_sample']['point']:+.5f} [{r['in_sample']['lo']:+.5f},{r['in_sample']['hi']:+.5f}] "
                  f"| CV {r['cv']['point']:+.5f} [{r['cv']['lo']:+.5f},{r['cv']['hi']:+.5f}]", flush=True)
    return res


def t4_tasks():
    import run_t4_analyze as T4   # registers walker/cheetah in the pipeline
    t4 = json.load(open(T4.OUT))
    res = {}
    for task in ('walker', 'cheetah'):
        o = t4[task]
        ev = C.load_states(task, 'primary', 'evaluation')
        kl = ev['kl'].astype(np.float64)
        cols = dict(E10=ev['estate_10'], kl=kl, rec=ev['rec'].astype(np.float64),
                    emarec=P.ema(ev['rec'], o['alpha_emarec']), C=ct_per_traj_median(kl, o['gamma']))
        mask, n_ep, eps, sf, Wt = setup(ev)
        r = cv_row(cols, mask, ['kl', 'rec', 'emarec'], ['C'], n_ep, eps, sf, Wt)
        res[task] = {'a_T4_C_beyond_controls': r, '_meta': dict(n_sites=int(mask.sum()), n_episodes=n_ep,
                                                                 gamma=o['gamma'], alpha_emarec=o['alpha_emarec'])}
        print(f"[CV] {task} T4: in-sample {r['in_sample']['point']:+.5f} | CV {r['cv']['point']:+.5f} "
              f"[{r['cv']['lo']:+.5f},{r['cv']['hi']:+.5f}]", flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--t4', action='store_true')
    a = ap.parse_args()
    P.seed_everything(0)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.update(t4_tasks() if a.t4 else main_tasks())
    json.dump(out, open(OUT, 'w'), indent=1, default=P._json_default)
