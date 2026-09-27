#!/usr/bin/env python3
"""Sprint P4 -- cheap probe controls on the corrected pipeline.

  PC-scrubbed probe: project raw h_t onto the orthogonal complement of the
    top-50 PCs of probe-fit h_t, refit the logistic probe (same labels/settings).
  MLP probe: one hidden layer (64 ReLU units) on standardised h_t, Adam,
    early stopping on 10 held-out PROBE-FIT EPISODES (episode-level split, no
    step-level split), patience 5 epochs.
Reported on evaluation episodes: AUROC vs 1[KL_t > calibration median], and
R^2(readout, C_t^past) with the calibration-selected gamma from P1's rule.

Usage: python run_sprint_p4.py [--smoke] [--tasks ...]
"""
import argparse
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.rerun import pipeline as P
from src.rerun.stats import GramSet

N_VAL_EP = 10
MAX_EPOCHS = 200
PATIENCE = 5


def fit_mlp(h_ep, y_ep):
    """h_ep: (E,T',D), y_ep: (E,T'). Last N_VAL_EP episodes validate."""
    tr_h, va_h = h_ep[:-N_VAL_EP].reshape(-1, h_ep.shape[-1]), h_ep[-N_VAL_EP:].reshape(-1, h_ep.shape[-1])
    tr_y, va_y = y_ep[:-N_VAL_EP].ravel(), y_ep[-N_VAL_EP:].ravel()
    sc = StandardScaler().fit(tr_h)
    Xtr, Xva = sc.transform(tr_h), sc.transform(va_h)
    clf = MLPClassifier(hidden_layer_sizes=(64,), random_state=0, batch_size=256,
                        learning_rate_init=1e-3)
    best, best_state, bad, hist = -1, None, 0, []
    rng = np.random.default_rng(0)
    for ep in range(MAX_EPOCHS):
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), 4096):
            idx = perm[b:b + 4096]
            clf.partial_fit(Xtr[idx], tr_y[idx], classes=[0, 1])
        auc = roc_auc_score(va_y, clf.predict_proba(Xva)[:, 1])
        hist.append(float(auc))
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = ([w.copy() for w in clf.coefs_], [b.copy() for b in clf.intercepts_])
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    clf.coefs_, clf.intercepts_ = best_state
    return clf, sc, dict(epochs=len(hist), best_val_auroc=float(best))


def main():
    from run_sprint_p1p2 import states_for, calib_params, pick_gamma, variant
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tasks', nargs='+', default=list(P.TASKS))
    a = ap.parse_args()
    P.seed_everything(0)
    torch.set_num_threads(os.cpu_count() or 1)
    res_path = os.path.join(P.out_dir(a.smoke), 'results.json')
    for task in a.tasks:
        label, path = P.model_paths(task)[0]
        if not a.smoke and not os.path.exists(path):
            print(f'MISSING {task}: {path}')
            continue
        S, _ = states_for(task, label, path, a.smoke, need_estate=True)
        cp = calib_params(S['calibration']['kl'])
        m = P.MIN_T
        hp_ep = S['probe']['h'][:, m:]
        yp_ep = (S['probe']['kl'][:, m:] > cp['median']).astype(int)
        D = hp_ep.shape[-1]
        hp = hp_ep.reshape(-1, D)
        mu = hp.mean(0)
        pcs = PCA(n_components=50, random_state=0).fit(hp - mu).components_

        def scrub(h):
            x = h.reshape(-1, D) - mu
            return (x - (x @ pcs.T) @ pcs).reshape(h.shape)

        ev, cal = S['evaluation'], S['calibration']
        mask_ev, mask_cal = P.site_mask(ev['kl'].shape), P.site_mask(cal['kl'].shape)
        y_ev = (ev['kl'][:, m:] > cp['median']).astype(int).ravel()
        out = {}
        readouts = {}
        clf, sc = P.fit_probe(hp, yp_ep.ravel())
        readouts['linear'] = (P.probe_scores(clf, sc, ev['h']), P.probe_scores(clf, sc, cal['h']))
        clf_s, sc_s = P.fit_probe(scrub(hp_ep).reshape(-1, D), yp_ep.ravel())
        readouts['pc_scrubbed'] = (P.probe_scores(clf_s, sc_s, scrub(ev['h'])),
                                   P.probe_scores(clf_s, sc_s, scrub(cal['h'])))
        mlp, sc_m, info = fit_mlp(hp_ep, yp_ep)
        f = lambda h: mlp.predict_proba(sc_m.transform(h.reshape(-1, D)))[:, 1].reshape(h.shape[:-1])
        readouts['mlp'] = (f(ev['h']), f(cal['h']))
        for name, (ro, ro_cal) in readouts.items():
            g, _ = pick_gamma(ro_cal, cal['kl'], cp, 'C_past', mask_cal)
            gs = GramSet(dict(ro=ro, cpast=variant('C_past', ev['kl'], cp, g),
                              kl=ev['kl'].astype(np.float64)), mask_ev)
            W = gs.boot_weights()
            out[name] = dict(eval_auroc=float(roc_auc_score(y_ev, ro[:, m:].ravel())),
                             gamma_c_past=g, r2_c_past=gs.r2_ci('ro', ['cpast'], W),
                             r2_kl=gs.r2_ci('ro', ['kl'], W))
        out['mlp']['training'] = info
        out['pc_scrubbed']['var_removed'] = float(
            PCA(n_components=50, random_state=0).fit(hp - mu).explained_variance_ratio_.sum())
        P.update_results(res_path, ['p4', task], out)
        print(task, {k: (round(v['eval_auroc'], 3), round(v['r2_c_past']['point'], 3))
                     for k, v in out.items()}, flush=True)


if __name__ == '__main__':
    main()
