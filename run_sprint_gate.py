#!/usr/bin/env python3
"""Sprint Step 2 -- W1 impact gate.

On each task's primary model, run the SAME fixed-seed episodes through both
action conventions and compare:
  correct: observe_step(h_{t-1}, z_{t-1}, a_{t-1}, obs_t)
  buggy:   observe_step(h_{t-1}, z_{t-1}, a_t,     obs_t)
Reports mean/median KL_t, mean Rec_t, per-step Pearson r of KL between
conventions, and probe AUROC under each convention (probe refit per convention
on the probe-fit episodes, label threshold = that convention's calibration
median). Gate (fixed in advance): proceed only if on ALL tasks r >= 0.90 and
|AUROC_correct - AUROC_buggy| <= 0.03.

Usage: python run_sprint_gate.py [--smoke]
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from src.rerun import pipeline as P

R_MIN, DAUC_MAX = 0.90, 0.03


def run_task(task, smoke):
    model = P.get_model(task, P.TASKS[task]['primary'], smoke)
    eps = {s: P.episodes_cached(task, s) for s in ('probe', 'calibration', 'gate')}
    res, kl_gate = {}, {}
    for conv in ('correct', 'buggy'):
        st = {s: P.log_states(model, eps[s], P.TORCH_SEEDS[s], convention=conv) for s in eps}
        m = P.MIN_T
        thresh = float(np.median(st['calibration']['kl'][:, m:]))
        hp = st['probe']['h'][:, m:].reshape(-1, st['probe']['h'].shape[-1])
        yp = (st['probe']['kl'][:, m:].ravel() > thresh).astype(int)
        clf, sc = P.fit_probe(hp, yp)
        g = st['gate']
        score = P.probe_scores(clf, sc, g['h'][:, m:]).ravel()
        yg = (g['kl'][:, m:].ravel() > thresh).astype(int)
        kl_gate[conv] = g['kl'][:, m:].ravel()
        res[conv] = dict(
            kl_mean=float(g['kl'][:, m:].mean()), kl_median=float(np.median(g['kl'][:, m:])),
            rec_mean=float(g['rec'][:, m:].mean()), calib_kl_median=thresh,
            probe_auroc=float(roc_auc_score(yg, score)),
            label_rate=float(yg.mean()))
    r = float(pearsonr(kl_gate['correct'], kl_gate['buggy'])[0])
    rho = float(np.corrcoef(np.argsort(np.argsort(kl_gate['correct'])),
                            np.argsort(np.argsort(kl_gate['buggy'])))[0, 1])
    dauc = res['correct']['probe_auroc'] - res['buggy']['probe_auroc']
    res.update(kl_pearson_r=r, kl_spearman_rho=rho, auroc_diff=float(dauc),
               passes=bool(r >= R_MIN and abs(dauc) <= DAUC_MAX),
               n_episodes=len(P.SPLITS['gate']), sites_per_episode=P.EP_LEN - P.MIN_T)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tasks', nargs='+', default=list(P.TASKS))
    a = ap.parse_args()
    P.seed_everything(0)
    torch.set_num_threads(os.cpu_count() or 1)
    out = {}
    for task in a.tasks:
        out[task] = run_task(task, a.smoke)
        t = out[task]
        print(f"{task:9s} r(KL)={t['kl_pearson_r']:.3f}  KL mean {t['correct']['kl_mean']:.2f} vs "
              f"{t['buggy']['kl_mean']:.2f}  AUROC {t['correct']['probe_auroc']:.3f} vs "
              f"{t['buggy']['probe_auroc']:.3f}  -> {'PASS' if t['passes'] else 'FAIL'}", flush=True)
    verdict = all(out[t]['passes'] for t in out) and set(out) == set(P.TASKS)
    out['_gate'] = dict(proceed_sept29=bool(verdict), r_min=R_MIN, dauc_max=DAUC_MAX,
                        smoke=a.smoke)
    d = P.out_dir(a.smoke)
    with open(os.path.join(d, 'gate.json'), 'w') as f:
        json.dump(out, f, indent=2)
    P.update_results(os.path.join(d, 'results.json'), ['w1_gate'], out)
    print('GATE:', 'PROCEED (Sept 29)' if verdict else 'STOP -> Oct 22')


if __name__ == '__main__':
    main()
