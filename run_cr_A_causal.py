#!/usr/bin/env python3.11
"""Camera-ready Part A, step 3: interventions on corrected states.

  steer  : Table 6 -- steer h'_t = h_t + lambda*sigma_v*u, lambda in {-2..2}, for v
           and the 50 canonical null directions (same absolute step), plus ablation
           h' = h - (h.u)u (Sec. 5.2 'z ~ -22' sanity check). 500 evaluation sites.
  noisy  : Sec. 8.2 -- same on primary models with sigma=0.1 observation noise fed
           to the model (E measured against the clean observations).
  pomdp  : Sec. 8.2 -- ablation of v on the 5 full/partial cartpole pairs, 600 sites.
sigma_v = sd(v.h) on calibration-episode states (held-out pool).

Usage: python run_cr_A_causal.py --mode steer --tasks cartpole [--labels primary]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.camera_ready import core as C
from src.camera_ready import causal as CZ
from src.rerun import pipeline as P
from run_sprint_p1p2 import calib_params

OUT = os.path.join(C.ROOT, 'A_raw')
SITE_SEED = 31


def fit_probe(task, label):
    pr, cal = C.load_states(task, label, 'probe'), C.load_states(task, label, 'calibration')
    cp = calib_params(cal['kl'])
    D = pr['h'].shape[-1]
    hp = pr['h'][:, C.MIN_T:].reshape(-1, D)
    clf, sc = P.fit_probe(hp, (pr['kl'][:, C.MIN_T:].ravel() > cp['median']).astype(int))
    v = P.probe_direction_raw(clf, sc).astype(np.float64)
    sigma_v = float((cal['h'][:, C.MIN_T:].reshape(-1, D) @ v).std())
    return clf, sc, v, sigma_v, hp


def run(task, label, path, kind, mode):
    t0 = time.time()
    model = C.load_any(path, kind)
    clf, sc, v, sigma_v, hp = fit_probe(task, label)
    if mode == 'noisy':
        ev = C.load_states(task, label, 'evaluation_noisy')
        ep = C.episodes(task, 'evaluation_noisy')
        sites = CZ.Sites(model, clf, sc, ev['h'], dict(obs=ep['obs_clean'], act=ep['act']), 500, SITE_SEED,
                         noisy_obs=ep['obs'], v=v)
    elif mode == 'taskg':
        # paper Sec. 5.2 / Task G site rule: 10 sites per trajectory, 60 trajectories,
        # sampled with p proportional to C_t + 0.1 (gamma 0.95, calibration median)
        from run_sprint_p1p2 import calib_params as _cp
        ev = C.load_states(task, label, 'evaluation')
        ep = C.episodes(task, 'evaluation')
        thr = _cp(C.load_states(task, label, 'calibration')['kl'])['median']
        ct = P.compute_ct(ev['kl'], thr, 0.95)
        rng = np.random.default_rng(0)
        T = ev['kl'].shape[1]
        valid = np.arange(C.MIN_T, T - max(P.HORIZONS) - 1)
        pick = []
        for e in range(60):
            w = ct[e, valid] + 0.1
            for t in rng.choice(valid, size=10, replace=False, p=w / w.sum()):
                pick.append((e, int(t)))
        sites = CZ.Sites(model, clf, sc, ev['h'], ep, None, None, pick=np.array(pick), v=v)
    else:
        ev = C.load_states(task, label, 'evaluation')
        ep = C.episodes(task, 'evaluation')
        sites = CZ.Sites(model, clf, sc, ev['h'], ep, 600 if mode == 'pomdp' else 500, SITE_SEED, v=v)
    steer = mode not in ('pomdp', 'taskg')
    eff_v = sites.effects(v, sigma_v, steer=steer)
    null = [sites.effects(u, sigma_v, steer=steer) for u in CZ.canonical_null()]
    out = dict(task=task, label=label, mode=mode, sigma_v=sigma_v, n_sites=len(sites.e),
               lambdas_sigma=CZ.LAMBDAS.tolist(), v=eff_v,
               base_means={k: float(x.mean()) for k, x in sites.base.items()},
               null50=CZ.score(eff_v, null), null_effects=null, seconds=time.time() - t0)
    with open(os.path.join(OUT, f'causal_{mode}_{task}_{label}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=P._json_default)
    s = out['null50']
    msg = (f"[{mode}] {task}/{label}: ablate r0 z={s['ablate_r0']['z']:+.1f} r10 z={s['ablate_r10']['z']:+.1f} "
           f"E z={s['ablate_es']['z']:+.2f}")
    if steer:
        msg = (f"[{mode}] {task}/{label}: steer prob r0 z={s['slope_r0']['z']:+.1f} r10 z={s['slope_r10']['z']:+.1f} "
               f"proj p0 z={s['slope_p0']['z']:+.1f} p10 z={s['slope_p10']['z']:+.1f} "
               f"E z={s['slope_es']['z']:+.2f} | " + msg.split(': ', 1)[1])
    print(msg + f'  ({out["seconds"]:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['steer', 'noisy', 'pomdp', 'taskg'], required=True)
    ap.add_argument('--tasks', nargs='+', default=['cartpole', 'reacher', 'pendulum'])
    ap.add_argument('--labels', nargs='+')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    P.seed_everything(0)
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    reg = C.registry()
    tasks = ['cartpole_pomdp'] if a.mode == 'pomdp' else a.tasks
    for task in tasks:
        for label, path, kind in reg[task]:
            if a.labels and label not in a.labels:
                continue
            if a.mode in ('noisy', 'taskg') and label != 'primary':
                continue
            if os.path.exists(os.path.join(OUT, f'causal_{a.mode}_{task}_{label}.json')):
                continue
            run(task, label, path, kind, a.mode)


if __name__ == '__main__':
    main()
