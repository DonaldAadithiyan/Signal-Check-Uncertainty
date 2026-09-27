#!/usr/bin/env python3
"""Sprint P3 -- dense-direction causal dissociation on the corrected pipeline.

For the unit probe direction v (raw h space) and each null direction u:
  steering  h'_t = h_t + lambda * u,  lambda in {-2,-1,0,+1,+2} * sigma_v
            (the SAME absolute step for v and every null direction: norm-matched)
  ablation  h'_t = h_t - (h_t . u) u
Readouts at each evaluation site t:
  probe readout at k=0 (the intervened h'_t) and at k=10 (posterior continuation
  with the real observations obs_{t+1..t+10} and the correct actions a_{t..t+9}),
  and E^state_10 by prior-only imagination from h'_t.
Every condition at a site uses the same torch sampling seed (common random numbers).
Steering statistic: OLS slope of the mean readout on lambda (in sigma units).
Ablation statistic: mean change vs the unintervened run. Both as z against:
  null50      canonical 50 random unit directions (outputs/canonical_null)
  null500     500 seeded random unit directions            (--big-nulls)
  subspace50  50 random unit directions in the orthogonal complement of the
              top-50 PCs of probe-fit h_t                  (--big-nulls)

sigma_v = std(h . v) over calibration-episode states. Sites: N_SITES evaluation
sites drawn uniformly (seeded) from t in [MIN_T, T-21].

Usage: python run_sprint_p3.py [--smoke] [--tasks ...] [--big-nulls] [--models primary|all]
"""
import argparse
import os

import numpy as np
import torch

from src.rerun import pipeline as P

LAMBDAS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
N_SITES = 500
SITE_SEED = 31
DIR_SEED_500 = 5_000
DIR_SEED_SUB = 5_001
CRN_SEED = 77
CANON = ['outputs/canonical_null/random_directions_50_dim256.npz',
         'kaggle/task12/canonical_null/random_directions_50_dim256.npz']
K_READ = 10


@torch.no_grad()
def run_condition(model, clf, sc, obs, act, h_sites, e_idx, t_idx, h_new):
    """Returns probe readout at k=0 and k=K_READ and E^state_10 per site."""
    rssm = model.rssm
    gen = torch.Generator().manual_seed(CRN_SEED)
    e = torch.as_tensor(e_idx, dtype=torch.long)
    t = torch.as_tensor(t_idx, dtype=torch.long)
    h = torch.as_tensor(h_new, dtype=torch.float32)
    emb_t = model.encoder(obs[e, t])
    z0, _ = P._sample_onehot(rssm, rssm.post_net(torch.cat([h, emb_t], -1)), gen)
    r0 = clf.predict_proba(sc.transform(h.numpy()))[:, 1]
    # posterior continuation
    hc, zc = h, z0
    for k in range(1, K_READ + 1):
        hc = P._gru(rssm, hc, zc, act[e, t + k - 1])
        emb = model.encoder(obs[e, t + k])
        zc, _ = P._sample_onehot(rssm, rssm.post_net(torch.cat([hc, emb], -1)), gen)
    r10 = clf.predict_proba(sc.transform(hc.numpy()))[:, 1]
    # prior imagination
    hi, zi, dist = h, z0, []
    for k in range(1, 11):
        hi = P._gru(rssm, hi, zi, act[e, t + k - 1])
        zi, _ = P._sample_onehot(rssm, rssm.prior_net(hi), gen)
        dec = model.decoder(torch.cat([hi, zi], -1))
        dist.append(torch.linalg.vector_norm(dec - obs[e, t + k], dim=-1))
    es = torch.stack(dist, 1).mean(1).numpy()
    return dict(r0=r0, r10=r10, es=es)


def direction_effects(model, clf, sc, obs, act, H, e_idx, t_idx, u, sigma_v, base):
    means = {k: [] for k in ('r0', 'r10', 'es')}
    for lam in LAMBDAS:
        out = base if lam == 0 else run_condition(model, clf, sc, obs, act, H, e_idx, t_idx,
                                                  H + lam * sigma_v * u[None])
        for k in means:
            means[k].append(float(out[k].mean()))
    abl = run_condition(model, clf, sc, obs, act, H, e_idx, t_idx, H - np.outer(H @ u, u))
    res = {}
    for k in means:
        res[f'slope_{k}'] = float(np.polyfit(LAMBDAS, means[k], 1)[0])
        res[f'means_{k}'] = means[k]
        res[f'ablate_{k}'] = float((abl[k] - base[k]).mean())
    return res


def z_against(stat, null_stats):
    n = np.asarray(null_stats)
    pct = float((n < stat).mean() * 100)
    return dict(value=float(stat), null_mean=float(n.mean()), null_std=float(n.std()),
                z=float((stat - n.mean()) / (n.std() + 1e-12)), percentile=pct, n=int(len(n)))


def unit_rows(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def run_model(task, label, path, smoke, big_nulls):
    from run_sprint_p1p2 import states_for, calib_params
    from sklearn.decomposition import PCA
    S, model = states_for(task, label, path, smoke, need_estate=(label == 'primary'))
    if model is None:
        model = P.get_model(task, path, smoke)
    cp = calib_params(S['calibration']['kl'])
    hp = S['probe']['h'][:, P.MIN_T:].reshape(-1, S['probe']['h'].shape[-1])
    clf, sc = P.fit_probe(hp, (S['probe']['kl'][:, P.MIN_T:].ravel() > cp['median']).astype(int))
    v = P.probe_direction_raw(clf, sc).astype(np.float64)
    hc = S['calibration']['h'][:, P.MIN_T:].reshape(-1, hp.shape[1])
    sigma_v = float((hc @ v).std())

    ev = S['evaluation']
    E, T = ev['kl'].shape
    rng = np.random.default_rng(SITE_SEED)
    ts = np.arange(P.MIN_T, T - max(P.HORIZONS) - 1)
    pool = np.stack(np.meshgrid(np.arange(E), ts, indexing='ij'), -1).reshape(-1, 2)
    pick = pool[rng.choice(len(pool), N_SITES, replace=False)]
    e_idx, t_idx = pick[:, 0], pick[:, 1]
    H = ev['h'][e_idx, t_idx].astype(np.float64)
    obs, act = torch.from_numpy(ev['obs']), torch.from_numpy(ev['act'])

    base = run_condition(model, clf, sc, obs, act, H, e_idx, t_idx, H)
    eff_v = direction_effects(model, clf, sc, obs, act, H, e_idx, t_idx, v, sigma_v, base)

    nulls = {}
    for p in CANON:
        if os.path.exists(p):
            nulls['null50'] = unit_rows(np.load(p)['directions'].astype(np.float64))
            break
    if big_nulls:
        r = np.random.default_rng(DIR_SEED_500)
        nulls['null500'] = unit_rows(r.standard_normal((500, H.shape[1])))
        pcs = PCA(n_components=50, random_state=0).fit(hp - hp.mean(0)).components_
        r = np.random.default_rng(DIR_SEED_SUB)
        X = r.standard_normal((50, H.shape[1]))
        X -= (X @ pcs.T) @ pcs
        nulls['subspace50'] = unit_rows(X)

    out = dict(label=label, sigma_v=sigma_v, n_sites=N_SITES, lambdas_sigma=LAMBDAS.tolist(),
               v=eff_v, base_means={k: float(base[k].mean()) for k in base}, nulls={})
    for name, U in nulls.items():
        effs = [direction_effects(model, clf, sc, obs, act, H, e_idx, t_idx, u, sigma_v, base)
                for u in U]
        out['nulls'][name] = {stat: z_against(eff_v[stat], [e[stat] for e in effs])
                              for stat in eff_v if not stat.startswith('means_')}
        s = out['nulls'][name]
        print(f"  {task}/{label} {name}: slope r0 z={s['slope_r0']['z']:+.1f}  "
              f"r10 z={s['slope_r10']['z']:+.1f}  E z={s['slope_es']['z']:+.1f} | ablate r0 "
              f"z={s['ablate_r0']['z']:+.1f} r10 z={s['ablate_r10']['z']:+.1f} "
              f"E z={s['ablate_es']['z']:+.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tasks', nargs='+', default=list(P.TASKS))
    ap.add_argument('--big-nulls', action='store_true')
    ap.add_argument('--models', choices=['primary', 'all'], default='primary')
    a = ap.parse_args()
    P.seed_everything(0)
    torch.set_num_threads(os.cpu_count() or 1)
    res_path = os.path.join(P.out_dir(a.smoke), 'results.json')
    for task in a.tasks:
        models = P.model_paths(task)
        if a.models == 'primary' or a.smoke:
            models = models[:1]
        for label, path in models:
            if not a.smoke and not os.path.exists(path):
                print(f'MISSING {task} {label}: {path}', flush=True)
                continue
            out = run_model(task, label, path, a.smoke, a.big_nulls and label == 'primary')
            P.update_results(res_path, ['p3', task, label], out)


if __name__ == '__main__':
    main()
