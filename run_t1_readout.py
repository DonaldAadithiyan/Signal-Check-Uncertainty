#!/usr/bin/env python3.11
"""Camera-ready Task 1 (ICAART 2027 #388, reviewer R2): direct predictive value of
the learned readout for imagination error E_{t,10}.

Same data and definitions as the paper's Table 4 (Phase 1 + addendum Sec. 1.1):
  * frozen primary checkpoints; probe fit on 60% of each model's training log
    (stratified, random_state=0), v = probe_direction (unit, raw h space);
  * Phase-1 evaluation set rebuilt with the ORIGINAL code path
    (run_phase1_external_validation.collect_trajectories, seed 2024, 100 trajectories,
    6000 sites, torch.manual_seed(2024) first) -- identical RNG consumption order to
    run_addendum_phase1_hardening.rebuild_site_dataset; the only change is that h_t
    and trajectory-level arrays are also retained;
  * C_t: per-trajectory median, gamma 0.95/0.70/0.90; EMARec alpha by the paper's
    rule (0.02/0.5/0.02); controls {KL, Rec, EMARec} (+ ensemble disagreement on
    cartpole, as in Table 4);
  * Delta R^2 and its CI: incremental_r2 / trajectory_stratified_bootstrap_incremental_r2
    from run_addendum_phase1_hardening.py (1000 trajectory resamples, seed 0).
Readout = v^T h_t (projection; affine-equivalent to the probe logit). The probe
probability is reported as a secondary row. Null: each of the 50 canonical random
directions u in place of v (u^T h_t); z and percentile of v against the 50.

Hard stop: the rebuilt C_t Delta R^2 must equal the published Table 4 point values.
"""
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct, probe_direction
from run_phase1_external_validation import (
    ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, ensemble_disagreement_series, MIN_T, MAX_HORIZON,
    HORIZONS, SEED, N_TRAJ)
from run_addendum_phase1_hardening import incremental_r2, trajectory_stratified_bootstrap_incremental_r2

OUT = 'outputs/camera_ready'
DATA = os.path.join(OUT, 'task_data')
CANON = 'outputs/canonical_null/random_directions_50_dim256.npz'
TASKS = ['cartpole', 'reacher', 'pendulum']
PUBLISHED_T4 = {'cartpole': (0.0280, 0.0154, 0.0433), 'reacher': (0.0182, 0.0083, 0.0309),
                'pendulum': (0.0019, 0.0001, 0.0051)}
N_BOOT = 1000


def rebuild(task, spec, cfg):
    """Copy of run_addendum_phase1_hardening.rebuild_site_dataset (same RNG order),
    additionally returning h, t per site and the full trajectories."""
    torch.manual_seed(SEED)
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])
    ensemble_models = None
    if spec['ensemble'] and all(os.path.exists(p) for p in spec['ensemble']):
        ensemble_models = [load_model(p)[0] for p in spec['ensemble']]
    y = binarise_by_median(tr['kl'])
    idx_tr, _ = train_test_split(np.arange(len(tr['h'])), test_size=0.40, stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])
    trajs = collect_trajectories(model, spec, N_TRAJ, cfg)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64), gamma=spec['gamma_ct'])
        trj['probe'] = clf.predict_proba(scaler.transform(trj['h']))[:, 1]
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)
        trj['ens_dis'] = (ensemble_disagreement_series(ensemble_models, trj['obs'], cfg)
                          if ensemble_models is not None else np.full(len(trj['kl']), np.nan))
    sites = []
    for ti, trj in enumerate(trajs):
        for t in range(MIN_T, len(trj['obs']) - MAX_HORIZON - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED)
    if len(sites) > 6000:
        sel = rng.choice(len(sites), 6000, replace=False)
        sites = [sites[i] for i in sel]
    rows = {k: [] for k in ['ct', 'kl', 'recon', 'ema_recon', 'ens_dis', 'probe', 'traj_id', 't', 'h']}
    E = {K: [] for K in HORIZONS}
    for (ti, t) in sites:
        trj = trajs[ti]
        d, _, _ = imagined_vs_real_obs(model, trj, t, MAX_HORIZON, spec['domain'])
        if len(d) < MAX_HORIZON:
            continue
        for K in HORIZONS:
            E[K].append(e_state(d, K))
        for k in ('ct', 'kl', 'recon', 'ema_recon', 'ens_dis', 'probe', 'h'):
            rows[k].append(trj[k][t])
        rows['traj_id'].append(ti)
        rows['t'].append(t)
    rows = {k: np.array(v) for k, v in rows.items()}
    E = {K: np.array(v, np.float64) for K, v in E.items()}
    have_ens = not np.all(np.isnan(rows['ens_dis']))
    return dict(rows=rows, E=E, have_ens=have_ens, trajs=trajs, clf=clf, scaler=scaler,
                ema_alpha=ema_alpha, model=model)


def save_task_data(task, R, v):
    os.makedirs(DATA, exist_ok=True)
    rows, trajs = R['rows'], R['trajs']
    T = min(len(t['kl']) for t in trajs)
    np.savez(os.path.join(DATA, f'{task}_sites.npz'), v=v, ema_alpha=R['ema_alpha'], have_ens=R['have_ens'],
             **{f'site_{k}': v_ for k, v_ in rows.items()}, **{f'E{K}': R['E'][K] for K in HORIZONS})
    np.savez(os.path.join(DATA, f'{task}_trajs.npz'),
             **{k: np.stack([t[k][:T] for t in trajs]) for k in ('obs', 'act', 'h', 'z', 'kl', 'recon', 'rew',
                                                                   'ct', 'probe', 'ema_recon', 'ens_dis')})


def boot_r(x, y, tid, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(tid)
    groups = [np.where(tid == i)[0] for i in ids]
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))])
        out.append(pearsonr(x[idx], y[idx])[0])
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


def signal_stats(rows, x, target, base, boot=True):
    r = float(pearsonr(x, target)[0])
    rr = dict(rows); rr['ct'] = x
    if boot:
        point, lo, hi, _, n_deg, _ = trajectory_stratified_bootstrap_incremental_r2(rr, target, base, True)
        return dict(r=r, r_ci=boot_r(x, target, rows['traj_id']), r2=r * r,
                    delta_r2=dict(point=point, lo=lo, hi=hi, n_degenerate=n_deg))
    return dict(r=r, r2=r * r, delta_r2=float(incremental_r2(rr, target, base, True)))


def zpct(val, null):
    null = np.asarray(null)
    return dict(z=float((val - null.mean()) / (null.std() + 1e-12)), percentile=float((null < val).mean() * 100),
                null_mean=float(null.mean()), null_sd=float(null.std()), null_max=float(null.max()))


def main():
    np.random.seed(0)
    torch.set_num_threads(4)
    cfg = XS_CONFIG.copy()
    U = np.load(CANON)['directions'].astype(np.float64)
    if U.shape != (50, 256) or not np.allclose(np.linalg.norm(U, axis=1), 1, atol=1e-5):
        raise AssertionError(f'canonical null malformed: {U.shape}')
    res = {}
    for task in TASKS:
        print(f'[T1] {task}: rebuilding Phase-1 site dataset ...', flush=True)
        R = rebuild(task, ENVS[task], cfg)
        rows, y = R['rows'], R['E'][10]
        base = ['kl', 'recon', 'ema_recon'] + (['ens_dis'] if R['have_ens'] else [])
        # hard stop: reproduce published Table 4
        ct = trajectory_stratified_bootstrap_incremental_r2(rows, y, base, R['have_ens'])
        pub = PUBLISHED_T4[task]
        if round(ct[0], 4) != pub[0] or round(ct[1], 4) != pub[1] or round(ct[2], 4) != pub[2]:
            raise AssertionError(f'{task}: rebuilt Table 4 {ct[:3]} != published {pub}; STOP')
        v = probe_direction(R['clf'], R['scaler']).astype(np.float64)
        v /= np.linalg.norm(v)
        save_task_data(task, R, v)
        H = rows['h'].astype(np.float64)
        ro = H @ v
        out = dict(n_sites=int(len(y)), n_trajectories=int(len(np.unique(rows['traj_id']))), controls=base,
                   ema_alpha=R['ema_alpha'], gamma=ENVS[task]['gamma_ct'],
                   C_t=signal_stats(rows, rows['ct'].astype(np.float64), y, base),
                   readout_projection=signal_stats(rows, ro, y, base),
                   readout_probability=signal_stats(rows, rows['probe'].astype(np.float64), y, base),
                   r2_controls=float(incremental_r2(dict(rows, ct=np.zeros(len(y))), y, base, True) * 0))
        from sklearn.linear_model import LinearRegression
        Xc = np.column_stack([rows[c] for c in base])
        out['r2_controls'] = float(LinearRegression().fit(Xc, y).score(Xc, y))
        null = [signal_stats(rows, H @ u, y, base, boot=False) for u in U]
        out['null'] = dict(
            abs_r=zpct(abs(out['readout_projection']['r']), [abs(n['r']) for n in null]),
            r2=zpct(out['readout_projection']['r2'], [n['r2'] for n in null]),
            delta_r2=zpct(out['readout_projection']['delta_r2']['point'], [n['delta_r2'] for n in null]),
            per_direction=null)
        # reference: C_t's standalone R^2 and dR2 against the same null distribution
        out['null']['C_t_vs_null_r2'] = zpct(out['C_t']['r2'], [n['r2'] for n in null])
        out['null']['C_t_vs_null_delta_r2'] = zpct(out['C_t']['delta_r2']['point'], [n['delta_r2'] for n in null])
        res[task] = out
        p, c = out['readout_projection'], out['C_t']
        print(f"  C_t: r {c['r']:+.3f} R2 {c['r2']:.3f} dR2 {c['delta_r2']['point']:+.4f} "
              f"[{c['delta_r2']['lo']:+.4f},{c['delta_r2']['hi']:+.4f}] (matches Table 4)", flush=True)
        print(f"  readout: r {p['r']:+.3f} R2 {p['r2']:.3f} dR2 {p['delta_r2']['point']:+.4f} "
              f"[{p['delta_r2']['lo']:+.4f},{p['delta_r2']['hi']:+.4f}]  null z: R2 {out['null']['r2']['z']:+.2f} "
              f"(pct {out['null']['r2']['percentile']:.0f}), dR2 {out['null']['delta_r2']['z']:+.2f} "
              f"(pct {out['null']['delta_r2']['percentile']:.0f})", flush=True)
        with open(os.path.join(OUT, 'T1_readout.json'), 'w') as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    main()
