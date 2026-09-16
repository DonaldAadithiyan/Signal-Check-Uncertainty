#!/usr/bin/env python3.11
"""
Task 4 (gap-closing spec) — Distribution-shift test.

Earns the target claim's "...and distribution shift" clause as a claim distinct
from Phase 4's partial observability. Design, verbatim from the spec: reuse
Phase 1's external-validation machinery (imagined-vs-real rollout, E^state) and
test whether C_t's incremental-R^2 advantage over KL/Recon/EMA, and the dense-
direction causal steering effect (on both probe readout and E^state), hold,
strengthen, or weaken under Set-B-style Gaussian observation noise (sigma=0.1)
relative to the existing Set A (clean) condition -- on all three tasks, not just
cartpole (the original Set A/B pipeline in src/data/collect.py is cartpole-only;
this reuses src/env/dmc_wrapper.DMCEnv's identical noisy-observation mechanism,
already parametrised for any dm_control domain, to get the reacher/pendulum cases
for free without new data-collection code).

This is a genuinely different manipulation from Phase 4: partial observability
changes what the model can see AT ALL (fewer input dimensions); this changes the
INPUT DISTRIBUTION while keeping the observation space and task identical --
Gaussian noise added to every observation dimension the model already sees,
matching the original workshop paper's own Set A/B construction exactly
(CartpoleEnv(noisy=True, noise_std=0.1) / DMCEnv(..., noisy=True, noise_std=0.1)).

Everything downstream of environment construction is copied from
run_phase1_external_validation.py's run_task() unmodified (same site sampling,
same E^state construction, same incremental-R^2 regression, same bootstrap CI
routine) so that Set A vs Set B differ ONLY in the noisy=True/False flag passed
to the environment -- not in any analysis-side difference that could itself
explain a Set A/B gap.

Part 2 (causal steering under shift) reuses Phase 6's dose-response protocol
(src/probe/intervention.py's probe_direction + random_matched_direction, the
50-direction empirical null) at the SAME lambda sweep, run on Set-B trajectories
instead of Set A's, to test whether the (already null-confirmed-decisive-on-
probe / null-confirmed-NOT-significant-on-E^state) causal effects replicate
under this shift.
"""

import os
import json
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from run_phase1_external_validation import (
    ENVS, load_model, collect_trajectories as _collect_trajectories_clean,
    imagined_vs_real_obs, e_state, fit_ema_alpha, ema_series,
    ensemble_disagreement_series, bootstrap_pearson_ci,
    MIN_T, MAX_HORIZON, HORIZONS, SEED, N_TRAJ, N_ENSEMBLE,
)
K_HEADLINE = 10 if 10 in HORIZONS else HORIZONS[-1]
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct, probe_direction, random_matched_direction

OUT_DIR = 'outputs/task4_distribution_shift'
FIG_DIR = 'outputs/figures'
NOISE_STD = 0.1     # matches src/config.py's existing Set-B noise_std exactly
N_NULL = 20          # reduced from Phase 6's 50 -- the causal steering effect is a
                     # REPLICATION check here (does an already-established Phase 6
                     # effect hold under shift), not a first discovery, so a smaller
                     # null is an explicit, stated precision/cost tradeoff (mirrors
                     # Task G's own null-protocol cost/precision tradeoff at 200 vs
                     # 500 sites). Full 50-direction null would be ~2.5x the cost of
                     # the E^state rollout-based steering test below, which is the
                     # dominant cost in this script (each steer_effect call re-runs
                     # a MAX_HORIZON-step imagination rollout twice per site).
N_STEER_SITES = 150  # reduced from Phase 6's 600 for the same reason -- stated
                     # explicitly as a smaller-sample replication, not a rerun at
                     # matched power; if the effect is borderline under shift this
                     # should be flagged for a larger follow-up rather than
                     # over-interpreted at this n.
LAMBDAS = [-2, -1, 0, 1, 2]   # sigma_v multiples, matches Phase 6 exactly


def make_env_shift(spec, seed, noisy):
    if spec['env_cls'] == 'cartpole':
        return CartpoleEnv(task=spec['task'], noisy=noisy, noise_std=NOISE_STD, seed=seed)
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=noisy,
                  noise_std=NOISE_STD, seed=seed)


def collect_trajectories_shift(model, spec, n_traj, cfg, noisy, seed=SEED):
    """Identical body to run_phase1_external_validation.collect_trajectories,
    except environment construction goes through make_env_shift so noisy=True
    can be requested -- the original only ever calls make_env with noisy=False
    baked in. Kept as a literal copy (not a monkeypatch) so this file has no
    hidden coupling to the original's internal state.

    Task 1 fix propagated here: torch.manual_seed must be set before RSSM
    sampling (torch.distributions.Categorical.sample() draws from the global
    unseeded RNG otherwise) -- see run_phase1_external_validation.collect_
    trajectories for the root-cause verification. Without this, Set A vs Set B
    comparisons in this script would be confounded by irreproducible sampling
    noise on top of the actual noise-injection manipulation being tested."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    trajs = []
    for ep in range(n_traj):
        env = make_env_shift(spec, seed=seed + ep, noisy=noisy)
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        act_dim = model.act_dim
        obs_l, act_l, h_l, z_l, kl_l, recon_l, rew_l = [], [], [], [], [], [], []
        done, step = False, 0
        rng = np.random.default_rng(seed + ep)
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                emb = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, emb)
                kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                dec = model.decoder(torch.cat([h, z], dim=-1)).squeeze(0).cpu().numpy()
                recon = float(np.sum((dec - obs) ** 2))

                obs_l.append(obs.copy()); act_l.append(a.copy())
                h_l.append(h.squeeze(0).cpu().numpy().copy())
                z_l.append(post_l.squeeze(0).cpu().numpy().copy())
                kl_l.append(kl); recon_l.append(recon)

                obs, rew, done = env.step(a)
                rew_l.append(rew)
                step += 1
        trajs.append(dict(
            obs=np.array(obs_l, np.float32), act=np.array(act_l, np.float32),
            h=np.array(h_l, np.float32), z=np.array(z_l, np.float32),
            kl=np.array(kl_l, np.float32), recon=np.array(recon_l, np.float32),
            rew=np.array(rew_l, np.float32),
        ))
    return trajs


def run_condition(task, spec, cfg, noisy, clf, scaler, ema_alpha, ensemble_models):
    """Runs the H1/H2 pipeline (incremental R^2 @ K_HEADLINE) on either the
    clean (noisy=False, 'Set A') or noise-shifted (noisy=True, 'Set B') condition,
    reusing the already-fit probe/scaler/EMA-alpha (fit ONCE on the ORIGINAL
    clean training states, never refit per condition -- otherwise a Set B
    regression would be circularly re-calibrated to the shift itself rather than
    testing whether the EXISTING representation/probe still works under shift)."""
    trajs = collect_trajectories_shift(model=spec['_model'], spec=spec, n_traj=N_TRAJ,
                                        cfg=cfg, noisy=noisy, seed=SEED + (5000 if noisy else 0))
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64),
                                gamma=spec['gamma_ct'])
        trj['probe'] = clf.predict_proba(scaler.transform(trj['h']))[:, 1]
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)
        if ensemble_models is not None:
            trj['ens_dis'] = ensemble_disagreement_series(ensemble_models, trj['obs'], cfg)
        else:
            trj['ens_dis'] = np.full(len(trj['kl']), np.nan)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - MAX_HORIZON - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED + (7000 if noisy else 0))
    if len(sites) > 6000:
        sel = rng.choice(len(sites), 6000, replace=False)
        sites = [sites[i] for i in sel]

    rows = {k: [] for k in ['ct', 'kl', 'recon', 'ema_recon', 'ens_dis', 'probe']}
    e_state_by_K = {K: [] for K in HORIZONS}
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, _, _ = imagined_vs_real_obs(spec['_model'], trj, t, MAX_HORIZON, spec['domain'])
        if len(state_dist_full) < MAX_HORIZON:
            continue
        for K in HORIZONS:
            e_state_by_K[K].append(e_state(state_dist_full, K))
        rows['ct'].append(trj['ct'][t]); rows['kl'].append(trj['kl'][t])
        rows['recon'].append(trj['recon'][t]); rows['ema_recon'].append(trj['ema_recon'][t])
        rows['ens_dis'].append(trj['ens_dis'][t]); rows['probe'].append(trj['probe'][t])
    for k in rows:
        rows[k] = np.array(rows[k], dtype=np.float64)
    for K in HORIZONS:
        e_state_by_K[K] = np.array(e_state_by_K[K], dtype=np.float64)

    have_ensemble = not np.all(np.isnan(rows['ens_dis']))
    target = e_state_by_K[K_HEADLINE]
    baseline_cols = ['kl', 'recon', 'ema_recon'] + (['ens_dis'] if have_ensemble else [])
    X_full = np.column_stack([rows['ct']] + [rows[c] for c in baseline_cols])
    X_ctrl = np.column_stack([rows[c] for c in baseline_cols])
    Xs_full = StandardScaler().fit_transform(X_full)
    Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
    r2_full = LinearRegression().fit(Xs_full, target).score(Xs_full, target)
    r2_ctrl = LinearRegression().fit(Xs_ctrl, target).score(Xs_ctrl, target)
    incremental = r2_full - r2_ctrl

    point, lo, hi = bootstrap_pearson_ci(rows['ct'], target, seed=1)
    r_ct_kl, _ = pearsonr(rows['ct'], rows['kl'])

    # ── causal steering: probe direction, dose-response on probe + E^state ──
    v = probe_direction(clf, scaler)
    sigma_v = np.std([float(h @ v) for trj in trajs for h in trj['h']])
    steer_sites_idx = rng.choice(len(sites), min(N_STEER_SITES, len(sites)), replace=False)
    steer_sites = [sites[i] for i in steer_sites_idx]

    def steer_effect(direction, lam, site_list):
        d_probe, d_estate = [], []
        for (ti, t) in site_list:
            trj = trajs[ti]
            h_t = trj['h'][t]
            h_steer = h_t + lam * sigma_v * direction
            ps_base = trj['probe'][t]
            ps_steer = clf.predict_proba(scaler.transform(h_steer.reshape(1, -1)))[0, 1]
            d_probe.append(ps_steer - ps_base)
            # E^state effect: re-decode/imagine from steered h (single-step proxy,
            # matching Phase 6's own protocol exactly for direct comparability)
            trj_mod = dict(trj)
            trj_mod_h = trj['h'].copy()
            trj_mod_h[t] = h_steer
            trj_mod['h'] = trj_mod_h
            dist_steer, _, _ = imagined_vs_real_obs(spec['_model'], trj_mod, t, MAX_HORIZON, spec['domain'])
            dist_base, _, _ = imagined_vs_real_obs(spec['_model'], trj, t, MAX_HORIZON, spec['domain'])
            if len(dist_steer) >= K_HEADLINE and len(dist_base) >= K_HEADLINE:
                d_estate.append(e_state(dist_steer, K_HEADLINE) - e_state(dist_base, K_HEADLINE))
        return float(np.mean(d_probe)), float(np.mean(d_estate)) if d_estate else float('nan')

    dose_probe, dose_estate = {}, {}
    for lam in LAMBDAS:
        dp, de = steer_effect(v, lam, steer_sites)
        dose_probe[lam] = dp
        dose_estate[lam] = de

    # empirical null for the lambda=+2 slope endpoint (matches Phase 6's null scope)
    rng_null = np.random.default_rng(9191)
    null_probe_hi, null_estate_hi = [], []
    for _ in range(N_NULL):
        vr = random_matched_direction(rng_null, v.shape[0])
        dp, de = steer_effect(vr, 2, steer_sites)
        null_probe_hi.append(dp)
        null_estate_hi.append(de)
    z_probe = (dose_probe[2] - np.mean(null_probe_hi)) / (np.std(null_probe_hi) + 1e-12)
    de_valid = [x for x in null_estate_hi if np.isfinite(x)]
    z_estate = ((dose_estate[2] - np.mean(de_valid)) / (np.std(de_valid) + 1e-12)
                if len(de_valid) >= 10 and np.isfinite(dose_estate[2]) else float('nan'))

    return dict(
        n_sites=len(target), incremental_r2=float(incremental),
        r2_full=float(r2_full), r2_ctrl=float(r2_ctrl),
        pearson_r_ct_estate=float(point), ci_lo=float(lo), ci_hi=float(hi),
        r_ct_kl=float(r_ct_kl),
        dose_probe=dose_probe, dose_estate=dose_estate,
        causal_z_probe_lam2=float(z_probe), causal_z_estate_lam2=float(z_estate),
        null_probe_mean=float(np.mean(null_probe_hi)), null_probe_std=float(np.std(null_probe_hi)),
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    for task, spec in ENVS.items():
        print(f"\n{'='*78}\n{task.upper()} — TASK 4 DISTRIBUTION SHIFT\n{'='*78}")
        model, obs_dim, act_dim = load_model(spec['checkpoint'])
        spec = dict(spec)
        spec['_model'] = model

        tr = dict(np.load(spec['training_states']))
        ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])
        ensemble_models = None
        if spec['ensemble'] and all(os.path.exists(p) for p in spec['ensemble']):
            ensemble_models = [load_model(p)[0] for p in spec['ensemble']]

        # probe fit ONCE on original clean training states -- shared by both conditions
        from sklearn.model_selection import train_test_split
        y = binarise_by_median(tr['kl'])
        idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                           stratify=y, random_state=0)
        clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])

        print("  --- Set A (clean, noisy=False) ---")
        res_a = run_condition(task, spec, cfg, noisy=False, clf=clf, scaler=scaler,
                               ema_alpha=ema_alpha, ensemble_models=ensemble_models)
        print(f"    incremental R^2={res_a['incremental_r2']:+.4f}  "
              f"r(C_t,E^state)={res_a['pearson_r_ct_estate']:+.4f} "
              f"[{res_a['ci_lo']:+.4f},{res_a['ci_hi']:+.4f}]  "
              f"causal_z(probe,lam=2)={res_a['causal_z_probe_lam2']:+.2f}  "
              f"causal_z(E^state,lam=2)={res_a['causal_z_estate_lam2']:+.2f}")

        print("  --- Set B (noise-shifted, sigma=0.1) ---")
        res_b = run_condition(task, spec, cfg, noisy=True, clf=clf, scaler=scaler,
                               ema_alpha=ema_alpha, ensemble_models=ensemble_models)
        print(f"    incremental R^2={res_b['incremental_r2']:+.4f}  "
              f"r(C_t,E^state)={res_b['pearson_r_ct_estate']:+.4f} "
              f"[{res_b['ci_lo']:+.4f},{res_b['ci_hi']:+.4f}]  "
              f"causal_z(probe,lam=2)={res_b['causal_z_probe_lam2']:+.2f}  "
              f"causal_z(E^state,lam=2)={res_b['causal_z_estate_lam2']:+.2f}")

        verdict = ('STRENGTHENS' if res_b['incremental_r2'] > res_a['incremental_r2'] * 1.1
                   else 'DEGRADES' if res_b['incremental_r2'] < res_a['incremental_r2'] * 0.9
                   else 'PRESERVED')
        print(f"  VERDICT (incremental R^2, Set A -> Set B): {verdict}")

        results[task] = dict(set_a=res_a, set_b=res_b, verdict_incremental_r2=verdict)

    out_path = os.path.join(OUT_DIR, 'task4_distribution_shift.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"  {'task':<12}{'R2 (A)':>10}{'R2 (B)':>10}{'verdict':>14}"
          f"{'z_probe (A)':>13}{'z_probe (B)':>13}{'z_Estate (A)':>14}{'z_Estate (B)':>14}")
    for task, r in results.items():
        a, b = r['set_a'], r['set_b']
        print(f"  {task:<12}{a['incremental_r2']:>10.4f}{b['incremental_r2']:>10.4f}"
              f"{r['verdict_incremental_r2']:>14}"
              f"{a['causal_z_probe_lam2']:>13.2f}{b['causal_z_probe_lam2']:>13.2f}"
              f"{a['causal_z_estate_lam2']:>14.2f}{b['causal_z_estate_lam2']:>14.2f}")


if __name__ == '__main__':
    main()
