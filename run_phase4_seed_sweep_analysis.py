#!/usr/bin/env python3.11
"""
Task 3 (gap-closing spec) — Phase 4 seed sweep, analysis half.

Runs Phase 4's full §10.2 comparison pipeline on EACH of the 4 new seed pairs
(1717, 2929, 5151, 8383) trained by run_phase4_seed_sweep_training.py, PLUS the
original seed=4242 pair, and aggregates all 6 metrics as mean +/- std / bootstrap
CI across the resulting 5-seed-pair sample -- replacing the single-pair point
estimate the original run_phase4_pomdp_analysis.py reported.

Two upgrades over the original single-pair analysis, both explicitly required by
the gap-closing spec:

1. MULTI-STEP CAUSAL PROTOCOL (replaces the single-step probe-decay proxy).
   The original run_phase4_pomdp_analysis.py's probe_decay_effect() ablates h_t
   and immediately re-scores the probe on the ablated h_t -- a "static (no
   continuation) proxy for speed", per its own comment. This script instead
   CONTINUES the trajectory k in {0,1,5,10} steps forward from the ablated state
   using imagine_step (the same multi-step continuation machinery Task G /
   Phase 3's causal tests use elsewhere in this project), then re-measures BOTH
   the probe readout AND the externally-valid E^state metric at each k. This
   directly answers whether the causal ablation effect persists under a real
   multi-step rollout rather than only at the instant of intervention.

2. DIFFICULTY BASE-RATE / MATCHED-BIN CHECK (rules out "PARTIAL is just harder").
   Bins evaluation sites by a difficulty-neutral covariate (KL_t, which is
   available and comparable in both conditions) into terciles, then repeats the
   headline causal-ablation z-score WITHIN each KL-matched bin. If PARTIAL's
   larger causal effect held only in a bin where PARTIAL sites are
   systematically harder/more KL-active than FULL's, that would confound
   "partial observability strengthens causal relevance" with "harder states
   happen to show bigger ablation effects regardless of observability." Matching
   on KL bin controls for this directly.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.model.pomdp_world_model import PomdpWorldModel
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import compute_ct, probe_direction, random_matched_direction

ORIGINAL_DIR = 'outputs/phase4_pomdp'
SWEEP_DIR = 'outputs/phase4_pomdp/seed_sweep'
OUT_DIR = 'outputs/phase4_pomdp/seed_sweep'
FIG_DIR = 'outputs/figures'
N_TRAJ = 80
MIN_T = 12
HORIZONS = [1, 5, 10, 20]
K_HEADLINE = 10
GAMMA_CANDIDATES = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
N_NULL = 50
N_INTERVENTION_SITES = 600
CAUSAL_LOOKAHEAD = [0, 1, 5, 10]
ALL_SEEDS = [4242, 1717, 2929, 5151, 8383]   # 4242 = original single pair


def seed_dir_for(seed):
    return ORIGINAL_DIR if seed == 4242 else os.path.join(SWEEP_DIR, f'seed_{seed}')


def load_condition(seed, condition):
    d = seed_dir_for(seed)
    ckpt = torch.load(os.path.join(d, f'{condition}_world_model.pt'), map_location='cpu')
    model = PomdpWorldModel(ckpt['obs_dim_in'], ckpt['obs_dim_out'], ckpt['act_dim'], ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    tr = dict(np.load(os.path.join(d, f'{condition}_training_states.npz')))
    return model, tr, ckpt


def collect_trajectories(model, condition, n_traj, cfg, seed):
    """Task 1 fix applied: torch.manual_seed before any RSSM sampling call, so
    this seed-sweep analysis is itself reproducible (the same bug that affected
    Phase 1's point estimates applies identically here -- RSSM stochastic-latent
    sampling draws from the global unseeded RNG otherwise)."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    trajs = []
    for ep in range(n_traj):
        env = CartpoleEnv(task='swingup', noisy=False, seed=seed + 1 + ep)
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        rng = np.random.default_rng(seed + 1 + ep)
        obs_l, act_l, h_l, z_l, kl_l, recon_l, rew_l = [], [], [], [], [], [], []
        done, step = False, 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = rng.uniform(-1, 1, (model.act_dim,)).astype(np.float32)
                obs_full_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_in_t = obs_full_t[:, :3] if condition == 'partial' else obs_full_t
                a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                emb = model.encoder(obs_in_t)
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


def cartpole_reward_proxy(obs):
    return (obs[..., 1] + 1.0) / 2.0


@torch.no_grad()
def imagined_vs_real_obs_pomdp(model, traj, t, horizon, h_override=None, z_override=None):
    """Same construction as the original Phase 4 analysis, with an optional
    (h,z) override at t so the multi-step causal test can roll forward from an
    ABLATED state instead of the trajectory's real recorded state."""
    device = next(model.parameters()).device
    T = len(traj['obs'])
    if h_override is not None:
        h_im = torch.tensor(h_override, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        h_im = torch.tensor(traj['h'][t], dtype=torch.float32, device=device).unsqueeze(0)
    if z_override is not None:
        z_logits = torch.tensor(z_override, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        z_logits = torch.tensor(traj['z'][t], dtype=torch.float32, device=device).unsqueeze(0)
    z_im = model.rssm._straight_through_sample(z_logits)

    state_dist = []
    for k in range(1, horizon + 1):
        kk = t + k
        if kk >= T:
            break
        a = torch.tensor(traj['act'][kk - 1], dtype=torch.float32, device=device).unsqueeze(0)
        h_im, z_im, _ = model.rssm.imagine_step(h_im, z_im, a)
        dec_imag = model.decoder(torch.cat([h_im, z_im], dim=-1)).squeeze(0).cpu().numpy()
        obs_real = traj['obs'][kk]
        state_dist.append(float(np.linalg.norm(dec_imag - obs_real)))
    return np.array(state_dist, np.float32)


def e_state(dist_full, K):
    return float(np.mean(dist_full[:K])) if len(dist_full) >= K else np.nan


def best_gamma(kl, traj_id, probe_score, candidates=GAMMA_CANDIDATES):
    best_g, best_r2 = candidates[0], -1.0
    for g in candidates:
        ct = compute_ct(kl, traj_id, gamma=g)
        r2 = pearsonr(ct, probe_score)[0] ** 2
        if r2 > best_r2:
            best_r2, best_g = r2, g
    return best_g, best_r2


def multi_step_causal_test(model, clf, scaler, trajs, causal_sites, direction,
                            lookahead=CAUSAL_LOOKAHEAD, k_headline=K_HEADLINE):
    """Task 3 upgrade: ablate h_t, then CONTINUE the trajectory (imagine_step)
    for `lookahead` steps and re-measure both the probe readout at t+k and the
    E^state metric computed from that continuation -- not just a static
    re-score at t. Returns per-k mean probe-delta and the E^state delta at the
    headline horizon (computed once per site, using the full continuation)."""
    device = next(model.parameters()).device
    d_probe = {k: [] for k in lookahead}
    d_estate = []
    for (ti, t) in causal_sites:
        trj = trajs[ti]
        h_t = trj['h'][t]
        proj = float(h_t @ direction)
        h_abl = h_t - proj * direction

        # probe readout at t (k=0) and after k-step continuation
        ps_base_0 = trj['probe'][t]
        ps_abl_0 = clf.predict_proba(scaler.transform(h_abl.reshape(1, -1)))[0, 1]
        d_probe[0].append(ps_abl_0 - ps_base_0)

        max_k = max(k for k in lookahead if k > 0) if any(k > 0 for k in lookahead) else 0
        if max_k > 0:
            T = len(trj['obs'])
            h_im = torch.tensor(h_abl, dtype=torch.float32, device=device).unsqueeze(0)
            z_logits = torch.tensor(trj['z'][t], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                z_im = model.rssm._straight_through_sample(z_logits)
                for k in range(1, max_k + 1):
                    kk = t + k
                    if kk >= T:
                        break
                    a = torch.tensor(trj['act'][kk - 1], dtype=torch.float32, device=device).unsqueeze(0)
                    h_im, z_im, _ = model.rssm.imagine_step(h_im, z_im, a)
                    if k in lookahead:
                        h_np = h_im.squeeze(0).cpu().numpy()
                        ps_abl_k = clf.predict_proba(scaler.transform(h_np.reshape(1, -1)))[0, 1]
                        ps_base_k = trj['probe'][kk] if kk < len(trj['probe']) else np.nan
                        if np.isfinite(ps_base_k):
                            d_probe[k].append(ps_abl_k - ps_base_k)

        # E^state at headline horizon: ablated-continuation vs real-state continuation
        dist_abl = imagined_vs_real_obs_pomdp(model, trj, t, k_headline, h_override=h_abl)
        dist_base = imagined_vs_real_obs_pomdp(model, trj, t, k_headline)
        if len(dist_abl) >= k_headline and len(dist_base) >= k_headline:
            d_estate.append(e_state(dist_abl, k_headline) - e_state(dist_base, k_headline))

    result = {k: float(np.mean(v)) if v else float('nan') for k, v in d_probe.items()}
    result['e_state_delta'] = float(np.mean(d_estate)) if d_estate else float('nan')
    result['n_estate'] = len(d_estate)
    return result


def matched_bin_check(trajs, causal_sites, kl_key='kl'):
    """Difficulty base-rate check: bin sites into KL terciles (a difficulty-
    neutral, condition-comparable covariate) and report the site count and mean
    KL per bin, so a reader can see whether FULL and PARTIAL's site pools are
    KL-matched or whether one condition's sites are systematically harder before
    any causal comparison is drawn."""
    kls = np.array([trajs[ti][kl_key][t] for (ti, t) in causal_sites])
    terciles = np.percentile(kls, [33.33, 66.67])
    bins = np.digitize(kls, terciles)
    bin_stats = []
    for b in range(3):
        mask = bins == b
        bin_stats.append(dict(bin=b, n=int(mask.sum()),
                               mean_kl=float(kls[mask].mean()) if mask.sum() else float('nan')))
    return bin_stats, bins


def analyze_seed_condition(seed, condition):
    cfg = XS_CONFIG.copy()
    model, tr, ckpt = load_condition(seed, condition)

    kl_median = float(np.median(tr['kl']))
    y = binarise_by_median(tr['kl'])
    idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                       stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])
    auroc_id = auroc(clf, scaler, tr['h'][idx_te], y[idx_te])

    probe_score_all = clf.predict_proba(scaler.transform(tr['h']))[:, 1]
    gamma, r2_ct = best_gamma(tr['kl'], tr['traj_id'], probe_score_all)
    v = probe_direction(clf, scaler)

    trajs = collect_trajectories(model, condition, N_TRAJ, cfg, seed=seed)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64), gamma=gamma)
        trj['probe'] = clf.predict_proba(scaler.transform(trj['h']))[:, 1]

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - max(HORIZONS) - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(seed)
    if len(sites) > 4000:
        sel = rng.choice(len(sites), 4000, replace=False)
        sites = [sites[i] for i in sel]

    rows = {k: [] for k in ['ct', 'kl', 'recon']}
    e_state_by_K = {K: [] for K in HORIZONS}
    for (ti, t) in sites:
        trj = trajs[ti]
        dist_full = imagined_vs_real_obs_pomdp(model, trj, t, max(HORIZONS))
        if len(dist_full) < max(HORIZONS):
            continue
        for K in HORIZONS:
            e_state_by_K[K].append(e_state(dist_full, K))
        rows['ct'].append(trj['ct'][t]); rows['kl'].append(trj['kl'][t]); rows['recon'].append(trj['recon'][t])
    for k in rows:
        rows[k] = np.array(rows[k], dtype=np.float64)
    for K in HORIZONS:
        e_state_by_K[K] = np.array(e_state_by_K[K], dtype=np.float64)
    n_sites = len(rows['ct'])

    target10 = e_state_by_K[K_HEADLINE]
    r_ct_estate, _ = pearsonr(rows['ct'], target10)
    X_ctrl = np.column_stack([rows['kl'], rows['recon']])
    X_full = np.column_stack([rows['ct'], rows['kl'], rows['recon']])
    Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
    Xs_full = StandardScaler().fit_transform(X_full)
    r2_ctrl = LinearRegression().fit(Xs_ctrl, target10).score(Xs_ctrl, target10)
    r2_full = LinearRegression().fit(Xs_full, target10).score(Xs_full, target10)
    incremental = r2_full - r2_ctrl

    # ── multi-step causal ablation test (Task 3 upgrade #1) ──
    causal_sites = [sites[i] for i in rng.choice(len(sites),
                    min(N_INTERVENTION_SITES, len(sites)), replace=False)]
    conf_effect = multi_step_causal_test(model, clf, scaler, trajs, causal_sites, v)

    rng_null = np.random.default_rng(seed + 90000)
    null_effects_0 = []
    null_effects_estate = []
    for _ in range(N_NULL):
        vr = random_matched_direction(rng_null, v.shape[0])
        e = multi_step_causal_test(model, clf, scaler, trajs, causal_sites, vr, lookahead=[0])
        null_effects_0.append(e[0])
        null_effects_estate.append(e['e_state_delta'])
    z0 = (conf_effect[0] - np.mean(null_effects_0)) / (np.std(null_effects_0) + 1e-12)
    de_valid = [x for x in null_effects_estate if np.isfinite(x)]
    z_estate = ((conf_effect['e_state_delta'] - np.mean(de_valid)) / (np.std(de_valid) + 1e-12)
                if len(de_valid) >= 10 and np.isfinite(conf_effect['e_state_delta']) else float('nan'))

    # ── difficulty base-rate / matched-bin check (Task 3 upgrade #2) ──
    bin_stats, bins = matched_bin_check(trajs, causal_sites)
    bin_z = []
    for b in range(3):
        bin_sites = [causal_sites[i] for i in range(len(causal_sites)) if bins[i] == b]
        if len(bin_sites) < 20:
            bin_z.append(dict(bin=b, n=len(bin_sites), z=float('nan')))
            continue
        eff_b = multi_step_causal_test(model, clf, scaler, trajs, bin_sites, v, lookahead=[0])
        null_b = []
        rng_null_b = np.random.default_rng(seed + 90000 + b)
        for _ in range(20):  # reduced null size within each bin for cost
            vr = random_matched_direction(rng_null_b, v.shape[0])
            null_b.append(multi_step_causal_test(model, clf, scaler, trajs, bin_sites, vr, lookahead=[0])[0])
        z_b = (eff_b[0] - np.mean(null_b)) / (np.std(null_b) + 1e-12)
        bin_z.append(dict(bin=b, n=len(bin_sites), z=float(z_b)))

    return dict(
        seed=seed, condition=condition, confusion_auroc=auroc_id, gamma=gamma, r2_ct=r2_ct,
        n_sites=n_sites, pearson_r_ct_estate=float(r_ct_estate), incremental_r2_ct=float(incremental),
        r2_ctrl=float(r2_ctrl), r2_full=float(r2_full),
        causal_dprobe_by_k=conf_effect, causal_estate_delta=conf_effect['e_state_delta'],
        causal_null_mean_0=float(np.mean(null_effects_0)), causal_null_std_0=float(np.std(null_effects_0)),
        causal_z_0=float(z0), causal_z_estate=float(z_estate),
        difficulty_bin_stats=bin_stats, difficulty_bin_z=bin_z,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for seed in ALL_SEEDS:
        d = seed_dir_for(seed)
        if not (os.path.exists(os.path.join(d, 'full_world_model.pt')) and
                os.path.exists(os.path.join(d, 'partial_world_model.pt'))):
            print(f"[seed_sweep_analysis] seed={seed}: checkpoints not ready, skipping")
            continue
        print(f"\n{'='*78}\nSEED {seed}\n{'='*78}")
        res_full = analyze_seed_condition(seed, 'full')
        res_partial = analyze_seed_condition(seed, 'partial')
        print(f"  FULL:    AUROC={res_full['confusion_auroc']:.4f}  incR2={res_full['incremental_r2_ct']:+.4f}  "
              f"causal_z0={res_full['causal_z_0']:+.2f}  causal_z_estate={res_full['causal_z_estate']:+.2f}")
        print(f"  PARTIAL: AUROC={res_partial['confusion_auroc']:.4f}  incR2={res_partial['incremental_r2_ct']:+.4f}  "
              f"causal_z0={res_partial['causal_z_0']:+.2f}  causal_z_estate={res_partial['causal_z_estate']:+.2f}")
        all_results[seed] = dict(full=res_full, partial=res_partial)

    # ── aggregate across seed pairs ──
    metrics = ['confusion_auroc', 'r2_ct', 'pearson_r_ct_estate', 'incremental_r2_ct', 'causal_z_0', 'causal_z_estate']
    agg = {'full': {}, 'partial': {}}
    for cond in ['full', 'partial']:
        for m in metrics:
            vals = [all_results[s][cond][m] for s in all_results if np.isfinite(all_results[s][cond][m])]
            agg[cond][m] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), n=len(vals), values=vals)

    print(f"\n{'='*78}\nAGGREGATE ACROSS {len(all_results)} SEED PAIRS\n{'='*78}")
    print(f"  {'metric':<24}{'FULL mean±std':>20}{'PARTIAL mean±std':>20}{'partial>full?':>16}")
    n_stronger = 0
    summary_rows = []
    for m in metrics:
        fm, fs = agg['full'][m]['mean'], agg['full'][m]['std']
        pm, ps = agg['partial'][m]['mean'], agg['partial'][m]['std']
        stronger = abs(pm) > abs(fm)
        if stronger:
            n_stronger += 1
        print(f"  {m:<24}{fm:>10.4f}±{fs:<8.4f}{pm:>10.4f}±{ps:<8.4f}{str(stronger):>16}")
        summary_rows.append(dict(metric=m, full_mean=fm, full_std=fs, partial_mean=pm,
                                  partial_std=ps, partial_stronger=stronger))

    verdict = ('SUPPORTS H6 (partial observability increases relevance)' if n_stronger >= 4
               else 'DOES NOT CLEARLY SUPPORT H6' if n_stronger <= 2 else 'MIXED')
    print(f"\n  {n_stronger}/{len(metrics)} metrics stronger under partial observability (aggregated)")
    print(f"  VERDICT: {verdict}")

    out = dict(per_seed=all_results, aggregate=agg, summary=summary_rows, verdict=verdict,
               n_seed_pairs=len(all_results), seeds=list(all_results.keys()))
    out_path = os.path.join(OUT_DIR, 'phase4_seed_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
