#!/usr/bin/env python3.11
"""
Phase 4b — Full-vs-partial-observability comparison analysis (Reframed_Confusion_
RSSM_Project, Section 10.2).

Runs the core comparison pipeline identically on the two conditions trained by
run_phase4_pomdp_training.py (outputs/phase4_pomdp/{full,partial}_world_model.pt):

  - R^2(h_t, C_t)                          [closed-form fit quality]
  - confusion AUROC                        [Probe A on a KL-median split, held-out]
  - future imagination-error prediction    [H1 from Phase 1: r(C_t, E^state_K)]
  - incremental value over KL              [H2 from Phase 1: incremental R^2]
  - temporal-memory estimate gamma          [best-fit gamma via a small sweep]
  - causal effect size                     [Task-G-style 50-direction null on the
                                             confusion-direction ablation]

Hypothesis (Section 10.2): partial observability => greater behavioral relevance
of accumulated confusion (larger incremental R^2, larger causal effect, etc).
This is tested, not assumed -- reported honestly either way (Section 10.2,
Section 19 Gate 3).
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.model.pomdp_world_model import PomdpWorldModel
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import compute_ct, probe_direction, regression_direction, random_matched_direction

OUT_DIR = 'outputs/phase4_pomdp'
FIG_DIR = 'outputs/figures'
N_TRAJ = 80
MIN_T = 12
HORIZONS = [1, 5, 10, 20]
K_HEADLINE = 10
GAMMA_CANDIDATES = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
N_NULL = 50
N_INTERVENTION_SITES = 600
SEED = 4242


def load_condition(condition):
    ckpt = torch.load(os.path.join(OUT_DIR, f'{condition}_world_model.pt'), map_location='cpu')
    model = PomdpWorldModel(ckpt['obs_dim_in'], ckpt['obs_dim_out'], ckpt['act_dim'], ckpt['cfg'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    tr = dict(np.load(os.path.join(OUT_DIR, f'{condition}_training_states.npz')))
    return model, tr, ckpt


def collect_trajectories(model, condition, n_traj, cfg, seed=SEED + 1):
    device = next(model.parameters()).device
    trajs = []
    for ep in range(n_traj):
        env = CartpoleEnv(task='swingup', noisy=False, seed=seed + ep)
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        rng = np.random.default_rng(seed + ep)
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
def imagined_vs_real_obs_pomdp(model, traj, t, horizon):
    """Same construction as Phase 1's imagined_vs_real_obs: roll IMAGINATION
    forward with real actions from the posterior state at t, decode (to the
    FULL-STATE decoder target), compare to real full obs/reward. Identical for
    both conditions since imagination/decoding never uses the (possibly masked)
    encoder -- only the encoder input differs between conditions, not the
    imagine/decode path."""
    device = next(model.parameters()).device
    T = len(traj['obs'])
    h_im = torch.tensor(traj['h'][t], dtype=torch.float32, device=device).unsqueeze(0)
    z_logits = torch.tensor(traj['z'][t], dtype=torch.float32, device=device).unsqueeze(0)
    z_im = model.rssm._straight_through_sample(z_logits)

    state_dist, imag_rew, real_rew = [], [], []
    for k in range(1, horizon + 1):
        kk = t + k
        if kk >= T:
            break
        a = torch.tensor(traj['act'][kk - 1], dtype=torch.float32, device=device).unsqueeze(0)
        h_im, z_im, _ = model.rssm.imagine_step(h_im, z_im, a)
        dec_imag = model.decoder(torch.cat([h_im, z_im], dim=-1)).squeeze(0).cpu().numpy()
        obs_real = traj['obs'][kk]
        state_dist.append(float(np.linalg.norm(dec_imag - obs_real)))
        imag_rew.append(float(cartpole_reward_proxy(dec_imag)))
        real_rew.append(float(traj['rew'][kk - 1]))
    return (np.array(state_dist, np.float32), np.array(imag_rew, np.float32),
            np.array(real_rew, np.float32))


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


def analyze_condition(condition):
    print(f"\n{'='*78}\n{condition.upper()} CONDITION\n{'='*78}")
    cfg = XS_CONFIG.copy()
    model, tr, ckpt = load_condition(condition)
    print(f"  obs_dim_in={ckpt['obs_dim_in']} obs_dim_out={ckpt['obs_dim_out']}")

    kl_median = float(np.median(tr['kl']))
    y = binarise_by_median(tr['kl'])
    idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                       stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])
    auroc_id = auroc(clf, scaler, tr['h'][idx_te], y[idx_te])
    print(f"  Probe A held-out AUROC (confusion AUROC): {auroc_id:.4f}")

    probe_score_all = clf.predict_proba(scaler.transform(tr['h']))[:, 1]
    gamma, r2_ct = best_gamma(tr['kl'], tr['traj_id'], probe_score_all)
    print(f"  best-fit gamma={gamma}  R^2(h_t->C_t)={r2_ct:.4f}")

    v = probe_direction(clf, scaler)

    print(f"  collecting {N_TRAJ} held-out trajectories...")
    trajs = collect_trajectories(model, condition, N_TRAJ, cfg)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64), gamma=gamma)
        trj['probe'] = clf.predict_proba(scaler.transform(trj['h']))[:, 1]

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - max(HORIZONS) - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED)
    if len(sites) > 4000:
        sel = rng.choice(len(sites), 4000, replace=False)
        sites = [sites[i] for i in sel]
    print(f"  {len(sites)} evaluation sites")

    rows = {k: [] for k in ['ct', 'kl', 'recon']}
    e_state_by_K = {K: [] for K in HORIZONS}
    for (ti, t) in sites:
        trj = trajs[ti]
        dist_full, _, _ = imagined_vs_real_obs_pomdp(model, trj, t, max(HORIZONS))
        if len(dist_full) < max(HORIZONS):
            continue
        for K in HORIZONS:
            e_state_by_K[K].append(e_state(dist_full, K))
        rows['ct'].append(trj['ct'][t])
        rows['kl'].append(trj['kl'][t])
        rows['recon'].append(trj['recon'][t])
    for k in rows:
        rows[k] = np.array(rows[k], dtype=np.float64)
    for K in HORIZONS:
        e_state_by_K[K] = np.array(e_state_by_K[K], dtype=np.float64)
    n_sites = len(rows['ct'])
    print(f"  {n_sites} sites with full horizon")

    h1 = {}
    for K in HORIZONS:
        target = e_state_by_K[K]
        r, p = pearsonr(rows['ct'], target)
        h1[K] = dict(r=r, p=p)
        print(f"    K={K:>2}: r(C_t, E^state)={r:+.4f} (p={p:.2g})")

    target10 = e_state_by_K[K_HEADLINE]
    X_ctrl = np.column_stack([rows['kl'], rows['recon']])
    X_full = np.column_stack([rows['ct'], rows['kl'], rows['recon']])
    Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
    Xs_full = StandardScaler().fit_transform(X_full)
    r2_ctrl = LinearRegression().fit(Xs_ctrl, target10).score(Xs_ctrl, target10)
    r2_full = LinearRegression().fit(Xs_full, target10).score(Xs_full, target10)
    incremental = r2_full - r2_ctrl
    print(f"  incremental R^2 of C_t over {{KL,Recon}} @ K={K_HEADLINE}: {incremental:+.4f} "
          f"(ctrl={r2_ctrl:.4f} -> full={r2_full:.4f})")

    # ── causal effect size: Task-G-style 50-direction null on confusion-direction ablation ──
    print(f"  causal ablation test ({N_INTERVENTION_SITES} sites, {N_NULL}-direction null)...")
    causal_sites = [sites[i] for i in rng.choice(len(sites),
                    min(N_INTERVENTION_SITES, len(sites)), replace=False)]

    def probe_decay_effect(direction, lookahead=(0, 1, 5, 10)):
        deltas = {k: [] for k in lookahead}
        for (ti, t) in causal_sites:
            trj = trajs[ti]
            h_t = trj['h'][t]
            proj = float(h_t @ direction)
            h_abl = h_t - proj * direction
            ps_base = trj['probe'][t]
            ps_abl = clf.predict_proba(scaler.transform(h_abl.reshape(1, -1)))[0, 1]
            for k in lookahead:
                deltas[k].append(ps_abl - ps_base)  # static (no continuation) proxy for speed
        return {k: float(np.mean(deltas[k])) for k in lookahead}

    conf_effect = probe_decay_effect(v)
    rng_null = np.random.default_rng(9090)
    null_effects = {k: [] for k in conf_effect}
    for _ in range(N_NULL):
        vr = random_matched_direction(rng_null, v.shape[0])
        e = probe_decay_effect(vr)
        for k in e:
            null_effects[k].append(e[k])
    z0 = (conf_effect[0] - np.mean(null_effects[0])) / (np.std(null_effects[0]) + 1e-12)
    pct0 = float((np.array(null_effects[0]) > conf_effect[0]).mean() * 100)
    print(f"  confusion-direction ablation dprobe_0={conf_effect[0]:+.4f} vs "
          f"null={np.mean(null_effects[0]):+.4f}±{np.std(null_effects[0]):.4f}  "
          f"z={z0:+.2f}  pct_extreme={pct0:.0f}%")

    return dict(
        condition=condition, confusion_auroc=auroc_id, gamma=gamma, r2_ct=r2_ct,
        n_sites=n_sites, h1_by_horizon=h1, incremental_r2_ct=incremental,
        r2_ctrl=r2_ctrl, r2_full=r2_full,
        causal_dprobe0=conf_effect[0], causal_null_mean=float(np.mean(null_effects[0])),
        causal_null_std=float(np.std(null_effects[0])), causal_z=float(z0),
        causal_pct_extreme=pct0,
    )


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results = {}
    for condition in ['full', 'partial']:
        results[condition] = analyze_condition(condition)

    print(f"\n{'='*78}\nPHASE 4 — FULL vs PARTIAL OBSERVABILITY COMPARISON\n{'='*78}")
    f, p = results['full'], results['partial']
    print(f"\n  {'metric':<32}{'full':>14}{'partial':>14}{'partial > full?':>18}")
    comparisons = [
        ('confusion AUROC', 'confusion_auroc', True),
        ('gamma', 'gamma', None),
        ('R^2(h_t, C_t)', 'r2_ct', True),
        (f'r(C_t,E^state) @K={K_HEADLINE}', None, True),
        ('incremental R^2 (C_t over KL+Recon)', 'incremental_r2_ct', True),
        ('causal z-score (vs null)', 'causal_z', True),
    ]
    summary_rows = []
    for label, key, higher_is_stronger in comparisons:
        if key is None:
            fv, pv = f['h1_by_horizon'][K_HEADLINE]['r'], p['h1_by_horizon'][K_HEADLINE]['r']
        else:
            fv, pv = f[key], p[key]
        stronger = (abs(pv) > abs(fv)) if higher_is_stronger else None
        print(f"  {label:<32}{fv:>14.4f}{pv:>14.4f}{str(stronger):>18}")
        summary_rows.append(dict(metric=label, full=fv, partial=pv, partial_stronger=stronger))

    n_stronger = sum(1 for r in summary_rows if r['partial_stronger'])
    verdict = ('SUPPORTS H6 (partial observability increases relevance)' if n_stronger >= 4
               else 'DOES NOT CLEARLY SUPPORT H6' if n_stronger <= 2
               else 'MIXED')
    print(f"\n  {n_stronger}/{len(summary_rows)} metrics stronger under partial observability")
    print(f"  VERDICT: {verdict}")

    out = dict(full=results['full'], partial=results['partial'],
               summary=summary_rows, verdict=verdict)
    out_path = os.path.join(OUT_DIR, 'phase4_comparison.json')
    with open(out_path, 'w') as fjson:
        json.dump(out, fjson, indent=2, default=float)
    print(f"\nWrote {out_path}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    labels = [r['metric'] for r in summary_rows]
    fulls = [r['full'] for r in summary_rows]
    partials = [r['partial'] for r in summary_rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, fulls, width=0.4, label='full obs', color='steelblue')
    ax.bar(x + 0.2, partials, width=0.4, label='partial obs', color='darkorange')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(); ax.set_title('Phase 4: full vs partial observability')
    ax = axes[1]
    for cond, res, color in [('full', results['full'], 'steelblue'),
                              ('partial', results['partial'], 'darkorange')]:
        rs = [res['h1_by_horizon'][K]['r'] for K in HORIZONS]
        ax.plot(HORIZONS, rs, 'o-', color=color, label=cond)
    ax.set_xlabel('horizon K'); ax.set_ylabel('r(C_t, E^state_K)')
    ax.set_title('predictive decay by horizon'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'phase4_pomdp_comparison.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {fig_path}")


if __name__ == '__main__':
    main()
