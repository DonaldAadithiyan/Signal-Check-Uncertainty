#!/usr/bin/env python3.11
"""
Task R — Multi-seed replication for reacher and pendulum.

Cartpole has 5-seed means + bootstrap CIs; reacher and pendulum each rest on a single
trained model despite now carrying load-bearing weight (pendulum has its own mechanistic
subsection from Task N). This trains 2-3 additional seeds each (3 total per env, lighter
than cartpole's 5) and reruns the core measurement set from Tasks D/J per seed:
  Set C AUROC, within-task confound AUROC, C_t best-γ & R², null-space angle, frac top-10 PC.

Reports mean ± std (median + range for small n) alongside the existing single-seed numbers.
Specifically flags: does pendulum's Set C INVERSION replicate across seeds (strengthening
Task N's mechanism as a stable env property), and does reacher's weaker within-task confound
(0.578) hold or move.

Reuses run_second_env.py's analysis helpers; resumable per (env, seed). CPU, XS.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.training.replay_buffer import EpisodeReplayBuffer
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import compute_ct, bootstrap_auroc_ci
# reuse the exact analysis helpers used for the single-seed reacher/pendulum runs
from run_second_env import collect_env, build_set_c, null_space_geometry, _within_task

ENVS = {'reacher': ('reacher', 'easy'), 'pendulum': ('pendulum', 'swingup')}
SEEDS = [1, 2, 3]                 # seed 0 already exists from Tasks D/J
GAMMAS = [0.70, 0.80, 0.90, 0.95, 0.99]
N_EP = 20
ROOT = 'outputs/multiseed_env'


def train_seed(cfg, domain, task, seed, ckpt, states_path):
    """Env-parametrised, seed-parametrised training (mirrors run_second_env.train_on_env)."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = DMCEnv(domain, task, seed=seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len, batch, warmup, max_steps = cfg['seq_len'], cfg['batch_size'], cfg['warmup_steps'], cfg['total_env_steps']
    log_h, log_z, log_kl, log_recon, log_traj = [], [], [], [], []
    step_count, traj_id, t0 = 0, 0, time.time()
    ep_obs, ep_act = [], []
    h_inf = torch.zeros(1, cfg['rssm_deter']); z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
    obs = env.reset(); ep_obs.append(obs.copy())
    print(f"[train {domain}_{task} seed{seed}] obs={env.obs_dim} act={env.act_dim}", flush=True)
    while step_count < max_steps:
        action = np.random.uniform(-1, 1, size=(env.act_dim,)).astype(np.float32)
        model.eval()
        with torch.no_grad():
            ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0); at = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            emb = model.encoder(ot)
            h_inf, z_inf, prior_l, post_l = model.rssm.observe_step(h_inf, z_inf, at, emb)
            dec = model.decoder(torch.cat([h_inf, z_inf], dim=-1))
            kl_val = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            rc_val = F.mse_loss(dec, ot, reduction='none').sum().item()
        log_h.append(h_inf.squeeze(0).numpy().copy()); log_z.append(post_l.squeeze(0).numpy().copy())
        log_kl.append(kl_val); log_recon.append(rc_val); log_traj.append(traj_id)
        obs_new, _, done = env.step(action); ep_act.append(action.copy()); step_count += 1
        if done or (len(ep_act) >= cfg['episode_max_steps']):
            ep_obs.append(obs_new.copy()); buffer.add_episode(ep_obs[:-1], ep_act)
            traj_id += 1; ep_obs, ep_act = [], []
            h_inf = torch.zeros(1, cfg['rssm_deter']); z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
            obs = env.reset(); ep_obs.append(obs.copy())
        else:
            obs = obs_new; ep_obs.append(obs.copy())
        if step_count >= warmup and len(buffer) >= seq_len * batch:
            model.train()
            ob, ab = buffer.sample(batch, seq_len, device='cpu')
            loss, _, _ = model.compute_loss(ob, ab, kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip']); optim.step()
        if step_count % 20000 == 0:
            print(f"  {domain} s{seed} step {step_count:,}/{max_steps:,} kl={np.mean(log_kl[-500:]):.2f} "
                  f"{(time.time()-t0)/60:.0f}m", flush=True)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'obs_dim': env.obs_dim, 'act_dim': env.act_dim}, ckpt)
    states = dict(h=np.array(log_h, np.float32), z=np.array(log_z, np.float32),
                  kl=np.array(log_kl, np.float32), recon=np.array(log_recon, np.float32),
                  traj_id=np.array(log_traj, np.int64))
    np.savez(states_path, **states)
    return model, states


def analyze_seed(env_key, model, states, cfg):
    """Core measurement set: same as Tasks D/J."""
    domain, task = ENVS[env_key]
    h, z, kl, recon, traj = states['h'], states['z'], states['kl'], states['recon'], states['traj_id']
    N = len(h)
    y = binarise_by_median(kl); kl_median = float(np.median(kl))
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h[tr_idx], y[tr_idx])

    # eval sets (fresh, same as run_second_env)
    np.random.seed(42)
    set_a = collect_env(model, DMCEnv(domain, task, noisy=False, seed=100), N_EP, cfg)
    set_b = collect_env(model, DMCEnv(domain, task, noisy=True, noise_std=cfg['noise_std'], seed=200), N_EP, cfg)
    set_c = build_set_c(set_a, set_b)
    set_c_within = _within_task(set_a)

    auroc_c = auroc(clf, sc, set_c['h'], set_c['labels'])
    auroc_within = auroc(clf, sc, set_c_within['h'], set_c_within['labels'])
    scores_c = clf.predict_proba(sc.transform(set_c['h']))[:, 1]
    ci_c = bootstrap_auroc_ci(set_c['labels'], scores_c, seed=0)

    # C_t characterisation
    probe_te = clf.predict_proba(sc.transform(h[te_idx]))[:, 1]
    best_r2, best_g = -1, None
    for g in GAMMAS:
        ct = compute_ct(kl, traj, gamma=g, kl_median=kl_median)[te_idx]
        r2 = r2_score(probe_te, LinearRegression().fit(ct.reshape(-1, 1), probe_te).predict(ct.reshape(-1, 1)))
        if r2 > best_r2:
            best_r2, best_g = r2, g

    angle, frac = null_space_geometry(h, kl)
    return dict(auroc_c=float(auroc_c), ci_setc=list(ci_c), auroc_within_task=float(auroc_within),
                best_gamma=float(best_g), best_ct_r2=float(best_r2),
                nullspace_angle=float(angle), frac_in_top10=float(frac))


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(ROOT, exist_ok=True)

    for env_key, (domain, task) in ENVS.items():
        cfg_e = cfg.copy()
        pe = DMCEnv(domain, task, seed=0)
        cfg_e['obs_dim'], cfg_e['act_dim'] = pe.obs_dim, pe.act_dim
        for seed in SEEDS:
            d = os.path.join(ROOT, f'{env_key}_seed{seed}')
            os.makedirs(d, exist_ok=True)
            mpath = os.path.join(d, 'metrics.json')
            if os.path.exists(mpath):
                print(f"[{env_key} seed{seed}] metrics.json exists — skip"); continue
            ckpt = os.path.join(d, 'model.pt'); spath = os.path.join(d, 'states.npz')
            if os.path.exists(ckpt) and os.path.exists(spath):
                ck = torch.load(ckpt, map_location='cpu')
                model = WorldModel(ck['obs_dim'], ck['act_dim'], ck['cfg']); model.load_state_dict(ck['model_state']); model.eval()
                states = dict(np.load(spath))
            else:
                model, states = train_seed(cfg_e, domain, task, seed, ckpt, spath)
            m = analyze_seed(env_key, model, states, cfg_e)
            m['env'] = env_key; m['seed'] = seed
            json.dump(m, open(mpath, 'w'), indent=2)
            print(f"[{env_key} seed{seed}] SetC={m['auroc_c']:.3f} within={m['auroc_within_task']:.3f} "
                  f"γ={m['best_gamma']} R²={m['best_ct_r2']:.3f} angle={m['nullspace_angle']:.1f}", flush=True)

    aggregate()


def aggregate():
    # single-seed baselines from Tasks D/J
    base = {'reacher': dict(auroc_c=0.619, auroc_within_task=0.578, best_gamma=0.70, best_ct_r2=0.216,
                            nullspace_angle=89.4, frac_in_top10=0.0017),
            'pendulum': dict(auroc_c=0.322, auroc_within_task=0.437, best_gamma=0.90, best_ct_r2=0.886,
                             nullspace_angle=88.1, frac_in_top10=0.0149)}
    print("\n" + "=" * 78)
    print("TASK R — MULTI-SEED REPLICATION (reacher + pendulum)")
    print("=" * 78)
    out = {}
    for env_key in ENVS:
        rows = [base[env_key] | {'seed': 0, 'source': 'D/J single'}]
        for seed in SEEDS:
            mp = os.path.join(ROOT, f'{env_key}_seed{seed}', 'metrics.json')
            if os.path.exists(mp):
                rows.append(json.load(open(mp)) | {'source': f'seed{seed}'})
        keys = ['auroc_c', 'auroc_within_task', 'best_gamma', 'best_ct_r2', 'nullspace_angle', 'frac_in_top10']
        print(f"\n  {env_key.upper()}  (n={len(rows)} seeds incl. original)")
        print(f"    {'metric':<24}{'per-seed values':<34}{'mean ± std':>18}")
        env_summary = {}
        for k in keys:
            vals = np.array([r[k] for r in rows], float)
            env_summary[k] = dict(values=vals.tolist(), mean=float(vals.mean()), std=float(vals.std()),
                                  median=float(np.median(vals)), min=float(vals.min()), max=float(vals.max()))
            vstr = ' '.join(f'{v:.3f}' for v in vals)
            print(f"    {k:<24}{vstr:<34}{vals.mean():>9.3f} ± {vals.std():<7.3f}")
        out[env_key] = env_summary
        # flags
        sc_vals = np.array([r['auroc_c'] for r in rows])
        if env_key == 'pendulum':
            inv = (sc_vals < 0.5).sum()
            print(f"    → Set C inversion replicates: {inv}/{len(sc_vals)} seeds below 0.5 "
                  f"({'STABLE' if inv >= len(sc_vals)-1 else 'seed-variable'})")
        if env_key == 'reacher':
            wv = np.array([r['auroc_within_task'] for r in rows])
            print(f"    → within-task confound: {wv.mean():.3f} ± {wv.std():.3f} "
                  f"(range [{wv.min():.3f}, {wv.max():.3f}])")
    json.dump(out, open(os.path.join(ROOT, 'aggregate.json'), 'w'), indent=2, default=float)
    print(f"\n  Saved: {os.path.join(ROOT, 'aggregate.json')}")


if __name__ == '__main__':
    import sys
    if '--aggregate-only' in sys.argv:
        aggregate()
    else:
        main()
