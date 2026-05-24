"""
Collect Sets A, B, C using a frozen world model.
Set A: in-distribution (same env, random policy)
Set B: near-OOD (Gaussian noise added to observations)
Set C: contrastive states from A+B based on recon error percentiles
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.env.wrapper import CartpoleEnv


def collect_states(model, env, n_episodes, cfg):
    """
    Run n_episodes with a random policy and frozen world model.
    Returns dict with h, z, kl, recon, obs arrays (all CPU numpy).
    """
    device = next(model.parameters()).device
    model.eval()

    all_h, all_z, all_kl, all_recon, all_obs = [], [], [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)

        done  = False
        step  = 0

        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                action = np.random.uniform(-1, 1, size=(cfg['act_dim'],)).astype(np.float32)

                obs_t = torch.tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
                a_t   = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

                embed = model.encoder(obs_t)
                h, z, prior_logits, post_logits = model.rssm.observe_step(h, z, a_t, embed)

                decoded   = model.decoder(torch.cat([h, z], dim=-1))
                kl_val    = model.rssm.kl_divergence(
                    post_logits, prior_logits, free_bits=0.0).item()
                recon_val = F.mse_loss(decoded, obs_t, reduction='none').sum().item()

                all_h.append(h.squeeze(0).cpu().numpy().copy())
                all_z.append(post_logits.squeeze(0).cpu().numpy().copy())
                all_kl.append(kl_val)
                all_recon.append(recon_val)
                all_obs.append(obs.copy())

                obs, _, done = env.step(action)
                step += 1

        if (ep + 1) % 5 == 0:
            print(f"  collected episode {ep + 1}/{n_episodes}")

    return {
        'h':     np.array(all_h,     dtype=np.float32),
        'z':     np.array(all_z,     dtype=np.float32),
        'kl':    np.array(all_kl,    dtype=np.float32),
        'recon': np.array(all_recon, dtype=np.float32),
        'obs':   np.array(all_obs,   dtype=np.float32),
    }


def build_set_c(set_a: dict, set_b: dict, percentile_low=20, percentile_high=80):
    """
    C1 (label=0, low uncertainty): Set B states with bottom-percentile recon error.
    C2 (label=1, high uncertainty): Set A states with top-percentile recon error.
    """
    b_recon       = set_b['recon']
    b_thresh_low  = np.percentile(b_recon, percentile_low)
    c1_mask       = b_recon <= b_thresh_low

    a_recon        = set_a['recon']
    a_thresh_high  = np.percentile(a_recon, percentile_high)
    c2_mask        = a_recon >= a_thresh_high

    c1_idx = np.where(c1_mask)[0]
    c2_idx = np.where(c2_mask)[0]

    rng = np.random.default_rng(42)
    if len(c1_idx) > 200:
        c1_idx = rng.choice(c1_idx, 200, replace=False)
    if len(c2_idx) > 200:
        c2_idx = rng.choice(c2_idx, 200, replace=False)

    n_c1, n_c2 = len(c1_idx), len(c2_idx)

    set_c = {
        'h':     np.concatenate([set_b['h'][c1_idx],     set_a['h'][c2_idx]]),
        'z':     np.concatenate([set_b['z'][c1_idx],     set_a['z'][c2_idx]]),
        'kl':    np.concatenate([set_b['kl'][c1_idx],    set_a['kl'][c2_idx]]),
        'recon': np.concatenate([set_b['recon'][c1_idx], set_a['recon'][c2_idx]]),
        'obs':   np.concatenate([set_b['obs'][c1_idx],   set_a['obs'][c2_idx]]),
        'labels': np.array([0] * n_c1 + [1] * n_c2, dtype=np.int32),
        'group':  np.array(['C1'] * n_c1 + ['C2'] * n_c2),
    }

    print(f"  Set C: {n_c1} C1 (OOD-but-accurate) + {n_c2} C2 (ID-but-failing) = {n_c1+n_c2} total")
    return set_c


def collect_all_sets(model, cfg, seed=42):
    np.random.seed(seed)

    print("[collect] Set A (in-distribution)...")
    env_a = CartpoleEnv(noisy=False, seed=seed)
    set_a = collect_states(model, env_a, cfg['n_eval_episodes'], cfg)
    print(f"  Set A: {len(set_a['h'])} states | mean KL={set_a['kl'].mean():.4f}")

    print("[collect] Set B (near-OOD, noisy obs)...")
    env_b = CartpoleEnv(noisy=True, noise_std=cfg['noise_std'], seed=seed + 1)
    set_b = collect_states(model, env_b, cfg['n_eval_episodes'], cfg)
    print(f"  Set B: {len(set_b['h'])} states | mean KL={set_b['kl'].mean():.4f}")

    print("[collect] Set C (contrastive)...")
    set_c = build_set_c(set_a, set_b)

    return set_a, set_b, set_c
