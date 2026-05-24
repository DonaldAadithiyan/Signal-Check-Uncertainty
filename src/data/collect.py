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


def collect_states_with_ensemble(main_model, ensemble_models, env, n_episodes, cfg):
    """
    RWM-U-style ensemble collection. All models step through the same trajectory
    in lockstep — each building its own h_t from the full observation sequence.
    At each step, variance across ensemble models' decoded predictions is the
    disagreement score. This matches the RWM-U baseline methodology.
    """
    device = next(main_model.parameters()).device
    all_models = [main_model] + ensemble_models
    for m in all_models:
        m.eval()

    all_h, all_z, all_kl, all_recon, all_obs, all_ens_var = [], [], [], [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        # Each model maintains its own h_t and z_t
        hs = [torch.zeros(1, cfg['rssm_deter'],                       device=device)
              for _ in all_models]
        zs = [torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
              for _ in all_models]

        done = False
        step = 0

        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                action = np.random.uniform(-1, 1, size=(cfg['act_dim'],)).astype(np.float32)
                obs_t  = torch.tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
                a_t    = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

                decoded_preds = []
                for i, m in enumerate(all_models):
                    embed = m.encoder(obs_t)
                    hs[i], zs[i], prior_l, post_l = m.rssm.observe_step(hs[i], zs[i], a_t, embed)
                    dec = m.decoder(torch.cat([hs[i], zs[i]], dim=-1))
                    decoded_preds.append(dec.squeeze(0).cpu().numpy())

                    if i == 0:  # main model metrics
                        kl_val    = m.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                        recon_val = F.mse_loss(dec, obs_t, reduction='none').sum().item()
                        h_np = hs[0].squeeze(0).cpu().numpy().copy()
                        z_np = post_l.squeeze(0).cpu().numpy().copy()

                # Ensemble disagreement = variance across ensemble models' predictions
                ens_preds = np.stack(decoded_preds[1:], axis=0)  # exclude main model
                ens_var   = ens_preds.var(axis=0).mean()

                all_h.append(h_np)
                all_z.append(z_np)
                all_kl.append(kl_val)
                all_recon.append(recon_val)
                all_obs.append(obs.copy())
                all_ens_var.append(float(ens_var))

                obs, _, done = env.step(action)
                step += 1

        if (ep + 1) % 5 == 0:
            print(f"  collected episode {ep + 1}/{n_episodes}")

    return {
        'h':       np.array(all_h,       dtype=np.float32),
        'z':       np.array(all_z,       dtype=np.float32),
        'kl':      np.array(all_kl,      dtype=np.float32),
        'recon':   np.array(all_recon,   dtype=np.float32),
        'obs':     np.array(all_obs,     dtype=np.float32),
        'ens_var': np.array(all_ens_var, dtype=np.float32),
    }


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


def build_set_c(set_a: dict, set_b: dict, n_bins=10, per_bin=20, max_total=200):
    """
    KL-matched contrastive set. Pool A+B, bin by KL percentile, then within
    each bin split by recon error. C1 and C2 end up with matched KL distributions,
    differing only in whether the model is coping (low recon) or confused (high recon).

    C1 (label=0): low recon within KL bin — model handling the situation
    C2 (label=1): high recon within KL bin — model confused at the same KL level

    This prevents the KL-recon correlation from inflating Set C AUROC.
    """
    all_h     = np.concatenate([set_a['h'],     set_b['h']])
    all_z     = np.concatenate([set_a['z'],     set_b['z']])
    all_kl    = np.concatenate([set_a['kl'],    set_b['kl']])
    all_recon = np.concatenate([set_a['recon'], set_b['recon']])
    all_obs   = np.concatenate([set_a['obs'],   set_b['obs']])

    bin_edges = np.percentile(all_kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(all_kl, bin_edges[1:-1])

    rng = np.random.default_rng(42)
    c1_indices, c2_indices = [], []

    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) < 4:
            continue
        recon_b     = all_recon[idx]
        low_thresh  = np.percentile(recon_b, 25)
        high_thresh = np.percentile(recon_b, 75)
        c1_cand = idx[recon_b <= low_thresh]
        c2_cand = idx[recon_b >= high_thresh]
        n_pick  = min(per_bin, len(c1_cand), len(c2_cand))
        if n_pick == 0:
            continue
        c1_indices.extend(rng.choice(c1_cand, n_pick, replace=False).tolist())
        c2_indices.extend(rng.choice(c2_cand, n_pick, replace=False).tolist())

    if len(c1_indices) > max_total:
        c1_indices = rng.choice(c1_indices, max_total, replace=False).tolist()
    if len(c2_indices) > max_total:
        c2_indices = rng.choice(c2_indices, max_total, replace=False).tolist()

    n_c1, n_c2 = len(c1_indices), len(c2_indices)

    set_c = {
        'h':      np.concatenate([all_h[c1_indices],     all_h[c2_indices]]),
        'z':      np.concatenate([all_z[c1_indices],     all_z[c2_indices]]),
        'kl':     np.concatenate([all_kl[c1_indices],    all_kl[c2_indices]]),
        'recon':  np.concatenate([all_recon[c1_indices], all_recon[c2_indices]]),
        'obs':    np.concatenate([all_obs[c1_indices],   all_obs[c2_indices]]),
        'labels': np.array([0] * n_c1 + [1] * n_c2, dtype=np.int32),
        'group':  np.array(['C1'] * n_c1 + ['C2'] * n_c2),
    }

    print(f"  Set C (KL-matched): {n_c1} C1 (coping) + {n_c2} C2 (confused) = {n_c1+n_c2} total")
    print(f"  C1 KL: mean={all_kl[c1_indices].mean():.2f}  "
          f"C2 KL: mean={all_kl[c2_indices].mean():.2f}")
    print(f"  C1 recon: mean={all_recon[c1_indices].mean():.3f}  "
          f"C2 recon: mean={all_recon[c2_indices].mean():.3f}")
    return set_c


def build_set_c_strong(model, cfg, set_a, n_bins=10, per_bin=20, max_total=200, seed=42):
    """
    Strong contrastive set using genuinely novel states.

    Novel states are collected from cartpole_balance — same 5-dim observation
    space as training (swingup), but completely different task dynamics. The model
    was never trained on balance; the pole starts upright and the distribution of
    states is entirely outside the swingup training distribution.

    C1 (label=0): novel (balance) states where the model is coping — low recon
    C2 (label=1): familiar (swingup) states where the model is confused — high recon

    KL-matched between groups so the probe cannot rely on KL magnitude.
    """
    print("  Collecting novel states from cartpole_balance...")
    env_novel  = CartpoleEnv(task='balance', noisy=False, seed=seed + 10)
    novel      = collect_states(model, env_novel, cfg['n_eval_episodes'], cfg)
    print(f"  Novel (balance): {len(novel['h'])} states | "
          f"mean KL={novel['kl'].mean():.3f}  mean recon={novel['recon'].mean():.3f}")

    # Pool novel + swingup (set_a) for KL-matched binning
    all_h     = np.concatenate([novel['h'],     set_a['h']])
    all_z     = np.concatenate([novel['z'],     set_a['z']])
    all_kl    = np.concatenate([novel['kl'],    set_a['kl']])
    all_recon = np.concatenate([novel['recon'], set_a['recon']])
    all_obs   = np.concatenate([novel['obs'],   set_a['obs']])
    # Source tag: 0 = novel/balance, 1 = familiar/swingup
    source    = np.array([0] * len(novel['h']) + [1] * len(set_a['h']), dtype=np.int32)

    bin_edges = np.percentile(all_kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(all_kl, bin_edges[1:-1])

    rng = np.random.default_rng(seed)
    c1_indices, c2_indices = [], []

    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        # C1 candidates: novel states (source=0) with low recon in this KL bin
        novel_in_bin  = idx[source[idx] == 0]
        swingup_in_bin = idx[source[idx] == 1]
        if len(novel_in_bin) < 2 or len(swingup_in_bin) < 2:
            continue

        recon_novel   = all_recon[novel_in_bin]
        recon_swingup = all_recon[swingup_in_bin]

        c1_cand = novel_in_bin[recon_novel   <= np.percentile(recon_novel,   40)]
        c2_cand = swingup_in_bin[recon_swingup >= np.percentile(recon_swingup, 60)]

        n_pick = min(per_bin, len(c1_cand), len(c2_cand))
        if n_pick == 0:
            continue
        c1_indices.extend(rng.choice(c1_cand, n_pick, replace=False).tolist())
        c2_indices.extend(rng.choice(c2_cand, n_pick, replace=False).tolist())

    if len(c1_indices) > max_total:
        c1_indices = rng.choice(c1_indices, max_total, replace=False).tolist()
    if len(c2_indices) > max_total:
        c2_indices = rng.choice(c2_indices, max_total, replace=False).tolist()

    n_c1, n_c2 = len(c1_indices), len(c2_indices)

    set_c = {
        'h':      np.concatenate([all_h[c1_indices],     all_h[c2_indices]]),
        'z':      np.concatenate([all_z[c1_indices],     all_z[c2_indices]]),
        'kl':     np.concatenate([all_kl[c1_indices],    all_kl[c2_indices]]),
        'recon':  np.concatenate([all_recon[c1_indices], all_recon[c2_indices]]),
        'obs':    np.concatenate([all_obs[c1_indices],   all_obs[c2_indices]]),
        'labels': np.array([0] * n_c1 + [1] * n_c2, dtype=np.int32),
        'group':  np.array(['C1'] * n_c1 + ['C2'] * n_c2),
    }

    print(f"  Set C (strong, KL-matched): {n_c1} C1 (novel+coping) + {n_c2} C2 (familiar+confused) = {n_c1+n_c2} total")
    print(f"  C1 KL: mean={all_kl[c1_indices].mean():.2f} ± {all_kl[c1_indices].std():.2f}  "
          f"C2 KL: mean={all_kl[c2_indices].mean():.2f} ± {all_kl[c2_indices].std():.2f}")
    print(f"  C1 recon: mean={all_recon[c1_indices].mean():.3f}  "
          f"C2 recon: mean={all_recon[c2_indices].mean():.3f}")
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
