"""
Training loop for the Mini-DreamerV3 world model.
Step-based: 1 gradient step per env step (after warmup).
Supports MPS (Apple Silicon GPU), CUDA, or CPU.
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.wrapper import CartpoleEnv
from src.model.world_model import WorldModel
from src.training.replay_buffer import EpisodeReplayBuffer


def train_world_model(cfg, seed=0):
    """
    Main training loop.
    Collects cfg['total_env_steps'] env steps; does 1 gradient step per env step after warmup.
    Logs (h_t, z_t, kl_t, recon_t) at every env step via online RSSM inference.
    Returns (model, training_states_dict).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg.get('device', 'cpu'))
    print(f"[trainer] device={device}")

    env   = CartpoleEnv(seed=seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    buffer  = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len = cfg['seq_len']
    batch   = cfg['batch_size']
    warmup  = cfg['warmup_steps']
    max_steps = cfg['total_env_steps']

    # Logging arrays (CPU numpy)
    log_h, log_z, log_kl, log_recon = [], [], [], []
    log_step, log_traj = [], []
    loss_history = []

    step_count   = 0
    traj_id      = 0
    t0           = time.time()

    # Episode state
    ep_obs  = []
    ep_act  = []
    h_inf   = torch.zeros(1, cfg['rssm_deter'], device=device)
    z_inf   = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)

    obs = env.reset()
    ep_obs.append(obs.copy())

    print(f"[trainer] {max_steps:,} steps | warmup={warmup:,} | seq_len={seq_len} | batch={batch}")

    while step_count < max_steps:
        # ── Online RSSM inference (eval, no grad) ────────────────────────
        action = np.random.uniform(-1, 1, size=(cfg['act_dim'],)).astype(np.float32)

        model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
            a_t   = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

            embed = model.encoder(obs_t)
            h_inf, z_inf, prior_logits, post_logits = model.rssm.observe_step(
                h_inf, z_inf, a_t, embed)

            decoded   = model.decoder(torch.cat([h_inf, z_inf], dim=-1))
            kl_val    = model.rssm.kl_divergence(post_logits, prior_logits, free_bits=0.0).item()
            recon_val = F.mse_loss(decoded, obs_t, reduction='none').sum().item()

        # Store on CPU numpy
        log_h.append(h_inf.squeeze(0).cpu().numpy().copy())
        log_z.append(post_logits.squeeze(0).cpu().numpy().copy())
        log_kl.append(kl_val)
        log_recon.append(recon_val)
        log_step.append(step_count)
        log_traj.append(traj_id)

        # ── Env step ─────────────────────────────────────────────────────
        obs_new, _, done = env.step(action)
        ep_act.append(action.copy())
        step_count += 1

        ep_done = done or (len(ep_act) >= cfg['episode_max_steps'])
        if ep_done:
            ep_obs.append(obs_new.copy())
            buffer.add_episode(ep_obs[:-1], ep_act)
            traj_id += 1
            ep_obs  = []
            ep_act  = []
            h_inf   = torch.zeros(1, cfg['rssm_deter'], device=device)
            z_inf   = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
            obs = env.reset()
            ep_obs.append(obs.copy())
        else:
            obs = obs_new
            ep_obs.append(obs.copy())

        # ── Gradient step ─────────────────────────────────────────────────
        if step_count >= warmup and len(buffer) >= seq_len * batch:
            model.train()
            obs_b, act_b = buffer.sample(batch, seq_len, device=str(device))
            loss, _, _ = model.compute_loss(obs_b, act_b,
                                            kl_free=cfg['kl_free'],
                                            kl_scale=cfg['kl_scale'])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optim.step()
            loss_history.append(loss.item())

        # ── Progress every 5K steps ───────────────────────────────────────
        if step_count % 5000 == 0:
            elapsed  = time.time() - t0
            avg_loss = float(np.mean(loss_history[-200:])) if loss_history else float('nan')
            avg_kl   = float(np.mean(log_kl[-500:]))
            eta_min  = (max_steps - step_count) / max(step_count, 1) * elapsed / 60
            print(f"  step {step_count:>7,}/{max_steps:,}  "
                  f"loss={avg_loss:.4f}  kl={avg_kl:.3f}  "
                  f"elapsed={elapsed/60:.1f}m  eta≈{eta_min:.0f}m  traj={traj_id}",
                  flush=True)

    # ── Save checkpoint ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(cfg['checkpoint_path']), exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'cfg': cfg}, cfg['checkpoint_path'])
    elapsed_total = time.time() - t0
    print(f"[trainer] done in {elapsed_total/60:.1f} min | {traj_id} episodes | "
          f"{len(loss_history)} grad steps", flush=True)

    training_states = {
        'h':          np.array(log_h,     dtype=np.float32),
        'z':          np.array(log_z,     dtype=np.float32),
        'kl':         np.array(log_kl,    dtype=np.float32),
        'recon':      np.array(log_recon, dtype=np.float32),
        'step_index': np.array(log_step,  dtype=np.int64),
        'traj_id':    np.array(log_traj,  dtype=np.int64),
    }
    return model, training_states


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m  = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m
