#!/usr/bin/env python3.11
"""
Train world model from scratch, saving checkpoints at specified step intervals.

Used for §6.2 checkpoint verification: z_gate saturation grows during training,
giving real variance in (mean z_gate, probe-PC angle) across checkpoints.
Expected: early checkpoints have lower z_gate and lower probe-PC angle;
late checkpoints saturate toward z_gate≈0.94 and angle≈88°.

Saves to outputs/checkpoints/ckpt_{step}.pt for each checkpoint step.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import XS_CONFIG
from src.env.wrapper import CartpoleEnv
from src.model.world_model import WorldModel
from src.training.replay_buffer import EpisodeReplayBuffer


CHECKPOINT_STEPS = [5_000, 10_000, 20_000, 40_000, 70_000, 100_000]
SEED             = 42   # different from existing model (seed=0) to avoid cache collision
CKPT_DIR         = 'outputs/checkpoints'


def train_with_checkpoints(cfg, seed=SEED, checkpoint_steps=CHECKPOINT_STEPS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg.get('device', 'cpu'))
    os.makedirs(CKPT_DIR, exist_ok=True)

    env   = CartpoleEnv(seed=seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    buffer    = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len   = cfg['seq_len']
    batch     = cfg['batch_size']
    warmup    = cfg['warmup_steps']
    max_steps = cfg['total_env_steps']

    step_count   = 0
    traj_id      = 0
    t0           = time.time()
    loss_history = []
    ckpt_set     = set(checkpoint_steps)

    h_inf = torch.zeros(1, cfg['rssm_deter'], device=device)
    z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
    obs   = env.reset()
    ep_obs, ep_act = [obs.copy()], []

    print(f"[checkpoint_training] seed={seed}  device={device}")
    print(f"  Checkpoints at: {checkpoint_steps}")

    while step_count < max_steps:
        action = np.random.uniform(-1, 1, size=(cfg['act_dim'],)).astype(np.float32)

        model.eval()
        with torch.no_grad():
            obs_t  = torch.tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
            a_t    = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            embed  = model.encoder(obs_t)
            h_inf, z_inf, prior_logits, post_logits = model.rssm.observe_step(
                h_inf, z_inf, a_t, embed)
            decoded   = model.decoder(torch.cat([h_inf, z_inf], dim=-1))

        obs_new, _, done = env.step(action)
        ep_act.append(action.copy())
        step_count += 1

        ep_done = done or (len(ep_act) >= cfg['episode_max_steps'])
        if ep_done:
            ep_obs.append(obs_new.copy())
            buffer.add_episode(ep_obs[:-1], ep_act)
            traj_id += 1
            ep_obs, ep_act = [], []
            h_inf = torch.zeros(1, cfg['rssm_deter'], device=device)
            z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
            obs   = env.reset()
            ep_obs.append(obs.copy())
        else:
            obs = obs_new
            ep_obs.append(obs.copy())

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

        # Save checkpoint
        if step_count in ckpt_set:
            ckpt_path = os.path.join(CKPT_DIR, f'ckpt_{step_count:06d}.pt')
            torch.save({'model_state': model.state_dict(), 'cfg': cfg,
                        'step': step_count}, ckpt_path)
            elapsed = time.time() - t0
            avg_loss = float(np.mean(loss_history[-200:])) if loss_history else float('nan')
            print(f"  [ckpt saved] step={step_count:>7,}  loss={avg_loss:.4f}  "
                  f"elapsed={elapsed/60:.1f}m", flush=True)

        if step_count % 10000 == 0:
            elapsed = time.time() - t0
            eta = (max_steps - step_count) / max(step_count, 1) * elapsed / 60
            print(f"  step {step_count:>7,}/{max_steps:,}  "
                  f"elapsed={elapsed/60:.1f}m  eta≈{eta:.0f}m", flush=True)

    elapsed_total = time.time() - t0
    print(f"[checkpoint_training] done in {elapsed_total/60:.1f} min", flush=True)


if __name__ == '__main__':
    cfg = XS_CONFIG.copy()
    train_with_checkpoints(cfg)
    print(f"\nCheckpoints saved: {[f'ckpt_{s:06d}.pt' for s in CHECKPOINT_STEPS]}")
