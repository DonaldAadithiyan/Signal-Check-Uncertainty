#!/usr/bin/env python3.11
"""
Phase 4a — Train the controlled full-vs-partial-observability pair (Reframed_
Confusion_RSSM_Project, Section 10.1).

Task: cartpole-swingup (obs = position(3) + velocity(2), 5-dim). Two conditions,
trained with IDENTICAL architecture, optimizer, seq_len, batch size, training
budget, and random seed for the environment/data stream -- Section 10.1's "keep
as much as possible unchanged":

  FULL:    o_t = s_t            (encoder sees all 5 dims; decoder reconstructs
                                  all 5 -- this reproduces the existing cartpole
                                  setup exactly, retrained here rather than
                                  reusing the frozen checkpoint so that BOTH
                                  conditions share the identical data stream/seed,
                                  which the original checkpoint's training run
                                  does not guarantee against this script's own
                                  fresh env instantiation)
  PARTIAL: o_t = M s_t          (encoder sees position(3) only; decoder still
                                  reconstructs the full 5-dim state -- forces the
                                  model to infer hidden velocity from h_t/z_t,
                                  the genuine filtering/POMDP setup)

Both use PomdpWorldModel (src/model/pomdp_world_model.py) for a fully controlled
comparison -- FULL's obs_dim_in == obs_dim_out == 5 makes it architecturally
identical to WorldModel in everything but class name, so this is not comparing
apples to oranges.

Same total_env_steps, same replay capacity, same everything from XS_CONFIG,
same training-run seed. Saves both checkpoints + activation dumps for the
downstream analysis script (run_phase4_pomdp_analysis.py).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

from src.config import XS_CONFIG
from src.model.pomdp_world_model import PomdpWorldModel
from src.env.wrapper import CartpoleEnv
from src.training.replay_buffer import EpisodeReplayBuffer

SEED = 4242
VELOCITY_MASK_DIMS = [3, 4]   # obs layout: position(0,1,2) + velocity(3,4)
OUT_DIR = 'outputs/phase4_pomdp'


def mask_obs(obs_full, mask_dims):
    """obs_full: (..., 5). Zero out the masked dims (velocity)."""
    obs_masked = obs_full.copy() if isinstance(obs_full, np.ndarray) else obs_full.clone()
    obs_masked[..., mask_dims] = 0.0
    return obs_masked


def train_condition(cfg, condition, seed=SEED):
    """condition: 'full' or 'partial'. Returns (model, training_states)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(cfg.get('device', 'cpu'))
    env = CartpoleEnv(seed=seed)
    obs_dim_full = env.obs_dim
    act_dim = env.act_dim

    obs_dim_in = 3 if condition == 'partial' else obs_dim_full   # position-only vs full
    obs_dim_out = obs_dim_full   # ALWAYS reconstruct the full state (Section 10.1 design)

    model = PomdpWorldModel(obs_dim_in, obs_dim_out, act_dim, cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])

    seq_len = cfg['seq_len']
    batch = cfg['batch_size']
    warmup = cfg['warmup_steps']
    max_steps = cfg['total_env_steps']

    log_h, log_z, log_kl, log_recon, log_step, log_traj = [], [], [], [], [], []
    loss_history = []
    step_count, traj_id = 0, 0
    t0 = time.time()

    ep_obs, ep_act = [], []
    h_inf = torch.zeros(1, cfg['rssm_deter'], device=device)
    z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)

    obs = env.reset()
    ep_obs.append(obs.copy())

    print(f"[phase4/{condition}] obs_dim_in={obs_dim_in} obs_dim_out={obs_dim_out} "
          f"{max_steps:,} steps, seed={seed}")

    while step_count < max_steps:
        action = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)

        model.eval()
        with torch.no_grad():
            obs_full_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_in_t = obs_full_t.clone()
            if condition == 'partial':
                obs_in_t = obs_in_t[:, :3]   # position-only slice
            a_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

            embed = model.encoder(obs_in_t)
            h_inf, z_inf, prior_logits, post_logits = model.rssm.observe_step(
                h_inf, z_inf, a_t, embed)
            decoded = model.decoder(torch.cat([h_inf, z_inf], dim=-1))
            kl_val = model.rssm.kl_divergence(post_logits, prior_logits, free_bits=0.0).item()
            recon_val = torch.nn.functional.mse_loss(
                decoded, obs_full_t, reduction='none').sum().item()

        log_h.append(h_inf.squeeze(0).cpu().numpy().copy())
        log_z.append(post_logits.squeeze(0).cpu().numpy().copy())
        log_kl.append(kl_val)
        log_recon.append(recon_val)
        log_step.append(step_count)
        log_traj.append(traj_id)

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
            obs = env.reset()
            ep_obs.append(obs.copy())
        else:
            obs = obs_new
            ep_obs.append(obs.copy())

        if step_count >= warmup and len(buffer) >= seq_len * batch:
            model.train()
            obs_b, act_b = buffer.sample(batch, seq_len, device=str(device))
            obs_in_b = obs_b.clone()
            if condition == 'partial':
                obs_in_b = obs_in_b[:, :, :3]
            loss, _, _ = model.compute_loss(obs_in_b, obs_b, act_b,
                                             kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optim.step()
            loss_history.append(loss.item())

        if step_count % 5000 == 0:
            elapsed = time.time() - t0
            avg_loss = float(np.mean(loss_history[-200:])) if loss_history else float('nan')
            avg_kl = float(np.mean(log_kl[-500:]))
            eta_min = (max_steps - step_count) / max(step_count, 1) * elapsed / 60
            print(f"  [{condition}] step {step_count:>7,}/{max_steps:,}  "
                  f"loss={avg_loss:.4f}  kl={avg_kl:.3f}  elapsed={elapsed/60:.1f}m  "
                  f"eta≈{eta_min:.0f}m  traj={traj_id}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f'{condition}_world_model.pt')
    torch.save({'model_state': model.state_dict(), 'cfg': cfg,
                'obs_dim_in': obs_dim_in, 'obs_dim_out': obs_dim_out,
                'act_dim': act_dim, 'condition': condition, 'seed': seed}, ckpt_path)
    elapsed_total = time.time() - t0
    print(f"[phase4/{condition}] done in {elapsed_total/60:.1f} min | {traj_id} episodes | "
          f"{len(loss_history)} grad steps -> {ckpt_path}", flush=True)

    training_states = {
        'h': np.array(log_h, dtype=np.float32),
        'z': np.array(log_z, dtype=np.float32),
        'kl': np.array(log_kl, dtype=np.float32),
        'recon': np.array(log_recon, dtype=np.float32),
        'step_index': np.array(log_step, dtype=np.int64),
        'traj_id': np.array(log_traj, dtype=np.int64),
    }
    np.savez(os.path.join(OUT_DIR, f'{condition}_training_states.npz'), **training_states)
    return model, training_states


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    for condition in ['full', 'partial']:
        ckpt_path = os.path.join(OUT_DIR, f'{condition}_world_model.pt')
        states_path = os.path.join(OUT_DIR, f'{condition}_training_states.npz')
        if os.path.exists(ckpt_path) and os.path.exists(states_path):
            print(f"[phase4] {condition}: checkpoint + states already exist, skipping training")
            continue
        train_condition(cfg, condition, seed=SEED)

    print("\n[phase4] both conditions trained. Run run_phase4_pomdp_analysis.py next.")


if __name__ == '__main__':
    main()
