#!/usr/bin/env python3.11
"""Camera-ready Task 4: train XS world models on two additional DeepMind Control tasks
(chosen before training: walker-walk and cheetah-run; fallback ball_in_cup-catch only if
one fails to train, which would be reported).

Exact recipe of the existing models (run_multiseed_env.train_seed): XS_CONFIG,
100,000 env steps, uniform-random policy, Adam lr 3e-4, warmup 1000, batch 8 x seq 16,
replay capacity 600 episodes, grad clip 1.0, episodes truncated at 500 steps, seed 0.
Two deliberate differences, both corrections:
  * the online training log pairs o_t with a_{t-1} (zeros at episode start), the
    training convention (WorldModel.compute_loss); the old loggers fed a_t;
  * random.seed(seed) as well (EpisodeReplayBuffer.sample uses Python's random).
Usage: python run_t4_train.py walker walk
"""
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import XS_CONFIG
from src.env.dmc_wrapper import DMCEnv
from src.model.world_model import WorldModel
from src.training.replay_buffer import EpisodeReplayBuffer

ROOT = 'outputs/t4_generality'


def train(domain, task, seed=0):
    cfg = XS_CONFIG.copy()
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    env = DMCEnv(domain, task, seed=seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len, batch, warmup, max_steps = cfg['seq_len'], cfg['batch_size'], cfg['warmup_steps'], cfg['total_env_steps']
    log = {k: [] for k in ('h', 'z', 'kl', 'recon', 'traj_id')}
    step_count, traj_id, t0 = 0, 0, time.time()
    ep_obs, ep_act = [], []
    h_inf = torch.zeros(1, cfg['rssm_deter']); z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
    prev_a = np.zeros(env.act_dim, np.float32)
    obs = env.reset(); ep_obs.append(obs.copy())
    print(f'[t4 {domain}_{task} seed{seed}] obs={env.obs_dim} act={env.act_dim}', flush=True)
    while step_count < max_steps:
        action = np.random.uniform(-1, 1, size=(env.act_dim,)).astype(np.float32)
        model.eval()
        with torch.no_grad():
            ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            ap = torch.tensor(prev_a, dtype=torch.float32).unsqueeze(0)       # a_{t-1}: training convention
            emb = model.encoder(ot)
            h_inf, z_inf, prior_l, post_l = model.rssm.observe_step(h_inf, z_inf, ap, emb)
            dec = model.decoder(torch.cat([h_inf, z_inf], dim=-1))
            kl_val = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            rc_val = F.mse_loss(dec, ot, reduction='none').sum().item()
        log['h'].append(h_inf.squeeze(0).numpy().copy()); log['z'].append(post_l.squeeze(0).numpy().copy())
        log['kl'].append(kl_val); log['recon'].append(rc_val); log['traj_id'].append(traj_id)
        obs_new, _, done = env.step(action); ep_act.append(action.copy()); step_count += 1
        prev_a = action
        if done or (len(ep_act) >= cfg['episode_max_steps']):
            ep_obs.append(obs_new.copy()); buffer.add_episode(ep_obs[:-1], ep_act)
            traj_id += 1; ep_obs, ep_act = [], []
            h_inf = torch.zeros(1, cfg['rssm_deter']); z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
            prev_a = np.zeros(env.act_dim, np.float32)
            obs = env.reset(); ep_obs.append(obs.copy())
        else:
            obs = obs_new; ep_obs.append(obs.copy())
        if step_count >= warmup and len(buffer) >= seq_len * batch:
            model.train()
            ob, ab = buffer.sample(batch, seq_len, device='cpu')
            loss, _, _ = model.compute_loss(ob, ab, kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {step_count}')
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip']); optim.step()
        if step_count % 10000 == 0:
            print(f'  {domain} step {step_count:,}/{max_steps:,} kl={np.mean(log["kl"][-500:]):.2f} '
                  f'rec={np.mean(log["recon"][-500:]):.4f} {(time.time() - t0) / 60:.0f}m', flush=True)
    d = os.path.join(ROOT, f'{domain}_{task}')
    os.makedirs(d, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'obs_dim': env.obs_dim, 'act_dim': env.act_dim},
               os.path.join(d, 'model.pt'))
    np.savez(os.path.join(d, 'training_states.npz'),
             **{k: np.array(v, np.int64 if k == 'traj_id' else np.float32) for k, v in log.items()})
    print(f'[t4] saved {d} ({(time.time() - t0) / 60:.0f} min)', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    train(sys.argv[1], sys.argv[2])
