"""
Task B -- Training-time necessity, Kaggle runner.

Runs on Kaggle because 27 runs x 100,000 env steps (~55 CPU-min/run measured
locally on this project's MacBook Air, ~25h serial) is too large to run
locally within the project's usual iteration loop. Identical logic to the
local run_taskB_training_necessity.py -- copied here, not reimplemented, so
Kaggle and local results are byte-for-byte the same code path. Only the I/O
layer differs: each (task, condition, seed) cell's checkpoint/states/
histories are written to OUT_DIR the moment that cell finishes, so a
Kaggle session that hits its time limit and restarts skips every cell
already on disk and only computes what's missing.

See outputs/deliverables/task_B_training_necessity.md (project repo) for the
full spec this implements (constraint mechanics, refit schedule, ship rule).
"""

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.model.constrained_rssm_hook import install_projection_hook
from src.training.replay_buffer import EpisodeReplayBuffer
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import train_probe
from src.probe.intervention import probe_direction

OUT_DIR = os.environ.get('TASKB_OUT', '/kaggle/working/taskB_results')

SEEDS = [3101, 3102, 3103]
CONDITIONS = ['A_unconstrained', 'B_vconstrained', 'C_randomdir']
TASKS = {
    'cartpole': dict(env_cls='cartpole', domain='cartpole', task='swingup'),
    'reacher':  dict(env_cls='dmc', domain='reacher', task='easy'),
    'pendulum': dict(env_cls='dmc', domain='pendulum', task='swingup'),
}

WARMUP_STEPS = 10_000
REFIT_EVERY = 5_000
REFIT_WINDOW = 20_000


def set_scale(**kwargs):
    """Override module-level constants for a smoke test (mirrors task12_program's
    set_scale convention)."""
    g = globals()
    for k, v in kwargs.items():
        g[k] = v


def make_env(spec, seed):
    if spec['env_cls'] == 'cartpole':
        return CartpoleEnv(task=spec['task'], noisy=False, seed=seed)
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=False, seed=seed)


def fit_v_from_recent(h_recent, kl_recent):
    y = (kl_recent > np.median(kl_recent)).astype(np.int32)
    if len(np.unique(y)) < 2:
        return None
    clf, scaler = train_probe(h_recent, y, C=1.0)
    return probe_direction(clf, scaler)


def train_condition(task, spec, cfg, condition, seed, out_dir, total_env_steps):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # EpisodeReplayBuffer.sample() draws from Python's global `random` module;
    # np.random.seed/torch.manual_seed alone do not make the training loop
    # reproducible once gradient steps start (see run_taskB_training_necessity.py's
    # matching comment -- diagnosed via this task's own pilot repro check).
    random.seed(seed)

    env = make_env(spec, seed=seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg)
    hook = install_projection_hook(model.rssm)

    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len, batch = cfg['seq_len'], cfg['batch_size']
    max_steps = total_env_steps

    log_h, log_kl, log_recon, log_traj = [], [], [], []
    v_history = []
    loss_history = []
    prev_v = None
    rng_dir = np.random.default_rng(seed + 777)

    step_count, traj_id, t0 = 0, 0, time.time()
    ep_obs, ep_act = [], []
    h_inf = torch.zeros(1, cfg['rssm_deter'])
    z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
    obs = env.reset(); ep_obs.append(obs.copy())

    print(f"[taskB {task} {condition} seed{seed}] obs={env.obs_dim} act={env.act_dim} steps={max_steps}", flush=True)

    def maybe_refit(step_count):
        nonlocal prev_v
        if condition == 'A_unconstrained':
            return
        if step_count < WARMUP_STEPS:
            return
        if step_count % REFIT_EVERY != 0:
            return
        if condition == 'B_vconstrained':
            h_recent = np.array(log_h[-REFIT_WINDOW:], dtype=np.float32)
            kl_recent = np.array(log_kl[-REFIT_WINDOW:], dtype=np.float32)
            v = fit_v_from_recent(h_recent, kl_recent)
            if v is None:
                return
        else:
            v = rng_dir.standard_normal(cfg['rssm_deter']).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
        cos = None
        if prev_v is not None:
            cos = float(np.dot(v, prev_v) / (np.linalg.norm(v) * np.linalg.norm(prev_v) + 1e-12))
        v_history.append(dict(step=step_count, v=v.tolist(), cosine_to_prev=cos))
        hook.set_direction(v)
        prev_v = v

    while step_count < max_steps:
        action = np.random.uniform(-1, 1, size=(env.act_dim,)).astype(np.float32)
        model.eval()
        with torch.no_grad():
            ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            at = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            emb = model.encoder(ot)
            h_inf, z_inf, prior_l, post_l = model.rssm.observe_step(h_inf, z_inf, at, emb)
            dec = model.decoder(torch.cat([h_inf, z_inf], dim=-1))
            kl_val = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            rc_val = F.mse_loss(dec, ot, reduction='none').sum().item()

        log_h.append(h_inf.squeeze(0).numpy().copy())
        log_kl.append(kl_val)
        log_recon.append(rc_val)
        log_traj.append(traj_id)

        obs_new, _, done = env.step(action)
        ep_act.append(action.copy())
        step_count += 1

        if done or (len(ep_act) >= cfg['episode_max_steps']):
            ep_obs.append(obs_new.copy())
            buffer.add_episode(ep_obs[:-1], ep_act)
            traj_id += 1
            ep_obs, ep_act = [], []
            h_inf = torch.zeros(1, cfg['rssm_deter'])
            z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
            obs = env.reset(); ep_obs.append(obs.copy())
        else:
            obs = obs_new
            ep_obs.append(obs.copy())

        maybe_refit(step_count)

        if step_count >= cfg['warmup_steps'] and len(buffer) >= seq_len * batch:
            model.train()
            ob, ab = buffer.sample(batch, seq_len, device='cpu')
            loss, recon_l, kl_l = model.compute_loss(ob, ab, kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optim.step()
            loss_history.append(dict(step=step_count, loss=loss.item(), recon=recon_l, kl=kl_l))

        if step_count % 20000 == 0:
            avg_loss = np.mean([l['loss'] for l in loss_history[-200:]]) if loss_history else float('nan')
            print(f"  [{task}/{condition}/s{seed}] step {step_count:,}/{max_steps:,} "
                  f"loss={avg_loss:.4f} kl={np.mean(log_kl[-500:]):.3f} "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'model.pt')
    torch.save({'model_state': model.state_dict(), 'cfg': cfg,
                'obs_dim': env.obs_dim, 'act_dim': env.act_dim,
                'condition': condition, 'seed': seed}, ckpt_path)

    states = dict(h=np.array(log_h, np.float32), kl=np.array(log_kl, np.float32),
                  recon=np.array(log_recon, np.float32), traj_id=np.array(log_traj, np.int64))
    np.savez(os.path.join(out_dir, 'states.npz'), **states)

    with open(os.path.join(out_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f)
    with open(os.path.join(out_dir, 'v_history.json'), 'w') as f:
        json.dump(v_history, f)

    constraint_resid, constraint_resid_v_step = None, None
    for vh in reversed(v_history):
        h_after = log_h[vh['step']:]
        if len(h_after) > 0:
            v_t = torch.tensor(vh['v'], dtype=torch.float32)
            h_tail = torch.tensor(np.array(h_after, dtype=np.float32))
            constraint_resid = float((h_tail @ v_t).abs().mean().item())
            constraint_resid_v_step = vh['step']
            break

    elapsed_min = (time.time() - t0) / 60
    print(f"[taskB {task} {condition} seed{seed}] done in {elapsed_min:.1f}m | "
          f"{traj_id} episodes | {len(loss_history)} grad steps | "
          f"final |v^T h| = {constraint_resid}", flush=True)

    result = dict(task=task, condition=condition, seed=seed,
                  ckpt_path=ckpt_path, states_path=os.path.join(out_dir, 'states.npz'),
                  n_grad_steps=len(loss_history), n_episodes=traj_id,
                  final_loss=loss_history[-1]['loss'] if loss_history else None,
                  constraint_resid_final=constraint_resid,
                  constraint_resid_v_step=constraint_resid_v_step,
                  n_refits=len(v_history), elapsed_min=elapsed_min,
                  total_env_steps=max_steps)
    with open(os.path.join(out_dir, 'cell_result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)
    return result


def run_all(tasks=None, conditions=None, seeds=None, total_env_steps=100_000):
    """Runs every (task, condition, seed) cell not already on disk (a cell is
    'done' iff out_dir/cell_result.json exists), writing each result the
    moment it finishes -- resumable across Kaggle session restarts."""
    cfg = XS_CONFIG.copy()
    tasks = tasks or list(TASKS.keys())
    conditions = conditions or CONDITIONS
    seeds = seeds or SEEDS

    results = {}
    for task in tasks:
        spec = TASKS[task]
        for condition in conditions:
            for seed in seeds:
                key = f'{task}_{condition}_{seed}'
                out_dir = os.path.join(OUT_DIR, key)
                cell_path = os.path.join(out_dir, 'cell_result.json')
                if os.path.exists(cell_path):
                    print(f"[skip] {key} already done")
                    results[key] = json.load(open(cell_path))
                    continue
                results[key] = train_condition(task, spec, cfg, condition, seed, out_dir, total_env_steps)
    return results
