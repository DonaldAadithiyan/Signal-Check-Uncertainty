#!/usr/bin/env python3.11
"""
Task B — Training-time necessity.

Question: is the predictive-difficulty representation necessary for the
world model to learn its competence, or is it a by-product that learning
can do without or rebuild elsewhere?

Three conditions, trained fresh (never reusing old checkpoints), identical
code path:
  A. Unconstrained control -- standard training, projection hook disabled.
  B. v-constrained -- h_tilde_t = h_t - v_t(v_t^T h_t) applied at every GRU
     step (see src/model/constrained_rssm_hook.py), v_t a running online
     probe estimate, refit every 5,000 env steps on the most recent 20,000
     h_t (the training loop's own online-inference log -- the same log
     every other training script in this project already produces as
     training_states.npz; reused here as the refit source rather than
     resampling the replay buffer, since it IS h_t at the moment each step
     was taken, matching the spec's "most recent 20,000 h_t" requirement
     exactly).
  C. Matched random-direction control -- same projection machinery, a fresh
     random unit direction at the same warmup/refit schedule instead of a
     fitted probe.

3 tasks x 3 conditions x 3 seeds (3101, 3102, 3103) = 27 runs, 100,000 env
steps each. Declared fallback (decided now, before any run): if compute for
all 27 isn't available, run cartpole only (9 runs) -- see FALLBACK_CARTPOLE_ONLY.

No policy/actor-critic anywhere (matches this project's standing "no policy
exists in this XS pipeline" fact) -- return is not measured.
"""

import os
import json
import time
import random
import argparse
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

OUT_DIR = 'outputs/taskB_training_necessity'
SEEDS = [3101, 3102, 3103]           # new namespace, unused elsewhere in this project
CONDITIONS = ['A_unconstrained', 'B_vconstrained', 'C_randomdir']

WARMUP_STEPS = 10_000                # no constraint before this many env steps
REFIT_EVERY = 5_000                  # refit v_t / draw new random dir every N env steps
REFIT_WINDOW = 20_000                # most recent N h_t used for the refit

TASKS = {
    'cartpole': dict(env_cls='cartpole', domain='cartpole', task='swingup'),
    'reacher':  dict(env_cls='dmc', domain='reacher', task='easy'),
    'pendulum': dict(env_cls='dmc', domain='pendulum', task='swingup'),
}

# Declared fallback, decided now (per §3.5): if full 27-run compute isn't
# available, run cartpole only. Recorded here, not decided by how results
# look partway through. Overridden by --tasks on the command line if the
# caller has decided otherwise (e.g. running on Kaggle with more compute).
FALLBACK_CARTPOLE_ONLY = False


def make_env(spec, seed):
    if spec['env_cls'] == 'cartpole':
        return CartpoleEnv(task=spec['task'], noisy=False, seed=seed)
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=False, seed=seed)


def fit_v_from_recent(h_recent, kl_recent):
    """Standard logistic probe: L2, C=1, lbfgs, standardized -- identical
    settings to every prior probe in this project. Labels: KL_t > median of
    the recent window."""
    y = (kl_recent > np.median(kl_recent)).astype(np.int32)
    if len(np.unique(y)) < 2:
        return None
    clf, scaler = train_probe(h_recent, y, C=1.0)
    v = probe_direction(clf, scaler)
    return v


def train_condition(task, spec, cfg, condition, seed, out_dir):
    """Trains one (task, condition, seed) cell from scratch. Returns dict of
    loss history, v_t history (or random-dir history), final model, states."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # EpisodeReplayBuffer.sample() (src/training/replay_buffer.py) draws from
    # Python's global `random` module, not numpy/torch -- np.random.seed and
    # torch.manual_seed alone do NOT make training-loop reproducible once
    # gradient steps start consuming buffer.sample() calls (discovered via
    # this task's own pilot reproducibility check, §3.6.3: h/kl/recon first
    # diverged at exactly cfg['warmup_steps']=1000, the first gradient step).
    # This exact gap exists unfixed elsewhere in the project too (src/
    # training/trainer.py, run_multiseed_env.py never seed `random` either),
    # but those scripts never claimed training-loop byte-reproducibility --
    # only inference/analysis-time reproducibility on already-trained models
    # was audited and fixed (see rng_bug_audit.md). Task B is the first place
    # this project requires literal training-run reproducibility, so it is
    # fixed here rather than carried forward as an unexamined gap.
    random.seed(seed)

    env = make_env(spec, seed=seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg)
    hook = install_projection_hook(model.rssm)   # no-op until set_direction is called

    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len, batch, max_steps = cfg['seq_len'], cfg['batch_size'], cfg['total_env_steps']

    log_h, log_kl, log_recon, log_traj = [], [], [], []
    v_history = []             # list of (step, v_or_randomdir, cosine_to_prev)
    loss_history = []
    prev_v = None

    rng_dir = np.random.default_rng(seed + 777)   # condition C's own seeded RNG, per spec

    step_count, traj_id, t0 = 0, 0, time.time()
    ep_obs, ep_act = [], []
    h_inf = torch.zeros(1, cfg['rssm_deter'])
    z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
    obs = env.reset(); ep_obs.append(obs.copy())

    print(f"[taskB {task} {condition} seed{seed}] obs={env.obs_dim} act={env.act_dim}", flush=True)

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
        else:  # C_randomdir
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

    # constraint-effectiveness diagnostic on the LAST v/random-dir that actually
    # governed at least one post-refit step (pilot check §3.6.1): |v^T h_tilde|
    # on steps strictly AFTER that refit. A fixed-size tail window would dilute
    # the residual with steps governed by an earlier v or pre-constraint steps,
    # understating effectiveness; but if the very last refit lands exactly on
    # the final training step (e.g. max_steps is an exact multiple of
    # REFIT_EVERY), it never governs any logged step, so we fall back to the
    # most recent refit that does have data after it.
    constraint_resid = None
    constraint_resid_v_step = None
    for vh in reversed(v_history):
        h_after = log_h[vh['step']:]
        if len(h_after) > 0:
            v_t = torch.tensor(vh['v'], dtype=torch.float32)
            h_tail = torch.tensor(np.array(h_after, dtype=np.float32))
            constraint_resid = float((h_tail @ v_t).abs().mean().item())
            constraint_resid_v_step = vh['step']
            break

    print(f"[taskB {task} {condition} seed{seed}] done in {(time.time()-t0)/60:.1f}m | "
          f"{traj_id} episodes | {len(loss_history)} grad steps | "
          f"final |v^T h| = {constraint_resid}", flush=True)

    return dict(ckpt_path=ckpt_path, states_path=os.path.join(out_dir, 'states.npz'),
                n_grad_steps=len(loss_history), n_episodes=traj_id,
                final_loss=loss_history[-1]['loss'] if loss_history else None,
                constraint_resid_final=constraint_resid,
                constraint_resid_v_step=constraint_resid_v_step,
                n_refits=len(v_history))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=None,
                         help='override task list (default: all 3, unless FALLBACK_CARTPOLE_ONLY)')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--conditions', nargs='+', default=CONDITIONS)
    parser.add_argument('--pilot', action='store_true',
                         help='pilot mode: cartpole seed 3101 only, all 3 conditions')
    args = parser.parse_args()

    cfg = XS_CONFIG.copy()

    if args.pilot:
        tasks = ['cartpole']
        seeds = [3101]
        conditions = CONDITIONS
    else:
        tasks = args.tasks or (['cartpole'] if FALLBACK_CARTPOLE_ONLY else list(TASKS.keys()))
        seeds = args.seeds
        conditions = args.conditions

    manifest = {}
    for task in tasks:
        spec = TASKS[task]
        for condition in conditions:
            for seed in seeds:
                key = f'{task}_{condition}_{seed}'
                out_dir = os.path.join(OUT_DIR, key)
                ckpt_path = os.path.join(out_dir, 'model.pt')
                if os.path.exists(ckpt_path):
                    print(f"[skip] {key} already trained")
                    manifest[key] = dict(ckpt_path=ckpt_path, states_path=os.path.join(out_dir, 'states.npz'),
                                          skipped=True)
                    continue
                print(f"\n{'='*78}\n{key}\n{'='*78}")
                manifest[key] = train_condition(task, spec, cfg, condition, seed, out_dir)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=float)
    print(f"\nWrote manifest to {OUT_DIR}/manifest.json")


if __name__ == '__main__':
    main()
