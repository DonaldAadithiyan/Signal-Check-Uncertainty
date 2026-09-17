#!/usr/bin/env python3.11
"""
Task 6 (Path B, gap-closing spec) -- DreamerV3-style actor-critic training,
WITH ON-POLICY WORLD-MODEL FINE-TUNING (revised design, scope-expanded from
the original "reuse the frozen world model as-is" spec after an empirical
finding required it -- see below).

--- Why this differs from the original spec ---

The original design trained the actor/critic purely in imagination through
the EXISTING, FROZEN world models (no world-model weight updates). This was
tried first (see git history / GAP_CLOSING_RUN_LOG.md) and found not to work:
the frozen cartpole model was trained on 100K steps of uniform-random-action
data, and real-env testing showed random actions almost never reach anywhere
near cartpole-swingup's upright goal (max cos_pole ~= -0.72 over 500 random
steps, vs +1.0 = upright). The frozen model's training distribution barely
covers the goal region, so an actor trained purely against its frozen
dynamics had no reliable signal to discover real swing-up -- confirmed
empirically across 2000-8000 pure-imagination training steps with no reward
improvement.

Per an explicit decision to fix this rather than report it as a dead end,
this script now runs the STANDARD DreamerV3 alternating loop instead of a
one-shot frozen-model setup:

  repeat for N_OUTER_LOOPS:
    1. Collect real-env episodes under the CURRENT actor (with exploration
       noise), expanding the replay buffer beyond the original random-action-
       only training distribution.
    2. Fine-tune the world model (train_world_model's compute_loss, SAME
       architecture/objective as its original training) on the growing
       replay buffer -- this is what lets the model learn to represent
       states the earlier random-action collection never visited.
    3. Train the actor/critic via imagination through the UPDATED model,
       exactly as before.

This means the "frozen model" premise from the original Task 6 spec is
relaxed to "starts from the existing checkpoint, continues training
on-policy" -- a real scope expansion, done because the alternative (a policy
that provably cannot learn) makes the rest of Task 6 impossible to run
honestly. The base checkpoint, architecture, and hyperparameters are
unchanged; only the fact that training continues is new.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.model.actor_critic import Actor, Critic, state_repr, lambda_return, imagine_rollout
from src.training.replay_buffer import EpisodeReplayBuffer

OUT_DIR = 'outputs/task6_actor_critic'
IMAGINE_HORIZON = 15
N_OUTER_LOOPS = 12                  # alternating collect/fine-tune/imagine-train cycles
EPISODES_PER_COLLECT = 40           # real episodes collected per outer loop
WM_FINETUNE_STEPS_PER_LOOP = 150    # world-model gradient steps per outer loop
AC_TRAIN_STEPS_PER_LOOP = 600       # actor-critic imagination steps per outer loop
BATCH_SIZE = 256
WM_SEQ_LEN = 50
WM_BATCH = 32
GAMMA = 0.99
LAMBDA = 0.95
ACTOR_LR = 3e-5
CRITIC_LR = 8e-5
WM_LR = 3e-4
ENTROPY_SCALE = 3e-3
EXPLORATION_STD_START = 0.6         # extra action noise during real-env collection, annealed
EXPLORATION_STD_END = 0.15

TASK_SEEDS = {
    'cartpole': dict(
        domain='cartpole',
        checkpoints=[
            'outputs/checkpoints/world_model.pt',
            'outputs/multiseed/seed_1/world_model.pt',
            'outputs/multiseed/seed_2/world_model.pt',
            'outputs/multiseed/seed_3/world_model.pt',
            'outputs/multiseed/seed_4/world_model.pt',
        ],
    ),
    'reacher': dict(
        domain='reacher',
        checkpoints=[
            'outputs/second_env/reacher_easy_world_model.pt',
            'outputs/multiseed_env/reacher_seed1/model.pt',
            'outputs/multiseed_env/reacher_seed2/model.pt',
            'outputs/multiseed_env/reacher_seed3/model.pt',
        ],
    ),
    'pendulum': dict(
        domain='pendulum',
        checkpoints=[
            'outputs/third_env/pendulum_swingup_world_model.pt',
            'outputs/multiseed_env/pendulum_seed1/model.pt',
            'outputs/multiseed_env/pendulum_seed2/model.pt',
            'outputs/multiseed_env/pendulum_seed3/model.pt',
        ],
    ),
}


def make_env(domain, seed_i):
    from src.env.wrapper import CartpoleEnv
    from src.env.dmc_wrapper import DMCEnv
    if domain == 'cartpole':
        return CartpoleEnv(task='swingup', seed=seed_i)
    task = 'easy' if domain == 'reacher' else 'swingup'
    return DMCEnv(domain=domain, task=task, seed=seed_i)


def load_frozen_model(ckpt_path, device='cpu'):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck['cfg']
    obs_dim = ck.get('obs_dim', cfg.get('obs_dim'))
    act_dim = ck.get('act_dim', cfg.get('act_dim'))
    model = WorldModel(obs_dim, act_dim, cfg).to(device)
    model.load_state_dict(ck['model_state'])
    return model, obs_dim, act_dim, cfg


@torch.no_grad()
def collect_real_episodes(model, actor, domain, cfg, n_episodes, seed, explore_std, device='cpu'):
    """Collects real-env episodes under the CURRENT actor with added Gaussian
    exploration noise (on top of the actor's own stochasticity), appended to
    the replay buffer. Returns (h,z) initial-state samples for imagination
    training too, reusing this same rollout rather than a separate pass."""
    buffer_episodes = []
    h_samples, z_samples = [], []
    was_training = model.training
    model.eval()
    for i in range(n_episodes):
        env = make_env(domain, seed + i)
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        obs_l, act_l = [obs.copy()], []
        done, step = False, 0
        while not done and step < cfg['episode_max_steps']:
            state = state_repr(h, z)
            action, _ = actor.sample(state)
            noise = torch.randn_like(action) * explore_std
            action = (action + noise).clamp(-1.0, 1.0)
            a_np = action.squeeze(0).cpu().numpy().astype(np.float32)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            embed = model.encoder(obs_t)
            h, z, _, _ = model.rssm.observe_step(h, z, action, embed)
            h_samples.append(h.squeeze(0).clone()); z_samples.append(z.squeeze(0).clone())

            obs, _, done = env.step(a_np)
            obs_l.append(obs.copy()); act_l.append(a_np)
            step += 1
        buffer_episodes.append((obs_l[:-1], act_l))
    if was_training:
        model.train()
    return buffer_episodes, torch.stack(h_samples), torch.stack(z_samples)


def finetune_world_model(model, optim, buffer, n_steps, cfg, device='cpu'):
    model.train()
    losses = []
    for _ in range(n_steps):
        if len(buffer) < WM_SEQ_LEN * 2:
            break
        obs_b, act_b = buffer.sample(WM_BATCH, WM_SEQ_LEN, device=str(device))
        loss, recon_l, kl_l = model.compute_loss(obs_b, act_b, kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optim.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')


def train_actor_critic_steps(model, actor, critic, actor_opt, critic_opt, domain,
                              h_pool, z_pool, n_steps, shaped_reward=False):
    rewards_log = []
    for step in range(n_steps):
        idx = np.random.choice(len(h_pool), min(BATCH_SIZE, len(h_pool)), replace=len(h_pool) < BATCH_SIZE)
        h0 = h_pool[idx]; z0 = z_pool[idx]

        rollout = imagine_rollout(model.rssm, model.decoder, actor, domain, h0, z0, IMAGINE_HORIZON,
                                   shaped_reward=shaped_reward)
        states = state_repr(rollout['h'], rollout['z'])
        rewards = rollout['reward']

        with torch.no_grad():
            values_for_actor = critic(states)
            bootstrap = critic(state_repr(rollout['h'][-1], rollout['z'][-1]))
        actor_returns = lambda_return(rewards, torch.cat([values_for_actor, bootstrap.unsqueeze(0)], dim=0),
                                       gamma=GAMMA, lam=LAMBDA)
        entropy_bonus = ENTROPY_SCALE * rollout['log_prob'].mean()
        actor_loss = -(actor_returns.mean()) - entropy_bonus

        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
        actor_opt.step()

        with torch.no_grad():
            states_det = states.detach()
            rewards_det = rewards.detach()
            bootstrap_det = critic(state_repr(rollout['h'][-1].detach(), rollout['z'][-1].detach()))
        values = critic(states_det)
        values_target = torch.cat([values.detach(), bootstrap_det.unsqueeze(0)], dim=0)
        returns = lambda_return(rewards_det, values_target, gamma=GAMMA, lam=LAMBDA)
        critic_loss = F.mse_loss(values, returns.detach())

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
        critic_opt.step()

        rewards_log.append(rewards.mean().item())
    return float(np.mean(rewards_log)) if rewards_log else float('nan')


def train_task6(task, seed_idx, ckpt_path, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    model, obs_dim, act_dim, model_cfg = load_frozen_model(ckpt_path, device)
    domain = TASK_SEEDS[task]['domain']
    state_dim = model_cfg['rssm_deter'] + model_cfg['rssm_stoch'] * model_cfg['rssm_classes']

    actor = Actor(state_dim, act_dim).to(device)
    critic = Critic(state_dim).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    wm_opt = torch.optim.Adam(model.parameters(), lr=WM_LR)

    buffer = EpisodeReplayBuffer(capacity=2000)
    print(f"[task6/{task}/seed{seed_idx}] state_dim={state_dim} act_dim={act_dim} "
          f"domain={domain} outer_loops={N_OUTER_LOOPS}", flush=True)

    t0 = time.time()
    loop_log = []
    for loop in range(N_OUTER_LOOPS):
        frac = loop / max(N_OUTER_LOOPS - 1, 1)
        explore_std = EXPLORATION_STD_START + frac * (EXPLORATION_STD_END - EXPLORATION_STD_START)

        episodes, h_pool, z_pool = collect_real_episodes(
            model, actor, domain, model_cfg, EPISODES_PER_COLLECT,
            seed=seed + loop * 1000, explore_std=explore_std, device=device)
        for obs_l, act_l in episodes:
            buffer.add_episode(obs_l, act_l)

        wm_loss = finetune_world_model(model, wm_opt, buffer, WM_FINETUNE_STEPS_PER_LOOP, model_cfg, device)
        # pendulum trains on the dense shaped proxy (see reward_from_decoded_obs_torch's
        # docstring for why the exact reward gives no usable gradient here); cartpole/
        # reacher are unaffected (no shaped variant exists for them).
        use_shaped = (domain == 'pendulum')
        mean_reward = train_actor_critic_steps(model, actor, critic, actor_opt, critic_opt,
                                                domain, h_pool, z_pool, AC_TRAIN_STEPS_PER_LOOP,
                                                shaped_reward=use_shaped)
        # also report the REAL (unshaped) reward on the same imagined states, so
        # pendulum's log is comparable to cartpole/reacher and to real-env eval
        with torch.no_grad():
            idx = np.random.choice(len(h_pool), min(BATCH_SIZE, len(h_pool)), replace=len(h_pool) < BATCH_SIZE)
            eval_rollout = imagine_rollout(model.rssm, model.decoder, actor, domain,
                                            h_pool[idx], z_pool[idx], IMAGINE_HORIZON, shaped_reward=False)
            mean_reward_true = eval_rollout['reward'].mean().item()

        elapsed = time.time() - t0
        loop_log.append(dict(loop=loop, wm_loss=wm_loss, mean_imag_reward_train=mean_reward,
                              mean_imag_reward_true=mean_reward_true,
                              buffer_steps=len(buffer), explore_std=explore_std))
        print(f"  [{task}/seed{seed_idx}] loop {loop+1}/{N_OUTER_LOOPS} "
              f"wm_loss={wm_loss:.4f} mean_imag_reward(train)={mean_reward:.4f} "
              f"mean_imag_reward(true)={mean_reward_true:.4f} "
              f"buffer_steps={len(buffer)} explore_std={explore_std:.2f} "
              f"elapsed={elapsed/60:.1f}m", flush=True)

    out_dir = os.path.join(OUT_DIR, task, f'seed{seed_idx}')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(actor_state=actor.state_dict(), critic_state=critic.state_dict(),
                     model_state=model.state_dict(),   # fine-tuned world model, saved alongside
                     obs_dim=obs_dim, act_dim=act_dim, state_dim=state_dim,
                     model_cfg=model_cfg, domain=domain, ckpt_path=ckpt_path, seed=seed),
               os.path.join(out_dir, 'actor_critic.pt'))
    with open(os.path.join(out_dir, 'train_log.json'), 'w') as f:
        json.dump(loop_log, f, indent=2)
    elapsed_total = time.time() - t0
    print(f"[task6/{task}/seed{seed_idx}] done in {elapsed_total/60:.1f}m -> {out_dir}", flush=True)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    for task, spec in TASK_SEEDS.items():
        for seed_idx, ckpt_path in enumerate(spec['checkpoints']):
            out_path = os.path.join(OUT_DIR, task, f'seed{seed_idx}', 'actor_critic.pt')
            if os.path.exists(out_path):
                print(f"[task6] {task}/seed{seed_idx}: already trained, skipping")
                continue
            if not os.path.exists(ckpt_path):
                print(f"[task6] {task}/seed{seed_idx}: checkpoint {ckpt_path} not found, skipping")
                continue
            train_task6(task, seed_idx, ckpt_path, cfg, seed=1000 * (seed_idx + 1) + hash(task) % 1000)

    print("\n[task6] all actor-critic training complete.")


if __name__ == '__main__':
    main()
