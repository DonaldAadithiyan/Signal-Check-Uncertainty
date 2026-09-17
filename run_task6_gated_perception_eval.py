#!/usr/bin/env python3.11
"""
Task 6 (Path B, gap-closing spec) -- gated-perception evaluation.

Loads the actor-critic policies trained by run_task6_actor_critic_training.py
(13 policies: cartpole x5, reacher x4, pendulum x4 seeds) and evaluates the
confusion-gated observation-querying scheme against baselines, measuring
GENUINE TASK RETURN (not just recall, unlike Pillar 4/Task 4a's routing
result) at matched query budgets.

Design (per the spec):
  - Policy acts under open-loop latent imagination by default (h_t rolled
    forward via imagine_step, no real observation), real observation queried
    when the gating signal exceeds a budget-calibrated threshold.
  - Compared against: fixed-budget random querying, periodic querying,
    raw-KL threshold, reconstruction-error threshold -- not just one point.
  - Return-vs-budget CURVE (multiple query-budget fractions), not a single
    operating point.
  - Multi-seed throughout (5 cartpole, 4 reacher/pendulum -- matching the
    actor-critic training seed counts).
  - Empirical null: same threshold scheme using RANDOM directions in place
    of the confusion probe direction v, matching the causal-validation
    standard used everywhere else in this project (Task G / Phase 3 / Phase 6).

Genuine real reward (unshaped, exact) is used for ALL measurement here,
including pendulum -- the shaped reward was training-only.

Cost-tractability note: N_EPISODES_EVAL and N_NULL are set explicitly smaller
than Phase 6's addendum-hardened 500-site/50-direction standard -- stated as a
tradeoff, same rationale used for Task 3/Task 4's reduced nulls elsewhere in
this gap-closing pass (this is a genuinely new, 13-policy x 5-budget x
multi-condition evaluation, an order of magnitude more evaluate calls than
any single prior causal test in this project).
"""

import os
import json
import numpy as np
import torch

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.model.actor_critic import Actor, state_repr
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import probe_direction, random_matched_direction

OUT_DIR = 'outputs/task6_gated_perception'
EPISODE_LEN_EVAL = 500
N_EPISODES_EVAL = 15
QUERY_BUDGETS = [0.1, 0.3, 0.5, 0.7, 1.0]
N_NULL = 10
CONDITIONS = ['confusion', 'random_budget', 'periodic', 'kl_threshold', 'recon_threshold']

TASK_SEEDS = {
    'cartpole': dict(domain='cartpole', ac_dir='outputs/task6_actor_critic/cartpole',
                      training_states='outputs/data/training_states.npz', n_seeds=5),
    'reacher': dict(domain='reacher', ac_dir='outputs/task6_actor_critic/reacher',
                     training_states='outputs/second_env/reacher_easy_training_states.npz', n_seeds=4),
    'pendulum': dict(domain='pendulum', ac_dir='outputs/task6_actor_critic/pendulum',
                      training_states='outputs/third_env/pendulum_swingup_training_states.npz', n_seeds=4),
}


def make_env(domain, seed_i):
    from src.env.wrapper import CartpoleEnv
    from src.env.dmc_wrapper import DMCEnv
    if domain == 'cartpole':
        return CartpoleEnv(task='swingup', seed=int(seed_i))
    task = 'easy' if domain == 'reacher' else 'swingup'
    return DMCEnv(domain=domain, task=task, seed=int(seed_i))


def load_actor_and_model(task, seed_idx):
    ck = torch.load(os.path.join(TASK_SEEDS[task]['ac_dir'], f'seed{seed_idx}', 'actor_critic.pt'),
                     map_location='cpu')
    model = WorldModel(ck['obs_dim'], ck['act_dim'], ck['model_cfg'])
    model.load_state_dict(ck['model_state'])
    model.eval()
    actor = Actor(ck['state_dim'], ck['act_dim'])
    actor.load_state_dict(ck['actor_state'])
    actor.eval()
    return actor, model, ck


def fit_probe_and_direction(task, tr):
    y = binarise_by_median(tr['kl'])
    from sklearn.model_selection import train_test_split
    idx_tr, _ = train_test_split(np.arange(len(tr['h'])), test_size=0.40, stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])
    v = probe_direction(clf, scaler)
    kl_median = float(np.median(tr['kl']))
    return clf, scaler, v, kl_median


def run_episodes(actor, model, cfg, domain, condition, budget, direction, threshold,
                  n_episodes, base_seed):
    """Runs n_episodes under the given condition/budget, returns (mean_reward,
    std_reward, mean_query_rate). All using the REAL, exact reward from
    env.step (unshaped for every task, including pendulum).

    `direction`/`threshold` are only used by the 'confusion' condition (and
    its random-direction null variant): `direction` is the probe/null
    direction to project h onto, `threshold` is the budget-calibrated
    projection cutoff (for 'kl_threshold'/'recon_threshold' conditions,
    `threshold` is instead the KL/recon median used by those baselines)."""
    period = max(1, int(round(1.0 / budget))) if budget > 0 else 10**9
    rewards, query_rates = [], []

    for ep in range(n_episodes):
        env = make_env(domain, base_seed + ep)
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'])
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
        total_reward, n_queries, step = 0.0, 0, 0
        done = False
        last_kl, last_recon = 0.0, 0.0
        recon_hist = []

        while not done and step < EPISODE_LEN_EVAL:
            state = state_repr(h, z)
            with torch.no_grad():
                action = actor.act(state, deterministic=True)
            a_np = action.squeeze(0).numpy()

            if condition == 'confusion':
                score = float(h.squeeze(0).numpy() @ direction)
                should_query = score > threshold
            elif condition == 'random_budget':
                should_query = np.random.default_rng(base_seed * 1000 + ep * 100 + step).random() < budget
            elif condition == 'periodic':
                should_query = (step % period == 0)
            elif condition == 'kl_threshold':
                should_query = last_kl > threshold
            elif condition == 'recon_threshold':
                recon_med = np.median(recon_hist) if recon_hist else 0.0
                should_query = last_recon > recon_med
            else:
                raise ValueError(condition)

            obs_new, real_rew, done = env.step(a_np)
            total_reward += real_rew

            if should_query:
                n_queries += 1
                obs_t = torch.tensor(obs_new, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    embed = model.encoder(obs_t)
                    h, z, prior_l, post_l = model.rssm.observe_step(h, z, action, embed)
                    last_kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                    dec = model.decoder(torch.cat([h, z], dim=-1))
                    last_recon = torch.nn.functional.mse_loss(dec, obs_t, reduction='sum').item()
                recon_hist.append(last_recon)
            else:
                with torch.no_grad():
                    h, z, _ = model.rssm.imagine_step(h, z, action)

            obs = obs_new
            step += 1

        rewards.append(total_reward)
        query_rates.append(n_queries / max(step, 1))

    return float(np.mean(rewards)), float(np.std(rewards)), float(np.mean(query_rates))


def evaluate_task_seed(task, seed_idx):
    actor, model, ck = load_actor_and_model(task, seed_idx)
    domain = TASK_SEEDS[task]['domain']
    cfg = ck['model_cfg']
    tr = dict(np.load(TASK_SEEDS[task]['training_states']))
    clf, scaler, v, kl_median_true = fit_probe_and_direction(task, tr)
    proj = tr['h'] @ v

    seed_results = {}
    for budget in QUERY_BUDGETS:
        # calibrate confusion-direction threshold to this budget on the training distribution
        conf_thresh = float(np.percentile(proj, 100 * (1 - budget)))

        budget_results = {}
        for condition in CONDITIONS:
            kl_arg = conf_thresh if condition in ('confusion',) else kl_median_true
            mean_r, std_r, qrate = run_episodes(actor, model, cfg, domain, condition, budget,
                                                 v, kl_arg, N_EPISODES_EVAL, base_seed=seed_idx * 10000)
            budget_results[condition] = dict(mean_reward=mean_r, std_reward=std_r, query_rate=qrate)
            print(f"  [{task}/seed{seed_idx}] budget={budget:.1f} {condition:<16} "
                  f"reward={mean_r:+.4f}±{std_r:.4f} qrate={qrate:.3f}", flush=True)

        # empirical null: N_NULL random directions in place of v, same calibrated threshold logic
        rng_null = np.random.default_rng(9000 + seed_idx)
        null_rewards = []
        for _ in range(N_NULL):
            vr = random_matched_direction(rng_null, ck['state_dim'])
            proj_r = tr['h'] @ vr
            thresh_r = float(np.percentile(proj_r, 100 * (1 - budget)))
            mean_r, _, _ = run_episodes(actor, model, cfg, domain, 'confusion', budget,
                                         vr, thresh_r, N_EPISODES_EVAL, base_seed=seed_idx * 10000 + 5000)
            null_rewards.append(mean_r)
        null_mean, null_std = float(np.mean(null_rewards)), float(np.std(null_rewards))
        conf_reward = budget_results['confusion']['mean_reward']
        z = (conf_reward - null_mean) / (null_std + 1e-12)
        budget_results['null_mean'] = null_mean
        budget_results['null_std'] = null_std
        budget_results['null_z'] = float(z)
        print(f"  [{task}/seed{seed_idx}] budget={budget:.1f} NULL(n={N_NULL})     "
              f"reward={null_mean:+.4f}±{null_std:.4f} z(confusion vs null)={z:+.2f}", flush=True)

        seed_results[str(budget)] = budget_results
    return seed_results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    for task, spec in TASK_SEEDS.items():
        all_results[task] = {}
        for seed_idx in range(spec['n_seeds']):
            print(f"\n{'='*78}\n{task.upper()} seed{seed_idx}\n{'='*78}", flush=True)
            all_results[task][f'seed{seed_idx}'] = evaluate_task_seed(task, seed_idx)

            out_path = os.path.join(OUT_DIR, 'gated_perception_results.json')
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=float)

    print(f"\nWrote {os.path.join(OUT_DIR, 'gated_perception_results.json')}")


if __name__ == '__main__':
    main()
