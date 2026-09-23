"""
Task 12 -- Active Data Acquisition, world-model-accuracy checkpoint.

Implements outputs/deliverables/task_12_design_decisions.md. Run one task per
call: run_task('cartpole' | 'reacher' | 'pendulum'). Each (comparator) cell's
result is written to OUT_DIR as soon as it finishes, so a session that times
out keeps everything completed so far and a rerun skips finished cells.

Comparators, all under the same N_BUDGET real environment interactions:
  pointwise triggers (decide per step: spend a real step, or continue free in
  imagination) -- ct, kl, recon, ema_recon, ensemble_disagreement, random,
  plus the 50-direction empirical null (random_direction_i);
  full policy -- sekar (Plan2Explore: ensemble-disagreement reward, imagined
  actor-critic, deployed for real every step).

Spend rule (identical for every pointwise trigger): fire when the trigger's
value is at or above the rolling (1 - SPEND_RATE) quantile of its own recent
values. A fixed threshold calibrated up front is not usable: a trigger whose
distribution drifts during acquisition (the online-trained ensemble's
disagreement shrinks as its members converge) can fall below it permanently,
and since imagined steps are free the loop then never spends its budget -- this
exact hang was hit in local smoke testing. The rolling quantile makes every
trigger spend at the same rate, so they differ only in WHICH states they spend
on, which is the thing being compared.
"""

import os
import json
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.model.actor_critic import Actor, Critic, state_repr, lambda_return
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from src.training.replay_buffer import EpisodeReplayBuffer

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get('TASK12_OUT', '/kaggle/working/task12_results')

# ── fixed before any run (design doc) ────────────────────────────────────────
SEED = 2024 + 13000
N_BUDGET = 10_000
SPEND_RATE = 0.5
QUANTILE_WINDOW = 500
MIN_WINDOW = 50
MAX_LOOP_FACTOR = 50
FINETUNE_STEPS = 2_000
FINETUNE_BATCH = 8
FINETUNE_SEQ_LEN = 16
N_EVAL_TRAJ = 30
EVAL_HORIZON = 10
EVAL_START = 10
N_NULL = 50
N_ENSEMBLE = 3
SEKAR_TRAIN_LOOPS = 30
SEKAR_STEPS_PER_LOOP = 20
SEKAR_HORIZON = 15
SEKAR_BATCH = 16

TASKS = {
    'cartpole': dict(env_cls='cartpole', domain='cartpole', task='swingup', gamma_ct=0.95),
    'reacher':  dict(env_cls='dmc', domain='reacher', task='easy', gamma_ct=0.70),
    'pendulum': dict(env_cls='dmc', domain='pendulum', task='swingup', gamma_ct=0.90),
}
POINTWISE = ['ct', 'kl', 'recon', 'ema_recon', 'ensemble_disagreement', 'random']

# seed offsets -- acquisition/calibration/training seeds never overlap the
# evaluation seeds SEED .. SEED + N_EVAL_TRAJ - 1
EVAL_SEED = SEED
CALIB_SEED = SEED + 10_000
ACQ_SEED = SEED + 20_000
SEKAR_SEED = SEED + 30_000
FT_SEED = SEED + 40_000


def set_scale(**kw):
    """Override constants for local smoke tests only."""
    g = globals()
    for k, v in kw.items():
        assert k in g, k
        g[k] = v


def make_env(spec, seed):
    if spec['env_cls'] == 'cartpole':
        return CartpoleEnv(task=spec['task'], noisy=False, seed=seed)
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=False, seed=seed)


def load_model(task):
    ck = torch.load(os.path.join(ROOT, 'checkpoints', f'{task}_world_model.pt'), map_location='cpu')
    cfg = ck['cfg']
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], cfg)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def zero_state(cfg):
    return (torch.zeros(1, cfg['rssm_deter']),
            torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes']))


# ── ensemble (Plan2Explore: one-step latent predictors, disagreement = variance) ──

def build_ensemble(model, seed):
    torch.manual_seed(seed)
    d_in = model.rssm.deter + model.rssm.z_dim + model.act_dim
    return nn.ModuleList([
        nn.Sequential(nn.Linear(d_in, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                      nn.Linear(256, model.rssm.deter))
        for _ in range(N_ENSEMBLE)])


def ensemble_disagreement(ensemble, h, z, a):
    inp = torch.cat([h, z, a], dim=-1)
    preds = torch.stack([m(inp) for m in ensemble], dim=0)
    return preds.var(dim=0).mean(dim=-1)


def train_ensemble(ensemble, opt, h, z, a, h_next):
    # enable_grad: callers run inside no_grad for world-model inference
    with torch.enable_grad():
        inp = torch.cat([h, z, a], dim=-1).detach()
        loss = sum(F.mse_loss(m(inp), h_next.detach()) for m in ensemble)
        opt.zero_grad()
        loss.backward()
        opt.step()


# ── pointwise acquisition ────────────────────────────────────────────────────

class TriggerState:
    """Running KL median, C_t, and EMA(recon), maintained on every step."""

    def __init__(self, gamma_ct, ema_alpha=0.1):
        self.gamma_ct = gamma_ct
        self.ema_alpha = ema_alpha
        self.kl_hist = []
        self.ct = 0.0
        self.ema = None

    def update(self, kl, recon):
        self.kl_hist.append(kl)
        med = float(np.median(self.kl_hist[-2000:]))
        self.ct = self.gamma_ct * self.ct + (1.0 if kl > med else 0.0)
        self.ema = recon if self.ema is None else self.ema_alpha * recon + (1 - self.ema_alpha) * self.ema


def trigger_value(trigger, st, kl, recon, h, z, a_t, ensemble, direction, rng):
    if trigger == 'ct':
        return st.ct
    if trigger == 'kl':
        return kl
    if trigger == 'recon':
        return recon
    if trigger == 'ema_recon':
        return st.ema
    if trigger == 'ensemble_disagreement':
        return float(ensemble_disagreement(ensemble, h, z, a_t).item())
    if trigger == 'random':
        return float(rng.uniform())
    if trigger == 'random_direction':
        return float(h.squeeze(0).numpy() @ direction)
    raise ValueError(trigger)


@torch.no_grad()
def acquire_pointwise(model, spec, cfg, trigger, direction=None):
    rng = np.random.default_rng(ACQ_SEED)
    tie_rng = np.random.default_rng(ACQ_SEED + 1)
    ensemble = opt = None
    if trigger == 'ensemble_disagreement':
        ensemble = build_ensemble(model, SEED + 1)
        opt = torch.optim.Adam(ensemble.parameters(), lr=1e-3)

    n_env = 0
    env = make_env(spec, seed=ACQ_SEED + n_env)
    obs = env.reset()
    h, z = zero_state(cfg)
    st = TriggerState(spec['gamma_ct'])
    window = deque(maxlen=QUANTILE_WINDOW)
    ep_obs, ep_act, episodes = [obs.copy()], [], []
    spent, iters = 0, 0
    act_dim = model.act_dim

    while spent < N_BUDGET:
        iters += 1
        if iters > MAX_LOOP_FACTOR * N_BUDGET:
            raise RuntimeError(f'[{trigger}] spent only {spent}/{N_BUDGET} after {iters} steps')
        a = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
        a_t = torch.tensor(a).unsqueeze(0)
        emb = model.encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        h_new, z_new, prior_l, post_l = model.rssm.observe_step(h, z, a_t, emb)
        kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
        dec = model.decoder(torch.cat([h_new, z_new], dim=-1)).squeeze(0).numpy()
        recon = float(np.sum((dec - obs) ** 2))
        st.update(kl, recon)

        val = trigger_value(trigger, st, kl, recon, h_new, z_new, a_t, ensemble, direction, tie_rng)
        # relative jitter breaks exact ties (C_t plateaus) at random -- the
        # tie-breaking failure mode found in Task 8
        val += 1e-9 * (abs(val) + 1e-12) * tie_rng.standard_normal()
        window.append(val)
        thr = np.quantile(window, 1 - SPEND_RATE) if len(window) >= MIN_WINDOW else -np.inf

        if val >= thr:
            next_obs, _, done = env.step(a)
            ep_act.append(a.copy())
            ep_obs.append(next_obs.copy())
            spent += 1
            if ensemble is not None:
                train_ensemble(ensemble, opt, h, z, a_t, h_new)
            h, z, obs = h_new, z_new, next_obs
            if done or len(ep_act) >= cfg['episode_max_steps']:
                episodes.append((ep_obs[:-1], ep_act))
                n_env += 1
                env = make_env(spec, seed=ACQ_SEED + n_env)
                obs = env.reset()
                h, z = zero_state(cfg)
                ep_obs, ep_act = [obs.copy()], []
        else:
            h, z, _ = model.rssm.imagine_step(h_new, z_new, a_t)
    if len(ep_act) >= 2:
        episodes.append((ep_obs[:-1], ep_act))
    return episodes, dict(real_steps=spent, loop_steps=iters)


# ── Sekar et al. (Plan2Explore) ──────────────────────────────────────────────

def run_sekar(model, spec, cfg):
    """Train the exploration policy (its real training steps count against the
    budget), then deploy it for real for the remainder of the budget."""
    torch.manual_seed(SEKAR_SEED)
    rng = np.random.default_rng(SEKAR_SEED)
    for p in model.parameters():
        p.requires_grad_(False)
    deter, z_dim, act_dim = cfg['rssm_deter'], cfg['rssm_stoch'] * cfg['rssm_classes'], model.act_dim
    ensemble = build_ensemble(model, SEKAR_SEED)
    ens_opt = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    actor, critic = Actor(deter + z_dim, act_dim), Critic(deter + z_dim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=8e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=8e-5)

    n_env = 0
    env = make_env(spec, seed=SEKAR_SEED + n_env)
    obs = env.reset()
    h, z = zero_state(cfg)
    ep_obs, ep_act, episodes = [obs.copy()], [], []
    starts = deque(maxlen=500)
    spent = 0

    def real_step(explore):
        nonlocal obs, h, z, n_env, env, ep_obs, ep_act, spent
        with torch.no_grad():
            if explore:
                a = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
            else:
                a = actor.act(state_repr(h, z)).squeeze(0).numpy().astype(np.float32)
            a_t = torch.tensor(a).unsqueeze(0)
            emb = model.encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            h_new, z_new, _, _ = model.rssm.observe_step(h, z, a_t, emb)
        train_ensemble(ensemble, ens_opt, h, z, a_t, h_new)
        next_obs, _, done = env.step(a)
        ep_act.append(a.copy())
        ep_obs.append(next_obs.copy())
        spent += 1
        h, z, obs = h_new, z_new, next_obs
        starts.append((h.detach(), z.detach()))
        if done or len(ep_act) >= cfg['episode_max_steps']:
            episodes.append((ep_obs[:-1], ep_act))
            n_env += 1
            env = make_env(spec, seed=SEKAR_SEED + n_env)
            obs = env.reset()
            h, z = zero_state(cfg)
            ep_obs, ep_act = [obs.copy()], []

    for loop in range(SEKAR_TRAIN_LOOPS):
        for _ in range(SEKAR_STEPS_PER_LOOP):
            real_step(explore=(loop == 0))
        idx = rng.choice(len(starts), size=min(SEKAR_BATCH, len(starts)), replace=False)
        h_i = torch.cat([starts[i][0] for i in idx])
        z_i = torch.cat([starts[i][1] for i in idx])
        feats, rewards, entropies = [], [], []
        for _ in range(SEKAR_HORIZON):
            s = state_repr(h_i, z_i)
            a_i, logp = actor.sample(s)
            h_i, z_i, _ = model.rssm.imagine_step(h_i, z_i, a_i)
            rewards.append(ensemble_disagreement(ensemble, h_i, z_i, a_i))
            feats.append(state_repr(h_i, z_i))
            entropies.append(-logp)
        feats = torch.stack(feats)
        values = critic(feats)
        returns = lambda_return(torch.stack(rewards), torch.cat([values, values[-1:]], 0))
        # dynamics backprop through the imagined trajectory (Dreamer / Plan2Explore)
        actor_loss = -(returns.mean() + 1e-3 * torch.stack(entropies).mean())
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        critic_loss = F.mse_loss(critic(feats.detach()), returns.detach())
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

    train_steps = spent
    while spent < N_BUDGET:
        real_step(explore=False)
    if len(ep_act) >= 2:
        episodes.append((ep_obs[:-1], ep_act))
    for p in model.parameters():
        p.requires_grad_(True)
    return episodes, dict(real_steps=spent, policy_training_real_steps=train_steps)


# ── fine-tune + evaluate (identical for every comparator) ───────────────────

def finetune(model, episodes, cfg):
    torch.manual_seed(FT_SEED)
    import random
    random.seed(FT_SEED)
    buf = EpisodeReplayBuffer(capacity=max(500, len(episodes)))
    for o, a in episodes:
        buf.add_episode(o, a)
    if len(buf) < FINETUNE_SEQ_LEN * FINETUNE_BATCH:
        raise RuntimeError(f'only {len(buf)} acquired steps -- too few to fine-tune')
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    model.train()
    for _ in range(FINETUNE_STEPS):
        o, a = buf.sample(FINETUNE_BATCH, FINETUNE_SEQ_LEN)
        loss, _, _ = model.compute_loss(o, a, kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, spec, cfg):
    """Mean E^state (10-step imagined-vs-real decode error) on held-out
    uniform-random trajectories -- same construction as Phase 1."""
    torch.manual_seed(EVAL_SEED)
    out = []
    for ep in range(N_EVAL_TRAJ):
        env = make_env(spec, seed=EVAL_SEED + ep)
        rng = np.random.default_rng(EVAL_SEED + ep)
        obs = env.reset()
        h, z = zero_state(cfg)
        obs_l, act_l, hz = [obs.copy()], [], []
        for _ in range(cfg['episode_max_steps']):
            a = rng.uniform(-1, 1, (model.act_dim,)).astype(np.float32)
            a_t = torch.tensor(a).unsqueeze(0)
            emb = model.encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            h, z, _, _ = model.rssm.observe_step(h, z, a_t, emb)
            hz.append((h, z))
            act_l.append(a)
            obs, _, done = env.step(a)
            obs_l.append(obs.copy())
            if done:
                break
        for t0 in range(EVAL_START, len(act_l) - EVAL_HORIZON - 1, 25):
            h_i, z_i = hz[t0]
            d = []
            for k in range(1, EVAL_HORIZON + 1):
                h_i, z_i, _ = model.rssm.imagine_step(h_i, z_i, torch.tensor(act_l[t0 + k]).unsqueeze(0))
                dec = model.decoder(torch.cat([h_i, z_i], dim=-1)).squeeze(0).numpy()
                d.append(float(np.linalg.norm(dec - obs_l[t0 + k + 1])))
            out.append(np.mean(d))
    return float(np.mean(out)), len(out)


# ── orchestration ────────────────────────────────────────────────────────────

def _cell(task, name, fn):
    path = os.path.join(OUT_DIR, task, f'{name}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    t0 = time.time()
    res = fn()
    res['minutes'] = (time.time() - t0) / 60
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  [{task}] {name:>24s}  E^state={res['e_state']:.4f}  real={res.get('real_steps', 0):>6}  "
          f"({res['minutes']:.1f} min)", flush=True)
    return res


def run_task(task, include_null=True, include_sekar=True):
    spec = TASKS[task]
    cfg = XS_CONFIG.copy()
    cfg.update(torch.load(os.path.join(ROOT, 'checkpoints', f'{task}_world_model.pt'),
                          map_location='cpu')['cfg'])
    null_dirs = np.load(os.path.join(ROOT, 'canonical_null', 'random_directions_50_dim256.npz'))['directions']

    def pointwise(trigger, direction=None):
        def fn():
            model = load_model(task)
            eps, info = acquire_pointwise(model, spec, cfg, trigger, direction)
            model = finetune(model, eps, cfg)
            e, n = evaluate(model, spec, cfg)
            return dict(e_state=e, n_eval=n, **info)
        return fn

    def baseline():
        e, n = evaluate(load_model(task), spec, cfg)
        return dict(e_state=e, n_eval=n)

    def sekar():
        model = load_model(task)
        eps, info = run_sekar(model, spec, cfg)
        model = finetune(model, eps, cfg)
        e, n = evaluate(model, spec, cfg)
        return dict(e_state=e, n_eval=n, **info)

    res = {'baseline': _cell(task, 'baseline', baseline)}
    for trig in POINTWISE:
        res[trig] = _cell(task, trig, pointwise(trig))
    if include_sekar:
        res['sekar'] = _cell(task, 'sekar', sekar)
    if include_null:
        res['null'] = [_cell(task, f'null_{i:02d}', pointwise('random_direction', null_dirs[i]))
                       for i in range(N_NULL)]
    summary = summarize(task, res)
    with open(os.path.join(OUT_DIR, task, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def summarize(task, res):
    """Ship rule (design doc, fixed before any run): C_t has the lowest E^state
    of all pointwise triggers, AND C_t's improvement over random acquisition
    exceeds the 50-direction null's improvement distribution at z > 2."""
    e = {k: res[k]['e_state'] for k in ['baseline'] + POINTWISE + (['sekar'] if 'sekar' in res else [])}
    out = dict(task=task, e_state=e)
    others = [t for t in POINTWISE if t != 'ct']
    out['beats_baselines'] = all(e['ct'] < e[t] for t in others)
    if 'null' in res:
        null_imp = np.array([e['random'] - r['e_state'] for r in res['null']])
        ct_imp = e['random'] - e['ct']
        z = (ct_imp - null_imp.mean()) / (null_imp.std() + 1e-12)
        out['null'] = dict(ct_improvement=ct_imp, null_mean=float(null_imp.mean()),
                           null_std=float(null_imp.std()), z=float(z))
        out['clears_null'] = bool(z > 2)
        out['ship'] = bool(out['beats_baselines'] and out['clears_null'])
    print(f"\n[{task}] " + '  '.join(f'{k}={v:.4f}' for k, v in e.items()), flush=True)
    if 'null' in out:
        print(f"[{task}] null z={out['null']['z']:+.2f}  beats_baselines={out['beats_baselines']}  "
              f"-> {'SHIP' if out['ship'] else 'NO-SHIP'}", flush=True)
    return out
