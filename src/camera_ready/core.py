"""
Camera-ready (ICAART 2027 #388) corrected logging, built on src/rerun/pipeline.py.

Additions over the sprint pipeline:
  * episodes also carry the true environment state (qpos, qvel, and for reacher
    the target position) at the time each obs_t was emitted;
  * logging also saves prior and posterior entropy (sum over the 32 categoricals);
    the prior at step t is computed from h_t alone, i.e. BEFORE o_t is seen;
  * partial-observation (POMDP) cartpole models are supported: their encoder
    sees obs[..., :obs_dim_in] while the decoder reconstructs the full obs;
  * imagination returns per-step L2 error AND the imagined reward at step K=10
    (paper Sec. 10.3 reward-inflation analysis), using the paper's own reward
    reconstruction from decoded observations.

Action convention (W1): h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]), a_{-1} = 0. The
imagination handoff from a site t feeds a_t exactly once (step k feeds a_{t+k-1}).
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.rerun import pipeline as P
from src.rerun.pipeline import (SPLITS, TORCH_SEEDS, NOISE_STD, NOISE_SEED, MIN_T,
                                HORIZONS, IMAGINE_SEED_OFFSET, EP_LEN)

ROOT = 'outputs/camera_ready'
EP_DIR = os.path.join(ROOT, 'episodes')
ST_DIR = os.environ.get('CR_STATE_DIR', os.path.join(ROOT, 'states'))
# CR_CONVENTION=buggy reproduces the old a_t pairing (control only; never a result)
CONVENTION = os.environ.get('CR_CONVENTION', 'correct')
KMAX = max(HORIZONS)

# ─── model registry ──────────────────────────────────────────────────────────

POMDP_SEEDS = [4242, 1717, 2929, 5151, 8383]


def pomdp_path(cond, seed):
    if seed == 4242:
        return f'outputs/phase4_pomdp/{cond}_world_model.pt'
    return f'outputs/phase4_pomdp/seed_sweep/seed_{seed}/{cond}_world_model.pt'


def registry():
    """{task: [(label, path, kind)]}; kind in {'wm', 'pomdp'}."""
    reg = {}
    for task in P.TASKS:
        reg[task] = [(lab, p, 'wm') for lab, p in P.model_paths(task)]
    reg['cartpole_pomdp'] = [(f'{c}_{s}', pomdp_path(c, s), 'pomdp')
                             for s in POMDP_SEEDS for c in ('full', 'partial')]
    return reg


def env_task(task):
    return 'cartpole' if task == 'cartpole_pomdp' else task


def load_any(path, kind):
    if kind == 'wm':
        m = P.load_model(path)
        m.obs_dim_in = None
        return m
    from src.model.pomdp_world_model import PomdpWorldModel
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = PomdpWorldModel(ck['obs_dim_in'], ck['obs_dim_out'], ck['act_dim'], ck['cfg'])
    m.load_state_dict(ck['model_state'])
    m.eval()
    m.obs_dim_in = int(ck['obs_dim_in'])
    return m


def enc_input(model, obs):
    """Observation fed to the encoder (POMDP partial: position dims only)."""
    d = getattr(model, 'obs_dim_in', None)
    return obs if d is None else obs[..., :d]


# ─── episodes with true state ────────────────────────────────────────────────

def _physics_state(env, task):
    ph = env._env.physics
    st = [np.asarray(ph.data.qpos, np.float64).ravel(), np.asarray(ph.data.qvel, np.float64).ravel()]
    if task == 'reacher':
        st.append(np.asarray(ph.named.model.geom_pos['target'][:2], np.float64))
    return np.concatenate(st).astype(np.float32)


def generate_episodes(task, seeds, ep_len=EP_LEN):
    """Identical action/env seeding to pipeline.generate_episodes; additionally
    records the true physics state at the time each obs_t is emitted."""
    obs_all, act_all, rew_all, st_all = [], [], [], []
    for s in seeds:
        env = P.make_env(task, s)
        rng = np.random.default_rng(s)
        o = env.reset()
        obs_l, act_l, rew_l, st_l = [], [], [], []
        for _ in range(ep_len):
            a = rng.uniform(-1, 1, (env.act_dim,)).astype(np.float32)
            obs_l.append(o)
            st_l.append(_physics_state(env, task))
            act_l.append(a)
            o, r, done = env.step(a)
            rew_l.append(r)
            if done:
                break
        if len(obs_l) != ep_len:
            raise RuntimeError(f'{task} seed {s}: episode ended at {len(obs_l)} < {ep_len}')
        obs_all.append(np.stack(obs_l)); act_all.append(np.stack(act_l))
        rew_all.append(rew_l); st_all.append(np.stack(st_l))
    return dict(obs=np.stack(obs_all).astype(np.float32), act=np.stack(act_all).astype(np.float32),
                rew=np.asarray(rew_all, np.float32), true_state=np.stack(st_all),
                seeds=np.asarray(list(seeds), np.int64))


def episodes(task, split):
    task = env_task(task)
    if split == 'evaluation_noisy':
        ep = dict(episodes(task, 'evaluation'))
        rng = np.random.default_rng(NOISE_SEED)
        ep['obs_clean'] = ep['obs']
        ep['obs'] = (ep['obs'] + rng.standard_normal(ep['obs'].shape) * NOISE_STD).astype(np.float32)
        return ep
    path = os.path.join(EP_DIR, f'{task}_{split}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(EP_DIR, exist_ok=True)
    ep = generate_episodes(task, SPLITS[split])
    np.savez(path, **ep)
    return ep


# ─── RSSM helpers ────────────────────────────────────────────────────────────

def cat_entropy(rssm, logits):
    B = logits.shape[0]
    lp = torch.log_softmax(logits.view(B, rssm.stoch, rssm.classes), -1)
    return -(lp.exp() * lp).sum(-1).sum(-1)


@torch.no_grad()
def log_states_full(model, ep, torch_seed, batch=128):
    """Corrected-convention filtering over episodes. Returns h, z_idx, kl, rec,
    ent_prior, ent_post (all (E,T[,..]))."""
    rssm = model.rssm
    if CONVENTION == 'correct':
        feed = P.prev_actions(ep['act'])
    elif CONVENTION == 'buggy':
        feed = ep['act']
    else:
        raise ValueError(CONVENTION)
    gen = torch.Generator().manual_seed(int(torch_seed))
    E, T = ep['act'].shape[:2]
    out = dict(h=np.zeros((E, T, rssm.deter), np.float32), z_idx=np.zeros((E, T, rssm.stoch), np.uint8),
               kl=np.zeros((E, T), np.float32), rec=np.zeros((E, T), np.float32),
               ent_prior=np.zeros((E, T), np.float32), ent_post=np.zeros((E, T), np.float32))
    for b0 in range(0, E, batch):
        sl = slice(b0, min(E, b0 + batch))
        obs = torch.from_numpy(ep['obs'][sl])
        a_in = torch.from_numpy(feed[sl])
        B = obs.shape[0]
        h = torch.zeros(B, rssm.deter)
        z = torch.zeros(B, rssm.z_dim)
        emb_all = model.encoder(enc_input(model, obs))
        for t in range(T):
            h = P._gru(rssm, h, z, a_in[:, t])
            prior_l = rssm.prior_net(h)
            post_l = rssm.post_net(torch.cat([h, emb_all[:, t]], dim=-1))
            z, idx = P._sample_onehot(rssm, post_l, gen)
            dec = model.decoder(torch.cat([h, z], dim=-1))
            out['h'][sl, t] = h.numpy()
            out['z_idx'][sl, t] = idx.numpy().astype(np.uint8)
            out['kl'][sl, t] = rssm.kl_divergence(post_l, prior_l, free_bits=0.0).numpy()
            out['rec'][sl, t] = ((dec - obs[:, t]) ** 2).sum(-1).numpy()
            out['ent_prior'][sl, t] = cat_entropy(rssm, prior_l).numpy()
            out['ent_post'][sl, t] = cat_entropy(rssm, post_l).numpy()
    return out


# paper Sec. 10.3 / Phase 1 reward reconstruction from decoded observations
_REACHER_RADII = 0.06
_PENDULUM_COS_BOUND = 0.9902680687415704


def reward_from_obs(task, o):
    from dm_control.utils import rewards as R
    if task in ('cartpole', 'cartpole_pomdp'):
        return (o[..., 1] + 1.0) / 2.0
    if task == 'reacher':
        return R.tolerance(np.linalg.norm(o[..., 2:4], axis=-1), bounds=(0, _REACHER_RADII))
    if task == 'pendulum':
        return R.tolerance(o[..., 0], bounds=(_PENDULUM_COS_BOUND, 1.0))
    # no reward reconstruction for other tasks (only Sec. 10.3, which covers cartpole/reacher)
    return np.full(o.shape[:-1], np.nan)


@torch.no_grad()
def imagine_from(model, ep_obs, ep_act, h0, z0, e_idx, t_idx, K, gen, batch=4096, keep_last_dec=False,
                 target_obs=None):
    """Prior-only imagination from (h0, z0) at sites; step k feeds a_{t+k-1} and is
    compared with obs_{t+k}. Returns per-step L2 (N,K) and optionally decoded obs at
    step K."""
    rssm = model.rssm
    tgt = ep_obs if target_obs is None else target_obs
    N = len(t_idx)
    dist = np.zeros((N, K), np.float32)
    last = np.zeros((N, tgt.shape[-1]), np.float32) if keep_last_dec else None
    act = torch.from_numpy(ep_act)
    obs = torch.from_numpy(tgt)
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch))
        e = torch.as_tensor(e_idx[sl], dtype=torch.long)
        t = torch.as_tensor(t_idx[sl], dtype=torch.long)
        h = torch.as_tensor(h0[sl], dtype=torch.float32)
        z = torch.as_tensor(z0[sl], dtype=torch.float32)
        for k in range(1, K + 1):
            h = P._gru(rssm, h, z, act[e, t + k - 1])
            z, _ = P._sample_onehot(rssm, rssm.prior_net(h), gen)
            dec = model.decoder(torch.cat([h, z], dim=-1))
            dist[sl, k - 1] = torch.linalg.vector_norm(dec - obs[e, t + k], dim=-1).numpy()
        if keep_last_dec:
            last[sl] = dec.numpy()
    return dist, last


def estate_all_sites(model, ep, st, torch_seed, task):
    """E_{t,K} for every site t in [MIN_T, T-KMAX-1] (NaN elsewhere) and the
    imagined reward at step 10 (same rollout, K=20 > 10 so recomputed at 10)."""
    E, T = st['kl'].shape
    ts = np.arange(MIN_T, T - KMAX - 1)
    e_idx, t_idx = np.meshgrid(np.arange(E), ts, indexing='ij')
    e_idx, t_idx = e_idx.ravel(), t_idx.ravel()
    h0 = st['h'][e_idx, t_idx]
    z0 = P.z_from_idx(model.rssm, st['z_idx'][e_idx, t_idx]).numpy()
    gen = torch.Generator().manual_seed(int(torch_seed) + IMAGINE_SEED_OFFSET)
    tgt = ep.get('obs_clean', ep['obs'])
    dist, _ = imagine_from(model, ep['obs'], ep['act'], h0, z0, e_idx, t_idx, KMAX, gen, target_obs=tgt)
    out = {}
    for K in HORIZONS:
        arr = np.full((E, T), np.nan, np.float32)
        arr[e_idx, t_idx] = dist[:, :K].mean(1)
        out[f'estate_{K}'] = arr
    # imagined reward at k=10 (separate seeded rollout of length 10)
    gen10 = torch.Generator().manual_seed(int(torch_seed) + IMAGINE_SEED_OFFSET + 1)
    _, dec10 = imagine_from(model, ep['obs'], ep['act'], h0, z0, e_idx, t_idx, 10, gen10,
                            keep_last_dec=True, target_obs=tgt)
    r_im = np.full((E, T), np.nan, np.float32)
    r_im[e_idx, t_idx] = reward_from_obs(task, dec10)
    r_real = np.full((E, T), np.nan, np.float32)
    r_real[e_idx, t_idx] = ep['rew'][e_idx, t_idx + 10 - 1]   # reward of the transition producing obs_{t+10}
    out['imag_rew_10'] = r_im
    out['real_rew_10'] = r_real
    return out


def ensemble_disagreement(models, obs):
    """Paper Table 4 cartpole baseline: single-step decode from zero state by each
    ensemble member, variance across members, mean over obs dims. obs (E,T,d)."""
    with torch.no_grad():
        o = torch.from_numpy(obs.reshape(-1, obs.shape[-1]))
        preds = []
        for m in models:
            N = o.shape[0]
            h0 = torch.zeros(N, m.rssm.deter)
            z0 = torch.zeros(N, m.rssm.z_dim)
            a0 = torch.zeros(N, m.act_dim)
            emb = m.encoder(o)
            h_n, z_n, _, _ = m.rssm.observe_step(h0, z0, a0, emb)
            preds.append(m.decoder(torch.cat([h_n, z_n], -1)).numpy())
    return np.stack(preds).var(0).mean(-1).reshape(obs.shape[:2])


def state_path(task, label, split):
    return os.path.join(ST_DIR, f'{task}_{label}_{split}.npz')


def load_states(task, label, split):
    f = state_path(task, label, split)
    if not os.path.exists(f):
        raise FileNotFoundError(f'missing states: {f} (run run_cr_A_log.py first)')
    return dict(np.load(f))


def assert_finite(d, where):
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == 'f' and k not in (
                'estate_1', 'estate_5', 'estate_10', 'estate_20', 'imag_rew_10', 'real_rew_10'):
            if not np.all(np.isfinite(v)):
                raise FloatingPointError(f'non-finite values in {k} ({where})')


# ─── probe-fitting set (CR_PROBE) ────────────────────────────────────────────
# 'paper'  : the paper's probe -- logistic regression on 60% of the model's own
#            online training log (stratified by 1[KL > training-log median],
#            random_state=0), v = probe_direction (unit, raw h space). Applied to
#            corrected states. Isolates the effect of the logging fix.
# 'frozen' : probe fit on frozen-model probe-fit episodes (handoff protocol).
PROBE_MODE = os.environ.get('CR_PROBE', 'frozen')


def training_log_path(task, label):
    if task == 'cartpole_pomdp':
        cond, seed = label.split('_')
        if seed == '4242':
            return f'outputs/phase4_pomdp/{cond}_training_states.npz'
        return f'outputs/phase4_pomdp/seed_sweep/seed_{seed}/{cond}_training_states.npz'
    if label == 'primary':
        return {'cartpole': 'outputs/data/training_states.npz',
                'reacher': 'outputs/second_env/reacher_easy_training_states.npz',
                'pendulum': 'outputs/third_env/pendulum_swingup_training_states.npz'}[task]
    if task == 'cartpole':
        return f'outputs/multiseed/{label}/training_states.npz'
    return f'outputs/multiseed_env/{label}/states.npz'


def paper_probe(task, label):
    """Returns (clf, scaler, v_unit, h_fit) exactly as the paper's scripts fit it."""
    from sklearn.model_selection import train_test_split
    from src.probe.linear_probe import binarise_by_median, train_probe
    from src.probe.intervention import probe_direction
    tr = np.load(training_log_path(task, label))
    h, kl = tr['h'], tr['kl']
    y = binarise_by_median(kl)
    idx, _ = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h[idx], y[idx])
    v = probe_direction(clf, sc).astype(np.float64)
    return clf, sc, v / np.linalg.norm(v), h[idx]
