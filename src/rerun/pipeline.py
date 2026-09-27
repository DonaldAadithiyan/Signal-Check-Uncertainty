"""
Corrected analysis pipeline for the September sprint.

Every sprint analysis imports this module; nothing downstream regenerates its
own sites. Fixes relative to the pre-sprint scripts:

  W1  action convention. Online state logging pairs obs_t with a_{t-1} (zeros at
      t=0), exactly as WorldModel.forward_sequence/compute_loss do in training
      (src/model/world_model.py:51,66,105,136). The old loops paired obs_t with
      a_t. `log_states(..., convention='buggy')` reproduces the old pairing and
      exists only for the Step-2 impact gate.
  W2  C_t thresholds come from the calibration split and are a required
      argument (`compute_ct(kl, thresh, ...)` has no default).
  W5  splits are by episode, with fixed seed ranges (SPLITS). No step-level
      splits anywhere.
  W6  every stochastic draw uses an explicit generator: env seed = episode
      seed, action RNG = default_rng(episode seed), RSSM categorical samples
      from a torch.Generator seeded per (split, purpose). Results therefore
      do not depend on call order or on the global torch RNG.

Because the data-collection policy is uniform-random and independent of the
model, episodes (obs, actions, rewards) are generated once per task and
shared by every model of that task; the model is then run over them in batch.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault('MUJOCO_GL', 'disable')

from src.config import XS_CONFIG            # noqa: E402
from src.model.world_model import WorldModel  # noqa: E402

# ─── fixed protocol constants ────────────────────────────────────────────────

EP_LEN = 500                 # XS_CONFIG['episode_max_steps']; all dm_control episodes are longer
MIN_T = 12                   # burn-in: sites t < MIN_T are excluded from every analysis
HORIZONS = (1, 5, 10, 20)
K_HEADLINE = 10
GAMMA_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)
EMA_ALPHA_GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
N_BOOT = 1000
BOOT_SEED = 7

# episode seed ranges (env seed == action-RNG seed == episode seed)
SPLITS = {
    'probe':       range(10_000, 10_100),   # 100 episodes: probe fit, PCA, SAE
    'calibration': range(20_000, 20_050),   # 50 episodes: thresholds, gamma, alpha
    'evaluation':  range(30_000, 30_100),   # 100 episodes: every reported number
    'gate':        range(40_000, 40_050),   # 50 episodes: Step-2 W1 impact gate only
}
# torch sampling seeds, one per split x purpose (never reused)
TORCH_SEEDS = {'probe': 1_001, 'calibration': 2_001, 'evaluation': 3_001, 'gate': 4_001}
IMAGINE_SEED_OFFSET = 50_000

TASKS = {
    'cartpole': dict(domain='cartpole', task='swingup',
                     primary='outputs/checkpoints/world_model.pt',
                     seeds_glob='outputs/multiseed/seed_*/world_model.pt'),
    'reacher':  dict(domain='reacher', task='easy',
                     primary='outputs/second_env/reacher_easy_world_model.pt',
                     seeds_glob='outputs/multiseed_env/reacher_seed*/model.pt'),
    'pendulum': dict(domain='pendulum', task='swingup',
                     primary='outputs/third_env/pendulum_swingup_world_model.pt',
                     seeds_glob='outputs/multiseed_env/pendulum_seed*/model.pt'),
}


def seed_everything(seed):
    """Top-of-entry-point seeding (W6). Analyses below also use explicit
    generators, so this is a belt-and-braces guard for library code."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── models ──────────────────────────────────────────────────────────────────

def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(XS_CONFIG)
    cfg.update(ck.get('cfg', {}))
    obs_dim = ck.get('obs_dim', cfg.get('obs_dim'))
    act_dim = ck.get('act_dim', cfg.get('act_dim'))
    m = WorldModel(obs_dim, act_dim, cfg)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def model_paths(task):
    """[(label, path)] for the primary model and every multiseed model found."""
    import glob
    spec = TASKS[task]
    out = [('primary', spec['primary'])]
    for p in sorted(glob.glob(spec['seeds_glob'])):
        out.append((os.path.basename(os.path.dirname(p)), p))
    return out


# ─── episodes (model-independent) ────────────────────────────────────────────

def make_env(task, seed):
    spec = TASKS[task]
    if spec['domain'] == 'cartpole':
        from src.env.wrapper import CartpoleEnv
        return CartpoleEnv(task=spec['task'], noisy=False, seed=seed)
    from src.env.dmc_wrapper import DMCEnv
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=False, seed=seed)


def generate_episodes(task, seeds, ep_len=EP_LEN):
    """Random-policy episodes. Returns obs (E,T,obs_dim), act (E,T,act_dim),
    rew (E,T), seeds (E,). act[:, t] is taken AFTER obs[:, t]; rew[:, t] is
    the reward for that transition."""
    obs_all, act_all, rew_all = [], [], []
    for s in seeds:
        env = make_env(task, s)
        rng = np.random.default_rng(s)
        o = env.reset()
        obs_l, act_l, rew_l = [], [], []
        for _ in range(ep_len):
            a = rng.uniform(-1, 1, (env.act_dim,)).astype(np.float32)
            obs_l.append(o)
            act_l.append(a)
            o, r, done = env.step(a)
            rew_l.append(r)
            if done:
                break
        if len(obs_l) != ep_len:
            raise RuntimeError(f'{task} seed {s}: episode ended at {len(obs_l)} < {ep_len}')
        obs_all.append(np.stack(obs_l)); act_all.append(np.stack(act_l)); rew_all.append(rew_l)
    return dict(obs=np.stack(obs_all).astype(np.float32), act=np.stack(act_all).astype(np.float32),
                rew=np.asarray(rew_all, np.float32), seeds=np.asarray(list(seeds), np.int64))


def episodes_cached(task, split, cache_dir='outputs/sprint/episodes'):
    path = os.path.join(cache_dir, f'{task}_{split}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(cache_dir, exist_ok=True)
    ep = generate_episodes(task, SPLITS[split])
    np.savez(path, **ep)
    return ep


# ─── RSSM stepping with explicit generators ──────────────────────────────────

def _sample_onehot(rssm, logits, gen):
    """Hard categorical sample (the forward value of RSSM._straight_through_sample),
    drawn from `gen` instead of the global RNG. Returns (one-hot, indices)."""
    B = logits.shape[0]
    probs = torch.softmax(logits.view(B, rssm.stoch, rssm.classes), dim=-1)
    idx = torch.multinomial(probs.reshape(-1, rssm.classes), 1, generator=gen).view(B, rssm.stoch)
    return F.one_hot(idx, rssm.classes).float().view(B, rssm.z_dim), idx


def _gru(rssm, h, z, a):
    return rssm.gru(torch.cat([z, a], dim=-1), h)


def prev_actions(act):
    """a_{t-1} for every t, zeros at t=0 -- the training convention (W1)."""
    prev = np.zeros_like(act)
    prev[:, 1:] = act[:, :-1]
    return prev


@torch.no_grad()
def log_states(model, ep, torch_seed, convention='correct', batch=128):
    """Run the frozen model over episodes, filtering with real observations.

    convention='correct': h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]), posterior uses obs_t.
    convention='buggy'  : feeds a_t instead of a_{t-1} (the pre-sprint logging).

    Returns h (E,T,D) float32, z_idx (E,T,stoch) uint8, kl (E,T), rec (E,T).
    """
    if convention == 'correct':
        feed = prev_actions(ep['act'])
    elif convention == 'buggy':
        feed = ep['act']
    else:
        raise ValueError(convention)
    rssm = model.rssm
    gen = torch.Generator().manual_seed(int(torch_seed))
    E, T = ep['act'].shape[:2]
    H = np.zeros((E, T, rssm.deter), np.float32)
    Z = np.zeros((E, T, rssm.stoch), np.uint8)
    KL = np.zeros((E, T), np.float32)
    REC = np.zeros((E, T), np.float32)
    for b0 in range(0, E, batch):
        sl = slice(b0, min(E, b0 + batch))
        obs = torch.from_numpy(ep['obs'][sl])
        a_in = torch.from_numpy(feed[sl])
        B = obs.shape[0]
        h = torch.zeros(B, rssm.deter)
        z = torch.zeros(B, rssm.z_dim)
        emb_all = model.encoder(obs)
        for t in range(T):
            h = _gru(rssm, h, z, a_in[:, t])
            prior_l = rssm.prior_net(h)
            post_l = rssm.post_net(torch.cat([h, emb_all[:, t]], dim=-1))
            z, idx = _sample_onehot(rssm, post_l, gen)
            dec = model.decoder(torch.cat([h, z], dim=-1))
            H[sl, t] = h.numpy()
            Z[sl, t] = idx.numpy().astype(np.uint8)
            KL[sl, t] = rssm.kl_divergence(post_l, prior_l, free_bits=0.0).numpy()
            REC[sl, t] = ((dec - obs[:, t]) ** 2).sum(-1).numpy()
    return dict(h=H, z_idx=Z, kl=KL, rec=REC)


def z_from_idx(rssm, z_idx):
    t = torch.from_numpy(z_idx.astype(np.int64))
    return F.one_hot(t, rssm.classes).float().reshape(*z_idx.shape[:-1], rssm.z_dim)


@torch.no_grad()
def imagine_errors(model, ep, h0, z0, t_idx, e_idx, K, gen, batch=4096):
    """Prior-only imagination from (h0, z0) at sites (e_idx, t_idx), with the real
    actions. Under the correct convention h_t has NOT consumed a_t, so step k
    feeds a_{t+k-1} (a_t exactly once at the handoff) and is compared with
    obs_{t+k}. Returns per-step L2 distance (N, K)."""
    rssm = model.rssm
    N = len(t_idx)
    out = np.zeros((N, K), np.float32)
    act = torch.from_numpy(ep['act'])
    obs = torch.from_numpy(ep['obs'])
    for b0 in range(0, N, batch):
        sl = slice(b0, min(N, b0 + batch))
        e = torch.as_tensor(e_idx[sl], dtype=torch.long)
        t = torch.as_tensor(t_idx[sl], dtype=torch.long)
        h = torch.as_tensor(h0[sl])
        z = torch.as_tensor(z0[sl])
        for k in range(1, K + 1):
            h = _gru(rssm, h, z, act[e, t + k - 1])
            z, _ = _sample_onehot(rssm, rssm.prior_net(h), gen)
            dec = model.decoder(torch.cat([h, z], dim=-1))
            out[sl, k - 1] = torch.linalg.vector_norm(dec - obs[e, t + k], dim=-1).numpy()
    return out


def estate_all_sites(model, ep, st, torch_seed, kmax=max(HORIZONS)):
    """E^state_K for every site t in [MIN_T, T-kmax-1]; NaN elsewhere.
    Returns dict K -> (E,T) array."""
    E, T = st['kl'].shape
    ts = np.arange(MIN_T, T - kmax - 1)
    e_idx, t_idx = np.meshgrid(np.arange(E), ts, indexing='ij')
    e_idx, t_idx = e_idx.ravel(), t_idx.ravel()
    h0 = torch.from_numpy(st['h'][e_idx, t_idx])
    z0 = z_from_idx(model.rssm, st['z_idx'][e_idx, t_idx])
    gen = torch.Generator().manual_seed(int(torch_seed) + IMAGINE_SEED_OFFSET)
    dist = imagine_errors(model, ep, h0, z0, t_idx, e_idx, kmax, gen)
    out = {}
    for K in HORIZONS:
        arr = np.full((E, T), np.nan, np.float32)
        arr[e_idx, t_idx] = dist[:, :K].mean(1)
        out[K] = arr
    return out


# ─── history statistics ──────────────────────────────────────────────────────

def discounted(x, gamma):
    """y_t = sum_{i>=0} gamma^i x_{t-i}, within each episode. x: (E,T)."""
    y = np.zeros_like(x, dtype=np.float64)
    run = np.zeros(x.shape[0])
    for t in range(x.shape[1]):
        run = gamma * run + x[:, t]
        y[:, t] = run
    return y


def compute_ct(kl, thresh, gamma, include_current=True):
    """C_t = sum_{i>=0} gamma^i 1[KL_{t-i} > thresh] (include_current=True), or
    C_t^past = sum_{i>=1} ... (include_current=False). `thresh` is REQUIRED and
    must come from the calibration split (W2). kl: (E,T), reset per episode."""
    flags = (kl > thresh).astype(np.float64)
    c = discounted(flags, gamma)
    return c if include_current else c - flags


def compute_ct_continuous(kl, mu, sd, gamma, include_current=True):
    """Discounted sum of calibration-standardised KL."""
    zk = (kl - mu) / sd
    c = discounted(zk, gamma)
    return c if include_current else c - zk


def ema(x, alpha):
    """EMA within each episode, initialised at the episode's first value."""
    y = np.zeros_like(x, dtype=np.float64)
    run = x[:, 0].astype(np.float64)
    for t in range(x.shape[1]):
        run = alpha * x[:, t] + (1 - alpha) * run
        y[:, t] = run
    return y


# ─── statistics ──────────────────────────────────────────────────────────────

def ols_r2(X, y):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, np.float64) for c in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / ss) if ss > 0 else float('nan')


def site_mask(shape, kmax=None):
    E, T = shape
    m = np.zeros(shape, bool)
    m[:, MIN_T:(T - kmax - 1) if kmax else T] = True
    return m


def traj_bootstrap(stat_fn, n_ep, n_boot=N_BOOT, seed=BOOT_SEED):
    """stat_fn(episode_index_array) -> float. Episodes resampled with replacement.
    Returns (lo, hi, boots)."""
    rng = np.random.default_rng(seed)
    boots = np.array([stat_fn(rng.integers(0, n_ep, n_ep)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), boots


# ─── probe & geometry ────────────────────────────────────────────────────────

def fit_probe(h, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(h)
    clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs', random_state=0)
    clf.fit(sc.transform(h), y)
    return clf, sc


def probe_scores(clf, sc, h):
    flat = h.reshape(-1, h.shape[-1])
    return clf.predict_proba(sc.transform(flat))[:, 1].reshape(h.shape[:-1])


def probe_direction_raw(clf, sc):
    v = clf.coef_[0] / sc.scale_
    return (v / np.linalg.norm(v)).astype(np.float32)


def projection_fraction(w, components):
    return float(((components @ w) ** 2).sum())


def geometry(h_fit, clf, sc, ks=(10, 50), n_random=1000, seed=11):
    """‖proj_U w‖² of the unit probe direction onto the top-k PCs, in standardised
    and raw space, with a random-unit-vector baseline (W3)."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    D = h_fit.shape[1]
    R = rng.standard_normal((n_random, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    spaces = {
        'standardized': (sc.transform(h_fit), clf.coef_[0] / np.linalg.norm(clf.coef_[0])),
        'raw': (h_fit - h_fit.mean(0), probe_direction_raw(clf, sc).astype(np.float64)),
    }
    out = {}
    for name, (X, w) in spaces.items():
        pca = PCA(n_components=max(ks), random_state=0).fit(X)
        for k in ks:
            U = pca.components_[:k]
            f = projection_fraction(w, U)
            fr = ((R @ U.T) ** 2).sum(1)
            out[f'{name}_top{k}'] = dict(
                frac=f, subspace_angle_deg=float(np.degrees(np.arccos(np.sqrt(min(f, 1.0))))),
                random_mean=float(fr.mean()), random_p5=float(np.percentile(fr, 5)),
                percentile_vs_random=float((fr <= f).mean() * 100),
                var_explained=float(pca.explained_variance_ratio_[:k].sum()))
    return out


# ─── entry-point helpers ─────────────────────────────────────────────────────

def get_model(task, path, smoke=False):
    """Load a frozen checkpoint. smoke=True builds a RANDOMLY INITIALISED model
    of the right shape so the code path can be exercised without checkpoints;
    smoke outputs are written under outputs/sprint/smoke/ and are never results."""
    if not smoke:
        return load_model(path)
    env = make_env(task, 0)
    torch.manual_seed(0)
    return WorldModel(env.obs_dim, env.act_dim, dict(XS_CONFIG)).eval()


def out_dir(smoke):
    d = 'outputs/sprint/smoke' if smoke else 'outputs/sprint'
    os.makedirs(d, exist_ok=True)
    return d


def update_results(path, keys, value):
    """Merge `value` into results.json at nested `keys` (list of str)."""
    import json
    res = {}
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
    node = res
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    with open(path, 'w') as f:
        json.dump(res, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
