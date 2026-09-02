#!/usr/bin/env python3.11
"""
Phase 1 — External behavioral validation (Reframed_Confusion_RSSM_Project, Section 6).
THE GATE-1 EXPERIMENT. Everything downstream (SAE mechanism, POMDP, theory, scaling)
is contingent on this passing.

Central question this answers (Gap A / Section 3):
    Is C_t encoding something behaviorally meaningful, or just a transformation of
    the model's own KL-related training signal?

Method: at held-out sites t in real trajectories, roll IMAGINATION forward K steps
using the REAL action sequence, decode the imagined latents to observations, and
compare against the REAL observations/rewards obtained by executing those same
actions in the environment. This target is computed entirely from decoded
observations and real env reward — NOT from KL — so it is a genuine external
criterion (H1).

Targets (computed independently of the KL threshold that defines C_t):
    E^state_{t,K}  = (1/K) sum_k || obs_imag_{t+k} - obs_real_{t+k} ||   (decoded space)
    E^reward_{t,K} = | sum_k gamma^k r_imag_{t+k} - sum_k gamma^k r_real_{t+k} |
                     using an EXACT task-specific reward reconstruction from decoded
                     obs (dm_control's own rewards.tolerance() applied to the relevant
                     decoded observation components — reacher: finger-to-target
                     distance; pendulum: pole verticality; cartpole: upright proxy,
                     matching Task H's existing convention since dm_control's cartpole
                     reward additionally depends on non-observed contact/velocity
                     terms this model doesn't decode separately).

Baselines compared (Section 6.2, all 5 named in the doc):
    KL_t, Recon_t, EMARecon_t, EnsembleDisagreement_t, C_t

Regression (Section 6.2):
    E_imag ~ beta0 + beta1*C_t + beta2*KL_t + beta3*Recon_t + beta4*EMARecon_t
             + beta5*EnsembleDisagreement_t
Reports standalone AUROC/correlation per baseline AND incremental R^2 of C_t
after all four other baselines are already included (the actual answer to the
circularity objection — Section 6.2's closing point).

Also runs (Section 6.4, 6.5 — bundled here since they reuse the same trajectory
collection and imagined-vs-real machinery):
    6.4 Adversarial dissociation against Berger-et-al.-style reward-inflating drift
        (imagined reward high, real reward low, recon/KL not obviously spiking):
        is C_t elevated there, or does it miss this failure mode?
    6.5 Implicit behavioral sensitivity: does raw action magnitude / deviation from
        typical action scale already correlate with C_t with no intervention?
        (No distributional/ensembled critic or explicit planner exists in this XS
        pipeline, so policy-entropy and critic-disagreement sub-checks from 6.5 are
        not applicable here and are explicitly reported as N/A rather than skipped
        silently.)

Runs on the 3 EXISTING frozen models (cartpole, reacher, pendulum). No retraining.
CPU only. Every result reported per-task with bootstrap 95% CIs — no pooling-only
summary (Section 16).
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from dm_control.utils import rewards as dmc_rewards

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct, bootstrap_ci

# ─── config ───────────────────────────────────────────────────────────────────

N_TRAJ = 100
HORIZONS = [1, 5, 10, 20]
MAX_HORIZON = max(HORIZONS)
MIN_T = 12                 # minimum warmup before a site is eligible (needs KL history)
EMA_ALPHA = 0.10            # tuned on train split only, see fit_ema_alpha()
GAMMA_REWARD_DISCOUNT = 0.99  # discount used inside E^reward_{t,K} itself
N_ENSEMBLE = 3              # matches paper's headline ensemble size
N_BOOT = 1000
SEED = 2024
OUT_DIR = 'outputs/phase1_external_validation'
FIG_DIR = 'outputs/figures'

ENVS = {
    'cartpole': dict(
        env_cls='cartpole', domain='cartpole', task='swingup',
        checkpoint='outputs/checkpoints/world_model.pt',
        training_states='outputs/data/training_states.npz',
        ensemble=[f'outputs/checkpoints/ensemble_seed{i}.pt' for i in range(N_ENSEMBLE)],
        gamma_ct=0.95,
    ),
    'reacher': dict(
        env_cls='dmc', domain='reacher', task='easy',
        checkpoint='outputs/second_env/reacher_easy_world_model.pt',
        training_states='outputs/second_env/reacher_easy_training_states.npz',
        ensemble=None,  # no per-task ensemble on disk; ensemble baseline reported N/A
        gamma_ct=0.70,
    ),
    'pendulum': dict(
        env_cls='dmc', domain='pendulum', task='swingup',
        checkpoint='outputs/third_env/pendulum_swingup_world_model.pt',
        training_states='outputs/third_env/pendulum_swingup_training_states.npz',
        ensemble=None,
        gamma_ct=0.90,
    ),
}


# ─── model / env loading ──────────────────────────────────────────────────────

def load_model(ckpt_path):
    device = torch.device('cpu')
    ck = torch.load(ckpt_path, map_location=device)
    mcfg = ck['cfg']
    obs_dim = ck.get('obs_dim', mcfg.get('obs_dim'))
    act_dim = ck.get('act_dim', mcfg.get('act_dim'))
    m = WorldModel(obs_dim, act_dim, mcfg).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m, obs_dim, act_dim


def make_env(spec, seed):
    if spec['env_cls'] == 'cartpole':
        return CartpoleEnv(task=spec['task'], noisy=False, seed=seed)
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=False, seed=seed)


# ─── exact task-specific reward reconstruction from decoded obs ──────────────
# All three are EXACT reconstructions of the dm_control reward function from the
# relevant decoded observation components (verified against dm_control source),
# not approximations — except cartpole, which uses the same upright-reward PROXY
# already established in Task H, since dm_control cartpole's exact reward also
# depends on state not separately decoded here (documented, not hidden).

_REACHER_RADII = 0.06         # physics.named.model.geom_size[['target','finger'],0].sum()
_PENDULUM_COS_BOUND = 0.9902680687415704   # dm_control.suite.pendulum._COSINE_BOUND


def reward_from_decoded_obs(domain, decoded_obs):
    """decoded_obs: (..., obs_dim) real-valued decoded observation array.
    Returns reward proxy/exact value(s), same leading shape."""
    if domain == 'cartpole':
        cos_pole = decoded_obs[..., 1]
        return (cos_pole + 1.0) / 2.0
    elif domain == 'reacher':
        to_target = decoded_obs[..., 2:4]   # obs layout: position(2) + to_target(2) + velocity(2)
        dist = np.linalg.norm(to_target, axis=-1)
        return dmc_rewards.tolerance(dist, bounds=(0, _REACHER_RADII))
    elif domain == 'pendulum':
        orientation_zz = decoded_obs[..., 0]  # obs layout: orientation(2) + velocity(1)
        return dmc_rewards.tolerance(orientation_zz, bounds=(_PENDULUM_COS_BOUND, 1.0))
    else:
        raise ValueError(domain)


# ─── trajectory collection (real env, real reward, full obs/act/h/z/kl/recon) ─

def collect_trajectories(model, spec, n_traj, cfg, seed=SEED):
    device = next(model.parameters()).device
    trajs = []
    for ep in range(n_traj):
        env = make_env(spec, seed=seed + ep)
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        act_dim = model.act_dim
        obs_l, act_l, h_l, z_l, kl_l, recon_l, rew_l = [], [], [], [], [], [], []
        done, step = False, 0
        rng = np.random.default_rng(seed + ep)
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                emb = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, emb)
                kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                dec = model.decoder(torch.cat([h, z], dim=-1)).squeeze(0).cpu().numpy()
                recon = float(np.sum((dec - obs) ** 2))

                obs_l.append(obs.copy()); act_l.append(a.copy())
                h_l.append(h.squeeze(0).cpu().numpy().copy())
                z_l.append(post_l.squeeze(0).cpu().numpy().copy())
                kl_l.append(kl); recon_l.append(recon)

                obs, rew, done = env.step(a)
                rew_l.append(rew)
                step += 1
        trajs.append(dict(
            obs=np.array(obs_l, np.float32), act=np.array(act_l, np.float32),
            h=np.array(h_l, np.float32), z=np.array(z_l, np.float32),
            kl=np.array(kl_l, np.float32), recon=np.array(recon_l, np.float32),
            rew=np.array(rew_l, np.float32),
        ))
    return trajs


# ─── imagined-vs-real, in OBSERVATION space (the external target) ────────────

@torch.no_grad()
def imagined_vs_real_obs(model, traj, t, horizon, domain):
    """From site t, roll imagination forward `horizon` steps with the REAL action
    sequence, decode to obs at each step, and compare against the REAL observed
    obs/reward at the same steps. Returns per-k state distance and reward (imag,
    real) arrays, len = min(horizon, available).
    """
    device = next(model.parameters()).device
    T = len(traj['obs'])

    h_im = torch.tensor(traj['h'][t], dtype=torch.float32, device=device).unsqueeze(0)
    z_logits = torch.tensor(traj['z'][t], dtype=torch.float32, device=device).unsqueeze(0)
    z_im = model.rssm._straight_through_sample(z_logits)

    state_dist, imag_rew, real_rew = [], [], []
    for k in range(1, horizon + 1):
        kk = t + k
        if kk >= T:
            break
        a = torch.tensor(traj['act'][kk - 1], dtype=torch.float32, device=device).unsqueeze(0)
        h_im, z_im, _ = model.rssm.imagine_step(h_im, z_im, a)
        dec_imag = model.decoder(torch.cat([h_im, z_im], dim=-1)).squeeze(0).cpu().numpy()
        obs_real = traj['obs'][kk]

        state_dist.append(float(np.linalg.norm(dec_imag - obs_real)))
        imag_rew.append(float(reward_from_decoded_obs(domain, dec_imag)))
        real_rew.append(float(traj['rew'][kk - 1]))  # rew logged at collection step kk-1 == step producing obs[kk]

    return (np.array(state_dist, np.float32), np.array(imag_rew, np.float32),
            np.array(real_rew, np.float32))


def e_state(state_dist_full, K):
    return float(np.mean(state_dist_full[:K])) if len(state_dist_full) >= K else np.nan


def e_reward(imag_rew_full, real_rew_full, K, gamma=GAMMA_REWARD_DISCOUNT):
    if len(imag_rew_full) < K:
        return np.nan
    disc = gamma ** np.arange(1, K + 1)
    return float(abs(np.sum(disc * imag_rew_full[:K]) - np.sum(disc * real_rew_full[:K])))


def bootstrap_pearson_ci(x, y, n_boot=N_BOOT, seed=0):
    """Bootstrap CI for Pearson r(x,y), robust to degenerate resamples (e.g. a
    reward target that saturates to a near-constant value in some resamples,
    which makes pearsonr return nan for that draw — those draws are dropped
    rather than propagating nan into the reported CI)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    point = pearsonr(x, y)[0]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        r = pearsonr(x[idx], y[idx])[0]
        if np.isfinite(r):
            boots.append(r)
    if len(boots) < 20:
        return float(point), float('nan'), float('nan')
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ─── baselines ────────────────────────────────────────────────────────────────

def fit_ema_alpha(recon_train, kl_train, candidates=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """Pick alpha maximizing correlation of EMA(recon) with KL on TRAIN split only
    (Section 7.2: tune only on train/calibration, not final test)."""
    best_a, best_r = candidates[0], -1.0
    for a in candidates:
        ema = np.zeros_like(recon_train)
        run = recon_train[0]
        for i, x in enumerate(recon_train):
            run = a * x + (1 - a) * run
            ema[i] = run
        r = abs(pearsonr(ema, kl_train)[0])
        if r > best_r:
            best_r, best_a = r, a
    return best_a


def ema_series(recon, alpha):
    ema = np.zeros_like(recon)
    run = recon[0]
    for i, x in enumerate(recon):
        run = alpha * x + (1 - alpha) * run
        ema[i] = run
    return ema


@torch.no_grad()
def ensemble_disagreement_series(models, obs_arr, cfg):
    """Single-step ensemble disagreement at each obs (independent decode from
    zero state, matching the paper's existing ensemble_disagreement convention)."""
    device = next(models[0].parameters()).device
    preds = []
    for model in models:
        model.eval()
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=device)
        N = obs_t.shape[0]
        h0 = torch.zeros(N, cfg['rssm_deter'], device=device)
        z0 = torch.zeros(N, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        a0 = torch.zeros(N, model.act_dim, device=device)
        emb = model.encoder(obs_t)
        h_n, z_n, _, _ = model.rssm.observe_step(h0, z0, a0, emb)
        dec = model.decoder(torch.cat([h_n, z_n], dim=-1)).cpu().numpy()
        preds.append(dec)
    stacked = np.stack(preds, axis=0)
    return stacked.var(axis=0).mean(axis=-1)


# ─── main per-task pipeline ───────────────────────────────────────────────────

def run_task(task, spec, cfg):
    print(f"\n{'='*78}\n{task.upper()}\n{'='*78}")
    model, obs_dim, act_dim = load_model(spec['checkpoint'])

    tr = dict(np.load(spec['training_states']))
    kl_median = float(np.median(tr['kl']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])
    print(f"  EMA alpha tuned on train split: {ema_alpha}")

    ensemble_models = None
    if spec['ensemble'] and all(os.path.exists(p) for p in spec['ensemble']):
        ensemble_models = [load_model(p)[0] for p in spec['ensemble']]
        print(f"  loaded {len(ensemble_models)}-model ensemble")
    else:
        print(f"  no ensemble checkpoints for {task}; ensemble baseline reported N/A")

    # probe (Probe-A) fit fresh on this task's training states, exact original protocol
    y = binarise_by_median(tr['kl'])
    idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                       stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])

    print(f"  collecting {N_TRAJ} held-out trajectories (real env, real reward)...")
    trajs = collect_trajectories(model, spec, N_TRAJ, cfg)

    # attach C_t, probe score, EMA recon, ensemble disagreement to each trajectory
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64),
                                gamma=spec['gamma_ct'])
        trj['probe'] = clf.predict_proba(scaler.transform(trj['h']))[:, 1]
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)
        if ensemble_models is not None:
            trj['ens_dis'] = ensemble_disagreement_series(ensemble_models, trj['obs'], cfg)
        else:
            trj['ens_dis'] = np.full(len(trj['kl']), np.nan)

    # ── site enumeration ──
    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - MAX_HORIZON - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED)
    if len(sites) > 6000:
        sel = rng.choice(len(sites), 6000, replace=False)
        sites = [sites[i] for i in sel]
    print(f"  {len(sites)} evaluation sites (horizon up to {MAX_HORIZON})")

    # ── compute per-site: baselines at t, and E^state/E^reward at each K ──
    rows = {k: [] for k in ['ct', 'kl', 'recon', 'ema_recon', 'ens_dis', 'probe',
                             'action_mag']}
    e_state_by_K = {K: [] for K in HORIZONS}
    e_reward_by_K = {K: [] for K in HORIZONS}

    action_scale = np.std(np.concatenate([trj['act'] for trj in trajs]), axis=0).mean()

    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, imag_rew_full, real_rew_full = imagined_vs_real_obs(
            model, trj, t, MAX_HORIZON, spec['domain'])
        if len(state_dist_full) < MAX_HORIZON:
            continue
        for K in HORIZONS:
            e_state_by_K[K].append(e_state(state_dist_full, K))
            e_reward_by_K[K].append(e_reward(imag_rew_full, real_rew_full, K))
        rows['ct'].append(trj['ct'][t])
        rows['kl'].append(trj['kl'][t])
        rows['recon'].append(trj['recon'][t])
        rows['ema_recon'].append(trj['ema_recon'][t])
        rows['ens_dis'].append(trj['ens_dis'][t])
        rows['probe'].append(trj['probe'][t])
        rows['action_mag'].append(float(np.linalg.norm(trj['act'][t])) / (action_scale + 1e-8))

    for k in rows:
        rows[k] = np.array(rows[k], dtype=np.float64)
    for K in HORIZONS:
        e_state_by_K[K] = np.array(e_state_by_K[K], dtype=np.float64)
        e_reward_by_K[K] = np.array(e_reward_by_K[K], dtype=np.float64)

    n_sites = len(rows['ct'])
    print(f"  {n_sites} sites with full horizon available")

    have_ensemble = not np.all(np.isnan(rows['ens_dis']))

    # ── H1: standalone predictive validity of C_t per horizon ──
    h1 = {}
    for K in HORIZONS:
        target = e_state_by_K[K]
        r_state, p_state = pearsonr(rows['ct'], target)
        rs_state, ps_state = spearmanr(rows['ct'], target)
        _, ci_lo, ci_hi = bootstrap_pearson_ci(rows['ct'], target)
        r_ci = (r_state, ci_lo, ci_hi)
        target_r = e_reward_by_K[K]
        r_rew, p_rew = pearsonr(rows['ct'], target_r)
        _, ci_rew_lo, ci_rew_hi = bootstrap_pearson_ci(rows['ct'], target_r)
        r_rew_ci = (r_rew, ci_rew_lo, ci_rew_hi)

        # AUROC for high-vs-low future failure (top/bottom quartile of E^state)
        hi = target >= np.percentile(target, 75)
        lo = target <= np.percentile(target, 25)
        mask = hi | lo
        auc_state = roc_auc_score(hi[mask].astype(int), rows['ct'][mask]) if mask.sum() > 10 else np.nan

        h1[K] = dict(
            pearson_r_state=r_state, pearson_p_state=p_state,
            pearson_r_state_ci=r_ci, spearman_r_state=rs_state, spearman_p_state=ps_state,
            pearson_r_reward=r_rew, pearson_p_reward=p_rew, pearson_r_reward_ci=r_rew_ci,
            auroc_state_hilo=float(auc_state),
        )
        print(f"    K={K:>2}: r(C_t, E^state)={r_state:+.3f} (p={p_state:.2g}), "
              f"r(C_t, E^reward)={r_rew:+.3f} (p={p_rew:.2g}), "
              f"AUROC(hi/lo E^state)={auc_state:.3f}")

    # ── H2: baseline comparison + incremental regression, at the FIXED headline horizon (K=10 or max available) ──
    K_headline = 10 if 10 in HORIZONS else HORIZONS[-1]
    target = e_state_by_K[K_headline]
    baseline_names = ['ct', 'kl', 'recon', 'ema_recon'] + (['ens_dis'] if have_ensemble else [])

    standalone = {}
    for name in baseline_names:
        r, p = pearsonr(rows[name], target)
        standalone[name] = dict(pearson_r=r, pearson_p=p)

    # full regression with all baselines
    X_cols = ['ct', 'kl', 'recon', 'ema_recon'] + (['ens_dis'] if have_ensemble else [])
    X_full = np.column_stack([rows[c] for c in X_cols])
    Xs = StandardScaler().fit_transform(X_full)
    y_reg = target

    reg_full = LinearRegression().fit(Xs, y_reg)
    r2_full = reg_full.score(Xs, y_reg)

    # without C_t (controls only)
    ctrl_cols = [c for c in X_cols if c != 'ct']
    X_ctrl = np.column_stack([rows[c] for c in ctrl_cols])
    Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
    reg_ctrl = LinearRegression().fit(Xs_ctrl, y_reg)
    r2_ctrl = reg_ctrl.score(Xs_ctrl, y_reg)

    incremental_r2 = r2_full - r2_ctrl
    beta_ct = reg_full.coef_[X_cols.index('ct')]

    print(f"\n  H2 baseline comparison @ K={K_headline} (target=E^state):")
    for name, d in standalone.items():
        print(f"    standalone r({name}, E^state) = {d['pearson_r']:+.3f} (p={d['pearson_p']:.2g})")
    print(f"    full regression R^2 (all {len(X_cols)} predictors) = {r2_full:.4f}")
    print(f"    controls-only R^2 (excl. C_t)                    = {r2_ctrl:.4f}")
    print(f"    incremental R^2 from C_t                          = {incremental_r2:.4f}")
    print(f"    standardized beta_Ct in full regression           = {beta_ct:+.4f}")

    # ── H2 on reward target too ──
    target_rew = e_reward_by_K[K_headline]
    reg_full_r = LinearRegression().fit(Xs, target_rew)
    r2_full_r = reg_full_r.score(Xs, target_rew)
    reg_ctrl_r = LinearRegression().fit(Xs_ctrl, target_rew)
    r2_ctrl_r = reg_ctrl_r.score(Xs_ctrl, target_rew)
    incremental_r2_reward = r2_full_r - r2_ctrl_r
    beta_ct_reward = reg_full_r.coef_[X_cols.index('ct')]
    print(f"    [reward target] full R^2={r2_full_r:.4f}, ctrl R^2={r2_ctrl_r:.4f}, "
          f"incremental R^2 from C_t={incremental_r2_reward:.4f}, beta_Ct={beta_ct_reward:+.4f}")

    # ── 6.4 adversarial dissociation: Berger-style reward-inflating drift ──
    # drift state: imagined reward high, real reward low, recon/KL NOT obviously spiking
    drift_imag_rew, drift_real_rew, drift_recon, drift_kl, drift_ct = [], [], [], [], []
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, imag_rew_full, real_rew_full = imagined_vs_real_obs(
            model, trj, t, K_headline, spec['domain'])
        if len(imag_rew_full) < K_headline:
            continue
        drift_imag_rew.append(float(imag_rew_full[-1]))
        drift_real_rew.append(float(real_rew_full[-1]))
        drift_recon.append(float(trj['recon'][t]))
        drift_kl.append(float(trj['kl'][t]))
        drift_ct.append(float(trj['ct'][t]))
    drift_imag_rew = np.array(drift_imag_rew); drift_real_rew = np.array(drift_real_rew)
    drift_recon = np.array(drift_recon); drift_kl = np.array(drift_kl); drift_ct = np.array(drift_ct)

    reward_gap = drift_imag_rew - drift_real_rew
    is_drift = (reward_gap > np.percentile(reward_gap, 75)) & \
               (drift_recon <= np.percentile(drift_recon, 50)) & \
               (drift_kl <= np.percentile(drift_kl, 50))
    n_drift = int(is_drift.sum())
    if n_drift >= 10:
        ct_drift_mean = float(drift_ct[is_drift].mean())
        ct_other_mean = float(drift_ct[~is_drift].mean())
        ct_drift_ci = bootstrap_ci(drift_ct[is_drift], n_boot=N_BOOT)
        ct_other_ci = bootstrap_ci(drift_ct[~is_drift], n_boot=N_BOOT)
        catches_it = ct_drift_mean > ct_other_mean
    else:
        ct_drift_mean = ct_other_mean = float('nan')
        ct_drift_ci = ct_other_ci = (float('nan'),) * 3
        catches_it = None
    dissociation = dict(
        n_drift_states=n_drift, n_total=len(is_drift),
        ct_mean_in_drift=ct_drift_mean, ct_mean_elsewhere=ct_other_mean,
        ct_ci_in_drift=ct_drift_ci, ct_ci_elsewhere=ct_other_ci,
        catches_it=catches_it,
    )
    verdict_64 = ('N/A (too few drift states)' if n_drift < 10 else
                  ('CATCHES IT' if catches_it else 'MISSES IT'))
    print(f"\n  6.4 adversarial dissociation: {n_drift} reward-inflating drift states found. "
          f"C_t there={ct_drift_mean:.3f} vs elsewhere={ct_other_mean:.3f} -> {verdict_64}")

    # ── 6.5 implicit behavioral sensitivity: C_t vs action magnitude (no policy/critic exists) ──
    r_act, p_act = pearsonr(rows['ct'], rows['action_mag'])
    print(f"\n  6.5 implicit sensitivity: r(C_t, |action|/scale) = {r_act:+.3f} (p={p_act:.2g}). "
          f"[policy-entropy / critic-disagreement sub-checks: N/A — this XS pipeline has no "
          f"policy or critic, only a random-action data-collection procedure.]")

    result = dict(
        task=task, n_sites=n_sites, n_trajectories=N_TRAJ, ema_alpha=ema_alpha,
        gamma_ct=spec['gamma_ct'], have_ensemble=have_ensemble,
        h1_by_horizon=h1, K_headline=K_headline,
        h2_standalone=standalone,
        h2_r2_full=r2_full, h2_r2_ctrl=r2_ctrl, h2_incremental_r2_ct=incremental_r2,
        h2_beta_ct=float(beta_ct),
        h2_r2_full_reward=r2_full_r, h2_r2_ctrl_reward=r2_ctrl_r,
        h2_incremental_r2_ct_reward=incremental_r2_reward, h2_beta_ct_reward=float(beta_ct_reward),
        dissociation_6_4=dissociation, verdict_6_4=verdict_64,
        sensitivity_6_5=dict(r_action_mag=r_act, p_action_mag=p_act),
    )
    return result, (rows, e_state_by_K, e_reward_by_K)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    all_results = {}
    raw_for_fig = {}
    for task, spec in ENVS.items():
        res, raw = run_task(task, spec, cfg)
        all_results[task] = res
        raw_for_fig[task] = raw

    # ── Gate 1 verdict ──
    print(f"\n{'='*78}\nGATE 1 VERDICT\n{'='*78}")
    gate1_pass_per_task = {}
    for task, res in all_results.items():
        K = res['K_headline']
        r = res['h1_by_horizon'][K]['pearson_r_state']
        p = res['h1_by_horizon'][K]['pearson_p_state']
        inc = res['h2_incremental_r2_ct']
        passed = (p < 0.05 and abs(r) > 0.05) or inc > 0.005
        gate1_pass_per_task[task] = dict(passed=bool(passed), r=r, p=p, incremental_r2=inc)
        print(f"  {task}: r(C_t,E^state)@K={K}={r:+.3f} (p={p:.2g}), "
              f"incremental R^2 over 4 baselines={inc:+.4f} -> "
              f"{'PASS' if passed else 'FAIL'}")

    n_pass = sum(1 for v in gate1_pass_per_task.values() if v['passed'])
    overall = ('GATE 1 PASSES' if n_pass >= 2 else 'GATE 1 DOES NOT PASS')
    print(f"\n  {overall} ({n_pass}/{len(gate1_pass_per_task)} tasks show C_t predicting "
          f"external imagination failure beyond the 4 cheap baselines).")
    if n_pass < 2:
        print("  Per Section 19: STOP and reconsider the interpretation of \"confusion\" "
              "before proceeding to SAE/POMDP/scaling phases.")

    all_results['_gate1_verdict'] = dict(per_task=gate1_pass_per_task, overall=overall)

    out_path = os.path.join(OUT_DIR, 'phase1_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    # ── figure: E^state vs C_t scatter + horizon-decay curve, per task ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, task in enumerate(ENVS.keys()):
        rows, e_state_by_K, e_reward_by_K = raw_for_fig[task]
        K = all_results[task]['K_headline']
        ax = axes[0, i]
        ax.scatter(rows['ct'], e_state_by_K[K], s=5, alpha=0.2, c='steelblue', rasterized=True)
        r = all_results[task]['h1_by_horizon'][K]['pearson_r_state']
        ax.set_title(f'{task}: C_t vs E^state_K={K}\nr={r:+.3f}')
        ax.set_xlabel('C_t'); ax.set_ylabel(f'E^state (K={K})')
        ax.grid(alpha=0.3)

        ax2 = axes[1, i]
        rs = [all_results[task]['h1_by_horizon'][k]['pearson_r_state'] for k in HORIZONS]
        ax2.plot(HORIZONS, rs, 'o-')
        ax2.axhline(0, color='gray', lw=0.5)
        ax2.set_xlabel('horizon K'); ax2.set_ylabel('r(C_t, E^state_K)')
        ax2.set_title(f'{task}: predictive decay over horizon')
        ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'phase1_external_validation.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {fig_path}")


if __name__ == '__main__':
    main()
