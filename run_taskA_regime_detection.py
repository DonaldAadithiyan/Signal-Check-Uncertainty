#!/usr/bin/env python3.11
"""
Task A — Regime-Change Detection.

Question: when the environment's dynamics change in a way the frozen world
model was never trained on, does the endogenous predictive-difficulty
representation s_t = v^T h_t detect that change earlier than cheap sequential
detectors, at the same false-alarm rate?

NOT a rerun of Task 10 (state repair). Task 10 tested whether a trigger could
improve a repair operation; the repair mechanism (not detection) turned out
to be the bottleneck (h's transition is observation-blind by construction).
This is pure detection: real-observation filtering (observe_step) every
step, no repair, no intervention.

Preflight (Step A0, see outputs/deliverables/task_A_regime_detection_preflight.md):
confirmed Task 10's logs cannot be reused (open-loop imagination post-shift,
no raw h_t/KL_t trajectories saved). Fresh inference-only rollouts required.

Everything below is fixed per the handoff spec before any result is seen.
"""

import os
import json
import numpy as np
import torch

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from run_phase1_external_validation import fit_ema_alpha
from src.probe.linear_probe import train_probe
from src.probe.intervention import probe_direction

OUT_DIR = 'outputs/taskA_regime_detection'
FIG_DIR = 'outputs/figures'
CANONICAL_DIR_PATH = 'outputs/canonical_null/random_directions_50_dim256.npz'

SEED = 2024 + 20_000          # never used elsewhere in this project's seed namespace
T_SHIFT = 150
EPISODE_LEN = 250
BURNIN = 20
DETECTION_HORIZON = 20        # H: detection window post-shift
CENSOR_DELAY = 100

FAR_PRIMARY = 0.05
FAR_SECONDARY = [0.10, 0.20]
FAR_LIST = [FAR_PRIMARY] + FAR_SECONDARY

N_CALIB_NOSHIFT = 100
N_EVAL_SHIFT = 100
N_EVAL_NOSHIFT = 100
CALIB_SEED_BASE = 20000
EVAL_SHIFT_SEED_BASE = 30000
EVAL_NOSHIFT_SEED_BASE = 31000

N_NULL = 50
N_BOOT = 1000

TASKS = {
    'cartpole': dict(env_cls='cartpole', domain='cartpole', task='swingup',
                      base_checkpoint='outputs/checkpoints/world_model.pt',
                      base_training_states='outputs/data/training_states.npz',
                      gamma_ct=0.95, shift_param='gravity', shift_mult=2.0,
                      multiseed=[f'outputs/multiseed/seed_{i}/world_model.pt' for i in range(5)],
                      multiseed_states=[f'outputs/multiseed/seed_{i}/training_states.npz' for i in range(5)]),
    'reacher': dict(env_cls='dmc', domain='reacher', task='easy',
                     base_checkpoint='outputs/second_env/reacher_easy_world_model.pt',
                     base_training_states='outputs/second_env/reacher_easy_training_states.npz',
                     gamma_ct=0.70, shift_param='damping', shift_mult=8.0,
                     multiseed=['outputs/second_env/reacher_easy_world_model.pt'] +
                               [f'outputs/multiseed_env/reacher_seed{i}/model.pt' for i in (1, 2, 3)],
                     multiseed_states=['outputs/second_env/reacher_easy_training_states.npz'] +
                                       [f'outputs/multiseed_env/reacher_seed{i}/states.npz' for i in (1, 2, 3)]),
    'pendulum': dict(env_cls='dmc', domain='pendulum', task='swingup',
                      base_checkpoint='outputs/third_env/pendulum_swingup_world_model.pt',
                      base_training_states='outputs/third_env/pendulum_swingup_training_states.npz',
                      gamma_ct=0.90, shift_param='gravity', shift_mult=2.0,
                      multiseed=['outputs/third_env/pendulum_swingup_world_model.pt'] +
                                [f'outputs/multiseed_env/pendulum_seed{i}/model.pt' for i in (1, 2, 3)],
                      multiseed_states=['outputs/third_env/pendulum_swingup_training_states.npz'] +
                                        [f'outputs/multiseed_env/pendulum_seed{i}/states.npz' for i in (1, 2, 3)]),
}


def load_canonical_directions():
    d = dict(np.load(CANONICAL_DIR_PATH))
    directions = d['directions']
    assert directions.shape == (N_NULL, 256), directions.shape
    return directions


def load_model_generic(ckpt_path, cfg):
    device = torch.device('cpu')
    ck = torch.load(ckpt_path, map_location=device)
    mcfg = ck.get('cfg', cfg)
    obs_dim = ck.get('obs_dim', mcfg.get('obs_dim', cfg['obs_dim']))
    act_dim = ck.get('act_dim', mcfg.get('act_dim', cfg['act_dim']))
    m = WorldModel(obs_dim, act_dim, mcfg).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def make_env(spec, seed):
    if spec['env_cls'] == 'cartpole':
        return CartpoleEnv(task=spec['task'], noisy=False, seed=seed)
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=False, seed=seed)


def apply_shift(env, spec):
    physics = env._env.physics
    if spec['shift_param'] == 'gravity':
        physics.model.opt.gravity[2] = -9.81 * spec['shift_mult']
    elif spec['shift_param'] == 'damping':
        physics.model.dof_damping[:] = physics.model.dof_damping[:] * spec['shift_mult']
    else:
        raise ValueError(spec['shift_param'])


def fit_v_direction(training_states_path):
    """Fit the frozen probe direction v for this model, per project convention
    (probe never persisted to disk -- refit each time from that model's own
    training_states.npz, median-KL binarization, logistic C=1 lbfgs standardized,
    identical settings to every prior probe in this project)."""
    tr = dict(np.load(training_states_path))
    h = tr['h']
    kl = tr['kl']
    y = (kl > np.median(kl)).astype(np.int32)
    clf, scaler = train_probe(h, y, C=1.0)
    v = probe_direction(clf, scaler)   # unit vector, raw h-space
    return v


@torch.no_grad()
def rollout_episode(model, spec, cfg, seed, ema_alpha, kl_median, shift, v, directions):
    """Filters real observations every step (observe_step only, never
    imagine_step). Applies the shift at t=T_SHIFT if shift=True. Returns
    per-step KL, Recon, EMARecon, s_t = v^T h_t, and the 50 null projections,
    all length EPISODE_LEN."""
    device = next(model.parameters()).device
    env = make_env(spec, seed=seed)
    obs = env.reset()
    h = torch.zeros(1, cfg['rssm_deter'], device=device)
    z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
    act_dim = model.act_dim
    rng = np.random.default_rng(seed)

    kl_l, recon_l, ema_l, s_l = [], [], [], []
    null_l = [[] for _ in range(N_NULL)]
    ema_val = None
    shift_applied = False

    for step in range(EPISODE_LEN):
        a = rng.uniform(-1, 1, (act_dim,)).astype(np.float32)
        a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)

        if shift and step == T_SHIFT and not shift_applied:
            apply_shift(env, spec)
            shift_applied = True

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        emb = model.encoder(obs_t)
        h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, emb)
        kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
        dec = model.decoder(torch.cat([h, z], dim=-1)).squeeze(0).cpu().numpy()

        next_obs, rew, done = env.step(a)
        recon = float(np.sum((dec - next_obs) ** 2))
        ema_val = recon if ema_val is None else (ema_alpha * recon + (1 - ema_alpha) * ema_val)

        h_np = h.squeeze(0).cpu().numpy()
        s_t = float(h_np @ v)

        kl_l.append(kl); recon_l.append(recon); ema_l.append(ema_val); s_l.append(s_t)
        for j in range(N_NULL):
            null_l[j].append(float(h_np @ directions[j]))

        obs = next_obs
        if done:
            break

    n = len(kl_l)
    return dict(
        kl=np.array(kl_l), recon=np.array(recon_l), ema=np.array(ema_l),
        s=np.array(s_l), null=np.array(null_l)[:, :n] if len(null_l[0]) == n else np.array([nl for nl in null_l]),
        n_steps=n,
    )


# ─── detectors ──────────────────────────────────────────────────────────────

def cusum_series(x, k):
    s = np.zeros_like(x)
    running = 0.0
    for i, xi in enumerate(x):
        running = max(0.0, running + xi - k)
        s[i] = running
    return s


def page_hinkley_series(x, delta):
    s = np.zeros_like(x)
    running = 0.0
    m = x[0] if len(x) else 0.0
    cum_mean = 0.0
    for i, xi in enumerate(x):
        cum_mean += (xi - cum_mean) / (i + 1)
        running = max(0.0, running + (xi - cum_mean - delta))
        s[i] = running
    return s


def closed_form_ct(kl_series, gamma, kl_median):
    ct = np.zeros_like(kl_series)
    state = 0.0
    for i, k in enumerate(kl_series):
        state = gamma * state + (1.0 if k > kl_median else 0.0)
        ct[i] = state
    return ct


def orient_sign(calib_series_list):
    """Orient a null direction's sign so it rises after warmup on calibration
    (no-shift) data -- scale is irrelevant since every detector gets its own
    calibrated threshold, but sign matters for a 'greater than threshold'
    alarm rule. Uses the mean late-vs-early trend across calib episodes."""
    trend = 0.0
    for s in calib_series_list:
        s = s[BURNIN:]
        if len(s) < 2:
            continue
        trend += (s[len(s) // 2:].mean() - s[:len(s) // 2].mean())
    return 1.0 if trend >= 0 else -1.0


# ─── calibration + evaluation ──────────────────────────────────────────────

def calibrate_threshold(detector_calib_series, far):
    """FAR = per-episode false-alarm probability over steps [BURNIN, end) on
    calibration no-shift episodes. theta is the smallest threshold such that
    the fraction of calib episodes with any alarm equals (as closely as a
    grid search allows) `far`."""
    all_vals = np.concatenate([s[BURNIN:] for s in detector_calib_series])
    # search over percentiles of the pooled calibration distribution
    lo, hi = 0.0, 100.0
    candidates = np.percentile(all_vals, np.linspace(50, 100, 2001))
    best_theta, best_gap = candidates[-1], 1e9
    for theta in candidates:
        far_hat = np.mean([np.any(s[BURNIN:] > theta) for s in detector_calib_series])
        gap = abs(far_hat - far)
        if gap < best_gap:
            best_gap = gap
            best_theta = theta
    return float(best_theta)


def alarm_time(series, theta):
    above = series[BURNIN:] > theta
    if not np.any(above):
        return None
    return int(np.argmax(above)) + BURNIN


def evaluate_detector(name, calib_series, eval_shift_series, eval_noshift_series, far_list):
    out = {'thresholds': {}, 'far': {}}
    for far in far_list:
        theta = calibrate_threshold(calib_series, far)
        out['thresholds'][far] = theta

        # realized FAR on evaluation no-shift episodes
        alarms_noshift = [alarm_time(s, theta) for s in eval_noshift_series]
        realized_far = float(np.mean([a is not None for a in alarms_noshift]))

        # shift episodes
        alarm_times = [alarm_time(s, theta) for s in eval_shift_series]
        pre_shift_false_alarms = sum(1 for a in alarm_times if a is not None and a < T_SHIFT)
        detections = [a for a in alarm_times if a is not None and a >= T_SHIFT]
        detect_within_h = sum(1 for a in detections if a <= T_SHIFT + DETECTION_HORIZON)
        n_shift_eps = len(eval_shift_series)
        detect_prob_h = float(detect_within_h / n_shift_eps)

        delays = []
        for a in alarm_times:
            if a is None or a < T_SHIFT:
                delays.append(CENSOR_DELAY)  # undetected or pre-shift alarm: censored
            else:
                delays.append(min(a - T_SHIFT, CENSOR_DELAY))
        delays = np.array(delays, dtype=np.float64)

        out['far'][far] = dict(
            theta=theta, realized_far=realized_far,
            pre_shift_false_alarms=pre_shift_false_alarms,
            n_shift_episodes=n_shift_eps,
            detect_prob_within_H=detect_prob_h,
            censored_mean_delay=float(delays.mean()),
            delays_per_episode=delays.tolist(),
        )
    return out


def bootstrap_delay_diff(delays_a, delays_b, n_boot=N_BOOT, seed=0):
    """Paired bootstrap over episodes on (delays_a - delays_b); returns
    (mean_diff, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(delays_a); b = np.asarray(delays_b)
    n = len(a)
    diffs = a - b
    boot = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(diffs.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def preflight_validity_check(eval_shift_series_kl, far_results_by_detector):
    """Hard stop check: raw KL must show a clear post-shift rise, AND at
    least one detector must beat chance detection probability at 5% FAR."""
    pre = np.concatenate([s[BURNIN:T_SHIFT] for s in eval_shift_series_kl])
    post = np.concatenate([s[T_SHIFT:T_SHIFT + DETECTION_HORIZON] for s in eval_shift_series_kl])
    kl_rise = float(post.mean() - pre.mean())
    kl_rise_clear = kl_rise > 0 and post.mean() > pre.mean() + pre.std()

    any_beats_chance = False
    for name, res in far_results_by_detector.items():
        # chance detection probability within H at FAR f, under a Poisson-ish
        # arrival assumption, is approximated conservatively as f (a detector
        # firing at its calibrated FAR randomly in time would detect within a
        # H/  (EPISODE_LEN-T_SHIFT) fraction of shift episodes by chance --
        # we use the realized FAR itself as the conservative chance baseline).
        dp = res['far'][FAR_PRIMARY]['detect_prob_within_H']
        if dp > FAR_PRIMARY:
            any_beats_chance = True

    return dict(kl_rise=kl_rise, kl_rise_clear=bool(kl_rise_clear), any_detector_beats_chance=bool(any_beats_chance))


def run_task_on_model(task, spec, cfg, model, v, directions, ema_alpha, kl_median):
    print(f"    calibrating (no-shift, {N_CALIB_NOSHIFT} episodes)...")
    calib_runs = [rollout_episode(model, spec, cfg, CALIB_SEED_BASE + i, ema_alpha, kl_median,
                                   shift=False, v=v, directions=directions)
                  for i in range(N_CALIB_NOSHIFT)]

    print(f"    evaluating shift ({N_EVAL_SHIFT} episodes)...")
    eval_shift_runs = [rollout_episode(model, spec, cfg, EVAL_SHIFT_SEED_BASE + i, ema_alpha, kl_median,
                                        shift=True, v=v, directions=directions)
                        for i in range(N_EVAL_SHIFT)]

    print(f"    evaluating no-shift ({N_EVAL_NOSHIFT} episodes)...")
    eval_noshift_runs = [rollout_episode(model, spec, cfg, EVAL_NOSHIFT_SEED_BASE + i, ema_alpha, kl_median,
                                          shift=False, v=v, directions=directions)
                          for i in range(N_EVAL_NOSHIFT)]

    gamma = spec['gamma_ct']

    def get(runs, key):
        return [r[key] for r in runs]

    def get_null(runs, j):
        return [r['null'][j] for r in runs]

    detector_series = {}
    detector_series['s_t'] = dict(
        calib=get(calib_runs, 's'), eval_shift=get(eval_shift_runs, 's'), eval_noshift=get(eval_noshift_runs, 's'))
    detector_series['KL'] = dict(
        calib=get(calib_runs, 'kl'), eval_shift=get(eval_shift_runs, 'kl'), eval_noshift=get(eval_noshift_runs, 'kl'))
    detector_series['Recon'] = dict(
        calib=get(calib_runs, 'recon'), eval_shift=get(eval_shift_runs, 'recon'), eval_noshift=get(eval_noshift_runs, 'recon'))
    detector_series['EMARecon'] = dict(
        calib=get(calib_runs, 'ema'), eval_shift=get(eval_shift_runs, 'ema'), eval_noshift=get(eval_noshift_runs, 'ema'))

    kl_calib_pool = np.concatenate([s[BURNIN:] for s in detector_series['KL']['calib']])
    mu_kl, sig_kl = kl_calib_pool.mean(), kl_calib_pool.std()
    k_cusum = mu_kl + 0.5 * sig_kl
    delta_ph = 0.5 * sig_kl

    detector_series['CUSUM_KL'] = dict(
        calib=[cusum_series(s, k_cusum) for s in detector_series['KL']['calib']],
        eval_shift=[cusum_series(s, k_cusum) for s in detector_series['KL']['eval_shift']],
        eval_noshift=[cusum_series(s, k_cusum) for s in detector_series['KL']['eval_noshift']],
    )
    detector_series['PageHinkley_KL'] = dict(
        calib=[page_hinkley_series(s, delta_ph) for s in detector_series['KL']['calib']],
        eval_shift=[page_hinkley_series(s, delta_ph) for s in detector_series['KL']['eval_shift']],
        eval_noshift=[page_hinkley_series(s, delta_ph) for s in detector_series['KL']['eval_noshift']],
    )

    # secondary: closed-form C_t (reported, never used to decide shipping)
    detector_series['C_t'] = dict(
        calib=[closed_form_ct(s, gamma, kl_median) for s in detector_series['KL']['calib']],
        eval_shift=[closed_form_ct(s, gamma, kl_median) for s in detector_series['KL']['eval_shift']],
        eval_noshift=[closed_form_ct(s, gamma, kl_median) for s in detector_series['KL']['eval_noshift']],
    )

    results = {}
    for name, ds in detector_series.items():
        results[name] = evaluate_detector(name, ds['calib'], ds['eval_shift'], ds['eval_noshift'], FAR_LIST)

    # null: 50 random directions, signed on calibration data
    null_results = []
    for j in range(N_NULL):
        calib_j = get_null(calib_runs, j)
        sign = orient_sign(calib_j)
        calib_j = [sign * s for s in calib_j]
        eval_shift_j = [sign * s for s in get_null(eval_shift_runs, j)]
        eval_noshift_j = [sign * s for s in get_null(eval_noshift_runs, j)]
        null_results.append(evaluate_detector(f'null_{j}', calib_j, eval_shift_j, eval_noshift_j, [FAR_PRIMARY]))

    preflight = preflight_validity_check(detector_series['KL']['eval_shift'], results)

    return dict(results=results, null_results=null_results, preflight=preflight,
                k_cusum=k_cusum, delta_ph=delta_ph, mu_kl=mu_kl, sig_kl=sig_kl)


def run_task(task, spec, cfg, directions):
    print(f"\n{'='*78}\n{task.upper()} — TASK A REGIME DETECTION\n{'='*78}")
    tr_base = dict(np.load(spec['base_training_states']))
    ema_alpha = fit_ema_alpha(tr_base['recon'][:50000], tr_base['kl'][:50000])
    kl_median = float(np.median(tr_base['kl']))

    model_specs = list(zip(spec['multiseed'], spec['multiseed_states']))
    per_seed_results = []
    for i, (ckpt, states_path) in enumerate(model_specs):
        print(f"  --- seed {i} ({ckpt}) ---")
        torch.manual_seed(SEED + i)
        model = load_model_generic(ckpt, cfg)
        v = fit_v_direction(states_path)
        seed_out = run_task_on_model(task, spec, cfg, model, v, directions, ema_alpha, kl_median)
        seed_out['checkpoint'] = ckpt
        per_seed_results.append(seed_out)

    return dict(task=task, n_seeds=len(model_specs), per_seed=per_seed_results)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    directions = load_canonical_directions()
    print(f"Loaded {len(directions)} canonical null directions from {CANONICAL_DIR_PATH}")

    all_results = {}
    for task, spec in TASKS.items():
        all_results[task] = run_task(task, spec, cfg, directions)
        with open(os.path.join(OUT_DIR, f'{task}_results.json'), 'w') as f:
            json.dump(all_results[task], f, indent=2, default=float)
        print(f"  wrote {OUT_DIR}/{task}_results.json")

    with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nWrote all results to {OUT_DIR}/")


if __name__ == '__main__':
    main()
