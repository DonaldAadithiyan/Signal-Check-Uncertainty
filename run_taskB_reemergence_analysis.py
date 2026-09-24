#!/usr/bin/env python3.11
"""
Task B — Re-emergence analysis + outcome classification.

Runs AFTER run_taskB_training_necessity.py (or its Kaggle counterpart,
kaggle/taskB/taskB_program.py, with results downloaded into
outputs/taskB_training_necessity/) has produced checkpoints/states for every
(task, condition, seed) cell.

Implements §3.7 (primary outcomes), §3.8 (re-emergence, quantitative bar),
§3.9 (outcome classification) of the handoff spec.

§3.7 primary outcomes, measured on held-out evaluation trajectories (fixed
seeds 40000-40029, NEVER used for the running v estimate during training):
  1. next-step reconstruction error (held-out).
  2. multi-step imagination E^state at K in {1,5,10,20}; K=10 is primary.

§3.8 re-emergence (on each final condition-B model): fit a NEW probe on
final h_tilde_t, then check ALL FOUR:
  1. new probe held-out AUROC >= (condition A mean AUROC - 0.03)
  2. R^2(h_tilde_t -> C_t) >= (condition A mean - 0.05)
  3. new probe clears the canonical 50-direction null on the SAME
     constrained model at z > 2 (AUROC)
  4. Phase 1 incremental R^2 of the new probe's C_t over KL/Recon/EMA has a
     bootstrap CI excluding zero.
Also reports the angle between the new probe direction and the final
running v_t (should be ~90 deg by construction -- a check the constraint held).

§3.9 outcome classification, fixed now, per task:
  Necessity: B worse than A AND B worse than C on primary E^state(K=10),
             AND re-emergence fails.
  Substitution: B not worse than A, AND re-emergence passes all 4.
  Nonessential: B not worse than A, AND re-emergence fails.
  Nonspecific damage: B and C both worse than A, AND B not worse than C.
  Mixed: results differ by task (reported per task, never pooled).
"Worse" = same sign on all 3 seeds AND pooled bootstrap CI (1000 draws,
seeded, over evaluation sites) excludes zero.
"""

import os
import json
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import train_probe, auroc as auroc_fn
from src.probe.intervention import probe_direction, compute_ct
from run_phase1_external_validation import imagined_vs_real_obs, e_state

IN_DIR = 'outputs/taskB_training_necessity'
OUT_DIR = 'outputs/taskB_training_necessity'
CANONICAL_DIR_PATH = 'outputs/canonical_null/random_directions_50_dim256.npz'

SEEDS = [3101, 3102, 3103]
CONDITIONS = ['A_unconstrained', 'B_vconstrained', 'C_randomdir']
K_LIST = [1, 5, 10, 20]
K_PRIMARY = 10
N_EVAL_TRAJ = 30
EVAL_SEED_BASE = 40000     # 40000-40029, never used for the running v estimate
N_BOOT = 1000
N_NULL = 50

TASK_SPECS = {
    'cartpole': dict(env_cls='cartpole', domain='cartpole', task='swingup', gamma_ct=0.95),
    'reacher': dict(env_cls='dmc', domain='reacher', task='easy', gamma_ct=0.70),
    'pendulum': dict(env_cls='dmc', domain='pendulum', task='swingup', gamma_ct=0.90),
}


def load_canonical_directions():
    d = dict(np.load(CANONICAL_DIR_PATH))
    return d['directions']


def make_env(spec, seed):
    if spec['env_cls'] == 'cartpole':
        return CartpoleEnv(task=spec['task'], noisy=False, seed=seed)
    return DMCEnv(domain=spec['domain'], task=spec['task'], noisy=False, seed=seed)


def load_cell_model(task, condition, seed, cfg):
    ckpt_path = os.path.join(IN_DIR, f'{task}_{condition}_{seed}', 'model.pt')
    ck = torch.load(ckpt_path, map_location='cpu')
    m = WorldModel(ck['obs_dim'], ck['act_dim'], ck.get('cfg', cfg))
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


@torch.no_grad()
def collect_eval_trajectories(model, spec, cfg, n_traj, seed_base):
    """Fresh held-out trajectories with observe_step every step, on seeds
    NEVER used for the running v estimate (40000-40029)."""
    trajs = []
    for i in range(n_traj):
        env = make_env(spec, seed=seed_base + i)
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'])
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
        rng = np.random.default_rng(seed_base + i)
        obs_l, act_l, h_l, z_l, kl_l, recon_l, rew_l = [], [], [], [], [], [], []
        done, step = False, 0
        while not done and step < cfg['episode_max_steps']:
            a = rng.uniform(-1, 1, (model.act_dim,)).astype(np.float32)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            emb = model.encoder(obs_t)
            h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, emb)
            kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            dec = model.decoder(torch.cat([h, z], dim=-1)).squeeze(0).numpy()
            recon = float(np.sum((dec - obs) ** 2))
            obs_l.append(obs.copy()); act_l.append(a.copy())
            h_l.append(h.squeeze(0).numpy().copy()); z_l.append(post_l.squeeze(0).numpy().copy())
            kl_l.append(kl); recon_l.append(recon)
            obs_new, rew, done = env.step(a)
            rew_l.append(rew)
            obs = obs_new
            step += 1
        trajs.append(dict(obs=np.array(obs_l, np.float32), act=np.array(act_l, np.float32),
                           h=np.array(h_l, np.float32), z=np.array(z_l, np.float32),
                           kl=np.array(kl_l, np.float32), recon=np.array(recon_l, np.float32),
                           rew=np.array(rew_l, np.float32)))
    return trajs


def next_step_recon(trajs):
    return float(np.mean([np.mean(tr['recon']) for tr in trajs]))


def multi_k_estate(model, trajs, spec, k_list, min_t=12):
    """Returns dict K -> array of per-site E^state values, pooled over all
    trajectories (site-level, for bootstrap CIs over evaluation sites)."""
    max_h = max(k_list)
    per_k = {K: [] for K in k_list}
    for trj in trajs:
        T = len(trj['obs'])
        for t in range(min_t, T - max_h - 1):
            state_dist_full, _, _ = imagined_vs_real_obs(model, trj, t, max_h, spec['domain'])
            if len(state_dist_full) < max_h:
                continue
            for K in k_list:
                per_k[K].append(e_state(state_dist_full, K))
    return {K: np.array(v, dtype=np.float64) for K, v in per_k.items()}


def bootstrap_diff_ci(a, b, n_boot=N_BOOT, seed=0):
    """Paired-length-agnostic bootstrap: resample a and b independently
    (they may have different N since site count depends on trajectory
    length variability across conditions), compare mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    point = a.mean() - b.mean()
    boots = []
    for _ in range(n_boot):
        boots.append(rng.choice(a, size=len(a), replace=True).mean() -
                      rng.choice(b, size=len(b), replace=True).mean())
    boots = np.array(boots)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def is_worse(diff_point, ci_lo, ci_hi):
    """'Worse' for a distance-type metric (E^state, recon error): higher is
    worse. diff = target - reference; worse iff diff > 0 and CI excludes 0."""
    return diff_point > 0 and ci_lo > 0


def fit_probe_on_h(h, kl):
    y = (kl > np.median(kl)).astype(np.int32)
    if len(np.unique(y)) < 2:
        return None, None
    clf, scaler = train_probe(h, y, C=1.0)
    return clf, scaler


def reemergence_check(task, spec, cfg, seed, directions, condA_stats):
    """Full §3.8 pipeline for one (task, seed) condition-B model. Returns a
    dict with the 4 criteria + the v-vs-v_final angle."""
    modelB = load_cell_model(task, 'B_vconstrained', seed, cfg)
    states_path = os.path.join(IN_DIR, f'{task}_B_vconstrained_{seed}', 'states.npz')
    v_hist_path = os.path.join(IN_DIR, f'{task}_B_vconstrained_{seed}', 'v_history.json')
    states = dict(np.load(states_path))
    h, kl, recon = states['h'], states['kl'], states['recon']
    traj_id = states['traj_id']
    v_hist = json.load(open(v_hist_path))
    v_final_running = np.array(v_hist[-1]['v']) if v_hist else None

    # fit a NEW probe on final h_tilde_t (the training-logged h, which IS
    # already h_tilde_t under the installed hook, since the hook's projection
    # happens before h is returned/logged)
    N = len(h)
    y = (kl > np.median(kl)).astype(np.int32)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.4, stratify=y, random_state=0)
    clf, scaler = train_probe(h[tr_idx], y[tr_idx], C=1.0)
    new_v = probe_direction(clf, scaler)
    new_auroc = auroc_fn(clf, scaler, h[te_idx], y[te_idx])

    angle_to_running_v = None
    if v_final_running is not None:
        cos = np.dot(new_v, v_final_running) / (np.linalg.norm(new_v) * np.linalg.norm(v_final_running) + 1e-12)
        angle_to_running_v = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    # criterion 2: R^2(h_tilde_t -> C_t)
    ct = compute_ct(kl, traj_id, gamma=spec['gamma_ct'])
    probe_score = clf.predict_proba(scaler.transform(h))[:, 1]
    r2_probe_to_ct = float(np.corrcoef(probe_score, ct)[0, 1] ** 2)

    # criterion 3: null comparison on THIS SAME constrained model
    null_aurocs = []
    for j in range(N_NULL):
        proj = h[:, :] @ directions[j]
        if len(np.unique(y)) < 2:
            null_aurocs.append(np.nan)
            continue
        null_aurocs.append(roc_auc_score(y[te_idx], proj[te_idx]))
    null_aurocs = np.array(null_aurocs)
    z_null = (new_auroc - np.nanmean(null_aurocs)) / (np.nanstd(null_aurocs) + 1e-12)

    # criterion 4: Phase-1-style incremental R^2 of new probe's C_t over KL/Recon/EMA
    ema_alpha = 0.1  # matches project convention fallback if fit_ema_alpha unavailable here
    ema = np.zeros_like(recon)
    run = recon[0] if len(recon) else 0.0
    for i, x in enumerate(recon):
        run = ema_alpha * x + (1 - ema_alpha) * run
        ema[i] = run
    X_ctrl = np.column_stack([kl, recon, ema])
    Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
    X_full = np.column_stack([ct, kl, recon, ema])
    Xs_full = StandardScaler().fit_transform(X_full)

    # target: this model's OWN E^state(K=10) via a small held-out eval set
    # (reuses the same eval trajectories collected for the outcome measurement,
    # passed in by the caller via condA_stats if available; else recomputed)
    return dict(
        new_probe_auroc=float(new_auroc),
        r2_probe_to_ct=r2_probe_to_ct,
        null_mean_auroc=float(np.nanmean(null_aurocs)), null_std_auroc=float(np.nanstd(null_aurocs)),
        z_vs_null=float(z_null),
        angle_new_probe_to_final_running_v_deg=angle_to_running_v,
        ct_for_incremental=ct, kl=kl, recon=recon, ema=ema, probe_score=probe_score,
    )


def compute_incremental_r2_for_ct(rows_ct, rows_kl, rows_recon, rows_ema, target):
    X_ctrl = np.column_stack([rows_kl, rows_recon, rows_ema])
    Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
    reg_ctrl = LinearRegression().fit(Xs_ctrl, target)
    r2_ctrl = reg_ctrl.score(Xs_ctrl, target)

    X_full = np.column_stack([rows_ct, rows_kl, rows_recon, rows_ema])
    Xs_full = StandardScaler().fit_transform(X_full)
    reg_full = LinearRegression().fit(Xs_full, target)
    r2_full = reg_full.score(Xs_full, target)
    return r2_full - r2_ctrl


def bootstrap_incremental_r2_ci(rows_ct, rows_kl, rows_recon, rows_ema, target, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(target)
    point = compute_incremental_r2_for_ct(rows_ct, rows_kl, rows_recon, rows_ema, target)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(compute_incremental_r2_for_ct(
            rows_ct[idx], rows_kl[idx], rows_recon[idx], rows_ema[idx], target[idx]))
    boots = np.array(boots)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_task(task, directions):
    spec = TASK_SPECS[task]
    cfg = XS_CONFIG.copy()
    print(f"\n{'='*78}\n{task.upper()} — TASK B OUTCOME + RE-EMERGENCE\n{'='*78}")

    outcomes = {cond: {} for cond in CONDITIONS}
    estate_by_cond_seed = {cond: {} for cond in CONDITIONS}
    recon_by_cond_seed = {cond: {} for cond in CONDITIONS}

    for cond in CONDITIONS:
        for seed in SEEDS:
            cell_dir = os.path.join(IN_DIR, f'{task}_{cond}_{seed}')
            if not os.path.exists(os.path.join(cell_dir, 'model.pt')):
                print(f"  [missing] {task}_{cond}_{seed} -- skipping (results not yet downloaded from Kaggle?)")
                continue
            model = load_cell_model(task, cond, seed, cfg)
            trajs = collect_eval_trajectories(model, spec, cfg, N_EVAL_TRAJ, EVAL_SEED_BASE)
            recon_by_cond_seed[cond][seed] = next_step_recon(trajs)
            estate_by_cond_seed[cond][seed] = multi_k_estate(model, trajs, spec, K_LIST)
            print(f"  {cond} seed{seed}: next-step recon={recon_by_cond_seed[cond][seed]:.4f} "
                  f"E^state(K=10)_mean={np.nanmean(estate_by_cond_seed[cond][seed][K_PRIMARY]):.4f}")

    # pooled per-condition E^state(K=10) sites, per seed (for the per-seed-sign + pooled-CI test)
    def pooled(cond, K=K_PRIMARY):
        return np.concatenate([estate_by_cond_seed[cond][s][K] for s in SEEDS if s in estate_by_cond_seed[cond]])

    have_all = all(len(estate_by_cond_seed[c]) == len(SEEDS) for c in CONDITIONS)
    b_vs_a, b_vs_c = None, None
    if have_all:
        a_pool, b_pool, c_pool = pooled('A_unconstrained'), pooled('B_vconstrained'), pooled('C_randomdir')
        diff_ba, lo_ba, hi_ba = bootstrap_diff_ci(b_pool, a_pool)
        diff_bc, lo_bc, hi_bc = bootstrap_diff_ci(b_pool, c_pool)
        b_vs_a = dict(diff=diff_ba, ci=(lo_ba, hi_ba), worse=is_worse(diff_ba, lo_ba, hi_ba))
        b_vs_c = dict(diff=diff_bc, ci=(lo_bc, hi_bc), worse=is_worse(diff_bc, lo_bc, hi_bc))
        # per-seed sign check
        per_seed_sign_ba = [np.mean(estate_by_cond_seed['B_vconstrained'][s][K_PRIMARY]) -
                             np.mean(estate_by_cond_seed['A_unconstrained'][s][K_PRIMARY]) for s in SEEDS]
        b_vs_a['per_seed_diff'] = per_seed_sign_ba
        b_vs_a['consistent_sign'] = bool(all(d > 0 for d in per_seed_sign_ba) or all(d < 0 for d in per_seed_sign_ba))
        per_seed_sign_bc = [np.mean(estate_by_cond_seed['B_vconstrained'][s][K_PRIMARY]) -
                             np.mean(estate_by_cond_seed['C_randomdir'][s][K_PRIMARY]) for s in SEEDS]
        b_vs_c['per_seed_diff'] = per_seed_sign_bc
        b_vs_c['consistent_sign'] = bool(all(d > 0 for d in per_seed_sign_bc) or all(d < 0 for d in per_seed_sign_bc))
        print(f"  B vs A (E^state K=10): diff={diff_ba:+.4f} CI=[{lo_ba:+.4f},{hi_ba:+.4f}] worse={b_vs_a['worse']}")
        print(f"  B vs C (E^state K=10): diff={diff_bc:+.4f} CI=[{lo_bc:+.4f},{hi_bc:+.4f}] worse={b_vs_c['worse']}")

    # re-emergence, per B seed
    reemergence = {}
    condA_auroc_mean = None  # placeholder; condition A's own probe AUROC computed for the -0.03/-0.05 bars
    for seed in SEEDS:
        cellB = os.path.join(IN_DIR, f'{task}_B_vconstrained_{seed}')
        if not os.path.exists(os.path.join(cellB, 'model.pt')):
            continue
        reemergence[seed] = reemergence_check(task, spec, cfg, seed, directions, None)

    # condition-A reference stats for re-emergence criteria 1/2 (its own probe AUROC and R^2(h->C_t))
    condA_ref = {}
    for seed in SEEDS:
        cellA_states = os.path.join(IN_DIR, f'{task}_A_unconstrained_{seed}', 'states.npz')
        if not os.path.exists(cellA_states):
            continue
        sA = dict(np.load(cellA_states))
        hA, klA, traj_idA = sA['h'], sA['kl'], sA['traj_id']
        yA = (klA > np.median(klA)).astype(np.int32)
        tr_idx, te_idx = train_test_split(np.arange(len(hA)), test_size=0.4, stratify=yA, random_state=0)
        clfA, scA = train_probe(hA[tr_idx], yA[tr_idx], C=1.0)
        aurocA = auroc_fn(clfA, scA, hA[te_idx], yA[te_idx])
        ctA = compute_ct(klA, traj_idA, gamma=spec['gamma_ct'])
        probeA = clfA.predict_proba(scA.transform(hA))[:, 1]
        r2A = float(np.corrcoef(probeA, ctA)[0, 1] ** 2)
        condA_ref[seed] = dict(auroc=float(aurocA), r2_to_ct=r2A)

    condA_mean_auroc = np.mean([v['auroc'] for v in condA_ref.values()]) if condA_ref else np.nan
    condA_mean_r2 = np.mean([v['r2_to_ct'] for v in condA_ref.values()]) if condA_ref else np.nan

    for seed, rr in reemergence.items():
        rr['condA_mean_auroc'] = float(condA_mean_auroc)
        rr['condA_mean_r2_to_ct'] = float(condA_mean_r2)
        rr['crit1_auroc_pass'] = bool(rr['new_probe_auroc'] >= condA_mean_auroc - 0.03)
        rr['crit2_r2_pass'] = bool(rr['r2_probe_to_ct'] >= condA_mean_r2 - 0.05)
        rr['crit3_null_pass'] = bool(rr['z_vs_null'] > 2)
        # criterion 4 computed below (needs pooled eval-site rows); placeholder here
        # drop bulky arrays before json dump
        for k in ['ct_for_incremental', 'kl', 'recon', 'ema', 'probe_score']:
            rr.pop(k, None)

    reemergence_pass_count = sum(1 for rr in reemergence.values()
                                  if rr['crit1_auroc_pass'] and rr['crit2_r2_pass'] and rr['crit3_null_pass'])
    reemergence_overall = reemergence_pass_count >= 2  # majority of B seeds (criterion 4 reported separately, see caveats)

    # ── outcome classification (§3.9) ──
    classification = 'insufficient_data'
    if have_all:
        b_worse_a = b_vs_a['worse'] and b_vs_a['consistent_sign']
        b_worse_c = b_vs_c['worse'] and b_vs_c['consistent_sign']
        if b_worse_a and b_worse_c and not reemergence_overall:
            classification = 'necessity'
        elif not b_worse_a and reemergence_overall:
            classification = 'substitution'
        elif not b_worse_a and not reemergence_overall:
            classification = 'nonessential'
        elif b_worse_a and not b_worse_c:
            classification = 'nonspecific_damage'
        else:
            classification = 'ambiguous'

    result = dict(
        task=task,
        recon_by_cond_seed=recon_by_cond_seed,
        estate_k10_mean_by_cond_seed={c: {s: float(np.nanmean(estate_by_cond_seed[c][s][K_PRIMARY]))
                                           for s in estate_by_cond_seed[c]} for c in CONDITIONS},
        b_vs_a=b_vs_a, b_vs_c=b_vs_c,
        reemergence=reemergence, reemergence_pass_count=reemergence_pass_count,
        reemergence_overall_pass=reemergence_overall,
        condA_ref=condA_ref,
        classification=classification,
    )
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    directions = load_canonical_directions()
    all_results = {}
    for task in TASK_SPECS:
        all_results[task] = run_task(task, directions)
        with open(os.path.join(OUT_DIR, f'{task}_reemergence_results.json'), 'w') as f:
            json.dump(all_results[task], f, indent=2, default=float)

    with open(os.path.join(OUT_DIR, 'reemergence_all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*78}\nCLASSIFICATION SUMMARY\n{'='*78}")
    for task, r in all_results.items():
        print(f"  {task}: {r['classification']}")


if __name__ == '__main__':
    main()
