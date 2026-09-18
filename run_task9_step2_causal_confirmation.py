#!/usr/bin/env python3.11
"""
Task 9 -- Step 2: causal confirmation on the blind-spot candidate sites
(high atom activation, low C_t), gated on Step 0 (power) and Step 1
(natural characterization) both having produced usable, non-degenerate
results -- which they did (see step0/step1 results.json).

Rebuilds the identical FRESH_SEED trajectory set and site list Step 0 used
(same seed, same call order -> byte-reproducible per the project's RNG-bug
audit), re-derives the same group masks, then restricts to the CANDIDATE
sites only (high atom, low C_t) and tests: does amplifying h along the
atom's own SAE decoder direction, AT THESE SITES SPECIFICALLY, raise
E^state -- and is that effect distinguishable from:

  (1) a random unit direction (norm-matched null, same protocol as Phase 3/6)
  (2) a random OTHER SAE atom's decoder direction (mechanism-specificity;
      is it "any sparse direction" or specifically atom #612/#156)
  (3) a KL-matched direction (regression direction fit to predict KL_t from
      h_t on the training distribution -- does amplifying "generic KL-ness"
      reproduce the effect)
  (4) a C_t-matched direction (regression direction fit to predict C_t_t
      from h_t -- does amplifying "generic confusion-history-ness" reproduce
      the effect)
  (5) an off-manifold check: does the amplified h_t remain in-distribution
      (projection onto v within the training range of h.v), so the effect
      isn't just "any sufficiently large perturbation breaks the decoder"

Ship rule (Task 9 spec, decided before seeing results): the atom's own
direction must beat ALL FOUR matched controls (not just the naive random
direction) AND stay within the off-manifold sanity bound for this to ship
as a paper-worthy blind-spot causal result. Otherwise -> future-work pile,
reported honestly, not softened.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

from src.config import XS_CONFIG
from src.probe.sae import TopKSAE, get_all_activations
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import (
    compute_ct, probe_direction, regression_direction, random_matched_direction,
    amplify, project_out,
)
from run_phase1_external_validation import (
    ENVS as P1_ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, MIN_T, MAX_HORIZON, SEED,
)
from run_task9_step0_sample_size import (
    FRESH_SEED, N_TRAJ_FRESH, K_HEADLINE, N_ATOMS, K_SPARSE, SAE_PATH,
    ATOM_612, ATOM_156, load_sae,
)

OUT_DIR = 'outputs/task9_blind_spot'
N_NULL = 50
LAMBDA_SIGMA = 2.0   # amplification magnitude, in units of sigma_v (matches Phase 6's scale convention)


@torch.no_grad()
def continue_and_imagine(model, traj, t, h_new, horizon=K_HEADLINE):
    """Same construction as Phase 6/Phase 3's continuation: posterior-continue
    from t with h replaced by h_new (using the real obs at t to get z), then
    roll pure imagination forward `horizon` steps with real actions, decode,
    compare to real subsequent obs -> E^state."""
    device = next(model.parameters()).device
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    h = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    emb = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    z = model.rssm._straight_through_sample(post_l)

    T = len(traj['obs'])
    h_im, z_im = h.clone(), z.clone()
    state_dist = []
    for k in range(1, horizon + 1):
        kk = t + k
        if kk >= T:
            break
        a = torch.tensor(traj['act'][kk - 1], dtype=torch.float32, device=device).unsqueeze(0)
        h_im, z_im, _ = model.rssm.imagine_step(h_im, z_im, a)
        dec = model.decoder(torch.cat([h_im, z_im], dim=-1)).squeeze(0).cpu().numpy()
        state_dist.append(float(np.linalg.norm(dec - traj['obs'][kk])))
    return float(np.mean(state_dist)) if len(state_dist) == horizon else np.nan


def rebuild_candidate_sites(task, spec, cfg, atom_idx, sae, step0_check):
    """Reproduces Step 0's exact trajectory set, site list, and CANDIDATE
    mask (high atom / low C_t), returning trajs + the candidate (ti,t) list."""
    torch.manual_seed(FRESH_SEED)
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])
    p1_spec = P1_ENVS[task]
    trajs = collect_trajectories(model, p1_spec, N_TRAJ_FRESH, cfg, seed=FRESH_SEED)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64), gamma=spec['gamma_ct'])
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - MAX_HORIZON - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(FRESH_SEED)
    if len(sites) > 6000:
        sel = rng.choice(len(sites), 6000, replace=False)
        sites = [sites[i] for i in sel]

    h_arr = np.array([trajs[ti]['h'][t] for (ti, t) in sites], dtype=np.float32)
    acts = get_all_activations(sae, h_arr)
    atom_activation = acts[:, atom_idx]
    ct_arr = np.array([trajs[ti]['ct'][t] for (ti, t) in sites])

    fire_rate = step0_check['atom_fire_rate']
    if fire_rate < 0.50:
        high_atom = atom_activation > 0.0
    else:
        high_atom = atom_activation >= step0_check['atom_activation_threshold']
    low_ct = ct_arr <= step0_check['ct_threshold']

    candidate_mask = high_atom & low_ct
    candidate_sites = [s for s, keep in zip(sites, candidate_mask) if keep]
    return model, trajs, candidate_sites, tr


def effect_of_direction(model, trajs, sites, v, lam, base_cache):
    """Amplify h by lam*v at each site, measure E^state, return mean delta
    vs. the cached unablated baseline."""
    deltas = []
    for (ti, t) in sites:
        trj = trajs[ti]
        h_t = trj['h'][t]
        h_new = amplify(h_t.reshape(1, -1), v, lam).reshape(-1)
        e_new = continue_and_imagine(model, trj, t, h_new)
        e_base = base_cache[(ti, t)]
        if not (np.isnan(e_new) or np.isnan(e_base)):
            deltas.append(e_new - e_base)
    return float(np.mean(deltas)) if deltas else float('nan'), len(deltas)


def run_task_causal_confirmation(task, spec, cfg, sae, atom_idx, step0_check):
    print(f"\n{'='*78}\n{task.upper()} #{atom_idx} — STEP 2 CAUSAL CONFIRMATION\n{'='*78}")
    model, trajs, candidate_sites, tr = rebuild_candidate_sites(task, spec, cfg, atom_idx, sae, step0_check)
    print(f"  {len(candidate_sites)} candidate sites (high atom, low C_t)")
    if len(candidate_sites) < 30:
        return dict(task=task, atom=atom_idx, verdict='INSUFFICIENT_SITES', n_candidate=len(candidate_sites))

    with torch.no_grad():
        d_atom = sae.decoder.weight[:, atom_idx].numpy().copy()  # unit-norm

    y_kl = binarise_by_median(tr['kl'])
    idx_tr, _ = train_test_split(np.arange(len(tr['h'])), test_size=0.40, stratify=y_kl, random_state=0)
    clf, sc = train_probe(tr['h'][idx_tr], y_kl[idx_tr])
    sigma_v_atom = float((tr['h'] @ d_atom).std())
    lam = LAMBDA_SIGMA * sigma_v_atom

    kl_ridge = Ridge(alpha=1.0).fit(tr['h'], tr['kl'])
    kl_dir = (kl_ridge.coef_ / np.linalg.norm(kl_ridge.coef_)).astype(np.float32)

    ct_train = compute_ct(tr['kl'], np.zeros(len(tr['kl']), dtype=np.int64), gamma=spec['gamma_ct'])
    ct_ridge = Ridge(alpha=1.0).fit(tr['h'], ct_train)
    ct_dir = (ct_ridge.coef_ / np.linalg.norm(ct_ridge.coef_)).astype(np.float32)

    # baseline E^state at candidate sites (unablated), cached once
    base_cache = {}
    for (ti, t) in candidate_sites:
        trj = trajs[ti]
        base_cache[(ti, t)] = continue_and_imagine(model, trj, t, trj['h'][t])

    # (0) target: amplify along the atom's own decoder direction
    d_target, n_target = effect_of_direction(model, trajs, candidate_sites, d_atom, lam, base_cache)
    print(f"  TARGET (atom #{atom_idx} direction, lambda={lam:+.4f}): d_E^state={d_target:+.4f} (n={n_target})")

    # (1) empirical null: 50 random unit directions, matched lambda magnitude
    rng_null = np.random.default_rng(4044)
    null_random = []
    for i in range(N_NULL):
        vr = random_matched_direction(rng_null, d_atom.shape[0])
        d, _ = effect_of_direction(model, trajs, candidate_sites, vr, lam, base_cache)
        null_random.append(d)
    null_random = np.array(null_random)
    null_random = null_random[~np.isnan(null_random)]

    # (2) random OTHER SAE atoms (mechanism-specificity control)
    rng_atoms = np.random.default_rng(5055)
    other_atoms = rng_atoms.choice([a for a in range(N_ATOMS) if a != atom_idx], size=N_NULL, replace=False)
    null_other_atom = []
    for a in other_atoms:
        with torch.no_grad():
            d_other = sae.decoder.weight[:, a].numpy().copy()
        d, _ = effect_of_direction(model, trajs, candidate_sites, d_other, lam, base_cache)
        null_other_atom.append(d)
    null_other_atom = np.array(null_other_atom)
    null_other_atom = null_other_atom[~np.isnan(null_other_atom)]

    # (3) KL-matched direction
    d_kl, _ = effect_of_direction(model, trajs, candidate_sites, kl_dir, lam, base_cache)

    # (4) C_t-matched direction
    d_ct, _ = effect_of_direction(model, trajs, candidate_sites, ct_dir, lam, base_cache)

    def z_and_pct(target, null):
        if len(null) < 5:
            return float('nan'), float('nan')
        z = (target - null.mean()) / (null.std() + 1e-12)
        pct = float((null > target).mean() * 100) if target > null.mean() else float((null < target).mean() * 100)
        return float(z), pct

    z_rand, pct_rand = z_and_pct(d_target, null_random)
    z_atom, pct_atom = z_and_pct(d_target, null_other_atom)

    print(f"  random-direction null:  mean={null_random.mean():+.4f}±{null_random.std():.4f}  z={z_rand:+.2f}")
    print(f"  random-other-atom null: mean={null_other_atom.mean():+.4f}±{null_other_atom.std():.4f}  z={z_atom:+.2f}")
    print(f"  KL-matched direction:   d_E^state={d_kl:+.4f}")
    print(f"  C_t-matched direction:  d_E^state={d_ct:+.4f}")

    # (5) off-manifold check: is h_t + lam*d_atom within the training range of
    # projections onto d_atom? if the amplified projection falls far outside
    # [min,max] of (tr['h'] @ d_atom), the "effect" may just be decoder
    # breakdown off the data manifold rather than a meaningful causal effect.
    proj_train = tr['h'] @ d_atom
    proj_lo, proj_hi = float(proj_train.min()), float(proj_train.max())
    cand_h = np.array([trajs[ti]['h'][t] for (ti, t) in candidate_sites])
    proj_before = cand_h @ d_atom
    proj_after = proj_before + lam
    frac_off_manifold = float(((proj_after < proj_lo) | (proj_after > proj_hi)).mean())
    print(f"  off-manifold check: training proj range=[{proj_lo:.3f},{proj_hi:.3f}], "
          f"lambda={lam:+.4f}, frac amplified sites landing outside range={frac_off_manifold:.3f}")

    beats_random_dir = np.isfinite(z_rand) and abs(z_rand) > 2.0 and d_target > 0
    beats_other_atom = np.isfinite(z_atom) and abs(z_atom) > 2.0 and d_target > 0
    beats_kl_matched = d_target > d_kl and (d_kl <= 0 or d_target > 1.5 * d_kl)
    beats_ct_matched = d_target > d_ct and (d_ct <= 0 or d_target > 1.5 * d_ct)
    in_manifold = frac_off_manifold < 0.20

    all_conditions = beats_random_dir and beats_other_atom and beats_kl_matched and beats_ct_matched and in_manifold
    verdict = 'SHIP — clean, all controls cleared' if all_conditions else 'NO-SHIP — future work (controls not all cleared)'
    print(f"  conditions: beats_random_dir={beats_random_dir}  beats_other_atom={beats_other_atom}  "
          f"beats_kl_matched={beats_kl_matched}  beats_ct_matched={beats_ct_matched}  in_manifold={in_manifold}")
    print(f"  VERDICT: {verdict}")

    return dict(
        task=task, atom=int(atom_idx), n_candidate_sites=len(candidate_sites),
        lambda_sigma=LAMBDA_SIGMA, lambda_raw=float(lam),
        d_target=d_target, n_target=n_target,
        null_random_mean=float(null_random.mean()), null_random_std=float(null_random.std()), z_random=z_rand,
        null_other_atom_mean=float(null_other_atom.mean()), null_other_atom_std=float(null_other_atom.std()), z_other_atom=z_atom,
        d_kl_matched=d_kl, d_ct_matched=d_ct,
        off_manifold_frac=frac_off_manifold,
        conditions=dict(beats_random_dir=beats_random_dir, beats_other_atom=beats_other_atom,
                         beats_kl_matched=beats_kl_matched, beats_ct_matched=beats_ct_matched,
                         in_manifold=in_manifold),
        verdict=verdict,
    )


def main():
    cfg = XS_CONFIG.copy()
    sae = load_sae()
    with open(os.path.join(OUT_DIR, 'step0_sample_size_results.json')) as f:
        step0 = json.load(f)

    results = {}
    TASKS = {
        'cartpole': dict(training_states='outputs/data/training_states.npz',
                          checkpoint='outputs/checkpoints/world_model.pt'),
        'reacher':  dict(training_states='outputs/second_env/reacher_easy_training_states.npz',
                          checkpoint='outputs/second_env/reacher_easy_world_model.pt'),
    }
    for task, atom_idx in [('reacher', ATOM_612), ('cartpole', ATOM_156)]:
        spec = TASKS[task]
        spec = dict(spec)
        from run_phase1_external_validation import ENVS as P1_ENVS
        spec['gamma_ct'] = P1_ENVS[task]['gamma_ct']
        key = f'{task}_{atom_idx}'
        result = run_task_causal_confirmation(task, spec, cfg, sae, atom_idx, step0[key])
        results[key] = result

    out_path = os.path.join(OUT_DIR, 'step2_causal_confirmation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
