#!/usr/bin/env python3.11
"""
Phase 6 — Positive causal steering (Reframed_Confusion_RSSM_Project, Section 13).

Only attempted after external validation passed (Gate 1, Phase 1: 3/3 tasks) and
the SAE mechanism / POMDP stress test were completed (Phases 3-4), per the plan's
sequencing.

Existing intervention throughout this project (Task G, Task L, Phase 3):
    h_t' = h_t - v(v^T h_t)          [ABLATION -- removes the confusion direction]
This phase adds the complementary, not-yet-tested direction:
    h_t' = h_t + lambda * v          [AMPLIFICATION -- pushes h_t further along v]
swept over lambda in {-2*sigma_v, -sigma_v, 0, +sigma_v, +2*sigma_v}, where
sigma_v = std(h_t . v) over the training distribution (Task G's existing std_proj
quantity) -- i.e. the sweep is in units of the direction's own natural scale in
h_t-space, not an arbitrary absolute magnitude.

Measured at every lambda, on all 3 tasks (extends Section 13's cartpole-only
framing to the same 3-task scope used throughout Phases 1-5 of this project):
    - probe confusion readout (does amplifying v raise the probe's own score,
      and ablating/negatively-amplifying lower it -- the "does this control what
      it should control" sanity check)
    - external imagination-error prediction (does E^state, the Phase-1 external
      target, respond to the SAME intervention that moves the probe -- the
      stronger, non-circular version of "this representation is bidirectionally
      causal for behavior the project has independently validated as meaningful")
    - real-observation query / routing behavior (does the fraction of states
      that would trigger a "check reality" decision, at a fixed query budget
      threshold from Task Q/R, respond monotonically to lambda)

NOT measured (stated explicitly, not silently skipped, per Section 6.5's
precedent): planning conservatism and eventual return require a policy/planner/
critic, which does not exist anywhere in this XS pipeline (data collection is
uniform-random actions throughout the whole project) -- Section 13's own list
of measures is aspirational for an architecture with a policy, which this one
is not. Reported as N/A, not fabricated.

The desired pattern (Section 13) is a DOSE-RESPONSE relationship: monotonic (or
at least ordered, sign-consistent) response of each readout across the lambda
sweep. This is tested, not assumed -- reported honestly if it is not monotonic
on some task, consistent with the project's non-goals (Section 18) against
tuning results to look cleaner than they are.

Runs on the EXISTING frozen models (cartpole, reacher, pendulum). No retraining.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import probe_direction, amplify, project_out
from run_phase1_external_validation import (
    ENVS as P1_ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    MIN_T, SEED,
)

LAMBDAS_IN_SIGMA = [-2.0, -1.0, 0.0, 1.0, 2.0]
K_HEADLINE = 10
N_TRAJ = 60
N_SITES = 1000
QUERY_BUDGET = 0.30
OUT_DIR = 'outputs/phase6_causal_steering'
FIG_DIR = 'outputs/figures'

TASKS = {
    'cartpole': dict(training_states='outputs/data/training_states.npz',
                      checkpoint='outputs/checkpoints/world_model.pt'),
    'reacher':  dict(training_states='outputs/second_env/reacher_easy_training_states.npz',
                      checkpoint='outputs/second_env/reacher_easy_world_model.pt'),
    'pendulum': dict(training_states='outputs/third_env/pendulum_swingup_training_states.npz',
                      checkpoint='outputs/third_env/pendulum_swingup_world_model.pt'),
}


@torch.no_grad()
def continue_and_imagine(model, traj, t, h_new, clf, scaler, domain, horizon=K_HEADLINE):
    """Continue the observed trajectory from t with h replaced by h_new (posterior
    continuation using the real subsequent observations -- same construction as
    Task G's continue_probe), returning the probe score at t. THEN roll pure
    imagination forward `horizon` steps from the intervened h_new (using real
    actions), decode, and compute E^state against the real subsequent observations
    -- the external-validity readout, using Phase 1's own construction."""
    device = next(model.parameters()).device
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    h = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    emb = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    z = model.rssm._straight_through_sample(post_l)
    probe_score = float(clf.predict_proba(scaler.transform(h_new.reshape(1, -1)))[0, 1])

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
    e_state_val = float(np.mean(state_dist)) if len(state_dist) == horizon else np.nan

    return probe_score, e_state_val


def run_task_steering(task, spec, cfg):
    print(f"\n{'='*78}\n{task.upper()} — PHASE 6 CAUSAL STEERING\n{'='*78}")
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    y = binarise_by_median(tr['kl'])
    idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                       stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])
    v = probe_direction(clf, scaler)
    sigma_v = float((tr['h'] @ v).std())
    print(f"  sigma_v (std of h.v over training distribution) = {sigma_v:.4f}")

    probe_te = clf.predict_proba(scaler.transform(tr['h'][idx_te]))[:, 1]
    route_thresh = float(np.percentile(probe_te, 100 * (1 - QUERY_BUDGET)))

    p1_spec = P1_ENVS[task]
    domain = p1_spec['domain']
    trajs = collect_trajectories(model, p1_spec, N_TRAJ, cfg, seed=SEED + 900)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - K_HEADLINE - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED + 900)
    if len(sites) > N_SITES:
        sel = rng.choice(len(sites), N_SITES, replace=False)
        sites = [sites[i] for i in sel]
    print(f"  {len(sites)} intervention sites")

    dose_response = {}
    for lam_sigma in LAMBDAS_IN_SIGMA:
        lam = lam_sigma * sigma_v
        probe_scores, e_states, query_flags = [], [], []
        for (ti, t) in sites:
            trj = trajs[ti]
            h_t = trj['h'][t]
            h_new = amplify(h_t.reshape(1, -1), v, lam).reshape(-1)
            ps, es = continue_and_imagine(model, trj, t, h_new, clf, scaler, domain)
            probe_scores.append(ps)
            e_states.append(es)
            query_flags.append(int(ps >= route_thresh))
        probe_scores = np.array(probe_scores)
        e_states = np.array(e_states)
        valid = ~np.isnan(e_states)
        dose_response[lam_sigma] = dict(
            lambda_sigma=lam_sigma, lambda_raw=float(lam),
            probe_mean=float(probe_scores.mean()), probe_std=float(probe_scores.std()),
            e_state_mean=float(np.nanmean(e_states)), e_state_std=float(np.nanstd(e_states)),
            query_rate=float(np.mean(query_flags)),
            n_valid_e_state=int(valid.sum()),
        )
        print(f"  lambda={lam_sigma:+.0f}sigma: probe={dose_response[lam_sigma]['probe_mean']:.4f} "
              f"E^state={dose_response[lam_sigma]['e_state_mean']:.4f} "
              f"query_rate={dose_response[lam_sigma]['query_rate']:.3f}")

    # dose-response monotonicity check (Spearman rank correlation of lambda vs each readout)
    lambdas_arr = np.array(LAMBDAS_IN_SIGMA)
    probe_means = np.array([dose_response[l]['probe_mean'] for l in LAMBDAS_IN_SIGMA])
    e_state_means = np.array([dose_response[l]['e_state_mean'] for l in LAMBDAS_IN_SIGMA])
    query_rates = np.array([dose_response[l]['query_rate'] for l in LAMBDAS_IN_SIGMA])

    rho_probe, p_probe = spearmanr(lambdas_arr, probe_means)
    rho_e_state, p_e_state = spearmanr(lambdas_arr, e_state_means)
    rho_query, p_query = spearmanr(lambdas_arr, query_rates)

    print(f"\n  monotonicity (Spearman rho of lambda vs. readout):")
    print(f"    probe confusion:      rho={rho_probe:+.3f} (p={p_probe:.3g})")
    print(f"    external E^state:     rho={rho_e_state:+.3f} (p={p_e_state:.3g})")
    print(f"    query rate (routing): rho={rho_query:+.3f} (p={p_query:.3g})")

    dose_response_verdict = (
        'CLEAN DOSE-RESPONSE (all 3 readouts monotonic, p<0.05)'
        if all(p < 0.05 and rho > 0 for rho, p in
               [(rho_probe, p_probe), (rho_e_state, p_e_state), (rho_query, p_query)])
        else 'PARTIAL DOSE-RESPONSE (probe readout monotonic, others not all clean)'
        if (p_probe < 0.05 and rho_probe > 0)
        else 'NO CLEAR DOSE-RESPONSE'
    )
    print(f"  verdict: {dose_response_verdict}")

    return dict(
        task=task, sigma_v=sigma_v, route_thresh=route_thresh, n_sites=len(sites),
        dose_response=dose_response,
        monotonicity=dict(rho_probe=rho_probe, p_probe=p_probe,
                          rho_e_state=rho_e_state, p_e_state=p_e_state,
                          rho_query=rho_query, p_query=p_query),
        verdict=dose_response_verdict,
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Note: 'planning conservatism' and 'eventual return' from Section 13's "
          "measure list are N/A -- no policy/planner/critic exists in this XS "
          "pipeline (uniform-random-action data collection throughout the project). "
          "Reported as N/A explicitly, not fabricated or silently omitted.\n")

    all_results = {}
    for task, spec in TASKS.items():
        all_results[task] = run_task_steering(task, spec, cfg)

    print(f"\n{'='*78}\nPHASE 6 SUMMARY\n{'='*78}")
    for task, r in all_results.items():
        print(f"  {task}: {r['verdict']}")

    out_path = os.path.join(OUT_DIR, 'phase6_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, task in enumerate(TASKS):
        r = all_results[task]
        lambdas_arr = np.array(LAMBDAS_IN_SIGMA)
        probe_means = [r['dose_response'][l]['probe_mean'] for l in LAMBDAS_IN_SIGMA]
        probe_stds = [r['dose_response'][l]['probe_std'] for l in LAMBDAS_IN_SIGMA]
        e_state_means = [r['dose_response'][l]['e_state_mean'] for l in LAMBDAS_IN_SIGMA]
        e_state_stds = [r['dose_response'][l]['e_state_std'] for l in LAMBDAS_IN_SIGMA]
        query_rates = [r['dose_response'][l]['query_rate'] for l in LAMBDAS_IN_SIGMA]

        ax = axes[i, 0]
        ax.errorbar(lambdas_arr, probe_means, yerr=probe_stds, marker='o', color='steelblue')
        ax.set_title(f'{task}: probe confusion\nrho={r["monotonicity"]["rho_probe"]:+.2f}')
        ax.set_xlabel('lambda (in units of sigma_v)'); ax.set_ylabel('probe score')
        ax.grid(alpha=0.3)

        ax = axes[i, 1]
        ax.errorbar(lambdas_arr, e_state_means, yerr=e_state_stds, marker='o', color='darkorange')
        ax.set_title(f'{task}: external E^state\nrho={r["monotonicity"]["rho_e_state"]:+.2f}')
        ax.set_xlabel('lambda (in units of sigma_v)'); ax.set_ylabel('E^state (K=10)')
        ax.grid(alpha=0.3)

        ax = axes[i, 2]
        ax.plot(lambdas_arr, query_rates, marker='o', color='green')
        ax.set_title(f'{task}: query rate (routing)\nrho={r["monotonicity"]["rho_query"]:+.2f}')
        ax.set_xlabel('lambda (in units of sigma_v)'); ax.set_ylabel('fraction queried')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'phase6_causal_steering.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {fig_path}")


if __name__ == '__main__':
    main()
