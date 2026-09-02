#!/usr/bin/env python3.11
"""
Pre-Scaling Rigor Addendum §5.1-§5.5 — Phase 6 causal-steering hardening.

§5.1 50-random-direction empirical null for the dose-response SLOPE (not just
     raw dose-response p-values, which are uninformative on their own since the
     probe was fit to v). For each task, apply the identical lambda-sweep
     amplification to all 50 existing null directions, fit each direction's own
     linear slope (metric ~ lambda) for probe/E^state/query-rate, and report v's
     slope as a z-score/percentile against the null-direction slope distribution.
§5.2 Recompute sigma_v on the HELD-OUT intervention pool, not the training
     distribution, for scale-matching consistency with the rest of the project.
§5.3 Extend the probe-confusion dose-response to the full k in {0,1,5,10}
     lookahead protocol (Appendix B / Task G / Phase 3's standard), not k=0 only.
§5.4 Add next-step reconstruction/prediction error as a 4th dose-response
     readout -- does amplifying v actually degrade next-step prediction, or only
     move the internal readout?
§5.5 Add a manifold-distance/state-plausibility readout: ||h_t' - h_t|| (trivial,
     scales with |lambda|*sigma_v by construction) AND nearest-neighbor distance
     to the real held-out h_t population, for both v and the null directions, so
     any off-manifold effect at large lambda is visible directly.

Runs on the 3 EXISTING frozen models. No retraining. Reuses the 50-random-
direction null generator from src/probe/intervention.py.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import probe_direction, amplify, random_matched_direction
from run_phase1_external_validation import (
    ENVS as P1_ENVS, load_model, collect_trajectories, MIN_T, SEED,
)
from run_phase6_causal_steering import TASKS

LAMBDAS_IN_SIGMA = [-2.0, -1.0, 0.0, 1.0, 2.0]
LOOKAHEAD = [0, 1, 5, 10]
K_HEADLINE = 10
N_TRAJ = 60
N_SITES_MAIN = 500     # sites for v's own full multi-step + recon-error sweep
N_SITES_NULL = 200     # sites for the 50-direction null sweep (smaller, matches
                       # Task G's null-protocol cost/precision tradeoff)
N_NULL = 50
QUERY_BUDGET = 0.30
OUT_DIR = 'outputs/phase6_causal_steering'


@torch.no_grad()
def continue_multi_step(model, traj, t, h_new, clf, scaler, domain, lookahead=LOOKAHEAD):
    """Full multi-step protocol (Task G / Phase 3 standard): continue the
    observed trajectory from t with h replaced by h_new (posterior continuation
    on real subsequent observations), returning probe scores at k in lookahead.
    ALSO computes next-step (k=1) reconstruction error against the real next
    observation (Section 5.4), and pure-imagination E^state_K=10 (Section 13's
    external readout, unchanged from the original Phase 6 script)."""
    device = next(model.parameters()).device
    T = len(traj['obs'])
    t_end = min(T, t + max(lookahead) + 1)

    h = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    emb = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    z = model.rssm._straight_through_sample(post_l)

    hs = [h.squeeze(0).cpu().numpy().copy()]
    next_step_recon = None
    for k in range(t + 1, t_end):
        a = torch.tensor(traj['act'][k - 1], dtype=torch.float32, device=device).unsqueeze(0)
        obs_k = torch.tensor(traj['obs'][k], dtype=torch.float32, device=device).unsqueeze(0)
        emb = model.encoder(obs_k)
        h, z, _, _ = model.rssm.observe_step(h, z, a, emb)
        if k == t + 1:
            dec = model.decoder(torch.cat([h, z], dim=-1))
            next_step_recon = float(torch.nn.functional.mse_loss(
                dec, obs_k, reduction='none').sum().item())
        hs.append(h.squeeze(0).cpu().numpy().copy())
    hs = np.array(hs, np.float32)
    ps = clf.predict_proba(scaler.transform(hs))[:, 1]
    probe_by_k = {k: float(ps[k]) if k < len(ps) else np.nan for k in lookahead}

    # pure-imagination E^state_K=10 from the intervened h_new (same construction
    # as the original Phase 6 script)
    h_im = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t2 = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    emb2 = model.encoder(obs_t2)
    post_l2 = model.rssm.post_net(torch.cat([h_im, emb2], dim=-1))
    z_im = model.rssm._straight_through_sample(post_l2)
    state_dist = []
    for k in range(1, K_HEADLINE + 1):
        kk = t + k
        if kk >= T:
            break
        a = torch.tensor(traj['act'][kk - 1], dtype=torch.float32, device=device).unsqueeze(0)
        h_im, z_im, _ = model.rssm.imagine_step(h_im, z_im, a)
        dec = model.decoder(torch.cat([h_im, z_im], dim=-1)).squeeze(0).cpu().numpy()
        state_dist.append(float(np.linalg.norm(dec - traj['obs'][kk])))
    e_state_val = float(np.mean(state_dist)) if len(state_dist) == K_HEADLINE else np.nan

    return probe_by_k, next_step_recon, e_state_val


def linear_slope(x, y):
    """OLS slope of y on x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return np.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def run_task_hardening(task, spec, cfg):
    print(f"\n{'='*78}\n{task.upper()} — ADDENDUM §5.1-§5.5\n{'='*78}")
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    y = binarise_by_median(tr['kl'])
    idx_tr, idx_te = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                       stratify=y, random_state=0)
    clf, scaler = train_probe(tr['h'][idx_tr], y[idx_tr])
    v = probe_direction(clf, scaler)

    probe_te = clf.predict_proba(scaler.transform(tr['h'][idx_te]))[:, 1]
    route_thresh = float(np.percentile(probe_te, 100 * (1 - QUERY_BUDGET)))

    p1_spec = P1_ENVS[task]
    domain = p1_spec['domain']
    trajs = collect_trajectories(model, p1_spec, N_TRAJ, cfg, seed=SEED + 900)

    all_sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - K_HEADLINE - 1):
            all_sites.append((ti, t))
    rng = np.random.default_rng(SEED + 900)

    main_sel = rng.choice(len(all_sites), min(N_SITES_MAIN, len(all_sites)), replace=False)
    main_sites = [all_sites[i] for i in main_sel]
    remaining = [i for i in range(len(all_sites)) if i not in set(main_sel.tolist())]
    null_sel = rng.choice(remaining, min(N_SITES_NULL, len(remaining)), replace=False)
    null_sites = [all_sites[i] for i in null_sel]
    print(f"  {len(main_sites)} main sites, {len(null_sites)} null-sweep sites (disjoint)")

    # §5.2: recompute sigma_v on the HELD-OUT intervention pool (main_sites' h_t),
    # not the training distribution
    h_pool = np.array([trajs[ti]['h'][t] for (ti, t) in main_sites])
    sigma_v_heldout = float((h_pool @ v).std())
    sigma_v_train = float((tr['h'] @ v).std())
    print(f"  sigma_v: train-distribution={sigma_v_train:.4f}  "
          f"held-out-pool={sigma_v_heldout:.4f} "
          f"(ratio={sigma_v_heldout/sigma_v_train:.3f})")

    # nearest-neighbor model for §5.5 manifold-distance check, fit on a broader
    # held-out h_t sample (disjoint from main_sites/null_sites by construction
    # since it's drawn from the untouched remainder of the training states)
    nn_ref = tr['h'][idx_te][:5000]
    nn_model = NearestNeighbors(n_neighbors=1).fit(nn_ref)

    def sweep_direction(direction, sites, sigma, lookaheads=LOOKAHEAD):
        """Runs the full lambda sweep for one direction; returns per-lambda dict
        of {probe_by_k, next_step_recon, e_state, mean_dist_to_pop}."""
        out = {}
        for lam_sigma in LAMBDAS_IN_SIGMA:
            lam = lam_sigma * sigma
            probe_ks = {k: [] for k in lookaheads}
            recon_ns, e_states, dists = [], [], []
            h_edits = []
            for (ti, t) in sites:
                trj = trajs[ti]
                h_t = trj['h'][t]
                h_new = amplify(h_t.reshape(1, -1), direction, lam).reshape(-1)
                pk, rn, es = continue_multi_step(model, trj, t, h_new, clf, scaler, domain, lookaheads)
                for k in lookaheads:
                    probe_ks[k].append(pk[k])
                recon_ns.append(rn)
                e_states.append(es)
                h_edits.append(h_new)
            h_edits = np.array(h_edits)
            nn_dist, _ = nn_model.kneighbors(h_edits)
            out[lam_sigma] = dict(
                probe_by_k={k: float(np.nanmean(probe_ks[k])) for k in lookaheads},
                next_step_recon=float(np.nanmean(recon_ns)),
                e_state=float(np.nanmean(e_states)),
                nn_dist_to_pop=float(np.mean(nn_dist)),
                edit_norm=float(lam),  # ||h' - h|| = |lambda| since direction is unit-norm
            )
        return out

    print(f"  running v's full sweep on {len(main_sites)} main sites...")
    v_sweep = sweep_direction(v, main_sites, sigma_v_heldout)
    for lam in LAMBDAS_IN_SIGMA:
        d = v_sweep[lam]
        print(f"    lambda={lam:+.0f}sigma: probe_k0={d['probe_by_k'][0]:.4f} "
              f"probe_k10={d['probe_by_k'][10]:.4f} next_step_recon={d['next_step_recon']:.4f} "
              f"E^state={d['e_state']:.4f} nn_dist={d['nn_dist_to_pop']:.4f}")

    # §5.3: multi-step probe dose-response slopes for v
    v_slopes = {}
    lambdas_arr = np.array(LAMBDAS_IN_SIGMA)
    for k in LOOKAHEAD:
        ys = [v_sweep[l]['probe_by_k'][k] for l in LAMBDAS_IN_SIGMA]
        v_slopes[f'probe_k{k}'] = linear_slope(lambdas_arr, ys)
    v_slopes['next_step_recon'] = linear_slope(lambdas_arr, [v_sweep[l]['next_step_recon'] for l in LAMBDAS_IN_SIGMA])
    v_slopes['e_state'] = linear_slope(lambdas_arr, [v_sweep[l]['e_state'] for l in LAMBDAS_IN_SIGMA])
    v_slopes['nn_dist'] = linear_slope(lambdas_arr, [v_sweep[l]['nn_dist_to_pop'] for l in LAMBDAS_IN_SIGMA])

    print(f"\n  v's dose-response slopes: {v_slopes}")

    # §5.1: 50-random-direction null on the smaller null-sweep site pool
    print(f"\n  building {N_NULL}-direction empirical null on {len(null_sites)} sites...")
    rng_null = np.random.default_rng(6060)
    null_slopes = {key: [] for key in v_slopes}
    for i in range(N_NULL):
        vr = random_matched_direction(rng_null, v.shape[0])
        sweep_r = sweep_direction(vr, null_sites, sigma_v_heldout, lookaheads=[0, 10])
        for key in null_slopes:
            if key.startswith('probe_k') and int(key[7:]) not in (0, 10):
                # for tractability, null sweep only computes k in {0,10}; use
                # those two lookaheads' slopes as the null reference for the
                # corresponding endpoints, and skip mid-lookaheads (5,1) in the
                # null (still reported for v itself above).
                continue
            if key == 'probe_k0':
                ys = [sweep_r[l]['probe_by_k'][0] for l in LAMBDAS_IN_SIGMA]
            elif key == 'probe_k10':
                ys = [sweep_r[l]['probe_by_k'][10] for l in LAMBDAS_IN_SIGMA]
            elif key == 'next_step_recon':
                ys = [sweep_r[l]['next_step_recon'] for l in LAMBDAS_IN_SIGMA]
            elif key == 'e_state':
                ys = [sweep_r[l]['e_state'] for l in LAMBDAS_IN_SIGMA]
            elif key == 'nn_dist':
                ys = [sweep_r[l]['nn_dist_to_pop'] for l in LAMBDAS_IN_SIGMA]
            else:
                continue
            null_slopes[key].append(linear_slope(lambdas_arr, ys))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N_NULL} null directions done", flush=True)

    null_summary = {}
    for key in ['probe_k0', 'probe_k10', 'next_step_recon', 'e_state', 'nn_dist']:
        d = np.array([x for x in null_slopes[key] if np.isfinite(x)])
        if len(d) < 5:
            continue
        c = v_slopes[key]
        z = (c - d.mean()) / (d.std() + 1e-12)
        pct = float((d < c).mean() * 100) if c > d.mean() else float((d > c).mean() * 100)
        null_summary[key] = dict(v_slope=c, null_mean=float(d.mean()), null_std=float(d.std()),
                                  z=float(z), pct_extreme=pct)
        print(f"    {key}: v_slope={c:+.5f}  null={d.mean():+.5f}±{d.std():.5f}  "
              f"z={z:+.2f}  pct_extreme={pct:.0f}%")

    return dict(
        task=task, sigma_v_train=sigma_v_train, sigma_v_heldout=sigma_v_heldout,
        n_main_sites=len(main_sites), n_null_sites=len(null_sites),
        v_sweep=v_sweep, v_slopes=v_slopes, null_summary=null_summary,
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    for task, spec in TASKS.items():
        all_results[task] = run_task_hardening(task, spec, cfg)

    print(f"\n{'='*78}\nADDENDUM §5 SUMMARY\n{'='*78}")
    for task, r in all_results.items():
        ns = r['null_summary']
        print(f"  {task}: sigma_v train={r['sigma_v_train']:.4f} -> held-out={r['sigma_v_heldout']:.4f}")
        for key in ['probe_k0', 'probe_k10', 'e_state', 'next_step_recon', 'nn_dist']:
            if key in ns:
                print(f"    {key}: z={ns[key]['z']:+.2f} pct={ns[key]['pct_extreme']:.0f}%")

    out_path = os.path.join(OUT_DIR, 'addendum_5_1_5_5_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
