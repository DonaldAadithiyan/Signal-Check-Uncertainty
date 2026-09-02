#!/usr/bin/env python3.11
"""
Phase 3b — Atom-level causal ablation & causal external-validity test
(Reframed_Confusion_RSSM_Project, Section 9.5-9.7).

Depends on run_phase3_sae_decomposition.py having produced:
  outputs/phase3_sae/sae_seed0.pt          (pooled TopK SAE, seed 0)
  outputs/phase3_sae/phase3_results.json   (feature_behavior: best externally-
                                             predictive atom per task, from §9.3/9.4)

§9.5 Atom-level causal ablation
    For the top candidate atom per task (from §9.3's r(atom, E^state) ranking),
    ablate its DECODER COLUMN direction from h_t (same mechanism as Task G's
    confusion-direction ablation: h' = h - (h.d)d for unit decoder column d),
    continue the trajectory, and measure the probe-readout effect at k in {0,1,5,10}
    against the EXISTING 50-random-direction empirical null (reuses
    src/probe/intervention.py's helpers and Task G's protocol exactly).

§9.6 Causal external-validity test (the key NEW experiment this phase adds)
    Beyond the probe-readout effect (which Task G already established for the
    dense confusion direction v), ask whether ablating the atom changes the
    EXTERNAL failure prediction itself: does E^state_{t,K} computed from the
    ablated h_t's continued imagination differ from the unablated continuation?
    The strong version of the claim requires BOTH:
        atom ablated -> probe/confusion readout drops (already tested in §9.5)
        atom ablated -> the imagined trajectory's E^state-relevant behavior changes
    We do not claim this if only the probe readout changes.

§9.7 Decisive targeted test
    If the atom is the non-redundant, externally-predictive component identified
    in §9.3/9.4 (positive incremental R^2 over KL/Recon/EMA), test whether
    ablating it REMOVES that incremental advantage: recompute the incremental-R^2
    test (same protocol as run_phase3_sae_decomposition.py's feature_behavior_and_
    incremental) but using the ABLATED continuation's re-encoded activation in
    place of the atom's original activation. If the incremental R^2 collapses
    toward zero, the atom is causally load-bearing for the SPECIFIC advantage
    C_t/features have over KL-alone -- not just correlated with it.

Runs on the EXISTING frozen models + the EXISTING trained SAE. No retraining.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.probe.sae import TopKSAE
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct, random_matched_direction
from run_phase1_external_validation import (
    ENVS as P1_ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, MIN_T, SEED,
)
from run_phase3_sae_decomposition import TASKS, N_ATOMS, K_SPARSE

N_NULL = 50
LOOKAHEAD = [0, 1, 5, 10]
K_HEADLINE = 10
N_TRAJ_CAUSAL = 30
OUT_DIR = 'outputs/phase3_sae'
SAE_PATH = os.path.join(OUT_DIR, 'sae_seed0.pt')
RESULTS_PATH = os.path.join(OUT_DIR, 'phase3_results.json')


def load_sae():
    sae = TopKSAE(d_in=256, n_atoms=N_ATOMS, k=K_SPARSE)
    sae.load_state_dict(torch.load(SAE_PATH, map_location='cpu'))
    sae.eval()
    return sae


@torch.no_grad()
def continue_probe_and_external(model, cfg, traj, t, h_new, clf, sc, domain, horizon=K_HEADLINE):
    """Continue the trajectory from t with h replaced by h_new (posterior
    continuation, using real observations -- same as Task G's continue_probe),
    THEN also roll pure IMAGINATION forward `horizon` steps from the continued
    state to get an ablated-condition E^state. Returns (probe_scores[0..max(LOOKAHEAD)],
    e_state_ablated)."""
    device = next(model.parameters()).device
    T = len(traj['obs'])
    t_end = min(T, t + max(LOOKAHEAD) + 1)

    h = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    emb = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    z = model.rssm._straight_through_sample(post_l)

    hs = [h.squeeze(0).cpu().numpy().copy()]
    zs = [z.squeeze(0).cpu().numpy().copy()]
    for k in range(t + 1, t_end):
        a = torch.tensor(traj['act'][k - 1], dtype=torch.float32, device=device).unsqueeze(0)
        obs_k = torch.tensor(traj['obs'][k], dtype=torch.float32, device=device).unsqueeze(0)
        emb = model.encoder(obs_k)
        h, z, _, _ = model.rssm.observe_step(h, z, a, emb)
        hs.append(h.squeeze(0).cpu().numpy().copy())
        zs.append(z.squeeze(0).cpu().numpy().copy())
    hs = np.array(hs, np.float32)
    ps = clf.predict_proba(sc.transform(hs))[:, 1]

    # roll pure imagination forward `horizon` steps from the ABLATED h at t (using
    # real actions), decode, compare to real obs -- the causal external-validity
    # target (same construction as Phase 1's imagined_vs_real_obs, but starting
    # from the intervened h_new instead of the natural posterior h_t)
    h_im = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    z_im = z  # reuse the posterior z at t computed above (intervention is on h only)
    state_dist = []
    for k in range(1, horizon + 1):
        kk = t + k
        if kk >= T:
            break
        a = torch.tensor(traj['act'][kk - 1], dtype=torch.float32, device=device).unsqueeze(0)
        h_im, z_im, _ = model.rssm.imagine_step(h_im, z_im, a)
        dec_imag = model.decoder(torch.cat([h_im, z_im], dim=-1)).squeeze(0).cpu().numpy()
        state_dist.append(float(np.linalg.norm(dec_imag - traj['obs'][kk])))
    e_state_ablated = float(np.mean(state_dist)) if len(state_dist) == horizon else np.nan

    return ps, e_state_ablated


def effect_for_direction(model, cfg, trajs, sites, v, clf, sc, domain, ps_base_cache, e_state_base_cache):
    dprobe = {k: [] for k in LOOKAHEAD}
    d_e_state = []
    for (ti, t) in sites:
        trj = trajs[ti]
        h_t = trj['h'][t]
        proj = float(h_t @ v)
        h_abl = h_t - proj * v
        ps, e_state_abl = continue_probe_and_external(model, cfg, trj, t, h_abl, clf, sc, domain)
        ps_base = ps_base_cache[(ti, t)]
        e_state_base = e_state_base_cache[(ti, t)]
        for k in LOOKAHEAD:
            if k < len(ps) and k < len(ps_base):
                dprobe[k].append(ps[k] - ps_base[k])
        if not (np.isnan(e_state_abl) or np.isnan(e_state_base)):
            d_e_state.append(e_state_abl - e_state_base)
    out = {f'dprobe_{k}': float(np.mean(dprobe[k])) for k in LOOKAHEAD}
    out['d_e_state'] = float(np.mean(d_e_state)) if d_e_state else float('nan')
    out['n_e_state'] = len(d_e_state)
    return out


def run_task_causal(task, spec, cfg, sae, best_atom_info):
    print(f"\n{'='*78}\n{task.upper()} — §9.5-9.7 CAUSAL FEATURE TESTS\n{'='*78}")
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    y = binarise_by_median(tr['kl'])
    idx_tr, _ = train_test_split(np.arange(len(tr['h'])), test_size=0.40,
                                  stratify=y, random_state=0)
    clf, sc = train_probe(tr['h'][idx_tr], y[idx_tr])
    kl_median = float(np.median(tr['kl']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])

    p1_spec = P1_ENVS[task]
    domain = p1_spec['domain']
    trajs = collect_trajectories(model, p1_spec, N_TRAJ_CAUSAL, cfg, seed=SEED + 500)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64),
                                gamma=spec['gamma_ct'])
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - K_HEADLINE - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED + 500)
    if len(sites) > 800:
        sel = rng.choice(len(sites), 800, replace=False)
        sites = [sites[i] for i in sel]
    print(f"  {len(sites)} intervention sites")

    # baseline (unablated) continuations, cached once
    ps_base_cache, e_state_base_cache = {}, {}
    for (ti, t) in sites:
        trj = trajs[ti]
        ps, e_st = continue_probe_and_external(model, cfg, trj, t, trj['h'][t], clf, sc, domain)
        ps_base_cache[(ti, t)] = ps
        e_state_base_cache[(ti, t)] = e_st

    # ── §9.5/9.6: ablate the top externally-predictive atom's decoder column ──
    atom_idx = best_atom_info['atom']
    with torch.no_grad():
        d_atom = sae.decoder.weight[:, atom_idx].numpy().copy()  # already unit-norm
    print(f"  target atom #{atom_idx} (r(atom,E^state)={best_atom_info['r_external']:+.3f} "
          f"from §9.3/9.4)")

    conf = effect_for_direction(model, cfg, trajs, sites, d_atom, clf, sc, domain,
                                 ps_base_cache, e_state_base_cache)
    print(f"  atom ablation effect: dprobe_0={conf['dprobe_0']:+.4f}  "
          f"d_E^state={conf['d_e_state']:+.4f} (n={conf['n_e_state']})")

    # empirical null: 50 random directions
    print(f"  building {N_NULL}-direction empirical null...")
    rng_null = np.random.default_rng(3033)
    null = {f'dprobe_{k}': [] for k in LOOKAHEAD}
    null['d_e_state'] = []
    for i in range(N_NULL):
        vr = random_matched_direction(rng_null, d_atom.shape[0])
        e = effect_for_direction(model, cfg, trajs, sites, vr, clf, sc, domain,
                                  ps_base_cache, e_state_base_cache)
        for key in null:
            null[key].append(e[key])
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N_NULL} null directions done", flush=True)
    for key in null:
        null[key] = np.array(null[key])

    null_summary = {}
    for key in [f'dprobe_{k}' for k in LOOKAHEAD] + ['d_e_state']:
        c = conf[key]
        d = null[key][~np.isnan(null[key])]
        if len(d) < 5:
            null_summary[key] = dict(confusion=float(c), null_mean=float('nan'),
                                      null_std=float('nan'), z=float('nan'), pct_extreme=float('nan'))
            continue
        z = (c - d.mean()) / (d.std() + 1e-12)
        pct = float((d > c).mean() * 100) if c < 0 else float((d < c).mean() * 100)
        null_summary[key] = dict(confusion=float(c), null_mean=float(d.mean()),
                                  null_std=float(d.std()), z=float(z), pct_extreme=pct)
        print(f"    {key}: atom={c:+.4f}  null={d.mean():+.4f}±{d.std():.4f}  "
              f"z={z:+.1f}  pct_extreme={pct:.0f}%")

    catches_probe = null_summary['dprobe_0']['z'] < -2.0 if np.isfinite(null_summary['dprobe_0']['z']) else False
    catches_external = (np.isfinite(null_summary['d_e_state']['z'])
                        and abs(null_summary['d_e_state']['z']) > 2.0)
    verdict_96 = ('BOTH probe AND external readout causally affected (strong evidence)'
                  if catches_probe and catches_external else
                  'probe readout affected but external readout NOT clearly affected '
                  '(NOT claiming external causal effect)' if catches_probe else
                  'neither readout clearly affected beyond the random-direction null')
    print(f"  §9.6 verdict: {verdict_96}")

    # ── §9.7: does ablating the atom remove C_t's incremental advantage over KL? ──
    # recompute incremental R^2 using ABLATED atom activations (re-encode the
    # ablated h through the SAE) vs baseline (unablated) atom activations, both
    # against E^reward-free target E^state, controlling for KL/Recon/EMA.
    h_base = np.array([trajs[ti]['h'][t] for (ti, t) in sites])
    kl_vals = np.array([trajs[ti]['kl'][t] for (ti, t) in sites])
    recon_vals = np.array([trajs[ti]['recon'][t] for (ti, t) in sites])
    ema_vals = np.array([trajs[ti]['ema_recon'][t] for (ti, t) in sites])
    target = np.array([e_state_base_cache[(ti, t)] for (ti, t) in sites])
    valid = ~np.isnan(target)
    h_base, kl_vals, recon_vals, ema_vals, target = (
        h_base[valid], kl_vals[valid], recon_vals[valid], ema_vals[valid], target[valid])

    with torch.no_grad():
        acts_base, _, _ = sae.encode(torch.tensor(h_base, dtype=torch.float32))
        atom_act_base = acts_base[:, atom_idx].numpy()

        h_ablated = h_base - (h_base @ d_atom)[:, None] * d_atom[None, :]
        acts_abl, _, _ = sae.encode(torch.tensor(h_ablated, dtype=torch.float32))
        atom_act_ablated = acts_abl[:, atom_idx].numpy()

    def incr_r2(atom_act):
        X_ctrl = np.column_stack([kl_vals, recon_vals, ema_vals])
        X_full = np.column_stack([atom_act, kl_vals, recon_vals, ema_vals])
        Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
        Xs_full = StandardScaler().fit_transform(X_full)
        r2_ctrl = LinearRegression().fit(Xs_ctrl, target).score(Xs_ctrl, target)
        r2_full = LinearRegression().fit(Xs_full, target).score(Xs_full, target)
        return r2_full - r2_ctrl

    incr_before = incr_r2(atom_act_base)
    incr_after = incr_r2(atom_act_ablated)
    print(f"  §9.7: incremental R^2 of atom-feature over {{KL,Recon,EMA}} for E^state: "
          f"before ablation={incr_before:+.4f}  after ablation={incr_after:+.4f}  "
          f"(atom activation post-ablation is near-zero by construction: "
          f"mean|act|={np.abs(atom_act_ablated).mean():.4f} vs "
          f"{np.abs(atom_act_base).mean():.4f} before)")

    removed = incr_before > 0.002 and incr_after < 0.3 * incr_before
    print(f"  -> ablation {'REMOVES' if removed else 'does NOT clearly remove'} "
          f"the incremental advantage")

    return dict(
        task=task, atom=int(atom_idx), n_sites=len(sites),
        ablation_effect=conf, null_summary=null_summary, verdict_9_6=verdict_96,
        incremental_r2_before_ablation=incr_before, incremental_r2_after_ablation=incr_after,
        removed_incremental_advantage=bool(removed),
        atom_activation_mean_before=float(np.abs(atom_act_base).mean()),
        atom_activation_mean_after=float(np.abs(atom_act_ablated).mean()),
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SAE_PATH):
        raise FileNotFoundError(f"{SAE_PATH} not found -- run run_phase3_sae_decomposition.py first")
    sae = load_sae()

    with open(RESULTS_PATH) as f:
        phase3 = json.load(f)

    all_results = {}
    for task, spec in TASKS.items():
        fb = phase3['feature_behavior'][task]
        best_atom_info = fb['top_atoms_by_r_external'][0]
        all_results[task] = run_task_causal(task, spec, cfg, sae, best_atom_info)

    print(f"\n{'='*78}\nPHASE 3B SUMMARY\n{'='*78}")
    for task, r in all_results.items():
        print(f"  {task}: atom #{r['atom']} -- §9.6: {r['verdict_9_6']}")
        print(f"           §9.7: incremental advantage removed by ablation? "
              f"{r['removed_incremental_advantage']} "
              f"({r['incremental_r2_before_ablation']:+.4f} -> "
              f"{r['incremental_r2_after_ablation']:+.4f})")

    out_path = os.path.join(OUT_DIR, 'phase3b_causal_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
