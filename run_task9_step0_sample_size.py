#!/usr/bin/env python3.11
"""
Task 9 (mechanism-guided blind-spot discovery) -- Step 0: sample-size /
power check, run FIRST, before any characterization or causal work.

Uses ONLY frozen, already-verified mechanisms (reacher atom #612, cartpole
atom #156, both from outputs/phase3_sae/sae_seed0.pt) -- no atom rediscovery,
no new SAE training.

Critical requirement: a FRESH held-out evaluation set, disjoint from every
split used to select, train, or causally test these atoms. Prior seed usage
in this project's Phase 3 pipeline:
  - Atom selection / feature-behavior dataset (collect_eval_trajectories_
    and_targets): seed=SEED (2024)
  - Causal ablation test (run_task_causal): seed=SEED+500 (2524)
This script uses seed=SEED+9000 (11024) -- a seed never used anywhere in
Phase 3's atom-selection or causal-testing pipeline, chosen specifically to
avoid circularity (reusing training/development data here would make the
experiment "we chose an atom because it causes error, then showed it causes
error").
"""

import os
import json
import numpy as np
import torch

from src.config import XS_CONFIG
from src.probe.sae import TopKSAE, get_all_activations
from src.probe.intervention import compute_ct
from run_phase1_external_validation import (
    ENVS as P1_ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, MIN_T, MAX_HORIZON, SEED,
)

OUT_DIR = 'outputs/task9_blind_spot'
FRESH_SEED = SEED + 9000   # never used in atom selection (SEED) or causal testing (SEED+500)
N_TRAJ_FRESH = 100         # matches Phase 1's own N_TRAJ scale
K_HEADLINE = 10
N_ATOMS = 2048
K_SPARSE = 48
SAE_PATH = 'outputs/phase3_sae/sae_seed0.pt'

ATOM_612 = 612   # reacher, KL-non-redundant, causal on external quality only
ATOM_156 = 156   # cartpole, KL-redundant, causal on both readouts


def load_sae():
    sae = TopKSAE(d_in=256, n_atoms=N_ATOMS, k=K_SPARSE)
    sae.load_state_dict(torch.load(SAE_PATH, map_location='cpu'))
    sae.eval()
    return sae


def build_fresh_dataset(task, spec, cfg, atom_idx, sae):
    """Collects a FRESH held-out set (seed=FRESH_SEED, never used in atom
    selection or causal testing) and computes per-site: atom activation,
    C_t, KL_t, Recon_t, and the accumulated imagination error E^state_{t,10}."""
    torch.manual_seed(FRESH_SEED)
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])

    p1_spec = P1_ENVS[task]
    trajs = collect_trajectories(model, p1_spec, N_TRAJ_FRESH, cfg, seed=FRESH_SEED)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64),
                                gamma=spec['gamma_ct'])
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

    h_list, kl_list, ct_list, recon_list, estate_list = [], [], [], [], []
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, _, _ = imagined_vs_real_obs(model, trj, t, MAX_HORIZON, p1_spec['domain'])
        if len(state_dist_full) < K_HEADLINE:
            continue
        h_list.append(trj['h'][t])
        kl_list.append(trj['kl'][t])
        ct_list.append(trj['ct'][t])
        recon_list.append(trj['recon'][t])
        estate_list.append(e_state(state_dist_full, K_HEADLINE))

    h_arr = np.array(h_list, dtype=np.float32)
    acts = get_all_activations(sae, h_arr)
    atom_activation = acts[:, atom_idx]

    return dict(
        h=h_arr, kl=np.array(kl_list, np.float64), ct=np.array(ct_list, np.float64),
        recon=np.array(recon_list, np.float64), e_state=np.array(estate_list, np.float64),
        atom_activation=atom_activation,
        n_trajectories=len(trajs), fresh_seed=FRESH_SEED,
    )


FIRE_RATE_SPARSE_CUTOFF = 0.50   # decided before inspecting either atom's own
                                  # group counts: an atom firing on <50% of
                                  # sites is "sparse-coded" for this purpose
                                  # and gets the fired/not-fired rule; an atom
                                  # firing on >=50% of sites gets the
                                  # percentile rule. This single cutoff is
                                  # applied identically to both atoms -- it is
                                  # not fit separately to each atom's numbers.


def sample_size_check(task, atom_idx, data, atom_pctile=75, ct_pctile=50):
    """Fixed rule, decided before looking at GROUP counts (though after
    observing each atom's own fire rate, which is a property of the atom
    itself, not of the blind-spot grouping this check exists to test):

    A TopK-SAE atom is sparse-coded (k=48 of 2048 atoms fire per site), so
    its activation distribution across sites can be a large mass point at
    exactly zero (the atom did not fire) plus a continuous tail above zero
    (it did) -- or, for an atom with a very high fire rate, close to a
    smooth continuous distribution with no meaningful mass at zero.
    Verified directly: reacher #612 fires on 22% of fresh sites (sparse);
    cartpole #156 fires on 100% (not sparse, in this classification). A
    SINGLE fixed rule cannot be correct for both regimes -- a percentile
    threshold on a sparse atom degenerates to 0.0 (selects everything above
    it, i.e. everything), and a fired/not-fired rule on a universally-firing
    atom is degenerate the other way (selects everything). The general
    principle, decided before applying it to either atom's specific numbers:
    if the atom's fire rate is below FIRE_RATE_SPARSE_CUTOFF, use "fired at
    all" (activation > 0) as the high-activation rule; otherwise use the
    percentile rule (top quartile, >= 75th percentile). 'C_t low' = bottom
    half (<=50th percentile) in both cases, unchanged."""
    atom = data['atom_activation']
    ct = data['ct']
    n = len(atom)

    fire_rate = float((atom > 0).mean())
    if fire_rate < FIRE_RATE_SPARSE_CUTOFF:
        atom_rule = 'sparse: activation > 0 (fired at all)'
        atom_thresh = 0.0
        high_atom = atom > atom_thresh
        low_atom = atom <= atom_thresh
    else:
        atom_rule = f'dense: top quartile (>= {atom_pctile}th percentile)'
        atom_thresh = np.percentile(atom, atom_pctile)
        high_atom = atom >= atom_thresh
        low_atom = atom < atom_thresh

    ct_thresh = np.percentile(ct, ct_pctile)
    low_ct = ct <= ct_thresh
    high_ct = ct > ct_thresh

    groups = dict(
        low_atom_low_ct=int((low_atom & low_ct).sum()),
        high_atom_high_ct=int((high_atom & high_ct).sum()),
        high_atom_low_ct_CANDIDATE=int((high_atom & low_ct).sum()),
        low_atom_high_ct_matched_control=int((low_atom & high_ct).sum()),
    )

    n_candidate = groups['high_atom_low_ct_CANDIDATE']
    n_control = groups['low_atom_high_ct_matched_control']

    # rough power calculation: using the already-established causal effect
    # sizes for these atoms (Cohen's-d-style, from the existing null-vs-
    # confusion-direction ablation tests) as the expected effect magnitude,
    # estimate whether n_candidate vs n_control has power to detect a
    # difference in E^state means at alpha=0.05 (two-sided), using the
    # ALREADY-OBSERVED E^state distribution's own std as the noise estimate
    # (conservative: uses this fresh dataset's actual variance, not an
    # assumed one).
    estate_std = float(np.std(data['e_state']))
    # effect size proxy: established causal ablation effect for this atom
    # (see docstring) -- d_e_state values from the already-verified causal
    # tests, used here only to estimate whether THIS group size could detect
    # an effect of THAT rough magnitude, not to re-assert the causal claim.
    # verified directly from outputs/phase3_sae/phase3b_causal_results.json
    # (reacher) and phase3b_causal_results_atom156.json (cartpole), the
    # ablation_effect.d_e_state fields -- NOT the incremental-R^2 fields,
    # which are a different quantity and were mistakenly used in an earlier
    # draft of this script.
    established_effect = dict(
        atom612=0.03207745676673949,   # reacher #612's verified ablation-mean delta E^state
        atom156=0.11115866609639488,   # cartpole #156's verified ablation-mean delta E^state
    )
    effect_key = 'atom612' if atom_idx == ATOM_612 else 'atom156'
    assumed_delta = established_effect[effect_key]

    # standard two-sample mean-difference power approximation (normal
    # approximation, not a full simulation -- explicitly a ROUGH estimate,
    # per the task's own framing)
    if n_candidate > 1 and n_control > 1 and estate_std > 0:
        se = estate_std * np.sqrt(1.0 / n_candidate + 1.0 / n_control)
        z_effect = assumed_delta / se
        # power at alpha=0.05 two-sided: Phi(z_effect - 1.96) + Phi(-z_effect - 1.96)
        from scipy.stats import norm
        power = float(norm.cdf(z_effect - 1.96) + norm.cdf(-z_effect - 1.96))
    else:
        z_effect, power = float('nan'), float('nan')

    return dict(
        task=task, atom=atom_idx, n_total_sites=n,
        atom_activation_threshold=float(atom_thresh), ct_threshold=float(ct_thresh),
        atom_fire_rate=fire_rate, atom_rule=atom_rule, ct_pctile_rule=ct_pctile,
        group_counts=groups, n_candidate=n_candidate, n_control=n_control,
        e_state_std_fresh_data=estate_std, assumed_effect_size=assumed_delta,
        rough_power_estimate=power, z_effect_snr=float(z_effect) if np.isfinite(z_effect) else None,
    )


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    sae = load_sae()

    results = {}

    print(f"\n{'='*78}\nSTEP 0 — SAMPLE-SIZE / POWER CHECK (fresh_seed={FRESH_SEED})\n{'='*78}")

    print("\n--- reacher #612 (primary blind-spot candidate) ---")
    data_reacher = build_fresh_dataset('reacher', P1_ENVS['reacher'], cfg, ATOM_612, sae)
    check_reacher = sample_size_check('reacher', ATOM_612, data_reacher)
    print(json.dumps(check_reacher, indent=2, default=float))
    results['reacher_612'] = check_reacher
    np.savez(os.path.join(OUT_DIR, 'fresh_data_reacher.npz'), **data_reacher)

    print("\n--- cartpole #156 (comparison, causal-on-both-readouts atom) ---")
    data_cartpole = build_fresh_dataset('cartpole', P1_ENVS['cartpole'], cfg, ATOM_156, sae)
    check_cartpole = sample_size_check('cartpole', ATOM_156, data_cartpole)
    print(json.dumps(check_cartpole, indent=2, default=float))
    results['cartpole_156'] = check_cartpole
    np.savez(os.path.join(OUT_DIR, 'fresh_data_cartpole.npz'), **data_cartpole)

    print(f"\n{'='*78}\nDECISION POINT\n{'='*78}")
    for key, check in results.items():
        n_cand = check['n_candidate']
        power = check['rough_power_estimate']
        power_ok = power is not None and np.isfinite(power) and power >= 0.5
        verdict = "PROCEED" if (n_cand >= 30 and power_ok) else "STOP — underpowered"
        power_str = f"{power:.3f}" if (power is not None and np.isfinite(power)) else str(power)
        print(f"  {key}: n_candidate={n_cand}, n_control={check['n_control']}, "
              f"rough_power={power_str}  ->  {verdict}")
        results[key]['decision'] = verdict

    out_path = os.path.join(OUT_DIR, 'step0_sample_size_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
