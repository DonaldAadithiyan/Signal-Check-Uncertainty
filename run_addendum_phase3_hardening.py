#!/usr/bin/env python3.11
"""
Pre-Scaling Rigor Addendum §2.1, §2.2, §2.3 — Phase 3 SAE hardening.

§2.1 Split-sample re-test of the "best atom" claim
    Phase 3's original feature_behavior_and_incremental() selected the best-of-20
    atom AND reported its correlation/incremental-R^2 on the SAME 3,000-site pool
    -- a max-over-20 selection with no correction, and the project's own numbers
    already show the winner's-curse signature (e.g. reacher's atom: +0.038 in
    Phase 3's original report vs +0.0038 in the independent causal-intervention
    pool, a 10x shrinkage). This re-runs with a clean split: partition sites into
    a selection half and a held-out half BEFORE selecting; report the held-out
    half's numbers as the headline, not the selection half's.

§2.2 v-membership check for the causally-tested atom
    Surfaces each held-out-confirmed atom's reconstruction-weight share in that
    task's v-decomposition (already computed as an intermediate of §9.1's ridge
    solve -- extracted here, not recomputed from scratch).

§2.3 Cross-SAE-seed replication
    Repeats the split-sample atom-selection-and-report procedure independently
    on SAE seeds 1 and 2 (not just seed 0), matching the class-level replication
    standard already used for the diffuse verdict in §9.1.

Runs on the 3 EXISTING trained SAEs (outputs/phase3_sae/sae_seed{0,1,2}.pt) and
the 3 EXISTING frozen RSSM checkpoints. No retraining.
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
from src.probe.sae import TopKSAE, get_all_activations
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import probe_direction
from run_phase3_sae_decomposition import (
    TASKS, N_ATOMS, K_SPARSE, N_TRAJ_EVAL, K_HEADLINE,
    collect_eval_trajectories_and_targets, decompose_direction,
)

OUT_DIR = 'outputs/phase3_sae'
TOP_N_CANDIDATES = 20
SEED = 5050


def load_sae(seed_idx):
    sae = TopKSAE(d_in=256, n_atoms=N_ATOMS, k=K_SPARSE)
    sae.load_state_dict(torch.load(os.path.join(OUT_DIR, f'sae_seed{seed_idx}.pt'),
                                     map_location='cpu'))
    sae.eval()
    return sae


def fit_confusion_direction(spec):
    tr = np.load(spec['training_states'])
    h, kl = tr['h'], tr['kl']
    y = binarise_by_median(kl)
    idx_tr, _ = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=0)
    clf, scaler = train_probe(h[idx_tr], y[idx_tr])
    return probe_direction(clf, scaler)


def split_sample_best_atom(sae, eval_data, top_n=TOP_N_CANDIDATES, seed=SEED):
    """§2.1: select the best atom on a SELECTION half, report its numbers on a
    disjoint HELD-OUT half. Returns (selection_stats, heldout_stats) for the
    atom chosen by the selection half."""
    n = len(eval_data['h'])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    sel_idx, hold_idx = perm[:half], perm[half:]

    def subset(idx):
        return {k: eval_data[k][idx] for k in eval_data}

    sel_data, hold_data = subset(sel_idx), subset(hold_idx)

    # candidate atoms: top-N by activation variance on the SELECTION half only
    acts_sel = get_all_activations(sae, sel_data['h'])
    var_per_atom = acts_sel.var(axis=0)
    candidates = np.argsort(-var_per_atom)[:top_n]

    # pick the best-correlating-with-E^state atom, SELECTION HALF ONLY
    best_atom, best_r_sel = None, -1.0
    sel_rows = {}
    for atom in candidates:
        a = acts_sel[:, atom]
        if a.std() < 1e-8:
            continue
        r_ext, p_ext = pearsonr(a, sel_data['e_state'])
        r_kl, _ = pearsonr(a, sel_data['kl'])
        r_ct, _ = pearsonr(a, sel_data['ct'])
        sel_rows[int(atom)] = dict(r_external=r_ext, p_external=p_ext, r_kl=r_kl, r_ct=r_ct)
        if abs(r_ext) > best_r_sel:
            best_r_sel, best_atom = abs(r_ext), int(atom)

    if best_atom is None:
        return None

    def incr_r2(data, atom_act):
        X_ctrl = np.column_stack([data['kl'], data['recon'], data['ema_recon']])
        X_full = np.column_stack([atom_act, data['kl'], data['recon'], data['ema_recon']])
        y = data['e_state']
        Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
        Xs_full = StandardScaler().fit_transform(X_full)
        r2_ctrl = LinearRegression().fit(Xs_ctrl, y).score(Xs_ctrl, y)
        r2_full = LinearRegression().fit(Xs_full, y).score(Xs_full, y)
        return r2_full - r2_ctrl

    a_sel = acts_sel[:, best_atom]
    selection_stats = dict(
        atom=best_atom, **sel_rows[best_atom],
        incremental_r2=incr_r2(sel_data, a_sel),
    )

    # now measure the SAME atom's numbers on the HELD-OUT half (the honest report)
    acts_hold = get_all_activations(sae, hold_data['h'])
    a_hold = acts_hold[:, best_atom]
    r_ext_h, p_ext_h = pearsonr(a_hold, hold_data['e_state'])
    r_kl_h, _ = pearsonr(a_hold, hold_data['kl'])
    r_ct_h, _ = pearsonr(a_hold, hold_data['ct'])
    heldout_stats = dict(
        atom=best_atom, r_external=r_ext_h, p_external=p_ext_h, r_kl=r_kl_h, r_ct=r_ct_h,
        incremental_r2=incr_r2(hold_data, a_hold),
    )

    return dict(selection=selection_stats, heldout=heldout_stats,
                n_selection=len(sel_idx), n_heldout=len(hold_idx))


def main():
    cfg = XS_CONFIG.copy()
    results = {}

    directions = {task: fit_confusion_direction(spec) for task, spec in TASKS.items()}

    for task, spec in TASKS.items():
        print(f"\n{'='*78}\n{task.upper()} — ADDENDUM §2.1/§2.2/§2.3\n{'='*78}")
        print(f"  collecting eval trajectories...")
        eval_data = collect_eval_trajectories_and_targets(task, spec, cfg, N_TRAJ_EVAL)
        print(f"  {len(eval_data['h'])} sites total")

        per_seed = {}
        for seed_i in range(3):
            sae = load_sae(seed_i)
            split_result = split_sample_best_atom(sae, eval_data)
            if split_result is None:
                continue

            sel, hold = split_result['selection'], split_result['heldout']
            shrinkage = (abs(hold['incremental_r2']) / abs(sel['incremental_r2'])
                        if abs(sel['incremental_r2']) > 1e-9 else float('nan'))

            # §2.2: v-membership share for the held-out-confirmed atom
            v = directions[task]
            decomp = decompose_direction(sae, v)
            atom_id = hold['atom']
            if atom_id in decomp['top10_atoms']:
                rank = decomp['top10_atoms'].index(atom_id)
                v_share = decomp['top10_shares'][rank]
            else:
                v_share = None  # not in top-10; negligible share

            print(f"  SAE seed {seed_i}: selected atom #{sel['atom']} "
                  f"(selection r_ext={sel['r_external']:+.3f}, incr_R2={sel['incremental_r2']:+.4f})")
            print(f"    held-out:  r_ext={hold['r_external']:+.3f} (p={hold['p_external']:.2g}), "
                  f"r_kl={hold['r_kl']:+.3f}, r_ct={hold['r_ct']:+.3f}, "
                  f"incr_R2={hold['incremental_r2']:+.4f}  "
                  f"(shrinkage factor from selection: {shrinkage:.2f}x)")
            print(f"    v-membership: atom's share of v's reconstruction weight = "
                  f"{f'{v_share*100:.2f}%' if v_share is not None else '<top-10 threshold (negligible)'}")

            per_seed[seed_i] = dict(
                selection=sel, heldout=hold, shrinkage_factor=shrinkage,
                v_reconstruction_share=v_share,
            )

        # cross-seed qualitative pattern check: does "near-zero KL correlation +
        # meaningful held-out external correlation" replicate across seeds?
        kl_non_redundant_flags = []
        for seed_i, r in per_seed.items():
            h = r['heldout']
            kl_non_redundant = abs(h['r_kl']) < 0.1 and abs(h['r_external']) > 0.1
            kl_non_redundant_flags.append(kl_non_redundant)
        replicates = all(kl_non_redundant_flags) if kl_non_redundant_flags else False
        any_holds = any(kl_non_redundant_flags) if kl_non_redundant_flags else False

        results[task] = dict(per_seed=per_seed,
                             kl_non_redundant_replicates_all_seeds=replicates,
                             kl_non_redundant_holds_any_seed=any_holds)
        print(f"\n  cross-seed: KL-non-redundant + externally-predictive pattern "
              f"replicates on ALL 3 SAE seeds? {replicates}  "
              f"(holds on at least 1 seed? {any_holds})")

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for task, r in results.items():
        seed0 = r['per_seed'].get(0)
        if seed0:
            sel_r2 = seed0['selection']['incremental_r2']
            hold_r2 = seed0['heldout']['incremental_r2']
            print(f"  {task}: seed0 selection incr_R2={sel_r2:+.4f} -> "
                  f"held-out incr_R2={hold_r2:+.4f}  "
                  f"(shrinkage {seed0['shrinkage_factor']:.2f}x)  "
                  f"replicates all seeds: {r['kl_non_redundant_replicates_all_seeds']}")

    out_path = os.path.join(OUT_DIR, 'addendum_2_1_2_2_2_3_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
