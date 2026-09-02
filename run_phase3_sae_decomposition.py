#!/usr/bin/env python3.11
"""
Phase 3 — SAE mechanistic decomposition (Reframed_Confusion_RSSM_Project, Section 8-9).

Runs on Gate 1 having passed (Phase 1: outputs/deliverables/phase1_external_validation.md,
3/3 tasks) and the mandatory rigor controls (Phase 2: temporal order matters for
explaining h_t on all 3 tasks, and for external prediction on 2/3).

The question this phase answers (Section 8.1):
    Which sparse features compose the representation of predictive difficulty, and
    which of those features predict EXTERNAL world-model failure rather than merely
    reproducing KL?

Pipeline:
  §8.2  Train a pooled TopK SAE on h_t across all 3 tasks (256-dim, unit-norm
        decoder columns, dead-feature AuxK recovery), across N_SAE_SEEDS seeds.
        Report mandatory metrics: L0, dead-atom %, variance explained.
  §9.1  Decompose the confusion direction v: project v onto the dictionary, rank
        atoms by |cos(atom, v)| and by their contribution to reconstructing v
        itself. Classify monosemantic / oligosemantic / diffuse using a fixed,
        pre-registered rule (not eyeballed after seeing the numbers): monosemantic
        if top-1 atom's coefficient share of ||v||-reconstruction >= 0.5;
        oligosemantic if top-4 share >= 0.8; else diffuse.
  §9.2  Cross-task composition: using the POOLED dictionary, compare which atoms
        fire most / most distinctively on each task's h_t, and how similar each
        task's "top atoms" are to v.
  §9.3  Feature behavior: for every atom, correlate its activation with KL_t, C_t,
        Recon_t, and (reusing Phase 1's already-collected external targets) with
        E^state_{t,K=10} on a fresh held-out trajectory sample per task.
  §9.4  Incremental feature analysis: for the top candidate feature(s), test
        whether they retain predictive power on E^state after controlling for
        KL_t, Recon_t, EMARecon_t (same protocol as Phase 1/2's incremental-R^2
        tests) -- the feature-level version of the circularity check.

Seeds: N_SAE_SEEDS=3 (Section 8.2's "at least 2-3"). Atom IDENTITY is not expected
to align across seeds (Section 8.2); this reports CLASS/BEHAVIOR-level replication
(does a "high similarity-to-v, high external validity" atom exist in every seed),
not "does atom #417 recur".

CPU only. No retraining of the base RSSMs.
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
from src.probe.sae import train_topk_sae, sae_metrics, get_all_activations
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct, probe_direction
from run_phase1_external_validation import (
    ENVS as P1_ENVS, load_model, collect_trajectories, imagined_vs_real_obs, e_state,
    fit_ema_alpha, ema_series, MIN_T, SEED,
)

N_ATOMS = 2048          # expansion ratio 8x for d_in=256
K_SPARSE = 48
N_EPOCHS = 40
BATCH_SIZE = 1024
N_SAE_SEEDS = 3
N_TRAJ_EVAL = 40         # trajectories per task for feature-behavior / incremental analysis
K_HEADLINE = 10
OUT_DIR = 'outputs/phase3_sae'
FIG_DIR = 'outputs/figures'

TASKS = {
    'cartpole': dict(training_states='outputs/data/training_states.npz',
                      checkpoint='outputs/checkpoints/world_model.pt', gamma_ct=0.95),
    'reacher':  dict(training_states='outputs/second_env/reacher_easy_training_states.npz',
                      checkpoint='outputs/second_env/reacher_easy_world_model.pt', gamma_ct=0.70),
    'pendulum': dict(training_states='outputs/third_env/pendulum_swingup_training_states.npz',
                      checkpoint='outputs/third_env/pendulum_swingup_world_model.pt', gamma_ct=0.90),
}


def load_pooled_h():
    """Pool h_t across all 3 tasks (Section 8.2 primary training set)."""
    pooled_h, pooled_task, pooled_kl, pooled_ct, pooled_recon = [], [], [], [], []
    per_task = {}
    for task, spec in TASKS.items():
        tr = dict(np.load(spec['training_states']))
        traj_id = tr.get('traj_id', np.zeros(len(tr['h']), dtype=np.int64))
        ct = compute_ct(tr['kl'], traj_id, gamma=spec['gamma_ct'])
        per_task[task] = dict(h=tr['h'], kl=tr['kl'], recon=tr['recon'], ct=ct, traj_id=traj_id)
        pooled_h.append(tr['h'])
        pooled_task.append(np.full(len(tr['h']), task))
        pooled_kl.append(tr['kl'])
        pooled_ct.append(ct)
        pooled_recon.append(tr['recon'])
    return (np.concatenate(pooled_h), np.concatenate(pooled_task),
            np.concatenate(pooled_kl), np.concatenate(pooled_ct),
            np.concatenate(pooled_recon), per_task)


def fit_confusion_directions(per_task):
    """Per-task probe direction v (Section 9.1 operates per-task, since v itself
    is task-specific -- confirmed by the workshop paper's own per-task geometry)."""
    directions = {}
    for task, d in per_task.items():
        y = binarise_by_median(d['kl'])
        idx_tr, _ = train_test_split(np.arange(len(d['h'])), test_size=0.40,
                                      stratify=y, random_state=0)
        clf, scaler = train_probe(d['h'][idx_tr], y[idx_tr])
        v = probe_direction(clf, scaler)
        directions[task] = dict(v=v, clf=clf, scaler=scaler)
    return directions


def decompose_direction(sae, v, task_h_mean_subtracted_scale=1.0):
    """Section 9.1: project v onto the dictionary. Since the SAE decoder columns
    are unit-norm atoms d_i, and v is itself unit-norm, we express v in the atom
    basis via least-squares (v ~ sum_i c_i d_i, c_i unconstrained sign) restricted
    to the TopK atoms that best reconstruct v alone (v encoded as if it were an
    input point, mean-centered by the SAE's own b_pre direction removed since v is
    a raw direction, not an activation)."""
    with torch.no_grad():
        v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
        # encode v as a direction: skip b_pre subtraction (v has no natural origin),
        # use raw encoder pre-activations to rank atoms by alignment
        pre_acts = torch.relu(sae.encoder(v_t)).squeeze(0)  # not literal recon target,
        # but a proxy for "which atoms' encoder rows point toward v"
        dec = sae.decoder.weight  # (d_in, n_atoms), unit-norm columns
        cos_sim = (dec.t() @ v_t.squeeze(0)) / (v_t.norm() + 1e-8)  # cos(atom_i, v)
        cos_sim = cos_sim.numpy()

        # least-squares reconstruction of v using ALL atoms (ridge-regularized to
        # handle the 2048 >> 256 rank deficiency), then rank by |coefficient|*||atom||
        # contribution to explained variance of v
        D = dec.numpy()  # (256, 2048), unit-norm columns
        # ridge solve: c = (D^T D + lambda I)^-1 D^T v
        lam = 1e-2
        DtD = D.T @ D
        c = np.linalg.solve(DtD + lam * np.eye(D.shape[1]), D.T @ v)
        contrib = np.abs(c)  # since ||d_i||=1, contribution to ||v||-direction ~ |c_i|
        contrib_share = contrib / (contrib.sum() + 1e-12)

    order = np.argsort(-contrib_share)
    top1_share = float(contrib_share[order[0]])
    top4_share = float(contrib_share[order[:4]].sum())

    if top1_share >= 0.5:
        classification = 'monosemantic'
    elif top4_share >= 0.8:
        classification = 'oligosemantic'
    else:
        classification = 'diffuse'

    return dict(
        classification=classification, top1_share=top1_share, top4_share=top4_share,
        top10_atoms=order[:10].tolist(), top10_shares=contrib_share[order[:10]].tolist(),
        top10_cos_sim=cos_sim[order[:10]].tolist(),
    )


def collect_eval_trajectories_and_targets(task, spec, cfg, n_traj):
    """Reuses Phase 1 machinery: real trajectories, imagined-vs-real E^state,
    plus KL/recon/C_t/EMA at each site -- the per-site feature-behavior dataset."""
    model, obs_dim, act_dim = load_model(spec['checkpoint'])
    tr = dict(np.load(spec['training_states']))
    ema_alpha = fit_ema_alpha(tr['recon'][:50000], tr['kl'][:50000])
    kl_median = float(np.median(tr['kl']))

    p1_spec = P1_ENVS[task]
    trajs = collect_trajectories(model, p1_spec, n_traj, cfg, seed=SEED)
    for trj in trajs:
        trj['ct'] = compute_ct(trj['kl'], np.zeros(len(trj['kl']), dtype=np.int64),
                                gamma=spec['gamma_ct'])
        trj['ema_recon'] = ema_series(trj['recon'], ema_alpha)

    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - K_HEADLINE - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(SEED)
    if len(sites) > 3000:
        sel = rng.choice(len(sites), 3000, replace=False)
        sites = [sites[i] for i in sel]

    h_list, kl_list, ct_list, recon_list, ema_list, target_list = [], [], [], [], [], []
    for (ti, t) in sites:
        trj = trajs[ti]
        state_dist_full, _, _ = imagined_vs_real_obs(model, trj, t, K_HEADLINE, p1_spec['domain'])
        if len(state_dist_full) < K_HEADLINE:
            continue
        h_list.append(trj['h'][t])
        kl_list.append(trj['kl'][t])
        ct_list.append(trj['ct'][t])
        recon_list.append(trj['recon'][t])
        ema_list.append(trj['ema_recon'][t])
        target_list.append(e_state(state_dist_full, K_HEADLINE))

    return dict(
        h=np.array(h_list, np.float32), kl=np.array(kl_list, np.float64),
        ct=np.array(ct_list, np.float64), recon=np.array(recon_list, np.float64),
        ema_recon=np.array(ema_list, np.float64), e_state=np.array(target_list, np.float64),
    )


def feature_behavior_and_incremental(sae, eval_data, top_n=20):
    """§9.3 + §9.4: for each atom (restricted to the top_n atoms by activation
    variance, for tractability), correlate activation with KL/C_t/Recon/E^state,
    then run the incremental-R^2 test for the single best externally-predictive
    candidate feature."""
    acts = get_all_activations(sae, eval_data['h'])
    var_per_atom = acts.var(axis=0)
    candidate_atoms = np.argsort(-var_per_atom)[:top_n]

    rows = []
    for atom in candidate_atoms:
        a = acts[:, atom]
        if a.std() < 1e-8:
            continue
        r_kl, _ = pearsonr(a, eval_data['kl'])
        r_ct, _ = pearsonr(a, eval_data['ct'])
        r_recon, _ = pearsonr(a, eval_data['recon'])
        r_ext, p_ext = pearsonr(a, eval_data['e_state'])
        rows.append(dict(atom=int(atom), r_kl=r_kl, r_ct=r_ct, r_recon=r_recon,
                          r_external=r_ext, p_external=p_ext,
                          activation_rate=float((a > 0).mean())))

    rows.sort(key=lambda r: -abs(r['r_external']))
    best = rows[0] if rows else None

    incremental = None
    if best is not None:
        atom = best['atom']
        a = acts[:, atom]
        X_ctrl = np.column_stack([eval_data['kl'], eval_data['recon'], eval_data['ema_recon']])
        X_full = np.column_stack([a, eval_data['kl'], eval_data['recon'], eval_data['ema_recon']])
        y = eval_data['e_state']
        Xs_ctrl = StandardScaler().fit_transform(X_ctrl)
        Xs_full = StandardScaler().fit_transform(X_full)
        r2_ctrl = LinearRegression().fit(Xs_ctrl, y).score(Xs_ctrl, y)
        r2_full = LinearRegression().fit(Xs_full, y).score(Xs_full, y)
        incremental = dict(atom=int(atom), r2_ctrl=r2_ctrl, r2_full=r2_full,
                            incremental_r2=r2_full - r2_ctrl)

    return rows, incremental


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 78)
    print("PHASE 3 — SAE MECHANISTIC DECOMPOSITION")
    print("=" * 78)

    print("\nLoading + pooling h_t across cartpole/reacher/pendulum...")
    pooled_h, pooled_task, pooled_kl, pooled_ct, pooled_recon, per_task = load_pooled_h()
    print(f"  pooled N={len(pooled_h)} ({dict((t, int((pooled_task==t).sum())) for t in TASKS)})")

    print("\nFitting per-task confusion directions v (Probe A protocol)...")
    directions = fit_confusion_directions(per_task)

    # ── §8.2: train pooled TopK SAE across N_SAE_SEEDS seeds ──
    saes, sae_metric_results = [], []
    for seed in range(N_SAE_SEEDS):
        print(f"\n--- training pooled TopK SAE, seed {seed} "
              f"(n_atoms={N_ATOMS}, k={K_SPARSE}) ---")
        sae, history = train_topk_sae(pooled_h, n_atoms=N_ATOMS, k=K_SPARSE,
                                       n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, seed=seed)
        metrics = sae_metrics(sae, pooled_h)
        print(f"  metrics: {metrics}")
        saes.append(sae)
        sae_metric_results.append(metrics)

    # ── §9.1: decompose v per task, per SAE seed ──
    print(f"\n{'='*78}\n§9.1 DICTIONARY COMPOSITION OF v\n{'='*78}")
    decomposition_results = {task: [] for task in TASKS}
    for task in TASKS:
        v = directions[task]['v']
        for seed_i, sae in enumerate(saes):
            dec = decompose_direction(sae, v)
            decomposition_results[task].append(dec)
            print(f"  {task} / SAE seed {seed_i}: {dec['classification']} "
                  f"(top1_share={dec['top1_share']:.3f}, top4_share={dec['top4_share']:.3f})")

    # class-level replication check: does the SAME classification occur across
    # all seeds for a task?
    class_replication = {}
    for task in TASKS:
        classes = [d['classification'] for d in decomposition_results[task]]
        class_replication[task] = dict(
            classes=classes, unanimous=len(set(classes)) == 1,
            majority=max(set(classes), key=classes.count),
        )
        print(f"  {task}: classes across {N_SAE_SEEDS} seeds = {classes} "
              f"-> majority = {class_replication[task]['majority']}")

    # ── §9.2: cross-task composition using the pooled dictionary (seed 0) ──
    print(f"\n{'='*78}\n§9.2 CROSS-TASK COMPOSITION (pooled dictionary, SAE seed 0)\n{'='*78}")
    sae0 = saes[0]
    cross_task = {}
    for task, d in per_task.items():
        acts = get_all_activations(sae0, d['h'][:20000])  # subsample for speed
        mean_act = acts.mean(axis=0)
        top_atoms = np.argsort(-mean_act)[:10]
        v = directions[task]['v']
        dec = decomposition_results[task][0]
        overlap_with_v = len(set(top_atoms.tolist()) & set(dec['top10_atoms']))
        cross_task[task] = dict(top_active_atoms=top_atoms.tolist(),
                                 overlap_with_v_top10=overlap_with_v)
        print(f"  {task}: top-10 most-active atoms overlap with v's top-10 "
              f"reconstruction atoms: {overlap_with_v}/10")

    # ── §9.3 + §9.4: feature behavior + incremental analysis, per task ──
    print(f"\n{'='*78}\n§9.3 + §9.4 FEATURE BEHAVIOR & INCREMENTAL ANALYSIS\n{'='*78}")
    feature_results = {}
    for task, spec in TASKS.items():
        print(f"\n--- {task}: collecting eval trajectories for feature behavior ---")
        eval_data = collect_eval_trajectories_and_targets(task, spec, cfg, N_TRAJ_EVAL)
        print(f"  {len(eval_data['h'])} sites")
        rows, incremental = feature_behavior_and_incremental(sae0, eval_data)
        feature_results[task] = dict(top_atoms_by_r_external=rows[:10], incremental=incremental)
        if rows:
            print(f"  best externally-predictive atom: #{rows[0]['atom']} "
                  f"r(atom, E^state)={rows[0]['r_external']:+.3f} "
                  f"(p={rows[0]['p_external']:.2g}); r(atom,KL)={rows[0]['r_kl']:+.3f}, "
                  f"r(atom,C_t)={rows[0]['r_ct']:+.3f}")
        if incremental:
            print(f"  incremental R^2 of atom #{incremental['atom']} over "
                  f"{{KL,Recon,EMARecon}}: {incremental['incremental_r2']:+.4f} "
                  f"(R^2 ctrl={incremental['r2_ctrl']:.4f} -> full={incremental['r2_full']:.4f})")

    # ── save everything ──
    results = dict(
        pooled_n=len(pooled_h), n_atoms=N_ATOMS, k_sparse=K_SPARSE, n_seeds=N_SAE_SEEDS,
        sae_metrics=sae_metric_results,
        decomposition=decomposition_results, class_replication=class_replication,
        cross_task=cross_task, feature_behavior=feature_results,
    )
    out_path = os.path.join(OUT_DIR, 'phase3_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    # save SAE checkpoints for later causal-intervention phase
    for i, sae in enumerate(saes):
        torch.save(sae.state_dict(), os.path.join(OUT_DIR, f'sae_seed{i}.pt'))
    print(f"Wrote {N_SAE_SEEDS} SAE checkpoints to {OUT_DIR}/sae_seed*.pt")

    # ── figure ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, task in enumerate(TASKS):
        ax = axes[i]
        shares = decomposition_results[task][0]['top10_shares']
        ax.bar(range(len(shares)), shares, color='steelblue')
        ax.set_title(f"{task}: v-reconstruction shares (seed 0)\n"
                      f"{decomposition_results[task][0]['classification']}")
        ax.set_xlabel('rank'); ax.set_ylabel('|contribution share|')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'phase3_sae_decomposition.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {fig_path}")


if __name__ == '__main__':
    main()
