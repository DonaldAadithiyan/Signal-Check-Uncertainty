#!/usr/bin/env python3.11
"""
Task T — Does a stronger (5-model) ensemble narrow the gap against the probe?

Reviewer R3TF: "Ensemble baseline uses only 2–3 models (cost-reduced for multi-seed runs) —
a stronger ensemble (5+ models, as is more typical in the cited baseline literature) might
narrow the gap reported against the probe."

The paper's headline dissociation (Table 1) uses a 3-model ensemble for the initial run and a
2-model ensemble for the cost-reduced multi-seed replication. Standard ensemble-disagreement
literature (Lakshminarayanan et al. 2017; Sekar et al. 2020) uses 5+ members. A weak baseline
would make the probe look better than a properly-resourced comparison — exactly what a more
skeptical reviewer could flag.

This sweeps ensemble size n = 2, 3, 4, 5 over the SAME evaluation sets, using the SAME
`ensemble_disagreement` implementation the paper already uses (variance across independently
trained models' decoded predictions). NOT a scale check: every member is the identical XS
configuration (256-dim GRU, ~12M params, XS_CONFIG hyperparameters). Only the member COUNT
changes.

Evaluation sets (all pre-existing on disk, no new collection):
  · Set C contrastive     — the headline KL-matched dissociation test
  · Set C within-balance  — the within-task control
  · Set A (in-distribution) vs novel-balance — direct novelty detection
Probe-A comparison numbers are read from the existing saved results, not recomputed, so the
probe side of the gap is untouched by this task.

To remove "which 2 models did you pick?" as a confound, for every n < 5 we evaluate ALL
C(5, n) member subsets and report mean ± sd across subsets alongside the specific
seeds-[0..n-1] subset the paper originally used.

Runs on frozen checkpoints. XS, CPU.
"""

import os
import sys
import json
import itertools

import numpy as np
import torch

sys.path.insert(0, '.')

from src.config import XS_CONFIG
from src.probe.linear_probe import ensemble_disagreement, auroc_direct

OUT_DIR = 'outputs/ensemble'
CK = 'outputs/checkpoints/ensemble_seed{}.pt'
ALL_SEEDS = [0, 1, 2, 3, 4]
SIZES = [2, 3, 4, 5]


def load_member(seed, cfg):
    from src.model.world_model import WorldModel
    path = CK.format(seed)
    ck = torch.load(path, map_location='cpu')
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg'])
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def novelty_auroc(models, set_a, novel, cfg):
    """Direct novelty detection: ensemble disagreement on in-distribution Set A vs novel
    (cartpole-balance) states. Label 1 = novel. Uses the same disagreement score."""
    pooled = {'obs': np.concatenate([set_a['obs'], novel['obs']]),
              'labels': np.concatenate([np.zeros(len(set_a['obs']), np.int32),
                                        np.ones(len(novel['obs']), np.int32)])}
    dis, _ = ensemble_disagreement(models, pooled, cfg)
    return float(auroc_direct(dis, pooled['labels']))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    have = [s for s in ALL_SEEDS if os.path.exists(CK.format(s))]
    missing = [s for s in ALL_SEEDS if s not in have]
    if missing:
        print(f"  ⚠ ensemble members not yet trained: seeds {missing}")
        print(f"    (run train_ensemble_expand.py first — evaluating with what exists: {have})")
    if len(have) < 2:
        print("  Need at least 2 members. Aborting.")
        return

    cfg = XS_CONFIG.copy()

    print("=" * 90)
    print("TASK T — ENSEMBLE SIZE SWEEP (Reviewer R3TF)")
    print("=" * 90)
    print(f"  members available: seeds {have}  |  all XS config "
          f"(deter={cfg['rssm_deter']}, ~12M params) — member COUNT is the only variable\n")

    print("  loading evaluation sets...")
    set_c = dict(np.load('outputs/data/set_c_contrastive.npz'))
    set_wb = dict(np.load('outputs/data/set_c_within_balance.npz'))
    set_a = dict(np.load('outputs/data/set_a_id.npz'))
    novel = dict(np.load('outputs/data/novel_rwmu.npz'))

    print("  loading ensemble members...")
    members = {s: load_member(s, cfg) for s in have}

    results = {'available_seeds': have, 'sizes': {}}

    for n in [x for x in SIZES if x <= len(have)]:
        subsets = list(itertools.combinations(have, n))
        rows = []
        for sub in subsets:
            ms = [members[s] for s in sub]
            _, auc_c = ensemble_disagreement(ms, set_c, cfg)
            _, auc_wb = ensemble_disagreement(ms, set_wb, cfg)
            auc_nov = novelty_auroc(ms, set_a, novel, cfg)
            rows.append(dict(seeds=list(sub), setc=float(auc_c),
                             within_balance=float(auc_wb), novelty=auc_nov))

        def agg(key):
            v = np.array([r[key] for r in rows])
            return dict(mean=float(v.mean()), std=float(v.std()),
                        min=float(v.min()), max=float(v.max()))

        # the subset the paper originally used for this size: seeds [0..n-1]
        paper_sub = tuple(range(n))
        paper_row = next((r for r in rows if tuple(r['seeds']) == paper_sub), None)

        results['sizes'][str(n)] = dict(
            n_subsets=len(subsets), per_subset=rows,
            setc=agg('setc'), within_balance=agg('within_balance'), novelty=agg('novelty'),
            paper_subset=paper_row)

        print(f"\n  n={n}  ({len(subsets)} subset{'s' if len(subsets) > 1 else ''} of {len(have)})")
        print(f"    {'metric':<24}{'mean±sd over subsets':>24}{'range':>20}"
              f"{'paper subset '+str(list(paper_sub)):>22}")
        for key, label in [('setc', 'Set C (contrastive)'),
                           ('within_balance', 'Set C within-balance'),
                           ('novelty', 'direct novelty (A vs novel)')]:
            a = results['sizes'][str(n)][key]
            pv = f"{paper_row[key]:.4f}" if paper_row else "—"
            print(f"    {label:<24}{a['mean']:>17.4f}±{a['std']:.4f}"
                  f"{'[' + format(a['min'], '.3f') + ', ' + format(a['max'], '.3f') + ']':>20}{pv:>22}")

    # ── the reviewer's question: does the gap narrow? ──
    print("\n" + "=" * 90)
    print("DOES A LARGER ENSEMBLE NARROW THE GAP AGAINST THE PROBE?")
    print("=" * 90)

    sizes_done = sorted(int(k) for k in results['sizes'])
    if len(sizes_done) >= 2:
        lo, hi = sizes_done[0], sizes_done[-1]
        print(f"\n  {'metric':<28}{'n='+str(lo):>12}{'n='+str(hi):>12}{'Δ (hi−lo)':>13}   trend")
        trends = {}
        for key, label in [('setc', 'Set C (contrastive)'),
                           ('within_balance', 'Set C within-balance'),
                           ('novelty', 'direct novelty')]:
            a = results['sizes'][str(lo)][key]['mean']
            b = results['sizes'][str(hi)][key]['mean']
            d = b - a
            trends[key] = dict(lo=a, hi=b, delta=float(d))
            arrow = 'stronger ensemble' if d > 0.005 else ('weaker' if d < -0.005 else 'flat')
            print(f"  {label:<28}{a:>12.4f}{b:>12.4f}{d:>+13.4f}   {arrow}")
        results['trend_lo_to_hi'] = dict(n_lo=lo, n_hi=hi, **trends)

        # Probe-A reference numbers (from existing saved results — NOT recomputed here)
        print("\n  Probe-A reference (unchanged, from existing saved results):")
        print("    Set C (contrastive) AUROC = 0.7144   [outputs/results/probe_results.csv]")
        print("    The dissociation claim is Probe-A ABOVE chance on Set C while the ensemble is not.")
        setc_hi = results['sizes'][str(hi)]['setc']
        print(f"\n    ensemble Set C at n={hi}: {setc_hi['mean']:.4f} ± {setc_hi['std']:.4f} "
              f"(range [{setc_hi['min']:.3f}, {setc_hi['max']:.3f}])")
        gap = 0.7144 - setc_hi['mean']
        results['probeA_setc_reference'] = 0.7144
        results['gap_probeA_minus_ensemble_at_max_n'] = float(gap)
        print(f"    Probe-A − ensemble gap at n={hi}: {gap:+.4f}")
        if setc_hi['mean'] >= 0.7144:
            note = (f"GAP CLOSED/REVERSED at n={hi}: the larger ensemble matches or beats Probe-A on "
                    f"Set C ({setc_hi['mean']:.4f} vs 0.7144). The headline dissociation claim needs "
                    f"revisiting in the text — report plainly.")
        elif gap < 0.05:
            note = (f"GAP NARROWED SUBSTANTIALLY at n={hi} (Probe-A − ensemble = {gap:+.4f}, under 0.05). "
                    f"The dissociation survives in sign but is much weaker than the 2–3-model comparison "
                    f"implies. The text should be adjusted to cite the 5-model number.")
        else:
            note = (f"GAP HOLDS at n={hi} (Probe-A − ensemble = {gap:+.4f}). The dissociation is not an "
                    f"artefact of a cost-reduced ensemble; citing the {hi}-model number is a strictly "
                    f"stronger version of the existing claim.")
        print(f"\n  {note}")
        results['_note'] = note

    out = os.path.join(OUT_DIR, 'task_t_ensemble_size.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {out}")

    csv_path = os.path.join(OUT_DIR, 'task_t_ensemble_size.csv')
    with open(csv_path, 'w') as f:
        f.write("n_members,seeds,setc_auroc,within_balance_auroc,novelty_auroc\n")
        for n in sorted(results['sizes'], key=int):
            for r in results['sizes'][n]['per_subset']:
                f.write(f"{n},\"{r['seeds']}\",{r['setc']:.6f},"
                        f"{r['within_balance']:.6f},{r['novelty']:.6f}\n")
    print(f"  CSV saved:     {csv_path}")


if __name__ == '__main__':
    main()
