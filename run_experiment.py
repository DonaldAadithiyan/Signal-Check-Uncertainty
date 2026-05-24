#!/usr/bin/env python3.11
"""
Phase 1 Signal Check — Mini-DreamerV3 Uncertainty Probe
Master orchestration script.

Usage:
    python3.11 run_experiment.py [--skip-ensemble] [--skip-probes]
"""

import os
import csv
import time
import argparse
import numpy as np
import torch

from src.config import XS_CONFIG
from src.training.trainer import train_world_model, load_model
from src.data.collect import collect_all_sets
from src.probe.linear_probe import (
    run_probe_a, run_probe_b, run_probe_c,
    run_block_analysis, run_ht_vs_zt, ensemble_disagreement,
)
from src.viz.figures import (
    plot_roc_curves, plot_block_heatmap, plot_kl_distribution,
)


# ─── Dev log helpers ─────────────────────────────────────────────────────────

DEV_LOG = 'DEV_LOG.md'


def log(msg, section=None):
    print(msg, flush=True)
    with open(DEV_LOG, 'a') as f:
        if section:
            f.write(f'\n## {section}\n')
        f.write(msg + '\n')


def log_table(rows, headers):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    with open(DEV_LOG, 'a') as f:
        f.write('\n'.join(lines) + '\n\n')


# ─── Phase 1: Train ──────────────────────────────────────────────────────────

def phase_train(cfg):
    if os.path.exists(cfg['checkpoint_path']) and os.path.exists(cfg['training_data_path']):
        log("[phase 1] checkpoint found — loading existing model")
        model = load_model(cfg)
        data  = dict(np.load(cfg['training_data_path']))
        return model, data

    log("Starting world model training...", section="Phase 1 — Training")
    t0 = time.time()
    model, states = train_world_model(cfg, seed=0)
    elapsed = time.time() - t0

    np.savez(cfg['training_data_path'], **states)
    log(f"Training complete in {elapsed/60:.1f} min | {len(states['h']):,} states logged")
    log(f"  mean KL={states['kl'].mean():.4f}  std={states['kl'].std():.4f}  "
        f"mean recon={states['recon'].mean():.4f}")
    return model, states


# ─── Phase 2: Collect eval sets ──────────────────────────────────────────────

def phase_collect(model, cfg):
    if (os.path.exists(cfg['set_a_path'])
            and os.path.exists(cfg['set_b_path'])
            and os.path.exists(cfg['set_c_path'])):
        log("[phase 2] evaluation sets found — loading")
        set_a = dict(np.load(cfg['set_a_path'], allow_pickle=True))
        set_b = dict(np.load(cfg['set_b_path'], allow_pickle=True))
        set_c = dict(np.load(cfg['set_c_path'], allow_pickle=True))
        return set_a, set_b, set_c

    log("Collecting evaluation sets...", section="Phase 2 — Data Collection")
    set_a, set_b, set_c = collect_all_sets(model, cfg)

    for path, data in [(cfg['set_a_path'], set_a),
                       (cfg['set_b_path'], set_b),
                       (cfg['set_c_path'], set_c)]:
        np.savez(path, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})

    log(f"Set A: {len(set_a['h'])} states | mean KL={set_a['kl'].mean():.4f}")
    log(f"Set B: {len(set_b['h'])} states | mean KL={set_b['kl'].mean():.4f}")
    log(f"Set C: {len(set_c['h'])} states | "
        f"C1={(set_c['labels']==0).sum()} C2={(set_c['labels']==1).sum()}")
    return set_a, set_b, set_c


# ─── Phase 3: Ensemble ───────────────────────────────────────────────────────

def phase_ensemble(cfg):
    flag  = 'outputs/checkpoints/ensemble_complete.flag'
    seeds = cfg['ensemble_seeds']

    if os.path.exists(flag):
        log("[phase 3] ensemble already trained — loading")
        return [_load_ensemble_model(cfg, seed=s) for s in seeds]

    log("Training ensemble (3 seeds)...", section="Phase 3 — Ensemble Training")
    models = []
    for s in seeds:
        ck = f"outputs/checkpoints/ensemble_seed{s}.pt"
        if os.path.exists(ck):
            log(f"  seed {s}: checkpoint exists — loading")
            m = _load_ensemble_model(cfg, ck_path=ck)
        else:
            log(f"  seed {s}: training...")
            t0    = time.time()
            cfg_s = {**cfg, 'checkpoint_path': ck}
            m, _  = train_world_model(cfg_s, seed=s)
            log(f"  seed {s}: done in {(time.time()-t0)/60:.1f} min")
        models.append(m)

    open(flag, 'w').close()
    return models


def _load_ensemble_model(cfg, seed=None, ck_path=None):
    from src.model.world_model import WorldModel
    if ck_path is None:
        ck_path = f"outputs/checkpoints/ensemble_seed{seed}.pt"
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
    m  = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


# ─── Phase 4: Probe analysis ─────────────────────────────────────────────────

def phase_probes(model, states, set_a, set_b, set_c, ensemble_models, cfg):
    log("Running probe analysis...", section="Phase 4 — Probe Analysis")

    log("  Probe A (KL gap)...")
    pa = run_probe_a(states, set_a, set_b, set_c)
    log(f"    ID={pa['auroc_id']:.4f}  A={pa['auroc_a']:.4f}  "
        f"B={pa['auroc_b']:.4f}  C={pa['auroc_c']:.4f}")

    log("  Probe B (rollout variance)...")
    pb = run_probe_b(model, states, set_a, set_b, set_c, cfg)
    log(f"    ID={pb['auroc_id']:.4f}  A={pb['auroc_a']:.4f}  "
        f"B={pb['auroc_b']:.4f}  C={pb['auroc_c']:.4f}")

    log("  Probe C (recon sanity check)...")
    pc = run_probe_c(states, set_a, set_b, set_c)
    log(f"    ID={pc['auroc_id']:.4f}  A={pc['auroc_a']:.4f}  "
        f"B={pc['auroc_b']:.4f}  C={pc['auroc_c']:.4f}")

    log("  Ensemble disagreement baseline...")
    _, ens_auroc = ensemble_disagreement(ensemble_models, set_c, cfg)
    log(f"    Set C AUROC={ens_auroc:.4f}")

    # ── AUROC results table
    log("\n### AUROC Results Table\n")
    headers = ['Probe', 'Train held-out', 'Set A (ID)', 'Set B (OOD)', 'Set C (contrastive)']
    rows = [
        ['Probe A (KL gap)',
         f"{pa['auroc_id']:.4f}", f"{pa['auroc_a']:.4f}",
         f"{pa['auroc_b']:.4f}", f"{pa['auroc_c']:.4f}"],
        ['Probe B (rollout var)',
         f"{pb['auroc_id']:.4f}", f"{pb['auroc_a']:.4f}",
         f"{pb['auroc_b']:.4f}", f"{pb['auroc_c']:.4f}"],
        ['Probe C (recon sanity)',
         f"{pc['auroc_id']:.4f}", f"{pc['auroc_a']:.4f}",
         f"{pc['auroc_b']:.4f}", f"{pc['auroc_c']:.4f}"],
        ['Ensemble baseline', 'N/A', 'N/A', 'N/A', f"{ens_auroc:.4f}"],
    ]
    log_table(rows, headers)
    with open(cfg['probe_results_path'], 'w', newline='') as f:
        csv.writer(f).writerows([headers] + rows)

    # ── Block analysis
    log("  Block (per-quarter h_t) analysis...")
    block = run_block_analysis(states, set_a)
    block_rows = [[q, f"{block[q]['auroc_train']:.4f}",
                   f"{block[q]['auroc_a']:.4f}", block[q]['dims']]
                  for q in block]
    log("\n### Per-Block AUROC\n")
    log_table(block_rows, ['Block', 'Train held-out', 'Set A', 'Dims'])
    with open(cfg['block_auroc_path'], 'w', newline='') as f:
        csv.writer(f).writerows([['Block', 'auroc_train', 'auroc_a', 'dims']] + block_rows)

    # ── h_t vs z_t
    log("  h_t vs z_t comparison...")
    hvz = run_ht_vs_zt(states, set_a, set_c)
    hvz_rows = [[name, f"{hvz[name]['auroc_train']:.4f}",
                 f"{hvz[name]['auroc_a']:.4f}", f"{hvz[name]['auroc_c']:.4f}"]
                for name in hvz]
    log("\n### h_t vs z_t AUROC\n")
    log_table(hvz_rows, ['Feature', 'Train held-out', 'Set A', 'Set C'])
    with open(cfg['ht_vs_zt_path'], 'w', newline='') as f:
        csv.writer(f).writerows([['feature', 'auroc_train', 'auroc_a', 'auroc_c']] + hvz_rows)

    return pa, block


# ─── Phase 5: Figures ────────────────────────────────────────────────────────

def phase_figures(pa, states, set_a, set_b, set_c, block, cfg):
    log("Generating figures...", section="Phase 5 — Figures")
    fdir = cfg['figures_dir']
    os.makedirs(fdir, exist_ok=True)
    plot_roc_curves(pa, states, set_a, set_b, set_c,
                    os.path.join(fdir, 'roc_curves.png'))
    plot_block_heatmap(block, os.path.join(fdir, 'block_heatmap.png'))
    plot_kl_distribution(states, set_a, set_b,
                         os.path.join(fdir, 'kl_distribution.png'))


# ─── Verdict ─────────────────────────────────────────────────────────────────

def verdict(pa):
    a_pass = pa['auroc_a'] > 0.72
    c_pass = pa['auroc_c'] > 0.63

    log("\n## Verdict\n")
    log(f"Criterion 1 — Probe A Set A AUROC > 0.72: "
        f"{pa['auroc_a']:.4f} → {'PASS ✓' if a_pass else 'FAIL ✗'}")
    log(f"Criterion 2 — Probe A Set C AUROC > 0.63: "
        f"{pa['auroc_c']:.4f} → {'PASS ✓' if c_pass else 'FAIL ✗'}")

    if a_pass and c_pass:
        log("\n**POSITIVE RESULT** — signal exists. Phase 2 and 3 are justified.")
    else:
        log("\n**NEGATIVE / PARTIAL RESULT** — signal weak or absent at this scale/budget.")
        log("Clean negative: motivates auxiliary training objective variant.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-ensemble', action='store_true')
    parser.add_argument('--skip-probes',   action='store_true')
    args = parser.parse_args()

    cfg = XS_CONFIG.copy()

    with open(DEV_LOG, 'w') as f:
        f.write("# Phase 1 Development Log\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
        f.write(f"**Device:** {cfg['device']}\n")
        f.write(f"**Config:** XS DreamerV3 — deter={cfg['rssm_deter']}, "
                f"stoch={cfg['rssm_stoch']}×{cfg['rssm_classes']}, "
                f"embed={cfg['embed_dim']}\n\n")
        f.write(f"**Training budget:** {cfg['total_env_steps']:,} env steps\n\n---\n\n")

    t_total = time.time()

    model, states   = phase_train(cfg)
    set_a, set_b, set_c = phase_collect(model, cfg)

    if not args.skip_ensemble:
        ensemble_models = phase_ensemble(cfg)
    else:
        log("[phase 3] ensemble skipped — using single model as trivial baseline")
        ensemble_models = [model]

    if not args.skip_probes:
        pa, block = phase_probes(model, states, set_a, set_b, set_c, ensemble_models, cfg)
        phase_figures(pa, states, set_a, set_b, set_c, block, cfg)
        verdict(pa)

    log(f"\n---\n**Total wall time:** {(time.time()-t_total)/3600:.2f} hours")
    log("Experiment complete.")


if __name__ == '__main__':
    main()
