#!/usr/bin/env python3.11
"""
Rebuild Set C with KL-matched construction and re-run all probes.
Does not retrain the world model or ensemble — loads existing checkpoints.
"""

import os
import csv
import numpy as np
import torch

from src.config import XS_CONFIG
from src.data.collect import build_set_c
from src.probe.linear_probe import (
    run_probe_a, run_probe_b, run_probe_c,
    run_block_analysis, run_ht_vs_zt, ensemble_disagreement,
)
from src.model.world_model import WorldModel


def load_model(cfg, ck_path):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
    m  = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def main():
    cfg = XS_CONFIG.copy()

    print("Loading training states and Sets A/B...")
    states = dict(np.load(cfg['training_data_path']))
    set_a  = dict(np.load(cfg['set_a_path'], allow_pickle=True))
    set_b  = dict(np.load(cfg['set_b_path'], allow_pickle=True))

    print("\nRebuilding Set C (KL-matched)...")
    set_c = build_set_c(set_a, set_b)

    # Verify KL distributions are matched
    labels = set_c['labels']
    c1_kl  = set_c['kl'][labels == 0]
    c2_kl  = set_c['kl'][labels == 1]
    print(f"\n  KL check — C1: {c1_kl.mean():.2f} ± {c1_kl.std():.2f}  "
          f"C2: {c2_kl.mean():.2f} ± {c2_kl.std():.2f}")

    # Save new set_c
    np.savez(cfg['set_c_path'],
             **{k: v for k, v in set_c.items() if isinstance(v, np.ndarray)})
    print("  Saved new set_c_contrastive.npz")

    print("\nLoading main model...")
    model = load_model(cfg, cfg['checkpoint_path'])

    print("Loading ensemble models...")
    seeds = cfg['ensemble_seeds']
    ensemble_models = [
        load_model(cfg, f"outputs/checkpoints/ensemble_seed{s}.pt")
        for s in seeds
    ]

    print("\n--- Running probes ---\n")

    print("Probe A (KL gap)...")
    pa = run_probe_a(states, set_a, set_b, set_c)
    print(f"  ID={pa['auroc_id']:.4f}  A={pa['auroc_a']:.4f}  "
          f"B={pa['auroc_b']:.4f}  C={pa['auroc_c']:.4f}")

    print("Probe B (rollout variance)...")
    pb = run_probe_b(model, states, set_a, set_b, set_c, cfg)
    print(f"  ID={pb['auroc_id']:.4f}  A={pb['auroc_a']:.4f}  "
          f"B={pb['auroc_b']:.4f}  C={pb['auroc_c']:.4f}")

    print("Probe C (recon sanity)...")
    pc = run_probe_c(states, set_a, set_b, set_c)
    print(f"  ID={pc['auroc_id']:.4f}  A={pc['auroc_a']:.4f}  "
          f"B={pc['auroc_b']:.4f}  C={pc['auroc_c']:.4f}")

    print("Ensemble disagreement...")
    _, ens_auroc = ensemble_disagreement(ensemble_models, set_c, cfg)
    print(f"  Set C AUROC={ens_auroc:.4f}")

    print("h_t vs z_t...")
    hvz = run_ht_vs_zt(states, set_a, set_c)
    for name in hvz:
        print(f"  {name}: train={hvz[name]['auroc_train']:.4f}  "
              f"A={hvz[name]['auroc_a']:.4f}  C={hvz[name]['auroc_c']:.4f}")

    print("\n=== CORRECTED AUROC TABLE ===\n")
    headers = ['Probe', 'Train held-out', 'Set A (ID)', 'Set B (OOD)', 'Set C (KL-matched)']
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
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    with open(cfg['probe_results_path'], 'w', newline='') as f:
        csv.writer(f).writerows([headers] + rows)
    print(f"\nSaved to {cfg['probe_results_path']}")


if __name__ == '__main__':
    main()
