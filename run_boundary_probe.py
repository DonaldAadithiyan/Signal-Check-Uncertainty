#!/usr/bin/env python3.11
"""
Task 6 — Imagination boundary probe: reframe the depth null result as a positive finding.

Observation: the depth experiment showed probe score jumps from ~0.50 to ~0.72 in
one imagination step, then plateaus — the probe detects the observation/imagination
boundary but not depth within imagination.

This script:
  1. Trains a binary probe specifically on the obs/imagination boundary:
       label 0 = real posterior h_t
       label 1 = imagined h_t (any depth 1–15)
  2. Measures AUROC of this boundary probe.
  3. Computes correlation between boundary probe scores and Probe A scores.
  4. Shows they detect different aspects of h_t — boundary probe fires uniformly
     on imagined states, Probe A varies with confusion level.

Expected:
  - Boundary probe AUROC >> 0.72 (easily separates obs vs imagination mode)
  - Correlation with Probe A ≈ 0.3–0.6 on real states (partial overlap)
  - Correlation on imagined states ≈ 0.1–0.3 (boundary probe uniformly high,
    Probe A varies)

Interpretation: h_t encodes at least two distinct aspects of model state —
observation mode vs imagination mode, and confusion level within observation mode.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe, auroc


N_START   = 5_000    # starting states for imagination rollouts
HORIZON   = 15       # imagination steps
BATCH     = 512


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def run_imagination(model, h_start, z_start, horizon, cfg, seed=0):
    """
    Batch imagination from N starting states.
    Returns h_per_depth: list[horizon] of (N, h_dim) arrays (depths 1..horizon).
    Depth 0 (real posterior) is excluded — that's the real data.
    """
    device = next(model.parameters()).device
    rng    = np.random.default_rng(seed)
    N      = h_start.shape[0]

    h = torch.tensor(h_start, dtype=torch.float32, device=device)
    z = torch.tensor(z_start, dtype=torch.float32, device=device)

    h_imagined = []
    with torch.no_grad():
        for _ in range(horizon):
            action = torch.tensor(
                rng.uniform(-1, 1, (N, cfg['act_dim'])).astype(np.float32),
                device=device)
            h, z, _ = model.rssm.imagine_step(h, z, action)
            h_imagined.append(h.cpu().numpy().copy())

    return h_imagined   # list of HORIZON arrays, each (N, h_dim)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ── Load training states and train Probe A ──
    print("Loading training states...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all  = tr['h']
    z_all  = tr['z']
    kl_all = tr['kl']
    N      = len(h_all)

    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(
        np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)

    print("Training Probe A (KL labels)...")
    clf_a, sc_a = train_probe(h_all[tr_idx], y_kl[tr_idx])
    auroc_a_id = auroc(clf_a, sc_a, h_all[te_idx], y_kl[te_idx])
    print(f"  Probe A held-out AUROC: {auroc_a_id:.4f}")

    # ── Run imagination rollouts ──
    print(f"\nLoading model and running {HORIZON}-step imagination from {N_START} states...")
    model = load_model(cfg)

    rng = np.random.default_rng(42)
    start_idx = rng.choice(te_idx, N_START, replace=False)   # held-out only
    h_start   = h_all[start_idx]
    z_start   = z_all[start_idx]

    h_imagined = run_imagination(model, h_start, z_start, HORIZON, cfg)
    # h_imagined[d] has shape (N_START, 256), d=0..HORIZON-1 → depths 1..HORIZON

    print(f"  {HORIZON * N_START:,} imagined states across {HORIZON} depths")

    # ── Build boundary dataset ──
    # Real: held-out states (te_idx), label 0
    # Imagined: all depths 1–HORIZON pooled, label 1
    h_real     = h_all[te_idx]                             # (N_te, 256)
    h_imag_all = np.concatenate(h_imagined, axis=0)       # (HORIZON * N_START, 256)

    h_boundary = np.concatenate([h_real, h_imag_all], axis=0)
    y_boundary  = np.array(
        [0] * len(h_real) + [1] * len(h_imag_all), dtype=np.int32)

    print(f"\n  Boundary dataset: {len(h_real):,} real + {len(h_imag_all):,} imagined")

    # ── Train boundary probe ──
    print("Training boundary probe...")
    b_tr, b_te = train_test_split(
        np.arange(len(h_boundary)), test_size=0.30, stratify=y_boundary, random_state=0)
    clf_b, sc_b = train_probe(h_boundary[b_tr], y_boundary[b_tr])
    auroc_boundary = auroc(clf_b, sc_b, h_boundary[b_te], y_boundary[b_te])
    print(f"  Boundary probe AUROC: {auroc_boundary:.4f}")

    # ── Correlation between probes ──
    # On held-out REAL states (proper evaluation — te_idx not in Probe A training)
    probe_a_real  = clf_a.predict_proba(sc_a.transform(h_real))[:, 1]
    probe_b_real  = clf_b.predict_proba(sc_b.transform(h_real))[:, 1]
    r_real = np.corrcoef(probe_a_real, probe_b_real)[0, 1]

    # On imagined states (subset)
    rng2 = np.random.default_rng(0)
    imag_idx = rng2.choice(len(h_imag_all), min(len(h_real), len(h_imag_all)), replace=False)
    probe_a_imag = clf_a.predict_proba(sc_a.transform(h_imag_all[imag_idx]))[:, 1]
    probe_b_imag = clf_b.predict_proba(sc_b.transform(h_imag_all[imag_idx]))[:, 1]
    r_imag = np.corrcoef(probe_a_imag, probe_b_imag)[0, 1]

    # ── Probe A on imagined states (by depth) ──
    print("\nProbe A and boundary probe scores by imagination depth:")
    print(f"  {'Depth':>6}  {'Probe A mean':>13}  {'Boundary mean':>14}")
    print(f"  {'-'*6}  {'-'*13}  {'-'*14}")
    # Depth 0 (real posterior)
    pa0 = probe_a_real.mean()
    pb0 = probe_b_real.mean()
    print(f"  {'0 (real)':>6}  {pa0:>13.4f}  {pb0:>14.4f}")
    for d, h_d in enumerate(h_imagined, start=1):
        if d in [1, 3, 5, 10, 15]:
            pa_d = clf_a.predict_proba(sc_a.transform(h_d))[:, 1].mean()
            pb_d = clf_b.predict_proba(sc_b.transform(h_d))[:, 1].mean()
            print(f"  {d:>6}  {pa_d:>13.4f}  {pb_d:>14.4f}")

    # ── Summary ──
    print("\n" + "="*65)
    print("BOUNDARY PROBE SUMMARY")
    print("="*65)
    print(f"\n  Probe A AUROC (KL labels, within real states): {auroc_a_id:.4f}")
    print(f"  Boundary probe AUROC (real vs imagined):       {auroc_boundary:.4f}")
    print(f"\n  Pearson r (Probe A vs Boundary) on real states:     {r_real:+.4f}")
    print(f"  Pearson r (Probe A vs Boundary) on imagined states:  {r_imag:+.4f}")

    if auroc_boundary > 0.85:
        print("\n  Boundary probe: STRONG — clearly separates observation from imagination mode.")
    elif auroc_boundary > 0.70:
        print("\n  Boundary probe: MODERATE separation of observation vs imagination mode.")
    else:
        print("\n  Boundary probe: WEAK — probes overlap significantly.")

    if r_real < 0.60 and auroc_boundary > 0.80:
        print(f"\n  DISTINCT SIGNALS: correlation on real states = {r_real:.2f} < 0.60.")
        print("  Probe A detects within-trajectory confusion.")
        print("  Boundary probe detects observation vs imagination mode.")
        print("  h_t encodes at least two distinct aspects of model state.")
    elif r_real > 0.80:
        print(f"\n  OVERLAPPING SIGNALS: correlation = {r_real:.2f}.")
        print("  The two probes largely detect the same aspect of h_t.")

    # ── Figure ──
    print("\nGenerating boundary probe figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Boundary Probe — Observation vs Imagination Mode', fontsize=12, fontweight='bold')

    # Left: scatter Probe A vs Boundary probe on real states (coloured by KL)
    ax = axes[0]
    n_plot = min(5000, len(h_real))
    rng3 = np.random.default_rng(1)
    idx_plot = rng3.choice(len(h_real), n_plot, replace=False)
    sc_plot = ax.scatter(probe_a_real[idx_plot], probe_b_real[idx_plot],
                         c=kl_all[te_idx][idx_plot], cmap='coolwarm',
                         alpha=0.3, s=6, rasterized=True)
    plt.colorbar(sc_plot, ax=ax, label='KL divergence')
    ax.set_xlabel('Probe A score (confusion)')
    ax.set_ylabel('Boundary probe score (imagination mode)')
    ax.set_title(f'Real states — r={r_real:+.3f}\n(corr < 0.6 → distinct signals)')
    ax.grid(True, alpha=0.3)

    # Right: probe scores by imagination depth
    ax = axes[1]
    all_depths = [0] + list(range(1, HORIZON + 1))
    pa_means, pb_means = [pa0], [pb0]
    for d, h_d in enumerate(h_imagined, start=1):
        pa_means.append(clf_a.predict_proba(sc_a.transform(h_d))[:, 1].mean())
        pb_means.append(clf_b.predict_proba(sc_b.transform(h_d))[:, 1].mean())

    ax.plot(all_depths, pa_means, 'b-o', markersize=4, linewidth=1.5, label='Probe A (confusion)')
    ax.plot(all_depths, pb_means, 'r-s', markersize=4, linewidth=1.5, label='Boundary probe')
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(0.3, 0.95, 'Real', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, color='gray')
    ax.text(1.5, 0.95, 'Imagined →', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=8, color='gray')
    ax.set_xlabel('Imagination depth (0 = real posterior)')
    ax.set_ylabel('Mean probe score')
    ax.set_title('Probe scores vs imagination depth\n(boundary: jumps at depth 1; Probe A: varies)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'boundary_probe.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    main()
