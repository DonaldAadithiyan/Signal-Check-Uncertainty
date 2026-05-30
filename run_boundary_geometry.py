#!/usr/bin/env python3.11
"""
Boundary direction geometry: what subspace does the obs/imagination boundary live in?

The boundary probe achieves 1.0000 AUROC with r=−0.015 correlation to Probe A.
These two probes live in orthogonal directions in h_t space. This script asks:

  - Where is the boundary direction relative to the PCA components?
    Hypothesis: boundary is in the HIGH-variance subspace (unlike confusion)
    because posterior-to-prior h_t shift is driven by observation content (large signal)

  - Are all three probe directions mutually orthogonal?
    Probe A (confusion), boundary probe, top PC1

  - What variance does the boundary direction explain vs the confusion direction?
    Boundary: large (should explain a large fraction of total h_t variance)
    Confusion: tiny (in the null space per the PCA analysis)

This completes the three-subspace decomposition:
  1. Observation content (top PCs, high variance, drives task representation)
  2. Imagination boundary (high variance, posterior vs prior mode)
  3. Within-trajectory confusion (low variance, near-null space)

All three linearly readable from h_t, all mutually orthogonal.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe


N_START  = 5_000
HORIZON  = 15
N_PCS    = 50


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def run_imagination_h(model, h_start, z_start, horizon, cfg, seed=0):
    device = next(model.parameters()).device
    rng    = np.random.default_rng(seed)
    N      = h_start.shape[0]
    h = torch.tensor(h_start, dtype=torch.float32, device=device)
    z = torch.tensor(z_start, dtype=torch.float32, device=device)
    h_list = []
    with torch.no_grad():
        for _ in range(horizon):
            a = torch.tensor(rng.uniform(-1,1,(N, cfg['act_dim'])).astype(np.float32), device=device)
            h, z, _ = model.rssm.imagine_step(h, z, a)
            h_list.append(h.cpu().numpy().copy())
    return np.concatenate(h_list, axis=0)  # (N*H, 256)


def angle_deg(v1, v2):
    """Angle in degrees between two unit vectors."""
    cos_val = abs(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))
    return np.degrees(np.arccos(np.clip(cos_val, 0.0, 1.0)))


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ── Load data and train Probe A ──
    print("Loading training states and training Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, z_all, kl_all = tr['h'], tr['z'], tr['kl']
    N = len(h_all)

    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf_a, sc_a = train_probe(h_all[tr_idx], y_kl[tr_idx])

    # Probe A direction in original h_t space
    probe_a_dir = clf_a.coef_[0] / sc_a.scale_
    probe_a_dir /= np.linalg.norm(probe_a_dir)

    # ── Run imagination and train boundary probe ──
    print(f"\nRunning {HORIZON}-step imagination from {N_START} held-out states...")
    model = load_model(cfg)
    rng = np.random.default_rng(42)
    start_idx  = rng.choice(te_idx, N_START, replace=False)
    h_imagined = run_imagination_h(model, h_all[start_idx], z_all[start_idx], HORIZON, cfg)

    h_real   = h_all[te_idx]   # (N_te, 256)
    h_bound  = np.concatenate([h_real, h_imagined], axis=0)
    y_bound  = np.array([0]*len(h_real) + [1]*len(h_imagined), dtype=np.int32)

    print("Training boundary probe...")
    clf_b, sc_b = train_probe(h_bound, y_bound)

    # Boundary direction in original h_t space
    boundary_dir = clf_b.coef_[0] / sc_b.scale_
    boundary_dir /= np.linalg.norm(boundary_dir)

    # ── PCA on full h_t (all training states) ──
    print(f"Fitting PCA ({N_PCS} components)...")
    h_scaled = sc_a.transform(h_all)
    pca = PCA(n_components=N_PCS, random_state=0).fit(h_scaled)

    # ── Angle analysis ──
    print("\n" + "="*65)
    print("BOUNDARY DIRECTION GEOMETRY")
    print("="*65)

    # Angles to top PCA components (in scaled space)
    probe_a_sc  = clf_a.coef_[0] / np.linalg.norm(clf_a.coef_[0])
    # boundary in scaled space: need to transform
    # boundary probe was trained on raw h_t (not scaled by sc_a)
    # use sc_b's scale for the boundary direction mapping
    boundary_sc = (clf_b.coef_[0] / sc_b.scale_) * sc_a.scale_
    boundary_sc /= np.linalg.norm(boundary_sc)

    print(f"\n  {'':30}  {'Confusion (Probe A)':>20}  {'Boundary probe':>16}")
    print(f"  {'-'*30}  {'-'*20}  {'-'*16}")
    cum_var_probe_a  = 0.0
    cum_var_boundary = 0.0
    print(f"  {'Component':30}  {'Angle (°)':>20}  {'Angle (°)':>16}")
    for k in range(10):
        a_confusion = angle_deg(probe_a_sc,  pca.components_[k])
        a_boundary  = angle_deg(boundary_sc, pca.components_[k])
        expl = pca.explained_variance_ratio_[k] * 100
        cum_var_probe_a  += (np.dot(probe_a_sc,  pca.components_[k]))**2
        cum_var_boundary += (np.dot(boundary_sc, pca.components_[k]))**2
        print(f"  PC{k+1:<3} ({expl:>4.1f}% var)             {a_confusion:>20.1f}  {a_boundary:>16.1f}")

    # Full sweep
    angles_a, angles_b = [], []
    for k in range(N_PCS):
        angles_a.append(angle_deg(probe_a_sc,  pca.components_[k]))
        angles_b.append(angle_deg(boundary_sc, pca.components_[k]))

    print(f"\n  Mean angle (PCs 1-10):   Confusion {np.mean(angles_a[:10]):.1f}°  "
          f"Boundary {np.mean(angles_b[:10]):.1f}°")
    print(f"  Mean angle (PCs 1-{N_PCS}):  Confusion {np.mean(angles_a):.1f}°  "
          f"Boundary {np.mean(angles_b):.1f}°")

    # Variance captured by each direction in top-k PCs
    def pct_in_top_k(vec_sc, k):
        proj = np.array([np.dot(vec_sc, pca.components_[i]) for i in range(k)])
        return proj.var() * k / (np.linalg.norm(vec_sc)**2 + 1e-9) * 100  # rough

    pf = lambda v, k: sum(np.dot(v, pca.components_[i])**2 for i in range(k))

    print(f"\n  Fraction of direction ||·||² in top PCA components:")
    print(f"  {'Top k PCs':>12}  {'Confusion (Probe A)':>20}  {'Boundary probe':>16}")
    for k in [1, 5, 10, 20, 50]:
        pf_a = pf(probe_a_sc,  k) / (np.linalg.norm(probe_a_sc)**2  + 1e-9) * 100
        pf_b = pf(boundary_sc, k) / (np.linalg.norm(boundary_sc)**2 + 1e-9) * 100
        print(f"  {'top '+str(k):>12}  {pf_a:>19.1f}%  {pf_b:>15.1f}%")

    # ── Mutual angles between probe directions ──
    a_ab = angle_deg(probe_a_dir, boundary_dir)
    print(f"\n  Mutual angles between probe directions:")
    print(f"    Confusion vs Boundary: {a_ab:.1f}°  (90°=orthogonal)")

    # Variance of each direction across h_t
    # variance = how spread the h_t distribution is in this direction
    proj_a  = h_all @ probe_a_dir
    proj_b  = h_all @ boundary_dir
    var_a   = proj_a.var()
    var_b   = proj_b.var()
    total_h_var = h_all.var(axis=0).sum()
    print(f"\n  h_t variance explained by each direction:")
    print(f"    Confusion (Probe A):   {var_a:.4f}  ({var_a/total_h_var*100:.2f}% of total)")
    print(f"    Boundary probe:        {var_b:.4f}  ({var_b/total_h_var*100:.2f}% of total)")
    print(f"    Total h_t variance:    {total_h_var:.4f}")
    print(f"    PC1 direction:         "
          f"{(h_all @ (pca.components_[0] / sc_a.scale_)).var():.4f}")

    # ── Mean h_t shift: real vs imagined ──
    real_mean  = h_real.mean(axis=0)
    imag_mean  = h_imagined.mean(axis=0)
    diff_vec   = real_mean - imag_mean
    diff_norm  = diff_vec / np.linalg.norm(diff_vec)
    a_diff_a  = angle_deg(diff_norm, probe_a_dir)
    a_diff_b  = angle_deg(diff_norm, boundary_dir)
    a_diff_pc1 = angle_deg(diff_norm, pca.components_[0] / sc_a.scale_
                           * sc_a.scale_)  # cancel
    print(f"\n  Mean h_t shift (real − imagined) direction:")
    print(f"    ||shift||:             {np.linalg.norm(diff_vec):.4f}")
    print(f"    Angle to boundary probe direction: {a_diff_b:.1f}°")
    print(f"    Angle to confusion direction:      {a_diff_a:.1f}°")

    # ── Interpretation ──
    print(f"\n  INTERPRETATION:")
    if np.mean(angles_a[:10]) > 80 and np.mean(angles_b[:10]) < 50:
        print("  THREE-SUBSPACE DECOMPOSITION CONFIRMED:")
        print(f"    Confusion (Probe A): {np.mean(angles_a[:10]):.1f}° from top PCs → null-space signal")
        print(f"    Boundary probe:      {np.mean(angles_b[:10]):.1f}° from top PCs → high-variance signal")
        print(f"    Mutual angle:        {a_ab:.1f}° → orthogonal signals")
        print("  h_t encodes: observation content (top PCs) + imagination mode (boundary)")
        print("  + within-trajectory confusion (null space). All three orthogonal.")
    elif a_ab > 75:
        print(f"  Confusion and boundary directions are approximately orthogonal ({a_ab:.1f}°).")
        print("  Subspace decomposition partially supported.")
    else:
        print(f"  Confusion and boundary directions are NOT orthogonal ({a_ab:.1f}°).")
        print("  Subspace decomposition not supported.")

    # ── Figure ──
    print("\nGenerating boundary geometry figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Boundary Direction Geometry — Three Orthogonal Subspaces',
                 fontsize=12, fontweight='bold')

    # Panel 1: Angle vs PC rank for both probes
    ax = axes[0]
    ax.plot(range(1, N_PCS+1), angles_a, 'b-o', markersize=3, linewidth=1.2,
            label=f'Confusion (Probe A)  mean={np.mean(angles_a[:10]):.1f}°')
    ax.plot(range(1, N_PCS+1), angles_b, 'r-s', markersize=3, linewidth=1.2,
            label=f'Boundary probe  mean={np.mean(angles_b[:10]):.1f}°')
    ax.axhline(90, color='gray', linestyle='--', linewidth=0.8, label='Orthogonal (90°)')
    ax.axhline(45, color='lightgray', linestyle=':', linewidth=0.8)
    ax.set_xlabel('PCA component rank')
    ax.set_ylabel('Angle with probe direction (°)')
    ax.set_title('Angle between probe directions and PCA components\n'
                 '(confusion → null-space; boundary → high-variance)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Panel 2: Variance fraction in top-k PCs
    ax = axes[1]
    k_vals = range(1, N_PCS+1)
    pf_a_cumul = [pf(probe_a_sc,  k) / (np.linalg.norm(probe_a_sc)**2  + 1e-9) * 100
                  for k in k_vals]
    pf_b_cumul = [pf(boundary_sc, k) / (np.linalg.norm(boundary_sc)**2 + 1e-9) * 100
                  for k in k_vals]
    ax.plot(k_vals, pf_a_cumul, 'b-', linewidth=1.5,
            label=f'Confusion (Probe A)  — 9% at top-50')
    ax.plot(k_vals, pf_b_cumul, 'r-', linewidth=1.5,
            label=f'Boundary probe')
    ax.set_xlabel('Number of top PCA components included')
    ax.set_ylabel('% of probe direction in top-k PCs')
    ax.set_title('Probe direction captured by top-k PCA components\n'
                 '(confusion stays near 0 — it lives in the null space)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(9, color='blue', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.text(2, 11, '9% (confusion)', color='blue', fontsize=7, alpha=0.7)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'boundary_geometry.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    main()
