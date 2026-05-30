#!/usr/bin/env python3.11
"""
Task 5 — Confusion vs novelty dissociation: 2×2 heatmap.

States split into 4 quadrants by (KL ≷ median) × (recon ≷ median):
  Q1 familiar/coping   (lo KL, lo recon)
  Q2 familiar/confused (lo KL, hi recon)
  Q3 novel/coping      (hi KL, lo recon)
  Q4 novel/confused    (hi KL, hi recon)

Probe A and ensemble disagreement are evaluated in each quadrant.

Expected dissociation:
  Probe A     — high in confused quadrants (right col), insensitive to KL level
  Ensemble    — high in novel quadrants (bottom row), insensitive to recon level

Data: Set A + Set B (both have ensemble disagreement as ens_var).
Probe trained on training_states (separate from evaluation pool).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ── Train Probe A ──
    print("Loading training states...")
    tr = dict(np.load(cfg['training_data_path']))
    y_tr = binarise_by_median(tr['kl'])
    tr_idx, _ = train_test_split(
        np.arange(len(tr['h'])), test_size=0.40, stratify=y_tr, random_state=0)
    clf, sc = train_probe(tr['h'][tr_idx], y_tr[tr_idx])
    print(f"  Probe trained on {len(tr_idx):,} states")

    # ── Load evaluation pool: Set A + Set B with ens_var ──
    print("\nLoading Set A + Set B (with ensemble disagreement)...")
    sa = dict(np.load('outputs/data/set_a_rwmu.npz'))
    sb = dict(np.load('outputs/data/set_b_rwmu.npz'))

    h     = np.concatenate([sa['h'],       sb['h']],       axis=0)
    kl    = np.concatenate([sa['kl'],      sb['kl']],      axis=0)
    recon = np.concatenate([sa['recon'],   sb['recon']],   axis=0)
    ens   = np.concatenate([sa['ens_var'], sb['ens_var']], axis=0)
    print(f"  {len(h):,} states  |  KL [{kl.min():.1f}, {kl.max():.1f}] mean={kl.mean():.1f}")
    print(f"  recon [{recon.min():.4f}, {recon.max():.3f}] mean={recon.mean():.4f}")
    print(f"  ens_var [{ens.min():.5f}, {ens.max():.4f}] mean={ens.mean():.5f}")

    # ── Probe scores ──
    probe = clf.predict_proba(sc.transform(h))[:, 1]

    # ── 4 quadrants by joint (KL, recon) median ──
    kl_med    = np.median(kl)
    recon_med = np.median(recon)
    print(f"\n  KL median:    {kl_med:.2f}")
    print(f"  recon median: {recon_med:.4f}")

    lo_kl  = kl    < kl_med
    hi_kl  = kl    >= kl_med
    lo_r   = recon < recon_med
    hi_r   = recon >= recon_med

    quad_masks = {
        'familiar_coping':   lo_kl & lo_r,
        'familiar_confused': lo_kl & hi_r,
        'novel_coping':      hi_kl & lo_r,
        'novel_confused':    hi_kl & hi_r,
    }

    # ── Print quadrant statistics ──
    print("\n" + "="*75)
    print("CONFUSION VS NOVELTY DISSOCIATION — 2×2 QUADRANT ANALYSIS")
    print("="*75)
    print(f"\n  {'Quadrant':<25} {'N':>6}  {'probe':>7}  {'ens_var':>8}  {'KL mean':>8}  {'recon mean':>10}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}")

    q_probe, q_ens, q_n = {}, {}, {}
    for name, mask in quad_masks.items():
        n = mask.sum()
        mp = probe[mask].mean()
        me = ens[mask].mean()
        mk = kl[mask].mean()
        mr = recon[mask].mean()
        q_probe[name] = mp
        q_ens[name]   = me
        q_n[name]     = n
        print(f"  {name:<25}  {n:>6}  {mp:>7.4f}  {me:>8.5f}  {mk:>8.2f}  {mr:>10.4f}")

    # ── 2×2 tables ──
    print("\n  Probe A — 2×2 mean score:")
    print(f"  {'':22}  {'Coping (lo recon)':>20}  {'Confused (hi recon)':>20}")
    print(f"  {'Familiar (lo KL)':22}  {q_probe['familiar_coping']:>20.4f}  {q_probe['familiar_confused']:>20.4f}")
    print(f"  {'Novel (hi KL)':22}  {q_probe['novel_coping']:>20.4f}  {q_probe['novel_confused']:>20.4f}")

    print("\n  Ensemble disagreement — 2×2 mean ens_var:")
    print(f"  {'':22}  {'Coping (lo recon)':>20}  {'Confused (hi recon)':>20}")
    print(f"  {'Familiar (lo KL)':22}  {q_ens['familiar_coping']:>20.5f}  {q_ens['familiar_confused']:>20.5f}")
    print(f"  {'Novel (hi KL)':22}  {q_ens['novel_coping']:>20.5f}  {q_ens['novel_confused']:>20.5f}")

    # ── Dissociation metrics ──
    # Sensitivity = how much the signal changes along each axis (averaged over the other)
    probe_recon_sens = 0.5 * (
        (q_probe['familiar_confused'] - q_probe['familiar_coping']) +
        (q_probe['novel_confused']    - q_probe['novel_coping'])
    )
    probe_kl_sens = 0.5 * (
        (q_probe['novel_coping']   - q_probe['familiar_coping']) +
        (q_probe['novel_confused'] - q_probe['familiar_confused'])
    )
    ens_recon_sens = 0.5 * (
        (q_ens['familiar_confused'] - q_ens['familiar_coping']) +
        (q_ens['novel_confused']    - q_ens['novel_coping'])
    )
    ens_kl_sens = 0.5 * (
        (q_ens['novel_coping']   - q_ens['familiar_coping']) +
        (q_ens['novel_confused'] - q_ens['familiar_confused'])
    )

    print("\n  Dissociation metrics (sensitivity = avg change along each axis):")
    print(f"  {'':30}  {'recon axis (confusion)':>22}  {'KL axis (novelty)':>18}")
    print(f"  {'Probe A':30}  {probe_recon_sens:>+22.4f}  {probe_kl_sens:>+18.4f}")
    print(f"  {'Ensemble disagreement':30}  {ens_recon_sens:>+22.5f}  {ens_kl_sens:>+18.5f}")

    probe_dominance  = abs(probe_recon_sens) / (abs(probe_recon_sens) + abs(probe_kl_sens) + 1e-9)
    ens_dominance    = abs(ens_kl_sens) / (abs(ens_kl_sens) + abs(ens_recon_sens) + 1e-9)
    print(f"\n  Probe A: {probe_dominance*100:.0f}% of sensitivity on recon axis (confusion)")
    print(f"  Ensemble: {ens_dominance*100:.0f}% of sensitivity on KL axis (novelty)")

    if probe_recon_sens > 0.05 and ens_kl_sens > ens_recon_sens:
        print("\n  CLEAN DISSOCIATION confirmed — probe tracks confusion, ensemble tracks novelty.")
    elif probe_recon_sens > 0.02:
        print("\n  PARTIAL DISSOCIATION — some separation of signals.")
    else:
        print("\n  WEAK DISSOCIATION — signals not clearly separated.")

    # ── Figure ──
    print("\nGenerating heatmap figure...")

    # Build 2×2 arrays (rows=KL, cols=recon)
    probe_mat = np.array([
        [q_probe['familiar_coping'], q_probe['familiar_confused']],
        [q_probe['novel_coping'],    q_probe['novel_confused']],
    ])
    ens_mat = np.array([
        [q_ens['familiar_coping'], q_ens['familiar_confused']],
        [q_ens['novel_coping'],    q_ens['novel_confused']],
    ])
    n_mat = np.array([
        [q_n['familiar_coping'], q_n['familiar_confused']],
        [q_n['novel_coping'],    q_n['novel_confused']],
    ])

    row_labels = ['Familiar\n(low KL)', 'Novel\n(high KL)']
    col_labels = ['Coping\n(low recon)', 'Confused\n(high recon)']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Confusion vs Novelty Dissociation', fontsize=13, fontweight='bold', y=1.02)

    for ax, mat, title, cmap, fmt in [
        (axes[0], probe_mat, 'Probe A (within-task confusion signal)', 'Blues', '.3f'),
        (axes[1], ens_mat,   'Ensemble disagreement (novelty signal)', 'Oranges', '.5f'),
    ]:
        vmin, vmax = mat.min(), mat.max()
        im = ax.imshow(mat, cmap=cmap, vmin=vmin * 0.9, vmax=vmax * 1.05, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_xlabel('Reconstruction error', fontsize=10)
        ax.set_ylabel('KL divergence', fontsize=10)
        ax.set_title(title, fontsize=10, pad=8)
        # Annotate cells with value and N
        for r in range(2):
            for c in range(2):
                val = mat[r, c]
                n   = n_mat[r, c]
                # white text on dark cells, dark on light
                brightness = (val - vmin) / (vmax - vmin + 1e-9)
                text_color = 'white' if brightness > 0.55 else 'black'
                ax.text(c, r, f'{val:{fmt}}\n(n={n:,})',
                        ha='center', va='center', fontsize=10,
                        color=text_color, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'confusion_novelty_heatmap.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    main()
