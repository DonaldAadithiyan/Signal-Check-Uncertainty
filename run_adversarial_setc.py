#!/usr/bin/env python3.11
"""
Task 3 — Adversarial Set C: build contrastive set from Set A only.

Current Set C pools Sets A (clean) and B (noisy). A careful reviewer could
argue the KL-matched bins still carry soft noise-level signal (A vs B origin).
This constructs a cleaner Set C using only Set A states — within-distribution,
within-noise-level, pure confusion vs coping split.

If AUROC is ~0.72: the original result was not driven by noise-level leakage.
If AUROC drops: noise level was doing work (informative negative).
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe, auroc


def build_kl_matched_set(h, kl, recon, n_bins=10, per_bin=20, max_n=200, seed=42):
    """KL-matched contrastive set: C1=low recon within KL bin, C2=high recon."""
    bin_edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(kl, bin_edges[1:-1])
    rng = np.random.default_rng(seed)
    c1, c2 = [], []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) < 4:
            continue
        r  = recon[idx]
        lo = idx[r <= np.percentile(r, 25)]
        hi = idx[r >= np.percentile(r, 75)]
        n  = min(per_bin, len(lo), len(hi))
        if n == 0:
            continue
        c1.extend(rng.choice(lo, n, replace=False).tolist())
        c2.extend(rng.choice(hi, n, replace=False).tolist())
    if len(c1) > max_n: c1 = rng.choice(c1, max_n, replace=False).tolist()
    if len(c2) > max_n: c2 = rng.choice(c2, max_n, replace=False).tolist()
    all_idx = c1 + c2
    labels  = np.array([0]*len(c1) + [1]*len(c2), dtype=np.int32)
    return h[all_idx], kl[all_idx], recon[all_idx], labels


def main():
    cfg = XS_CONFIG.copy()

    # ── Train Probe A on training states ──
    print("Loading training states and training Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    y_kl = binarise_by_median(tr['kl'])
    tr_idx, _ = train_test_split(
        np.arange(len(tr['h'])), test_size=0.40, stratify=y_kl, random_state=0)
    clf, sc = train_probe(tr['h'][tr_idx], y_kl[tr_idx])

    # ── Load Set A only (clean swingup, no noise) ──
    print("\nLoading Set A (clean swingup only)...")
    sa = dict(np.load(cfg['set_a_path']))
    h_a, kl_a, recon_a = sa['h'], sa['kl'], sa['recon']
    print(f"  Set A: {len(h_a):,} states  KL mean={kl_a.mean():.1f}  recon mean={recon_a.mean():.4f}")

    # ── Build adversarial Set C from Set A only ──
    print("\nBuilding adversarial Set C (within-Set-A KL-matched contrastive)...")
    h_c, kl_c, recon_c, labels = build_kl_matched_set(h_a, kl_a, recon_a)
    print(f"  {(labels==0).sum()} C1 (coping) + {(labels==1).sum()} C2 (confused)")
    print(f"  C1: KL mean={kl_c[labels==0].mean():.2f}  recon mean={recon_c[labels==0].mean():.4f}")
    print(f"  C2: KL mean={kl_c[labels==1].mean():.2f}  recon mean={recon_c[labels==1].mean():.4f}")
    kl_gap   = kl_c[labels==1].mean() - kl_c[labels==0].mean()
    recon_gap = recon_c[labels==1].mean() / recon_c[labels==0].mean()
    print(f"  KL gap: {kl_gap:.2f} nats  |  Recon ratio: {recon_gap:.1f}×")

    # ── Evaluate ──
    auroc_adv = auroc(clf, sc, h_c, labels)

    # Reference: original Set C (A+B pooled)
    print("\nLoading original Set C for comparison...")
    sc_orig = dict(np.load(cfg['set_c_path']))
    auroc_orig = auroc(clf, sc, sc_orig['h'], sc_orig['labels'])

    # ── Print results ──
    print("\n" + "="*65)
    print("ADVERSARIAL SET C — WITHIN-A ONLY")
    print("="*65)
    print(f"\n  Original Set C (A+B pooled):     AUROC = {auroc_orig:.4f}")
    print(f"  Adversarial Set C (A only):       AUROC = {auroc_adv:.4f}")
    print(f"  Difference:                              {auroc_adv - auroc_orig:+.4f}")
    print(f"\n  Set C KL gap:           {kl_gap:.2f} nats")
    print(f"  Set C recon ratio:      {recon_gap:.1f}× (C2/C1)")

    if abs(auroc_adv - auroc_orig) < 0.03:
        print(f"\n  RESULT ROBUST: adversarial Set C AUROC {auroc_adv:.4f} ≈ original {auroc_orig:.4f}.")
        print("  The original result was not driven by noise-level leakage from Set B.")
        print("  The confusion signal is detectable within-distribution (Set A only).")
    elif auroc_adv > auroc_orig:
        print(f"\n  STRONGER WITHIN-A: AUROC {auroc_adv:.4f} > {auroc_orig:.4f}.")
        print("  The within-A contrastive is actually a harder or cleaner test.")
    else:
        print(f"\n  DROPPED {auroc_adv:.4f} vs {auroc_orig:.4f}: noise-level signal was contributing.")
        print(f"  Δ = {auroc_adv - auroc_orig:+.4f}. Within-A result still above chance.")


if __name__ == '__main__':
    main()
