#!/usr/bin/env python3.11
"""
Task 2 — Train regression probe directly on C_t (γ=0.95).

Current state: probe trained on binary KL labels approximates C_t at R²=0.80.
This trains a Ridge regression probe directly on C_t values (continuous).

Questions:
  1. What R² does h_t achieve when trained with correct C_t supervision?
     R²=0.80 → R²=0.90+ means C_t is genuinely the right label, not a proxy.
  2. What AUROC does the directly-trained C_t probe achieve on Set C (KL-matched)?
     Higher than binary KL probe (0.72) → C_t is the right signal, not just KL.
  3. Practical recommendation: train on C_t if you want to use this.
"""

import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe, auroc


GAMMA    = 0.95
MAX_LAG  = 50


def compute_ct(kl, traj_id, gamma, max_lag):
    N  = len(kl)
    kl_median = np.median(kl)
    high_kl   = (kl > kl_median).astype(np.float32)
    ct = np.zeros(N, dtype=np.float32)
    for i in range(N):
        val = 0.0
        for lag in range(max_lag):
            j = i - lag
            if j < 0 or traj_id[j] != traj_id[i]:
                break
            val += (gamma ** lag) * high_kl[j]
        ct[i] = val
    return ct


def main():
    cfg = XS_CONFIG.copy()

    print("Loading training states...")
    tr      = dict(np.load(cfg['training_data_path']))
    h_all   = tr['h']
    kl_all  = tr['kl']
    traj_id = tr['traj_id']
    N       = len(h_all)

    print(f"Computing C_t (γ={GAMMA}, max_lag={MAX_LAG})...")
    ct_all = compute_ct(kl_all, traj_id, GAMMA, MAX_LAG)
    print(f"  C_t: mean={ct_all.mean():.3f}  std={ct_all.std():.3f}  "
          f"range=[{ct_all.min():.2f}, {ct_all.max():.2f}]")

    # ── Train/test split ──
    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(
        np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)

    # ── Binary KL probe baseline ──
    clf_kl, sc_kl = train_probe(h_all[tr_idx], y_kl[tr_idx])
    probe_scores_te = clf_kl.predict_proba(sc_kl.transform(h_all[te_idx]))[:, 1]
    r2_kl_vs_ct = r2_score(ct_all[te_idx], probe_scores_te)
    print(f"\n  Binary KL probe: R²(probe_score ~ C_t) = {r2_kl_vs_ct:.4f}  "
          f"(this is what we already knew: 0.80)")

    # ── Ridge regression probe on C_t ──
    print(f"\nTraining Ridge regression probe directly on C_t (γ={GAMMA})...")
    sc_ct  = StandardScaler()
    h_tr_s = sc_ct.fit_transform(h_all[tr_idx])
    h_te_s = sc_ct.transform(h_all[te_idx])

    ridge  = Ridge(alpha=1.0)
    ridge.fit(h_tr_s, ct_all[tr_idx])
    ct_pred_te = ridge.predict(h_te_s)

    r2_direct = r2_score(ct_all[te_idx], ct_pred_te)
    print(f"  R²(h_t → C_t, Ridge) on held-out: {r2_direct:.4f}")
    print(f"  R²(h_t → binary KL) [reference]:  {r2_kl_vs_ct:.4f}")
    print(f"  Improvement: {r2_direct - r2_kl_vs_ct:+.4f}")

    # ── AUROC on Set C using the C_t regression probe ──
    print("\nEvaluating on Set C (KL-matched contrastive)...")
    sc_data = dict(np.load(cfg['set_c_path']))
    h_c    = sc_data['h']
    labels_c = sc_data['labels']

    ct_scores_c = ridge.predict(sc_ct.transform(h_c))
    auroc_direct_c = roc_auc_score(labels_c, ct_scores_c)

    # Binary KL probe on Set C (reference)
    auroc_kl_c = auroc(clf_kl, sc_kl, h_c, labels_c)
    print(f"  AUROC (C_t regression probe, Set C): {auroc_direct_c:.4f}")
    print(f"  AUROC (binary KL probe, Set C):      {auroc_kl_c:.4f}  [reference: 0.7227]")
    print(f"  Δ AUROC: {auroc_direct_c - auroc_kl_c:+.4f}")

    # ── Gamma sweep on direct regression ──
    print("\nGamma sweep — R²(h_t → C_t) for different γ values:")
    gammas = [0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"  {'γ':>6}  {'R²(Ridge)':>10}  {'R²(binary_probe≈C_t)':>21}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*21}")
    best_r2, best_g = -1, None
    for g in gammas:
        ct_g   = compute_ct(kl_all, traj_id, g, MAX_LAG)
        ridge_g = Ridge(alpha=1.0)
        ridge_g.fit(h_tr_s, ct_g[tr_idx])
        r2_g = r2_score(ct_g[te_idx], ridge_g.predict(h_te_s))
        # Binary probe R² vs this C_t
        r2_probe_g = r2_score(ct_g[te_idx], probe_scores_te)
        print(f"  {g:>6.2f}  {r2_g:>10.4f}  {r2_probe_g:>21.4f}")
        if r2_g > best_r2:
            best_r2, best_g = r2_g, g

    print(f"\n  Best direct Ridge R²: {best_r2:.4f} at γ={best_g}")

    # ── Summary ──
    print("\n" + "="*65)
    print("DIRECT C_t PROBE SUMMARY")
    print("="*65)
    print(f"\n  R²(binary KL probe scores ≈ C_t, γ=0.95):   {r2_kl_vs_ct:.4f}")
    print(f"  R²(Ridge regression h_t → C_t, γ=0.95):      {r2_direct:.4f}")
    print(f"  Ceiling remaining:                             {1 - r2_direct:.4f}")
    print(f"\n  AUROC on Set C — binary KL probe:             {auroc_kl_c:.4f}")
    print(f"  AUROC on Set C — direct C_t regression probe: {auroc_direct_c:.4f}")

    if r2_direct > 0.90:
        print("\n  h_t encodes C_t at R²>0.90 with correct supervision.")
        print("  The binary KL proxy was leaving ~10 percentage points on the table.")
        print("  Practical recommendation: train directly on C_t for best performance.")
    elif r2_direct > r2_kl_vs_ct + 0.05:
        print(f"\n  Direct training on C_t improves R² by {r2_direct - r2_kl_vs_ct:+.4f}.")
        print("  C_t is a better supervision target than binary KL.")
    else:
        print(f"\n  Direct training on C_t gives marginal improvement ({r2_direct - r2_kl_vs_ct:+.4f}).")
        print("  The binary KL proxy was near-optimal for C_t prediction.")


if __name__ == '__main__':
    main()
