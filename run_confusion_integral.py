#!/usr/bin/env python3.11
"""
Confusion integral analysis: closed-form characterisation of what Probe A computes.

Two related questions:
  (1) Streak analysis: does probe score grow monotonically with the length
      of consecutive high-KL steps ending at the current state?
  (2) Confusion integral: does C_t = Σ_{i≥0} γ^i · 1[KL_{t-i} > median]
      predict probe score with high R²?

If C_t predicts probe score with R² > 0.7, the probe is approximately computing
a discounted count of recent confused steps — a closed-form interpretable
characterisation. No probing paper has produced this.

Expected:
  - Probe score monotonically increases with streak length L_t
  - R²(probe ~ C_t) > 0.7 (probe ≈ discounted confusion count)
  - R²(probe ~ KL_t alone) << R²(probe ~ C_t) (streak context matters)
  - Optimal γ for C_t prediction ≈ 0.90–0.99 (slow decay = long memory)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe


MAX_LAG = 50     # maximum look-back for confusion integral
GAMMAS  = [0.70, 0.80, 0.90, 0.95, 0.99]   # discount factors to sweep


def compute_streak_and_integral(kl, traj_id, high_kl, gammas, max_lag):
    """
    For every state i, compute:
      streak[i]: consecutive high-KL steps ending at i (within same trajectory)
      integrals[i, g]: C_t with discount gammas[g]
    """
    N = len(kl)
    streak    = np.zeros(N, dtype=np.int32)
    integrals = np.zeros((N, len(gammas)), dtype=np.float32)

    # Streak length (vectorisable with a single pass)
    for i in range(N):
        if high_kl[i]:
            if i > 0 and traj_id[i] == traj_id[i-1]:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        else:
            streak[i] = 0

    # Confusion integral: look back up to max_lag steps in same trajectory
    for i in range(N):
        for g_idx, gamma in enumerate(gammas):
            val = 0.0
            for lag in range(max_lag):
                j = i - lag
                if j < 0 or traj_id[j] != traj_id[i]:
                    break
                val += (gamma ** lag) * float(high_kl[j])
            integrals[i, g_idx] = val

    return streak, integrals


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ── Load training states and train Probe A ──
    print("Loading training states...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all    = tr['h']
    kl_all   = tr['kl']
    traj_id  = tr['traj_id']
    N        = len(h_all)

    y_kl = binarise_by_median(kl_all)
    kl_median = np.median(kl_all)
    high_kl   = (kl_all > kl_median).astype(np.int32)

    tr_idx, te_idx = train_test_split(
        np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y_kl[tr_idx])
    print(f"  Probe trained on {len(tr_idx):,} states")

    probe_scores_all = clf.predict_proba(sc.transform(h_all))[:, 1]
    probe_te   = probe_scores_all[te_idx]
    kl_te      = kl_all[te_idx]
    high_kl_te = high_kl[te_idx]

    # ── Compute streak and confusion integral on ALL states ──
    print(f"\nComputing streak lengths and confusion integrals (max_lag={MAX_LAG})...")
    streak_all, integrals_all = compute_streak_and_integral(
        kl_all, traj_id, high_kl, GAMMAS, MAX_LAG)

    streak_te    = streak_all[te_idx]
    integrals_te = integrals_all[te_idx]    # (N_te, len(GAMMAS))

    print(f"  Streak length stats: max={streak_te.max()}  mean={streak_te.mean():.2f}  "
          f"median={np.median(streak_te):.1f}")
    print(f"  KL median (threshold): {kl_median:.2f}")

    # ── R²: probe ~ KL_t alone (baseline) ──
    r2_kl = r2_score(probe_te, LinearRegression().fit(
        kl_te.reshape(-1, 1), probe_te).predict(kl_te.reshape(-1, 1)))

    # ── R²: probe ~ C_t (confusion integral) for each γ ──
    print("\n" + "="*65)
    print("CONFUSION INTEGRAL — R² ANALYSIS")
    print("="*65)
    print(f"\n  Baseline: R²(probe ~ KL_t alone) = {r2_kl:.4f}")
    print(f"\n  {'γ':>6}  {'R²(probe ~ C_t)':>16}  {'Δ vs KL baseline':>18}  {'C_t mean':>9}")
    print(f"  {'-'*6}  {'-'*16}  {'-'*18}  {'-'*9}")
    best_r2, best_gamma = -1, None
    for g_idx, gamma in enumerate(GAMMAS):
        ct = integrals_te[:, g_idx]
        r2 = r2_score(probe_te, LinearRegression().fit(
            ct.reshape(-1, 1), probe_te).predict(ct.reshape(-1, 1)))
        delta = r2 - r2_kl
        print(f"  {gamma:>6.2f}  {r2:>16.4f}  {delta:>+18.4f}  {ct.mean():>9.3f}")
        if r2 > best_r2:
            best_r2, best_gamma = r2, gamma

    # ── Best γ: joint regression probe ~ KL_t + C_t ──
    best_g_idx = GAMMAS.index(best_gamma)
    best_ct = integrals_te[:, best_g_idx]
    X_joint = np.column_stack([kl_te, best_ct])
    r2_joint = r2_score(probe_te, LinearRegression().fit(X_joint, probe_te).predict(X_joint))

    print(f"\n  Best γ: {best_gamma}  →  R²(probe ~ C_t) = {best_r2:.4f}")
    print(f"  Joint R²(probe ~ KL_t + C_t, γ={best_gamma}): {r2_joint:.4f}")

    if best_r2 > 0.70:
        print(f"\n  CLOSED-FORM CHARACTERISATION: the probe approximates a discounted")
        print(f"  confusion count C_t with γ={best_gamma}. R²={best_r2:.4f} >> KL baseline {r2_kl:.4f}.")
        print(f"  The probe is reading accumulated recent confusion, not just current KL.")
    elif best_r2 > 0.50:
        print(f"\n  PARTIAL: C_t explains probe better than KL alone (R²={best_r2:.4f} vs {r2_kl:.4f})")
        print(f"  but R² < 0.70. The confusion integral is an approximate characterisation.")
    else:
        print(f"\n  WEAK: C_t does not substantially predict probe score (R²={best_r2:.4f}).")

    # ── Accumulation curve: probe score vs streak length ──
    print("\n" + "="*65)
    print("ACCUMULATION CURVE — PROBE SCORE VS STREAK LENGTH")
    print("="*65)

    max_streak = min(streak_te.max(), 30)
    streak_bins = list(range(0, max_streak + 1))
    streak_means, streak_stds, streak_ns = [], [], []
    for s in streak_bins:
        mask = streak_te == s
        if mask.sum() >= 5:
            streak_means.append(probe_te[mask].mean())
            streak_stds.append(probe_te[mask].std())
            streak_ns.append(mask.sum())
        else:
            streak_means.append(float('nan'))
            streak_stds.append(float('nan'))
            streak_ns.append(0)

    print(f"\n  {'Streak L':>9}  {'N':>6}  {'Mean probe':>11}  {'± std':>8}")
    print(f"  {'-'*9}  {'-'*6}  {'-'*11}  {'-'*8}")
    for s in range(min(15, max_streak + 1)):
        if streak_ns[s] >= 5:
            print(f"  {s:>9}  {streak_ns[s]:>6}  {streak_means[s]:>11.4f}  "
                  f"±{streak_stds[s]:>7.4f}")

    valid = [s for s in streak_bins if streak_ns[s] >= 5]
    valid_means = [streak_means[s] for s in valid]
    if len(valid) >= 3:
        r_streak = np.corrcoef(valid, valid_means)[0, 1]
        print(f"\n  Pearson r (streak length vs probe score): {r_streak:+.4f}")
        if r_streak > 0.90:
            print("  STRONG MONOTONIC GROWTH — probe reads the confusion streak length.")
        elif r_streak > 0.70:
            print("  MODERATE growth — probe partially tracks streak length.")
        else:
            print("  WEAK — probe does not track streak length monotonically.")

    # Compare: R²(probe ~ streak) vs R²(probe ~ KL)
    streak_valid_mask = streak_te > 0  # only high-KL states have non-zero streak
    r2_streak = r2_score(probe_te, LinearRegression().fit(
        streak_te.reshape(-1,1), probe_te).predict(streak_te.reshape(-1,1)))
    print(f"\n  R²(probe ~ streak_length): {r2_streak:.4f}")
    print(f"  R²(probe ~ KL_t alone):    {r2_kl:.4f}")
    print(f"  R²(probe ~ C_t best):      {best_r2:.4f}")

    # ── Figure ──
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Confusion Integral — Probe as Discounted Confusion Counter',
                 fontsize=12, fontweight='bold')

    # Panel 1: R² vs gamma
    ax = axes[0]
    r2_vals = []
    for g_idx, gamma in enumerate(GAMMAS):
        ct = integrals_te[:, g_idx]
        r2 = r2_score(probe_te, LinearRegression().fit(
            ct.reshape(-1, 1), probe_te).predict(ct.reshape(-1, 1)))
        r2_vals.append(r2)
    ax.plot(GAMMAS, r2_vals, 'b-o', markersize=7, linewidth=2)
    ax.axhline(r2_kl, color='r', linestyle='--', linewidth=1.2,
               label=f'KL alone baseline (R²={r2_kl:.3f})')
    ax.set_xlabel('Discount factor γ')
    ax.set_ylabel('R² (probe ~ C_t)')
    ax.set_title('R² vs discount γ\n(peak γ = closed-form memory scale)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 2: scatter probe vs C_t (best gamma)
    ax = axes[1]
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(probe_te), min(3000, len(probe_te)), replace=False)
    ax.scatter(best_ct[idx_sub], probe_te[idx_sub], alpha=0.15, s=5,
               c='steelblue', rasterized=True)
    xs = np.linspace(best_ct.min(), best_ct.max(), 100)
    coeffs = np.polyfit(best_ct, probe_te, 1)
    ax.plot(xs, np.polyval(coeffs, xs), 'r-', linewidth=2,
            label=f'R²={best_r2:.3f}  (γ={best_gamma})')
    ax.set_xlabel(f'Confusion integral C_t (γ={best_gamma})')
    ax.set_ylabel('Probe A score')
    ax.set_title(f'Probe ≈ discounted confusion count\n(γ={best_gamma}, R²={best_r2:.3f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: accumulation curve (streak length vs probe score)
    ax = axes[2]
    valid_s = [s for s in streak_bins if streak_ns[s] >= 5]
    valid_m = [streak_means[s] for s in valid_s]
    valid_e = [streak_stds[s] / np.sqrt(streak_ns[s]) for s in valid_s]
    ax.errorbar(valid_s, valid_m, yerr=valid_e, fmt='b-o', markersize=5,
                linewidth=1.5, capsize=3)
    ax.axhline(probe_te[high_kl_te == 0].mean(), color='g', linestyle='--',
               linewidth=0.8, label=f'Low-KL baseline ({probe_te[high_kl_te==0].mean():.3f})')
    ax.axhline(probe_te.mean(), color='gray', linestyle=':', linewidth=0.8,
               label=f'Overall mean ({probe_te.mean():.3f})')
    ax.set_xlabel('Consecutive high-KL steps ending at t (streak L_t)')
    ax.set_ylabel('Mean probe A score')
    ax.set_title('Accumulation curve\n(monotonic growth = probe reads streak length)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'confusion_integral.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    main()
