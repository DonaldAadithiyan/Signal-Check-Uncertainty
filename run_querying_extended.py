#!/usr/bin/env python3.11
"""
Tasks 4 & 5 combined:

Task 4 — Extended active querying with proper baselines:
  Adds KL-threshold and recon-threshold policies to the Task 3 analysis.
  If probe outperforms BOTH scalar baselines at most query budgets,
  this proves trajectory-history context in h_t adds value beyond
  what any current-step scalar can provide.

Task 5 — Partial correlation analysis for the dissociation:
  The raw 2×2 showed partial dissociation (probe 58% recon-sensitive,
  ensemble 44% KL-sensitive). Partial correlation controls for the
  confound:
    partial_corr(probe, recon | KL) — probe sensitivity to recon after
      removing the KL component
    partial_corr(ensemble, KL | recon) — ensemble sensitivity to KL after
      removing the recon component
  If both partial correlations are significant, the dissociation holds
  in a statistically rigorous sense even though raw KL/recon are correlated.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe


BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


def recall_at_budget(scores_norm, kl, budget, high_kl_mask):
    """Fraction of high-KL events that get queried at given budget."""
    threshold = np.percentile(scores_norm, 100 * (1 - budget))
    queried   = scores_norm >= threshold
    recall    = (queried & high_kl_mask).sum() / (high_kl_mask.sum() + 1e-9)
    mean_imag_kl = kl[~queried].mean() if (~queried).sum() > 0 else float('nan')
    return recall, mean_imag_kl


def partial_correlation(x, y, z):
    """Pearson r(x, y) controlling for z (residualise both on z)."""
    z_col = z.reshape(-1, 1)
    res_x = x - LinearRegression().fit(z_col, x).predict(z_col)
    res_y = y - LinearRegression().fit(z_col, y).predict(z_col)
    r, p  = pearsonr(res_x, res_y)
    return r, p


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # TASK 4 — Extended active querying with scalar baselines
    # ══════════════════════════════════════════════════════════════════
    print("Loading training states and training Probe A (Task 4)...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all    = tr['h']
    kl_all   = tr['kl']
    recon_all = tr['recon']
    N        = len(h_all)

    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y_kl[tr_idx])

    h_te    = h_all[te_idx]
    kl_te   = kl_all[te_idx]
    recon_te = recon_all[te_idx]
    N_te    = len(te_idx)

    # Rank-normalise all scores to [0,1]
    probe_raw  = clf.predict_proba(sc.transform(h_te))[:, 1]
    probe_norm  = (rankdata(probe_raw)  / N_te).astype(np.float32)
    kl_norm     = (rankdata(kl_te)      / N_te).astype(np.float32)
    recon_norm  = (rankdata(recon_te)   / N_te).astype(np.float32)
    rng = np.random.default_rng(42)
    random_norm = rng.uniform(0, 1, N_te).astype(np.float32)

    kl_75th     = np.percentile(kl_te, 75)
    high_kl_mask = kl_te >= kl_75th

    print(f"  N_te={N_te:,}  KL 75th={kl_75th:.1f}  high-KL events={high_kl_mask.sum():,}")

    # Compute recall curves for all 5 policies
    policies = {
        'Probe A':     probe_norm,
        'KL oracle':   kl_norm,
        'Recon oracle': recon_norm,
        'Random':      random_norm,
    }

    recalls  = {k: [] for k in policies}
    imag_kls = {k: [] for k in policies}
    for budget in BUDGETS:
        for name, scores in policies.items():
            r, ik = recall_at_budget(scores, kl_te, budget, high_kl_mask)
            recalls[name].append(r)
            imag_kls[name].append(ik)

    # ── Print table ──
    print("\n" + "="*75)
    print("TASK 4 — ACTIVE QUERYING: RECALL vs BUDGET (4 POLICIES)")
    print("="*75)
    header = f"  {'Budget':>8}  " + "  ".join(f"{'Recall '+n[:6]:>12}" for n in policies)
    print(header)
    print("  " + "-"*8 + "  " + "  ".join("-"*12 for _ in policies))
    for i, b in enumerate(BUDGETS):
        row = f"  {b:>8.0%}  " + "  ".join(f"{recalls[n][i]:>12.3f}" for n in policies)
        print(row)

    # AUC of recall curve (trapezoid)
    print(f"\n  AUC (recall vs budget):")
    for name, r_list in recalls.items():
        auc = np.trapezoid(r_list, BUDGETS)
        print(f"    {name:<20}: {auc:.4f}")

    # Normalised vs random and oracle
    auc_probe  = np.trapezoid(recalls['Probe A'], BUDGETS)
    auc_kl     = np.trapezoid(recalls['KL oracle'], BUDGETS)
    auc_recon  = np.trapezoid(recalls['Recon oracle'], BUDGETS)
    auc_random = np.trapezoid(recalls['Random'], BUDGETS)
    print(f"\n  Normalised probe AUC (vs random, KL, recon upper bounds):")
    print(f"    (probe - random) / (KL_oracle - random):    "
          f"{(auc_probe-auc_random)/(auc_kl-auc_random+1e-9):.3f}")
    print(f"    (probe - random) / (recon_oracle - random): "
          f"{(auc_probe-auc_random)/(auc_recon-auc_random+1e-9):.3f}")

    # Probe vs KL difference at 30% budget
    r30_probe = recalls['Probe A'][BUDGETS.index(0.30)]
    r30_kl    = recalls['KL oracle'][BUDGETS.index(0.30)]
    r30_recon = recalls['Recon oracle'][BUDGETS.index(0.30)]
    r30_rand  = recalls['Random'][BUDGETS.index(0.30)]
    print(f"\n  At 30% query budget:")
    print(f"    Probe A:      recall={r30_probe:.3f}")
    print(f"    KL oracle:    recall={r30_kl:.3f}")
    print(f"    Recon oracle: recall={r30_recon:.3f}")
    print(f"    Random:       recall={r30_rand:.3f}")

    if r30_probe > r30_kl * 0.85:
        print(f"\n  Probe nearly matches KL oracle at 30% budget ({r30_probe:.3f} vs {r30_kl:.3f}).")
    if r30_probe > r30_recon:
        print(f"  Probe OUTPERFORMS recon oracle at 30% budget ({r30_probe:.3f} vs {r30_recon:.3f}).")
        print("  Trajectory-history context adds value beyond current-step recon signal.")
    elif r30_probe > r30_recon * 0.95:
        print(f"  Probe approximately matches recon oracle ({r30_probe:.3f} ≈ {r30_recon:.3f}).")
    else:
        print(f"  Recon oracle beats probe ({r30_recon:.3f} > {r30_probe:.3f}).")
        print("  Current-step recon is stronger than h_t trajectory history for querying.")

    # ══════════════════════════════════════════════════════════════════
    # TASK 5 — Partial correlation analysis
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("TASK 5 — PARTIAL CORRELATION ANALYSIS (DISSOCIATION FIX)")
    print("="*65)

    print("\nLoading Set A + Set B (with ens_var)...")
    sa = dict(np.load('outputs/data/set_a_rwmu.npz'))
    sb = dict(np.load('outputs/data/set_b_rwmu.npz'))
    h_eval    = np.concatenate([sa['h'],       sb['h']],       axis=0)
    kl_eval   = np.concatenate([sa['kl'],      sb['kl']],      axis=0)
    recon_eval = np.concatenate([sa['recon'],  sb['recon']],   axis=0)
    ens_eval  = np.concatenate([sa['ens_var'], sb['ens_var']], axis=0)

    probe_eval = clf.predict_proba(sc.transform(h_eval))[:, 1]

    # Raw correlations
    r_probe_recon, _ = pearsonr(probe_eval, recon_eval)
    r_probe_kl,    _ = pearsonr(probe_eval, kl_eval)
    r_ens_kl,      _ = pearsonr(ens_eval,   kl_eval)
    r_ens_recon,   _ = pearsonr(ens_eval,   recon_eval)

    print(f"\n  Raw correlations:")
    print(f"  {'':35}  {'r':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    print(f"  {'r(probe, recon)':35}  {r_probe_recon:>8.4f}")
    print(f"  {'r(probe, KL)':35}  {r_probe_kl:>8.4f}")
    print(f"  {'r(ensemble, KL)':35}  {r_ens_kl:>8.4f}")
    print(f"  {'r(ensemble, recon)':35}  {r_ens_recon:>8.4f}")
    print(f"  {'r(KL, recon)':35}  {pearsonr(kl_eval, recon_eval)[0]:>8.4f}")

    # Log-transform recon (heavily skewed)
    log_recon = np.log1p(recon_eval)
    log_ens   = np.log1p(ens_eval)
    log_kl    = np.log1p(kl_eval)

    # Partial correlations
    # Probe sensitivity to recon controlling for KL
    pc_probe_recon_given_kl,  p1 = partial_correlation(probe_eval, log_recon, log_kl)
    # Probe sensitivity to KL controlling for recon
    pc_probe_kl_given_recon,  p2 = partial_correlation(probe_eval, log_kl,    log_recon)
    # Ensemble sensitivity to KL controlling for recon
    pc_ens_kl_given_recon,    p3 = partial_correlation(log_ens,    log_kl,    log_recon)
    # Ensemble sensitivity to recon controlling for KL
    pc_ens_recon_given_kl,    p4 = partial_correlation(log_ens,    log_recon, log_kl)

    print(f"\n  Partial correlations (log-transformed, controlling for confound):")
    print(f"  {'':50}  {'partial r':>10}  {'p-value':>10}")
    print(f"  {'-'*50}  {'-'*10}  {'-'*10}")
    print(f"  {'r(probe, recon | KL)  [confusion signal]':50}  "
          f"{pc_probe_recon_given_kl:>10.4f}  {p1:>10.2e}")
    print(f"  {'r(probe, KL | recon)  [novelty leakage]':50}  "
          f"{pc_probe_kl_given_recon:>10.4f}  {p2:>10.2e}")
    print(f"  {'r(ensemble, KL | recon)  [novelty signal]':50}  "
          f"{pc_ens_kl_given_recon:>10.4f}  {p3:>10.2e}")
    print(f"  {'r(ensemble, recon | KL)  [confusion leakage]':50}  "
          f"{pc_ens_recon_given_kl:>10.4f}  {p4:>10.2e}")

    probe_confusion   = pc_probe_recon_given_kl
    probe_novelty     = pc_probe_kl_given_recon
    ens_novelty       = pc_ens_kl_given_recon
    ens_confusion     = pc_ens_recon_given_kl

    print(f"\n  Dissociation ratios (partial correlations):")
    print(f"  Probe:    recon|KL={probe_confusion:.4f}  vs  KL|recon={probe_novelty:.4f}  "
          f"ratio={probe_confusion/(abs(probe_novelty)+1e-6):.2f}x")
    print(f"  Ensemble: KL|recon={ens_novelty:.4f}  vs  recon|KL={ens_confusion:.4f}  "
          f"ratio={ens_novelty/(abs(ens_confusion)+1e-6):.2f}x")

    if probe_confusion > 0.10 and probe_confusion > abs(probe_novelty) * 1.2:
        print(f"\n  DISSOCIATION CONFIRMED (partial correlations):")
        print(f"  Probe retains significant recon sensitivity after controlling for KL.")
        if ens_novelty > 0.10 and ens_novelty > abs(ens_confusion) * 1.2:
            print(f"  Ensemble retains significant KL sensitivity after controlling for recon.")
            print(f"  The two signals are genuinely different axes of uncertainty.")
        else:
            print(f"  Ensemble KL sensitivity after controlling for recon is weaker.")
    elif probe_confusion > 0.05:
        print(f"\n  PARTIAL DISSOCIATION: probe has some recon signal beyond KL.")
    else:
        print(f"\n  WEAK: partial correlation near zero — probe tracks KL, not recon.")

    # ── Figures ──
    print("\nGenerating figures...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Precision-recall curves
    ax = axes[0]
    colors = {'Probe A': 'blue', 'KL oracle': 'green',
              'Recon oracle': 'orange', 'Random': 'red'}
    for name, r_list in recalls.items():
        ax.plot(BUDGETS, r_list, '-o', color=colors[name], markersize=4,
                linewidth=1.8, label=name)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Query budget (fraction of steps)')
    ax.set_ylabel('Recall of top-25% KL events')
    ax.set_title('Active querying: 4-policy comparison\n'
                 '(probe vs KL, recon, random baselines)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Partial correlation bar chart
    ax = axes[1]
    labels = ['r(probe,\nrecon|KL)', 'r(probe,\nKL|recon)',
              'r(ens,\nKL|recon)', 'r(ens,\nrecon|KL)']
    vals   = [pc_probe_recon_given_kl, pc_probe_kl_given_recon,
              pc_ens_kl_given_recon,   pc_ens_recon_given_kl]
    colors_bar = ['blue', 'steelblue', 'darkorange', 'moccasin']
    bars = ax.bar(labels, vals, color=colors_bar, edgecolor='black', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Partial correlation')
    ax.set_title('Partial correlations: dissociation after controlling for confound\n'
                 '(probe → recon; ensemble → KL)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005 * np.sign(val),
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'querying_extended.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    main()
