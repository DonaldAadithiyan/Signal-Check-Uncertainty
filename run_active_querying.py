#!/usr/bin/env python3.11
"""
Task 3 — Probe score as active querying signal: Pareto frontier analysis.

The probe detects when the model is confused (high KL). In an active querying
framework, you would collect a real observation when the model is confused
(probe > θ) and continue imagining when it is coping.

This script evaluates the probe as a query-routing oracle on existing trajectory
data (passive analysis — no actual agent required):

  For threshold θ ∈ [0.1, 0.9]:
    "Queried" states = states where probe(h_t) > θ
    "Imagined" states = states where probe(h_t) ≤ θ (model continues without real obs)

  Three baselines at each query rate:
    - Probe-gated:  query states with highest probe scores
    - KL-oracle:    query states with highest actual KL (best possible policy)
    - Random:       query states uniformly at random

  Metrics:
    - Query rate = fraction of steps where real obs collected
    - Mean KL of imagined states = confusion level the model is operating under
      without real observations (lower = better, model is only imagining on easy states)
    - Recall of high-KL events = fraction of top-25% KL states that get queried
      (higher = better — the agent correctly flags confused steps)

  Pareto frontier: for each query rate, which method has the lowest imagined KL?

  Expected:
    Probe-gated Pareto dominates random — at same query budget, the probe
    correctly identifies confused states that benefit from real observations.
    KL-oracle provides the upper bound.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe


THRESHOLDS = np.concatenate([
    np.linspace(0.05, 0.50, 10),
    np.linspace(0.55, 0.95, 9),
])


def pareto_metrics(scores, kl_all, threshold, high_kl_mask):
    """
    Compute querying metrics for a given threshold.

    scores:        probe scores (N,) — higher = more confused = should query
    threshold:     scalar, states with score > threshold get queried
    kl_all:        actual KL values (N,)
    high_kl_mask:  boolean mask for 'high KL events' to detect

    Returns: query_rate, mean_imagined_kl, recall_high_kl
    """
    queried = scores > threshold
    imagined = ~queried

    query_rate      = queried.mean()
    mean_imag_kl    = kl_all[imagined].mean() if imagined.sum() > 0 else float('nan')
    # Recall of high-KL events: fraction of high-KL states that get queried
    recall_high_kl  = (queried & high_kl_mask).sum() / (high_kl_mask.sum() + 1e-9)

    return query_rate, mean_imag_kl, recall_high_kl


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ── Load training states and train Probe A ──
    print("Loading training states and training Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all  = tr['h']
    kl_all = tr['kl']
    N      = len(h_all)

    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(
        np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y_kl[tr_idx])
    print(f"  Trained on {len(tr_idx):,} states")

    # Evaluate on held-out states only (avoids using probe training data)
    h_te   = h_all[te_idx]
    kl_te  = kl_all[te_idx]
    N_te   = len(te_idx)

    probe_scores = clf.predict_proba(sc.transform(h_te))[:, 1]
    # Rank-normalize all scores to [0,1] so the same threshold sweep works for all
    from scipy.stats import rankdata
    kl_scores     = (rankdata(kl_te)     / N_te).astype(np.float32)   # oracle
    probe_scores  = (rankdata(probe_scores) / N_te).astype(np.float32)
    rng = np.random.default_rng(42)
    random_scores = rng.uniform(0, 1, N_te).astype(np.float32)

    # High-KL events: top 25% of KL values
    kl_75th = np.percentile(kl_te, 75)
    high_kl_mask = kl_te >= kl_75th

    print(f"\n  Held-out states: {N_te:,}")
    print(f"  KL range: [{kl_te.min():.1f}, {kl_te.max():.1f}]  "
          f"mean={kl_te.mean():.1f}  75th pct={kl_75th:.1f}")
    print(f"  High-KL events (top 25%): {high_kl_mask.sum():,}")

    # ── Sweep thresholds ──
    probe_results  = []
    oracle_results = []
    random_results = []

    for θ in THRESHOLDS:
        probe_results.append(pareto_metrics(probe_scores,  kl_te, θ, high_kl_mask))
        oracle_results.append(pareto_metrics(kl_scores,    kl_te, θ, high_kl_mask))
        random_results.append(pareto_metrics(random_scores, kl_te, θ, high_kl_mask))

    probe_arr  = np.array(probe_results)   # (T, 3): query_rate, mean_imag_kl, recall
    oracle_arr = np.array(oracle_results)
    random_arr = np.array(random_results)

    # ── Print Pareto table ──
    print("\n" + "="*80)
    print("ACTIVE QUERYING — PARETO ANALYSIS")
    print("="*80)
    print("\nFor each query rate: mean KL of imagined (non-queried) states.")
    print("Lower mean imagined KL = probe correctly identified confused states to query.\n")

    # Select representative query rates for display
    target_rates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    print(f"  {'Query rate':>11}  {'Probe imag KL':>14}  {'Oracle imag KL':>15}  "
          f"{'Random imag KL':>15}  {'Probe recall':>13}  {'Oracle recall':>13}")
    print(f"  {'-'*11}  {'-'*14}  {'-'*15}  {'-'*15}  {'-'*13}  {'-'*13}")

    for target_rate in target_rates:
        # Find closest threshold to target query rate
        def find_closest(arr):
            diffs = np.abs(arr[:, 0] - target_rate)
            return arr[np.argmin(diffs)]
        p = find_closest(probe_arr)
        o = find_closest(oracle_arr)
        r = find_closest(random_arr)
        print(f"  {target_rate:>11.0%}  {p[1]:>14.3f}  {o[1]:>15.3f}  "
              f"{r[1]:>15.3f}  {p[2]:>13.3f}  {o[2]:>13.3f}")

    # ── Improvement over random ──
    print(f"\n  Improvement (probe vs random) at 30% query rate:")
    p30 = probe_arr[np.argmin(np.abs(probe_arr[:, 0] - 0.30))]
    r30 = random_arr[np.argmin(np.abs(random_arr[:, 0] - 0.30))]
    o30 = oracle_arr[np.argmin(np.abs(oracle_arr[:, 0] - 0.30))]
    print(f"    Probe:  imag KL={p30[1]:.3f}  recall={p30[2]:.3f}")
    print(f"    Random: imag KL={r30[1]:.3f}  recall={r30[2]:.3f}")
    print(f"    Oracle: imag KL={o30[1]:.3f}  recall={o30[2]:.3f}")
    kl_improvement = r30[1] - p30[1]
    recall_improvement = p30[2] - r30[2]
    print(f"\n    KL reduction vs random: {kl_improvement:+.3f} nats")
    print(f"    Recall improvement vs random: {recall_improvement:+.3f}")

    if kl_improvement > 0.5 or recall_improvement > 0.10:
        print("\n  POSITIVE: Probe-gated querying reduces confusion on imagined states.")
        print("  The probe correctly identifies which steps need real observations.")
    elif kl_improvement > 0.1 or recall_improvement > 0.03:
        print("\n  MARGINAL: Small but positive effect on querying efficiency.")
    else:
        print("\n  WEAK: Probe routing doesn't substantially improve over random querying.")

    # ── AUROC-style AUC of Pareto curve ──
    # Area under the recall-vs-query-rate curve (probe vs random vs oracle)
    # Higher area = better oracle at all query rates
    def auc_pareto(arr):
        rates = arr[:, 0]
        recall = arr[:, 2]
        sort_idx = np.argsort(rates)
        return np.trapezoid(recall[sort_idx], rates[sort_idx])

    auc_probe  = auc_pareto(probe_arr)
    auc_oracle = auc_pareto(oracle_arr)
    auc_random = auc_pareto(random_arr)

    print(f"\n  AUC (recall vs query-rate curve):")
    print(f"    Probe:  {auc_probe:.4f}")
    print(f"    Oracle: {auc_oracle:.4f}")
    print(f"    Random: {auc_random:.4f}")
    print(f"    Normalised (probe-random)/(oracle-random): "
          f"{(auc_probe-auc_random)/(auc_oracle-auc_random+1e-9):.3f}")

    # ── Figure: 2-panel Pareto plot ──
    print("\nGenerating Pareto frontier figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Active Querying — Probe as Confusion Oracle', fontsize=12, fontweight='bold')

    # Panel 1: query rate vs mean imagined KL
    ax = axes[0]
    ax.plot(probe_arr[:, 0],  probe_arr[:, 1],  'b-o', markersize=4, linewidth=1.5,
            label=f'Probe (AUC={auc_probe:.3f})')
    ax.plot(oracle_arr[:, 0], oracle_arr[:, 1], 'g--s', markersize=4, linewidth=1.5,
            label='Oracle (actual KL)', alpha=0.8)
    ax.plot(random_arr[:, 0], random_arr[:, 1], 'r:', markersize=4, linewidth=1.2,
            label='Random', alpha=0.7)
    ax.axhline(kl_te.mean(), color='gray', linestyle='--', linewidth=0.8,
               label=f'Global mean KL = {kl_te.mean():.1f}', alpha=0.6)
    ax.set_xlabel('Query rate (fraction of steps queried)')
    ax.set_ylabel('Mean KL of imagined states')
    ax.set_title('Pareto frontier: budget vs imagined confusion\n(lower=better — '
                 'probe keeps confused steps queried)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Panel 2: query rate vs recall of high-KL events
    ax = axes[1]
    ax.plot(probe_arr[:, 0],  probe_arr[:, 2],  'b-o', markersize=4, linewidth=1.5,
            label='Probe')
    ax.plot(oracle_arr[:, 0], oracle_arr[:, 2], 'g--s', markersize=4, linewidth=1.5,
            label='Oracle (actual KL)', alpha=0.8)
    ax.plot(random_arr[:, 0], random_arr[:, 2], 'r:', markersize=4, linewidth=1.2,
            label='Random', alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.6, alpha=0.4, label='Diagonal (random)')
    ax.set_xlabel('Query rate (fraction of steps queried)')
    ax.set_ylabel('Recall of top-25% KL events')
    ax.set_title('Probe as confusion oracle: recall vs budget\n(above diagonal = '
                 'probe better than random at same budget)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'active_querying.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    main()
