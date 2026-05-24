"""Generate all figures for Phase 1 report."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _get_roc(clf, scaler, X, y):
    if len(np.unique(y)) < 2:
        return None, None, float('nan')
    X_s  = scaler.transform(X)
    prob = clf.predict_proba(X_s)[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    return fpr, tpr, auc


def plot_roc_curves(probe_a_result, training_states, set_a, set_b, set_c, out_path):
    """ROC curves for Probe A on all three evaluation sets."""
    clf    = probe_a_result['clf']
    scaler = probe_a_result['scaler']
    kl_median = probe_a_result['kl_median']

    y_a = (set_a['kl'] > kl_median).astype(np.int32)
    y_b = (set_b['kl'] > kl_median).astype(np.int32)
    y_c = set_c['labels']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Probe A (KL gap) – ROC curves', fontsize=12)

    sets = [
        ('Set A held-out (ID)',   set_a['h'], y_a),
        ('Set B OOD',             set_b['h'], y_b),
        ('Set C contrastive',     set_c['h'], y_c),
    ]
    for ax, (title, X, y) in zip(axes, sets):
        fpr, tpr, auc = _get_roc(clf, scaler, X, y)
        if fpr is not None:
            ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC={auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(title)
        ax.legend(loc='lower right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[figures] saved {out_path}")


def plot_block_heatmap(block_results, out_path):
    """Bar chart of per-quarter AUROC."""
    quarters = list(block_results.keys())
    auroc_tr = [block_results[q]['auroc_train'] for q in quarters]
    auroc_a  = [block_results[q]['auroc_a']     for q in quarters]

    x = np.arange(len(quarters))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, auroc_tr, w, label='Train held-out', color='steelblue')
    ax.bar(x + w/2, auroc_a,  w, label='Set A',          color='coral')
    ax.axhline(0.5, color='k', linestyle='--', lw=1, label='Chance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{q}\n({block_results[q]["dims"]})' for q in quarters])
    ax.set_ylabel('AUROC')
    ax.set_title('Per-block contribution to KL probe (h_t quarters)')
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[figures] saved {out_path}")


def plot_kl_distribution(training_states, set_a, set_b, out_path):
    """Distribution of KL values for training vs Set A vs Set B."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = 50
    ax.hist(training_states['kl'], bins=bins, alpha=0.5, label='Training',  density=True)
    ax.hist(set_a['kl'],           bins=bins, alpha=0.5, label='Set A (ID)', density=True)
    ax.hist(set_b['kl'],           bins=bins, alpha=0.5, label='Set B (OOD)', density=True)
    ax.set_xlabel('KL divergence (nats)')
    ax.set_ylabel('Density')
    ax.set_title('KL gap distribution: Training vs Evaluation Sets')
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[figures] saved {out_path}")


def plot_training_curve(loss_log_path, out_path):
    """Plot training loss over time (if a loss log file exists)."""
    if not os.path.exists(loss_log_path):
        return
    losses = np.load(loss_log_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.6)
    window = max(1, len(losses) // 100)
    smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
    ax.plot(np.arange(window - 1, len(losses)), smoothed, color='red', lw=2)
    ax.set_xlabel('Gradient step')
    ax.set_ylabel('Loss')
    ax.set_title('World model training loss')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[figures] saved {out_path}")
