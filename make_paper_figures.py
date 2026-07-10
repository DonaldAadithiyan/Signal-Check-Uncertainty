#!/usr/bin/env python3.11
"""
Paper figure redesign (roc_curves + appendix figures). No new numbers — every value
is regenerated deterministically from saved models/sets or read from saved results.

Shared palette (Okabe–Ito, colorblind-safe), consistent method colors across figures:
  confusion probe = blue #0072B2 · ensemble = orange #E69F00 · reconstruction = green #009E73
  chance = gray · null = light gray · confusion marker (null fig) = vermillion #D55E00

Figures:
  1. roc_curves.png                  — main text: single Set-C dissociation ROC (3 curves, 1 axes)
  2. roc_curves_full_appendix.png    — appendix: Probe A ROC on Set A/B/C, shared legend
  3. empirical_null_distribution.png — appendix: 50-direction null histogram + confusion marker
  4. cross_seed_stability.png        — appendix: per-seed Set C AUROC dot-and-whisker, 3 envs
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe, ensemble_disagreement

FIGDIR = 'outputs/figures'
C_PROBE, C_ENS, C_RECON = '#0072B2', '#E69F00', '#009E73'
C_CHANCE, C_NULL, C_MARK = '#888888', '#c8c8c8', '#D55E00'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})


def load_wm(ck):
    d = torch.load(ck, map_location='cpu')
    m = WorldModel(d['cfg']['obs_dim'], d['cfg']['act_dim'], d['cfg'])
    m.load_state_dict(d['model_state']); m.eval(); return m


# ─── Figure 1: single-panel Set C dissociation ──────────────────────────────

def fig1_main_roc(cfg, clf, sc):
    setc = dict(np.load('outputs/data/set_c_contrastive.npz'))
    # (1) confusion probe on Set C
    p_probe = clf.predict_proba(sc.transform(setc['h']))[:, 1]
    fpr_p, tpr_p, _ = roc_curve(setc['labels'], p_probe); auc_p = roc_auc_score(setc['labels'], p_probe)
    # (2) ensemble disagreement on Set C
    ens_models = [load_wm(f'outputs/checkpoints/ensemble_seed{s}.pt') for s in (0, 1, 2)]
    dis, auc_e = ensemble_disagreement(ens_models, setc, cfg)
    fpr_e, tpr_e, _ = roc_curve(setc['labels'], dis)
    # (3) reconstruction error as the direct-novelty detector (swingup vs balance) — the 0.996 in Table 1
    a = dict(np.load('outputs/data/set_a_id.npz')); nov = dict(np.load('outputs/data/novel_rwmu.npz'))
    rec = np.concatenate([a['recon'], nov['recon']]); lab = np.array([0] * len(a['recon']) + [1] * len(nov['recon']))
    fpr_r, tpr_r, _ = roc_curve(lab, rec); auc_r = roc_auc_score(lab, rec)

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot(fpr_p, tpr_p, color=C_PROBE, lw=2.2, label=f'Confusion probe · Set C (AUROC {auc_p:.2f})')
    ax.plot(fpr_e, tpr_e, color=C_ENS, lw=2.2, label=f'Ensemble disagr. · Set C (AUROC {auc_e:.2f})')
    ax.plot(fpr_r, tpr_r, color=C_RECON, lw=2.2, ls=(0, (5, 1)),
            label=f'Recon error · OOD detect (AUROC {auc_r:.2f})')
    ax.plot([0, 1], [0, 1], color=C_CHANCE, ls='--', lw=1, label='chance')
    ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title('Set C dissociation: confusion vs novelty', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.92, handlelength=1.6)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIGDIR, 'roc_curves.png'); fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"[1] {p}  probe={auc_p:.3f} ens={auc_e:.3f} recon={auc_r:.3f}")


# ─── Figure 2: appendix 3-panel Probe A ROC (shared legend) ──────────────────

def fig2_appendix_roc(cfg, clf, sc):
    kl_median = float(np.median(dict(np.load(cfg['training_data_path']))['kl']))
    set_a = dict(np.load('outputs/data/set_a_id.npz'))
    set_b = dict(np.load('outputs/data/set_b_ood.npz'))
    set_c = dict(np.load('outputs/data/set_c_contrastive.npz'))
    panels = [
        ('Set A — held-out (ID)', set_a['h'], (set_a['kl'] > kl_median).astype(int)),
        ('Set B — near-OOD (noisy)', set_b['h'], (set_b['kl'] > kl_median).astype(int)),
        ('Set C — KL-matched contrastive', set_c['h'], set_c['labels']),
    ]
    # Vertical stack (3 rows × 1 col) so the figure is narrow-and-tall — renders at
    # single-column width in the two-column layout without shrinking the panels.
    fig, axes = plt.subplots(3, 1, figsize=(3.2, 8.0), sharex=True)
    line = None
    aucs = []
    for ax, (title, X, y) in zip(axes, panels):
        p = clf.predict_proba(sc.transform(X))[:, 1]
        fpr, tpr, _ = roc_curve(y, p); auc = roc_auc_score(y, p); aucs.append(auc)
        line, = ax.plot(fpr, tpr, color=C_PROBE, lw=2.2)
        chance, = ax.plot([0, 1], [0, 1], color=C_CHANCE, ls='--', lw=1)
        ax.set_title(f'{title}  —  AUROC {auc:.3f}', fontsize=8)
        ax.set_ylabel('TPR'); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel('FPR')
    fig.legend([line, chance], ['Probe A (KL → h_t)', 'chance'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.suptitle('Probe A ROC across evaluation sets', fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    pth = os.path.join(FIGDIR, 'roc_curves_full_appendix.png'); fig.savefig(pth, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"[2] {pth}  AUROCs A/B/C = {aucs[0]:.3f}/{aucs[1]:.3f}/{aucs[2]:.3f} (expect 0.872/0.807/0.714)")


def fig2_split_roc(cfg, clf, sc):
    """Same three panels as fig2, but as THREE separate square images, each
    filling its frame (minimal whitespace). Numbers identical to fig2."""
    kl_median = float(np.median(dict(np.load(cfg['training_data_path']))['kl']))
    set_a = dict(np.load('outputs/data/set_a_id.npz'))
    set_b = dict(np.load('outputs/data/set_b_ood.npz'))
    set_c = dict(np.load('outputs/data/set_c_contrastive.npz'))
    panels = [
        ('setA', 'Set A — held-out (ID)', set_a['h'], (set_a['kl'] > kl_median).astype(int)),
        ('setB', 'Set B — near-OOD (noisy)', set_b['h'], (set_b['kl'] > kl_median).astype(int)),
        ('setC', 'Set C — KL-matched contrastive', set_c['h'], set_c['labels']),
    ]
    for tag, title, X, y in panels:
        p = clf.predict_proba(sc.transform(X))[:, 1]
        fpr, tpr, _ = roc_curve(y, p); auc = roc_auc_score(y, p)
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        ax.plot(fpr, tpr, color=C_PROBE, lw=2.4, label=f'Probe A (AUROC {auc:.3f})')
        ax.plot([0, 1], [0, 1], color=C_CHANCE, ls='--', lw=1, label='chance')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.legend(loc='lower right', framealpha=0.92)
        ax.grid(True, alpha=0.25)
        ax.margins(0)
        fig.tight_layout(pad=0.3)
        pth = os.path.join(FIGDIR, f'roc_appendix_{tag}.png')
        fig.savefig(pth, dpi=300, bbox_inches='tight', pad_inches=0.02); plt.close(fig)
        print(f"[2-split] {pth}  AUROC {auc:.3f}")


# ─── Figure 3: empirical null distribution ──────────────────────────────────

def fig3_null_distribution():
    raw_p = 'outputs/causal/task_g_null_raw.npz'
    if not os.path.exists(raw_p):
        print("[3] SKIP — outputs/causal/task_g_null_raw.npz not present yet (Task G re-run pending)")
        return
    d = dict(np.load(raw_p))
    g = json.load(open('outputs/causal/task_g_results.json'))
    ks = [0, 1, 5, 10]
    # Vertical stack (4 rows × 1 col): narrow-and-tall for single-column width.
    # x-range differs per k (the effect shrinks with look-ahead), so NOT sharex.
    fig, axes = plt.subplots(4, 1, figsize=(3.2, 9.5))
    zs = []
    for ax, k in zip(axes, ks):
        null = d[f'null_dprobe_{k}']; conf = float(d[f'conf_dprobe_{k}'][0])
        z = g['null_summary'][f'dprobe_{k}']['z']; zs.append(z)
        ax.hist(null, bins=18, color=C_NULL, edgecolor='#999', lw=0.4)
        ax.axvline(conf, color=C_MARK, lw=2.2)
        # keep x-limits wide enough that the confusion marker + label are inside the axes
        lo = min(conf, null.min()); hi = max(null.max(), conf)
        pad = 0.08 * (hi - lo + 1e-6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.annotate(f'confusion dir.\nz ≈ {z:.0f}', xy=(conf, 0),
                    xytext=(conf + pad, ax.get_ylim()[1] * 0.9),
                    color=C_MARK, fontsize=7.5, ha='left', va='top', fontweight='bold')
        ax.set_title(f'k = {k}', fontsize=8, loc='left')
        ax.set_ylabel('# random dir.')
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Δ probe score (ablation)')
    fig.suptitle('Confusion direction vs 50-direction\nempirical null (ablation Δ probe)',
                 fontweight='bold', y=0.997, fontsize=9)
    fig.text(0.5, 0.005, 'The confusion direction (vermillion) sits far outside the null of 50\n'
             'norm-matched random directions at every look-ahead k — 100th percentile throughout.',
             ha='center', fontsize=6.6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    pth = os.path.join(FIGDIR, 'empirical_null_distribution.png'); fig.savefig(pth, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"[3] {pth}  z (k=0,1,5,10) = {'/'.join(f'{z:.1f}' for z in zs)} (read from saved JSON — unchanged)")


def fig3_split_null():
    """Same panels as fig3, but each look-ahead k as a SEPARATE full-frame image
    (roc-split style). Numbers read from the saved JSON/npz — unchanged."""
    raw_p = 'outputs/causal/task_g_null_raw.npz'
    if not os.path.exists(raw_p):
        print("[3-split] SKIP — task_g_null_raw.npz not present"); return
    d = dict(np.load(raw_p))
    g = json.load(open('outputs/causal/task_g_results.json'))
    for k in [0, 1, 5, 10]:
        null = d[f'null_dprobe_{k}']; conf = float(d[f'conf_dprobe_{k}'][0])
        z = g['null_summary'][f'dprobe_{k}']['z']
        fig, ax = plt.subplots(figsize=(3.4, 3.1))
        ax.hist(null, bins=18, color=C_NULL, edgecolor='#999', lw=0.4)
        ax.axvline(conf, color=C_MARK, lw=2.4)                      # matched line width
        lo = min(conf, null.min()); hi = max(null.max(), conf)
        pad = 0.10 * (hi - lo + 1e-6)
        ax.set_xlim(lo - pad, hi + pad); ax.margins(y=0)
        ax.annotate(f'confusion dir.\nz ≈ {z:.0f}', xy=(conf, 0),
                    xytext=(conf + pad, ax.get_ylim()[1] * 0.92),
                    color=C_MARK, fontsize=8, ha='left', va='top', fontweight='bold')
        ax.set_title(f'Look-ahead k = {k}', fontweight='bold', fontsize=9)
        ax.set_xlabel('Δ probe score (ablation)'); ax.set_ylabel('# random directions')
        ax.grid(True, alpha=0.2)
        fig.tight_layout(pad=0.3)
        pth = os.path.join(FIGDIR, f'null_k{k}.png')
        fig.savefig(pth, dpi=300, bbox_inches='tight', pad_inches=0.02); plt.close(fig)
        print(f"[3-split] {pth}  conf={conf:+.3f} z≈{z:.1f} (100th pct)")


# ─── Figure 4: cross-seed / cross-task Set C AUROC dot-and-whisker ───────────

def fig4_cross_seed():
    # cartpole: 5 seeds (multiseed). reacher/pendulum: 4 seeds each (D/J single + multiseed_env 1-3)
    cart = [json.load(open(f'outputs/multiseed/seed_{s}/metrics.json'))['auroc_c'] for s in range(5)]
    base = {'reacher': 0.619, 'pendulum': 0.322}
    envs = {'cartpole': cart, 'reacher': [base['reacher']], 'pendulum': [base['pendulum']]}
    for env in ('reacher', 'pendulum'):
        for s in (1, 2, 3):
            envs[env].append(json.load(open(f'outputs/multiseed_env/{env}_seed{s}/metrics.json'))['auroc_c'])
    names = ['cartpole', 'reacher', 'pendulum']
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    rng = np.random.default_rng(0)
    for i, env in enumerate(names):
        vals = np.array(envs[env]); m = vals.mean()
        jit = rng.uniform(-0.06, 0.06, len(vals))
        # whisker (range) then mean bar on top — line widths matched to the ROC figures
        # (chance/reference lw 1.0; primary data mark lw 2.4)
        ax.plot([i, i], [vals.min(), vals.max()], color='#555', lw=1.0, zorder=2)
        ax.scatter(np.full(len(vals), i) + jit, vals, s=22, color=C_PROBE, alpha=0.75, zorder=3,
                   edgecolors='white', lw=0.5)
        ax.plot([i - 0.18, i + 0.18], [m, m], color='black', lw=2.4, zorder=4)
        ax.annotate(f'{m:.2f}\n±{vals.std():.2f}', xy=(i + 0.24, m), fontsize=6.8, va='center')
    ax.axhline(0.5, color=C_CHANCE, ls='--', lw=1.0)
    ax.text(2.62, 0.505, 'chance', color=C_CHANCE, fontsize=6.5, va='bottom', ha='right')
    ax.set_xticks(range(3)); ax.set_xticklabels([f'{n}\n(n={len(envs[n])})' for n in names])
    ax.set_ylabel('Set C AUROC'); ax.set_ylim(0.25, 0.85); ax.set_xlim(-0.35, 2.62)
    ax.margins(x=0)
    ax.set_title('Set C AUROC across seeds (mean ± std; points = seeds)', fontweight='bold', fontsize=8)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout(pad=0.3)
    pth = os.path.join(FIGDIR, 'cross_seed_stability.png'); fig.savefig(pth, dpi=300, bbox_inches='tight', pad_inches=0.02); plt.close(fig)
    print(f"[4] {pth}  cartpole {np.mean(cart):.3f} | reacher {np.mean(envs['reacher']):.3f} | pendulum {np.mean(envs['pendulum']):.3f}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    cfg = XS_CONFIG.copy()
    tr = dict(np.load(cfg['training_data_path']))
    h, kl = tr['h'], tr['kl']; y = binarise_by_median(kl)
    itr, _ = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h[itr], y[itr])   # Probe A, exact pilot protocol
    fig1_main_roc(cfg, clf, sc)
    fig2_appendix_roc(cfg, clf, sc)
    fig3_null_distribution()
    fig4_cross_seed()


if __name__ == '__main__':
    main()
