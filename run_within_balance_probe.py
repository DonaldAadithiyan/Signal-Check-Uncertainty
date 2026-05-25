#!/usr/bin/env python3.11
"""
Within-balance confound check.

Set C Strong mixes balance h_t vectors (C1) with swingup h_t vectors (C2).
After hundreds of trajectory steps, those two populations have different
distributional fingerprints in h_t regardless of uncertainty — a probe
detecting task identity would score above chance.

This experiment removes that confound entirely:
  C1 (label=0): balance states, low recon within KL bin  — model coping
  C2 (label=1): balance states, high recon within KL bin — model confused

Both groups are balance h_t vectors. Same task identity. Only confusion differs.
The probe is trained on swingup (training_states.npz) and never sees balance.

If AUROC > 0.55 on this set, the probe is detecting genuine internal confusion
that generalises across tasks — not task identity embedded in trajectory history.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.probe.linear_probe import run_probe_a, binarise_by_median, train_probe, auroc, _eval_labels


def build_within_balance_set(balance, n_bins=10, per_bin=20, max_total=200, seed=42):
    """
    Both C1 and C2 from balance trajectories only.
    KL-matched within bins, split by recon.
    """
    kl    = balance['kl']
    recon = balance['recon']

    bin_edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(kl, bin_edges[1:-1])
    rng = np.random.default_rng(seed)

    c1_idx, c2_idx = [], []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) < 4:
            continue
        r = recon[idx]
        c1 = idx[r <= np.percentile(r, 30)]
        c2 = idx[r >= np.percentile(r, 70)]
        n  = min(per_bin, len(c1), len(c2))
        if n == 0:
            continue
        c1_idx.extend(rng.choice(c1, n, replace=False).tolist())
        c2_idx.extend(rng.choice(c2, n, replace=False).tolist())

    if len(c1_idx) > max_total:
        c1_idx = rng.choice(c1_idx, max_total, replace=False).tolist()
    if len(c2_idx) > max_total:
        c2_idx = rng.choice(c2_idx, max_total, replace=False).tolist()

    n1, n2 = len(c1_idx), len(c2_idx)
    out = {
        'h':      np.concatenate([balance['h'][c1_idx],    balance['h'][c2_idx]]),
        'z':      np.concatenate([balance['z'][c1_idx],    balance['z'][c2_idx]]),
        'kl':     np.concatenate([balance['kl'][c1_idx],   balance['kl'][c2_idx]]),
        'recon':  np.concatenate([balance['recon'][c1_idx],balance['recon'][c2_idx]]),
        'obs':    np.concatenate([balance['obs'][c1_idx],  balance['obs'][c2_idx]]),
        'labels': np.array([0]*n1 + [1]*n2, dtype=np.int32),
    }
    print(f"  Within-balance set: {n1} C1 (coping) + {n2} C2 (confused) = {n1+n2} total")
    print(f"  C1 KL: {balance['kl'][c1_idx].mean():.2f} ± {balance['kl'][c1_idx].std():.2f}  "
          f"C2 KL: {balance['kl'][c2_idx].mean():.2f} ± {balance['kl'][c2_idx].std():.2f}")
    print(f"  C1 recon: {balance['recon'][c1_idx].mean():.3f}  "
          f"C2 recon: {balance['recon'][c2_idx].mean():.3f}")
    return out


def main():
    cfg = XS_CONFIG.copy()

    print("Loading training states (swingup)...")
    training = dict(np.load(cfg['training_data_path']))
    kl_median = np.median(training['kl'])

    print("Loading balance states (novel_rwmu.npz)...")
    balance = dict(np.load('outputs/data/novel_rwmu.npz'))
    print(f"  Balance states: {len(balance['h'])}  "
          f"mean KL={balance['kl'].mean():.2f}  mean recon={balance['recon'].mean():.3f}")

    print("\nBuilding within-balance contrastive set...")
    wb = build_within_balance_set(balance)

    # Train probe on swingup training states
    print("\nTraining Probe A on swingup training states...")
    h_tr, kl_tr = training['h'], training['kl']
    y_tr = binarise_by_median(kl_tr)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_train, y_test = train_test_split(
        h_tr, y_tr, test_size=0.40, stratify=y_tr, random_state=0)
    clf, scaler = train_probe(X_tr, y_train)

    # Evaluate
    auroc_id = auroc(clf, scaler, X_te, y_test)
    auroc_wb = auroc(clf, scaler, wb['h'], wb['labels'])

    # z_t probe on within-balance set
    print("Training z_t probe on swingup training states...")
    z_tr = training['z']
    X_ztr, X_zte, y_ztrain, y_ztest = train_test_split(
        z_tr, y_tr, test_size=0.40, stratify=y_tr, random_state=0)
    clf_z, scaler_z = train_probe(X_ztr, y_ztrain)
    auroc_zt_wb = auroc(clf_z, scaler_z, wb['z'], wb['labels'])

    # Load Set C (swingup) for comparison
    print("Loading Set C (within-swingup baseline)...")
    set_c = dict(np.load(cfg['set_c_path'], allow_pickle=True))
    auroc_c = auroc(clf, scaler, set_c['h'], set_c['labels'])

    # Load Set C Strong for comparison
    set_c_strong = dict(np.load('outputs/data/set_c_strong.npz', allow_pickle=True))
    auroc_c_strong = auroc(clf, scaler, set_c_strong['h'], set_c_strong['labels'])

    print("\n=== WITHIN-BALANCE CONFOUND CHECK ===\n")
    print("Both C1 and C2 are balance h_t vectors — same task identity, only confusion differs.")
    print("Probe trained on swingup only.\n")

    headers = ['Test set', 'C1 source', 'C2 source', 'Probe A AUROC', 'Interpretation']
    rows = [
        ['Set C (KL-matched)',     'swingup (low recon)', 'swingup (high recon)',
         f'{auroc_c:.4f}', 'Clean — same task, within-swingup'],
        ['Set C Strong',           'balance (low recon)', 'swingup (high recon)',
         f'{auroc_c_strong:.4f}', 'Confounded — task identity available'],
        ['Within-balance (this)',  'balance (low recon)', 'balance (high recon)',
         f'{auroc_wb:.4f}', 'Clean — same task, within-balance'],
    ]
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---']*len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    print(f"\n  Train held-out (swingup): {auroc_id:.4f}")
    print(f"  Within-balance h_t probe: {auroc_wb:.4f}")
    print(f"  Within-balance z_t probe: {auroc_zt_wb:.4f}")

    print("\n--- Interpretation ---")
    if auroc_wb > 0.60:
        print(f"  AUROC {auroc_wb:.4f} > 0.60 — probe detects confusion within balance.")
        print("  The Set C Strong result is NOT purely task identity detection.")
        print("  h_t encodes internal confusion that generalises across tasks.")
    elif auroc_wb > 0.55:
        print(f"  AUROC {auroc_wb:.4f} — weak signal. Probe partially generalises but not strongly.")
    else:
        print(f"  AUROC {auroc_wb:.4f} ≤ 0.55 — probe does NOT generalise within balance.")
        print("  Set C Strong was likely detecting task identity, not genuine confusion.")

    np.savez('outputs/data/set_c_within_balance.npz', **wb)
    print("\nSaved to outputs/data/set_c_within_balance.npz")


if __name__ == '__main__':
    main()
