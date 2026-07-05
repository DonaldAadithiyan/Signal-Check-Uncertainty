#!/usr/bin/env python3.11
"""
Task E — Sanity-check the suspiciously perfect obs/imagination boundary AUROC=1.0.

AUROC 1.0000 for a "linear probe" separating real (posterior) h_t from imagined
(prior) h_t should be interrogated: does a SINGLE scalar feature of h_t already
achieve ≈1.0? If so, the result is a magnitude effect, not a genuinely
distributed linear direction — the framing should say so.

Candidates tested (each alone, no fitted probe):
  * L2 norm ‖h_t‖
  * raw projection onto the single top PCA component of the pooled distribution
  * per-dimension best single coordinate of h_t
  * mean(h_t), std(h_t) across dims

Compared against the full trained linear boundary probe. Ships with N and a
bootstrap 95% CI on each AUROC.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import bootstrap_auroc_ci

N_START = 5_000
HORIZON = 15
OUT_DIR = 'outputs/causal'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def run_imagination(model, h_start, z_start, horizon, cfg, seed=0):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    N = h_start.shape[0]
    h = torch.tensor(h_start, dtype=torch.float32, device=device)
    z = torch.tensor(z_start, dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for _ in range(horizon):
            a = torch.tensor(rng.uniform(-1, 1, (N, cfg['act_dim'])).astype(np.float32), device=device)
            h, z, _ = model.rssm.imagine_step(h, z, a)
            out.append(h.cpu().numpy().copy())
    return out


def auroc_abs(scores, y):
    """AUROC of a scalar, taking the better orientation (|·| around 0.5)."""
    a = roc_auc_score(y, scores)
    return max(a, 1 - a)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading training states...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, z_all, kl_all = tr['h'], tr['z'], tr['kl']
    N = len(h_all)
    y_kl = binarise_by_median(kl_all)
    _, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)

    print(f"Imagination from {N_START} held-out states, horizon {HORIZON}...")
    model = load_model(cfg)
    rng = np.random.default_rng(42)
    start_idx = rng.choice(te_idx, min(N_START, len(te_idx)), replace=False)
    imagined = run_imagination(model, h_all[start_idx], z_all[start_idx], HORIZON, cfg)

    h_real = h_all[te_idx]
    h_imag = np.concatenate(imagined, axis=0)
    X = np.concatenate([h_real, h_imag], axis=0)
    y = np.array([0] * len(h_real) + [1] * len(h_imag), dtype=np.int32)
    print(f"  {len(h_real):,} real + {len(h_imag):,} imagined")

    # split for the full probe (fair comparison)
    b_tr, b_te = train_test_split(np.arange(len(X)), test_size=0.30, stratify=y, random_state=0)

    results = {'n_real': int(len(h_real)), 'n_imag': int(len(h_imag))}

    print("\n" + "=" * 70)
    print("TASK E — SINGLE-SCALAR SANITY CHECK ON BOUNDARY AUROC")
    print("=" * 70)
    print(f"\n  {'Feature':<34}  {'AUROC':>8}  {'95% CI':>20}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*20}")

    # Full trained linear probe
    clf, sc = train_probe(X[b_tr], y[b_tr])
    full_scores = clf.predict_proba(sc.transform(X[b_te]))[:, 1]
    p, lo, hi = bootstrap_auroc_ci(y[b_te], full_scores, seed=0)
    results['full_probe'] = [p, lo, hi]
    print(f"  {'Full linear probe (256-dim)':<34}  {p:>8.4f}  [{lo:.4f}, {hi:.4f}]")

    # candidate scalars (evaluated on the SAME held-out b_te for fairness)
    Xte, yte = X[b_te], y[b_te]

    scalars = {
        'L2 norm ||h_t||':        np.linalg.norm(Xte, axis=1),
        'mean(h_t)':              Xte.mean(axis=1),
        'std(h_t)':               Xte.std(axis=1),
        'max|h_t| coord':         np.abs(Xte).max(axis=1),
    }
    # top-PC projection (PCA fit on train pool, scaled)
    sc_pca = StandardScaler().fit(X[b_tr])
    pca = PCA(n_components=1, random_state=0).fit(sc_pca.transform(X[b_tr]))
    top_pc_proj = (sc_pca.transform(Xte) @ pca.components_[0])
    scalars['top-1 PC projection'] = top_pc_proj

    best_scalar = ('', 0.0)
    for name, s in scalars.items():
        a = auroc_abs(s, yte)
        # orient scores so higher = class 1, for CI
        if roc_auc_score(yte, s) < 0.5:
            s = -s
        p, lo, hi = bootstrap_auroc_ci(yte, s, seed=1)
        results[name] = [float(a), float(lo), float(hi)]
        print(f"  {name:<34}  {a:>8.4f}  [{lo:.4f}, {hi:.4f}]")
        if a > best_scalar[1]:
            best_scalar = (name, a)

    # best single raw coordinate (search over 256 dims on train, eval on test)
    aurocs_dim = []
    for d in range(X.shape[1]):
        aurocs_dim.append(auroc_abs(X[b_tr][:, d], y[b_tr]))
    best_dim = int(np.argmax(aurocs_dim))
    s = Xte[:, best_dim]
    if roc_auc_score(yte, s) < 0.5:
        s = -s
    a_dim = auroc_abs(Xte[:, best_dim], yte)
    p, lo, hi = bootstrap_auroc_ci(yte, s, seed=2)
    results[f'best single coord (dim {best_dim})'] = [float(a_dim), float(lo), float(hi)]
    print(f"  {'best single coord (dim '+str(best_dim)+')':<34}  {a_dim:>8.4f}  [{lo:.4f}, {hi:.4f}]")
    if a_dim > best_scalar[1]:
        best_scalar = (f'dim {best_dim}', a_dim)

    print("\n" + "-" * 70)
    print(f"  Full probe AUROC:       {results['full_probe'][0]:.4f}")
    print(f"  Best single scalar:     {best_scalar[1]:.4f}  ({best_scalar[0]})")
    if best_scalar[1] >= 0.99:
        print("\n  MAGNITUDE EFFECT: a single scalar already achieves ≈1.0 AUROC. The")
        print("  obs/imagination boundary is a 1-D magnitude phenomenon, not a genuinely")
        print("  distributed linear direction. The 'linear probe' framing for THIS result")
        print("  should be revised to say so plainly (it does not invalidate the finding).")
    elif best_scalar[1] >= 0.90:
        print("\n  MOSTLY SCALAR: a single scalar gets most of the way; the boundary is")
        print("  largely (but not purely) a magnitude effect.")
    else:
        print("\n  GENUINELY DISTRIBUTED: no single scalar comes close to the full probe.")
        print("  This positively strengthens the 'boundary lives in a real geometric")
        print("  direction' claim — it is not a trivial magnitude artefact.")

    with open(os.path.join(OUT_DIR, 'task_e_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_e_results.json')}")


if __name__ == '__main__':
    main()
