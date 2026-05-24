#!/usr/bin/env python3.11
"""
RWM-U-style ensemble baseline.

Each ensemble model processes the same observation trajectory in lockstep,
building its own h_t from the full sequence. Disagreement = variance across
models' decoded predictions at each timestep. This matches the RWM-U methodology.

Compares directly against the linear probe on the main model's h_t.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.data.collect import collect_states_with_ensemble
from src.probe.linear_probe import run_probe_a


def _build_set_c_with_ens_var(set_a, set_b, n_bins=10, per_bin=20, max_total=200):
    """KL-matched Set C, propagating ens_var alongside h/z/kl/recon/obs."""
    all_keys = ['h', 'z', 'kl', 'recon', 'obs', 'ens_var']
    pooled = {k: np.concatenate([set_a[k], set_b[k]]) for k in all_keys}
    all_kl    = pooled['kl']
    all_recon = pooled['recon']

    bin_edges = np.percentile(all_kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(all_kl, bin_edges[1:-1])
    rng = np.random.default_rng(42)
    c1_idx, c2_idx = [], []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) < 4:
            continue
        r = all_recon[idx]
        c1 = idx[r <= np.percentile(r, 25)]
        c2 = idx[r >= np.percentile(r, 75)]
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
    out = {k: np.concatenate([pooled[k][c1_idx], pooled[k][c2_idx]]) for k in all_keys}
    out['labels'] = np.array([0]*n1 + [1]*n2, dtype=np.int32)
    print(f"  Set C (KL-matched): {n1} C1 + {n2} C2 | "
          f"C1 KL={pooled['kl'][c1_idx].mean():.2f}  C2 KL={pooled['kl'][c2_idx].mean():.2f}")
    return out


def _build_set_c_strong_with_ens_var(novel, set_a, n_bins=10, per_bin=20, max_total=200):
    """KL-matched strong Set C (balance novel vs swingup confused), with ens_var."""
    all_keys = ['h', 'z', 'kl', 'recon', 'obs', 'ens_var']
    pooled   = {k: np.concatenate([novel[k], set_a[k]]) for k in all_keys}
    source   = np.array([0]*len(novel['h']) + [1]*len(set_a['h']), dtype=np.int32)
    all_kl   = pooled['kl']
    all_recon = pooled['recon']

    bin_edges = np.percentile(all_kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(all_kl, bin_edges[1:-1])
    rng = np.random.default_rng(42)
    c1_idx, c2_idx = [], []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        novel_b   = idx[source[idx] == 0]
        swingup_b = idx[source[idx] == 1]
        if len(novel_b) < 2 or len(swingup_b) < 2:
            continue
        c1 = novel_b[all_recon[novel_b]   <= np.percentile(all_recon[novel_b],   40)]
        c2 = swingup_b[all_recon[swingup_b] >= np.percentile(all_recon[swingup_b], 60)]
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
    out = {k: np.concatenate([pooled[k][c1_idx], pooled[k][c2_idx]]) for k in all_keys}
    out['labels'] = np.array([0]*n1 + [1]*n2, dtype=np.int32)
    print(f"  Set C Strong: {n1} C1 (novel/coping) + {n2} C2 (familiar/confused) | "
          f"C1 KL={pooled['kl'][c1_idx].mean():.2f}  C2 KL={pooled['kl'][c2_idx].mean():.2f}")
    return out


def load_model(cfg, ck_path):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
    m  = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def main():
    cfg = XS_CONFIG.copy()

    print("Loading models...")
    main_model = load_model(cfg, cfg['checkpoint_path'])
    ensemble_models = [
        load_model(cfg, f"outputs/checkpoints/ensemble_seed{s}.pt")
        for s in cfg['ensemble_seeds']
    ]
    print(f"  Main model + {len(ensemble_models)} ensemble models loaded")

    print("\n[RWM-U] Collecting Set A with ensemble running in lockstep...")
    from src.env.wrapper import CartpoleEnv
    env_a = CartpoleEnv(task='swingup', noisy=False, seed=42)
    set_a = collect_states_with_ensemble(main_model, ensemble_models, env_a,
                                         cfg['n_eval_episodes'], cfg)
    print(f"  Set A: {len(set_a['h'])} states | mean KL={set_a['kl'].mean():.3f} | "
          f"mean ens_var={set_a['ens_var'].mean():.5f}")

    print("\n[RWM-U] Collecting Set B (noisy) with ensemble in lockstep...")
    env_b = CartpoleEnv(task='swingup', noisy=True, noise_std=cfg['noise_std'], seed=43)
    set_b = collect_states_with_ensemble(main_model, ensemble_models, env_b,
                                         cfg['n_eval_episodes'], cfg)
    print(f"  Set B: {len(set_b['h'])} states | mean KL={set_b['kl'].mean():.3f} | "
          f"mean ens_var={set_b['ens_var'].mean():.5f}")

    print("\nBuilding Set C (KL-matched) from RWM-U-collected data...")
    set_c = _build_set_c_with_ens_var(set_a, set_b)

    print("\nCollecting novel balance states with ensemble in lockstep...")
    from src.env.wrapper import CartpoleEnv as CE
    env_novel = CE(task='balance', noisy=False, seed=52)
    novel = collect_states_with_ensemble(main_model, ensemble_models, env_novel,
                                         cfg['n_eval_episodes'], cfg)
    print(f"  Novel (balance): {len(novel['h'])} states | mean KL={novel['kl'].mean():.3f} | "
          f"mean ens_var={novel['ens_var'].mean():.5f}")

    print("\nBuilding Set C Strong with ens_var propagated...")
    set_c_strong = _build_set_c_strong_with_ens_var(novel, set_a)

    print("\n--- Probe A vs RWM-U Ensemble ---\n")

    training_states = dict(np.load(cfg['training_data_path']))
    kl_median = np.median(training_states['kl'])

    pa       = run_probe_a(training_states, set_a, set_b, set_c)
    pa_strong = run_probe_a(training_states, set_a, set_b, set_c_strong)

    def _safe_auroc(scores, labels):
        if len(np.unique(labels)) < 2:
            labels = (labels > np.median(labels)).astype(int)
        if len(np.unique(labels)) < 2:
            return float('nan')
        return roc_auc_score(labels, scores)

    def _kl_labels(kl_arr):
        y = (kl_arr > kl_median).astype(int)
        if len(np.unique(y)) < 2:
            y = (kl_arr > np.median(kl_arr)).astype(int)
        return y

    ens_a      = _safe_auroc(set_a['ens_var'],      _kl_labels(set_a['kl']))
    ens_b      = _safe_auroc(set_b['ens_var'],      _kl_labels(set_b['kl']))
    ens_c      = _safe_auroc(set_c['ens_var'],      set_c['labels'])
    ens_strong = _safe_auroc(set_c_strong['ens_var'], set_c_strong['labels'])

    print("=== RWM-U COMPARISON TABLE ===\n")
    print("Ensemble = trajectory-aware RWM-U style (lockstep, full h_t per model)\n")

    headers = ['Set', 'Probe A (h_t)', 'RWM-U Ensemble', 'Gap']
    rows = [
        ['Set A (ID)',          f"{pa['auroc_a']:.4f}", f"{ens_a:.4f}",
         f"{pa['auroc_a']-ens_a:+.4f}"],
        ['Set B (noisy OOD)',   f"{pa['auroc_b']:.4f}", f"{ens_b:.4f}",
         f"{pa['auroc_b']-ens_b:+.4f}"],
        ['Set C (KL-matched)',  f"{pa['auroc_c']:.4f}", f"{ens_c:.4f}",
         f"{pa['auroc_c']-ens_c:+.4f}"],
        ['Set C Strong',        f"{pa_strong['auroc_c']:.4f}", f"{ens_strong:.4f}",
         f"{pa_strong['auroc_c']-ens_strong:+.4f}"],
    ]

    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---']*len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    # Save updated sets with ens_var
    np.savez('outputs/data/set_a_rwmu.npz',  **set_a)
    np.savez('outputs/data/set_b_rwmu.npz',  **set_b)
    np.savez('outputs/data/novel_rwmu.npz',  **novel)
    print("\nSaved RWM-U collected sets to outputs/data/")


if __name__ == '__main__':
    main()
