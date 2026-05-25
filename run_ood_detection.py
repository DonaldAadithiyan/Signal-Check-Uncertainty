#!/usr/bin/env python3.11
"""
Direct OOD detection test: can single-model internal signals match the ensemble?

Prior result: ensemble (RWM-U) = 0.94, h_t probe = 0.61 on swingup vs balance.
All cross-task experiments used the hard contrastive within-balance set.
This runs the easier direct test: swingup (label 0) vs balance (label 1), no KL matching.

Two signal types:
  Unsupervised (no training needed): raw scalars computed from gates/derivatives
  Probe-based: probe trained on swingup KL labels, score used as OOD detector

Ensemble baseline: 0.94 (RWM-U, trajectory-aware, 3 models)
h_t probe baseline: 0.61 (from earlier direct OOD test)
KL oracle: ~0.95 (direct KL value, not from probe)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.env.wrapper import CartpoleEnv
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe, auroc


def load_model(cfg, ck_path):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def compute_gru_gates(gru_cell, inp, h):
    W_ih, W_hh = gru_cell.weight_ih, gru_cell.weight_hh
    b_ih, b_hh = gru_cell.bias_ih,   gru_cell.bias_hh
    deter = h.shape[-1]
    pre_ih = inp @ W_ih.T + b_ih
    pre_hh = h   @ W_hh.T + b_hh
    r_t = torch.sigmoid(pre_ih[:, :deter]        + pre_hh[:, :deter])
    z_t = torch.sigmoid(pre_ih[:, deter:2*deter] + pre_hh[:, deter:2*deter])
    n_t = torch.tanh(   pre_ih[:, 2*deter:]      + r_t * pre_hh[:, 2*deter:])
    return z_t, r_t, n_t


def collect_with_gates(model, env, n_episodes, cfg):
    device = next(model.parameters()).device
    model.eval()
    gru = model.rssm.gru
    all_h, all_z_state, all_z_gate, all_r_gate, all_n_gate = [], [], [], [], []
    all_kl, all_recon = [], []
    all_traj_id, all_step_idx = [], []

    for ep in range(n_episodes):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        done = False
        step = 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                action = np.random.uniform(-1, 1, size=(cfg['act_dim'],)).astype(np.float32)
                obs_t  = torch.tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
                a_t    = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                embed  = model.encoder(obs_t)
                inp    = torch.cat([z, a_t], dim=-1)
                z_gate, r_gate, n_gate = compute_gru_gates(gru, inp, h)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                dec       = model.decoder(torch.cat([h, z], dim=-1))
                kl_val    = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                recon_val = F.mse_loss(dec, obs_t, reduction='none').sum().item()
                all_h.append(h.squeeze(0).cpu().numpy().copy())
                all_z_state.append(z.squeeze(0).cpu().numpy().copy())   # posterior stochastic z_t
                all_z_gate.append(z_gate.squeeze(0).cpu().numpy().copy())
                all_r_gate.append(r_gate.squeeze(0).cpu().numpy().copy())
                all_n_gate.append(n_gate.squeeze(0).cpu().numpy().copy())
                all_kl.append(kl_val)
                all_recon.append(recon_val)
                all_traj_id.append(ep)
                all_step_idx.append(step)
                obs, _, done = env.step(action)
                step += 1
        if (ep + 1) % 5 == 0:
            print(f"    ep {ep+1}/{n_episodes}")

    data = {
        'h':          np.array(all_h,        dtype=np.float32),
        'z_state':    np.array(all_z_state,  dtype=np.float32),  # 1024-dim posterior stochastic
        'z_gate':     np.array(all_z_gate,   dtype=np.float32),
        'r_gate':     np.array(all_r_gate,   dtype=np.float32),
        'n_gate':     np.array(all_n_gate,   dtype=np.float32),
        'kl':         np.array(all_kl,       dtype=np.float32),
        'recon':      np.array(all_recon,    dtype=np.float32),
        'traj_id':    np.array(all_traj_id,  dtype=np.int32),
        'step_index': np.array(all_step_idx, dtype=np.int32),
    }
    # Add Δh_t for step >= 2
    step = data['step_index']
    valid = step >= 2
    idx   = np.where(valid)[0]
    dh = data['h'][idx] - data['h'][idx - 1]
    data_v = {k: data[k][valid] for k in data}
    data_v['dh'] = dh
    return data_v


def main():
    cfg = XS_CONFIG.copy()

    print("Loading model...")
    model = load_model(cfg, cfg['checkpoint_path'])

    print("\nCollecting swingup (200 ep)...")
    env_sw = CartpoleEnv(task='swingup', noisy=False, seed=42)
    sw = collect_with_gates(model, env_sw, 200, cfg)
    print(f"  {len(sw['h'])} steps | KL={sw['kl'].mean():.1f} | "
          f"recon={sw['recon'].mean():.3f}")

    print("\nCollecting balance (20 ep)...")
    env_bal = CartpoleEnv(task='balance', noisy=False, seed=52)
    bal = collect_with_gates(model, env_bal, cfg['n_eval_episodes'], cfg)
    print(f"  {len(bal['h'])} steps | KL={bal['kl'].mean():.1f} | "
          f"recon={bal['recon'].mean():.3f}")

    # ── Train probes on 60% of swingup with KL labels ──
    # Split BEFORE building OOD pool so evaluation swingup states are held-out.
    print("\nTraining probes on swingup KL labels...")
    y_sw   = binarise_by_median(sw['kl'])
    tr_idx, te_idx = train_test_split(
        np.arange(len(sw['h'])), test_size=0.40, stratify=y_sw, random_state=0)

    probes = {}
    for name, feat in [('h', 'h'), ('dh', 'dh'), ('z_state', 'z_state'),
                       ('z_gate', 'z_gate'), ('r_gate', 'r_gate'), ('n_gate', 'n_gate')]:
        clf, sc = train_probe(sw[feat][tr_idx], y_sw[tr_idx])
        probes[name] = (clf, sc)

    # ── Build OOD dataset: swingup=0, balance=1 ──
    # Use only held-out swingup states (te_idx) to avoid probe training data leakage.
    rng = np.random.default_rng(42)
    n_bal = len(bal['h'])
    n_te = len(te_idx)
    sw_ood_idx = te_idx[rng.choice(n_te, min(n_bal, n_te), replace=False)]

    def pool(key):
        return np.concatenate([sw[key][sw_ood_idx], bal[key]])

    n_sw_ood = len(sw_ood_idx)
    ood_labels = np.array([0]*n_sw_ood + [1]*n_bal, dtype=np.int32)

    def probe_ood_auc(name, feat):
        clf, sc = probes[name]
        scores = clf.predict_proba(sc.transform(pool(feat)))[:, 1]
        return roc_auc_score(ood_labels, scores)

    def raw_ood_auc(scores):
        return roc_auc_score(ood_labels, scores)

    # ── Unsupervised raw scalars ──
    print("\nComputing raw scalar OOD signals...")

    raw_signals = {
        'KL (oracle)':       pool('kl'),
        'Recon (oracle)':    pool('recon'),
        '||h_t||':           np.linalg.norm(pool('h'),      axis=1),
        '||Δh_t||':          np.linalg.norm(pool('dh'),     axis=1),
        'mean(z_t)':         pool('z_gate').mean(axis=1),
        'mean(r_t)':         pool('r_gate').mean(axis=1),
        'mean(n_t)':         np.abs(pool('n_gate')).mean(axis=1),
        'std(z_t)':          pool('z_gate').std(axis=1),
        'std(r_t)':          pool('r_gate').std(axis=1),
    }

    raw_aucs  = {k: raw_ood_auc(v) for k, v in raw_signals.items()}
    probe_aucs = {
        'h_t probe':      probe_ood_auc('h',       'h'),
        'Δh_t probe':     probe_ood_auc('dh',      'dh'),
        'z_t probe (stochastic)': probe_ood_auc('z_state', 'z_state'),
        'z_gate probe':   probe_ood_auc('z_gate',  'z_gate'),
        'r_gate probe':   probe_ood_auc('r_gate',  'r_gate'),
        'n_gate probe':   probe_ood_auc('n_gate',  'n_gate'),
    }

    # ── Print ──
    print("\n" + "="*60)
    print("DIRECT OOD DETECTION: swingup (0) vs balance (1)")
    print("="*60)
    print(f"\nBaseline:  Ensemble (RWM-U) = 0.9425  [from prior run, trajectory-aware]\n")

    print("--- Unsupervised (no training needed) ---")
    for k, v in sorted(raw_aucs.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 20)
        print(f"  {k:<22} {v:.4f}  {bar}")

    print("\n--- Probe-based (trained on swingup KL labels) ---")
    for k, v in sorted(probe_aucs.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 20)
        print(f"  {k:<22} {v:.4f}  {bar}")

    print("\n--- Distribution shift summary ---")
    for feat, label in [('kl','KL'), ('recon','recon'), ('z_gate','z_gate mean'),
                        ('r_gate','r_gate mean'), ('dh','||Δh_t||')]:
        if feat == 'dh':
            sw_val = np.linalg.norm(sw['dh'], axis=1).mean()
            bal_val = np.linalg.norm(bal['dh'], axis=1).mean()
        elif feat in ('z_gate','r_gate'):
            sw_val = sw[feat].mean()
            bal_val = bal[feat].mean()
        else:
            sw_val = sw[feat].mean()
            bal_val = bal[feat].mean()
        print(f"  {label:<18} swingup={sw_val:.4f}  balance={bal_val:.4f}  "
              f"ratio={bal_val/sw_val:.2f}x")

    best_raw   = max(raw_aucs.items(),   key=lambda x: x[1])
    best_probe = max(probe_aucs.items(), key=lambda x: x[1])
    gap_to_ensemble = 0.9425 - best_raw[1]
    print(f"\n  Best raw signal:   {best_raw[0]} = {best_raw[1]:.4f}")
    print(f"  Best probe signal: {best_probe[0]} = {best_probe[1]:.4f}")
    print(f"  Gap to ensemble:   {gap_to_ensemble:+.4f}")


if __name__ == '__main__':
    main()
