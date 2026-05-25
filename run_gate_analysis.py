#!/usr/bin/env python3.11
"""
GRU gate analysis: does the update gate z_t encode confusion task-agnostically?

The Δh_t probe transfers at 0.68 (step>=2) but we don't know why. The GRU
update gate z_t is the mechanistic explanation: when surprised (high KL),
the gate should fire harder, updating more h_t dimensions more aggressively.

GRU equations (PyTorch GRUCell):
  r_t = sigmoid(W_ir·x + W_hr·h + b_r)       reset gate
  z_t = sigmoid(W_iz·x + W_hz·h + b_z)       update gate
  n_t = tanh(W_in·x + r_t ⊙ (W_hn·h + b_n)) new gate / candidate
  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ n_t

We extract z_t and r_t at each step by recomputing the gate activations from the
GRUCell weights. These are bounded [0, 1] per dimension — semantically meaningful
in a way that raw h_t values are not.

Probes tested (all trained on swingup KL labels, tested on within-balance):
  1. z_t probe            — 256-dim update gate activations
  2. r_t probe            — 256-dim reset gate activations
  3. n_t probe            — 256-dim new gate (candidate hidden state)
  4. mean(z_t) raw        — scalar, no training needed
  5. [z_t; r_t] probe     — 512-dim: both gates together
  6. Δh_t probe           — 256-dim baseline from previous experiment

If z_t transfers better than Δh_t (0.68), it is the underlying mechanism.
If they're equal, Δh_t is a clean proxy for z_t activity.
If mean(z_t) works as a raw scalar, the signal is simple and unsupervised.
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
    """
    Extract update gate z_t, reset gate r_t, and new gate n_t from a GRUCell.
    PyTorch GRUCell weight layout: [r | z | n] stacked along dim 0.
    """
    W_ih = gru_cell.weight_ih   # (3*deter, inp_dim)
    W_hh = gru_cell.weight_hh   # (3*deter, deter)
    b_ih = gru_cell.bias_ih     # (3*deter,)
    b_hh = gru_cell.bias_hh     # (3*deter,)

    deter = h.shape[-1]

    pre_ih = inp @ W_ih.T + b_ih    # (batch, 3*deter)
    pre_hh = h   @ W_hh.T + b_hh   # (batch, 3*deter)

    r_t = torch.sigmoid(pre_ih[:, :deter]        + pre_hh[:, :deter])
    z_t = torch.sigmoid(pre_ih[:, deter:2*deter] + pre_hh[:, deter:2*deter])
    n_t = torch.tanh(   pre_ih[:, 2*deter:]      + r_t * pre_hh[:, 2*deter:])

    return z_t, r_t, n_t


def collect_with_gates(model, env, n_episodes, cfg):
    """Collect h_t, gates, kl, recon with episode markers."""
    device = next(model.parameters()).device
    model.eval()
    gru = model.rssm.gru

    all_h, all_z_gate, all_r_gate, all_n_gate = [], [], [], []
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

                embed = model.encoder(obs_t)

                # Compute gates BEFORE the GRU updates h
                inp = torch.cat([z, a_t], dim=-1)
                z_gate, r_gate, n_gate = compute_gru_gates(gru, inp, h)

                # Normal RSSM step
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                dec      = model.decoder(torch.cat([h, z], dim=-1))
                kl_val   = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                recon_val = F.mse_loss(dec, obs_t, reduction='none').sum().item()

                all_h.append(h.squeeze(0).cpu().numpy().copy())
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
            print(f"    episode {ep+1}/{n_episodes}")

    return {
        'h':          np.array(all_h,       dtype=np.float32),
        'z_gate':     np.array(all_z_gate,  dtype=np.float32),
        'r_gate':     np.array(all_r_gate,  dtype=np.float32),
        'n_gate':     np.array(all_n_gate,  dtype=np.float32),
        'kl':         np.array(all_kl,      dtype=np.float32),
        'recon':      np.array(all_recon,   dtype=np.float32),
        'traj_id':    np.array(all_traj_id, dtype=np.int32),
        'step_index': np.array(all_step_idx,dtype=np.int32),
    }


def add_derivatives(data):
    """Add Δh_t aligned on step >= 2 (consistent with curvature experiment)."""
    h    = data['h']
    step = data['step_index']
    valid = step >= 2
    idx   = np.where(valid)[0]
    dh = h[idx] - h[idx - 1]

    out = {k: data[k][valid] for k in data}
    out['dh'] = dh
    out['zr_gate'] = np.concatenate([data['z_gate'][valid], data['r_gate'][valid]], axis=1)
    return out


def build_contrastive_set(data, n_bins=10, per_bin=20, max_total=200, seed=42):
    kl, recon = data['kl'], data['recon']
    bin_edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(kl, bin_edges[1:-1])
    rng = np.random.default_rng(seed)
    c1_idx, c2_idx = [], []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) < 4:
            continue
        r  = recon[idx]
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
    all_idx = c1_idx + c2_idx
    labels  = np.array([0]*len(c1_idx) + [1]*len(c2_idx), dtype=np.int32)
    return {k: data[k][all_idx] for k in data if k not in ('traj_id', 'step_index')}, labels


def main():
    cfg = XS_CONFIG.copy()

    print("Loading model...")
    model = load_model(cfg, cfg['checkpoint_path'])

    print("\nCollecting swingup with gates (200 episodes)...")
    env_sw = CartpoleEnv(task='swingup', noisy=False, seed=42)
    raw_sw = collect_with_gates(model, env_sw, 200, cfg)
    sw = add_derivatives(raw_sw)
    print(f"  {len(sw['h'])} steps (step>=2)")
    print(f"  mean KL={sw['kl'].mean():.2f}")
    print(f"  mean z_gate activity={sw['z_gate'].mean():.4f}  "
          f"mean r_gate activity={sw['r_gate'].mean():.4f}")

    print("\nCollecting balance with gates (20 episodes)...")
    env_bal = CartpoleEnv(task='balance', noisy=False, seed=52)
    raw_bal = collect_with_gates(model, env_bal, cfg['n_eval_episodes'], cfg)
    bal = add_derivatives(raw_bal)
    print(f"  {len(bal['h'])} steps (step>=2)")
    print(f"  mean KL={bal['kl'].mean():.2f}")
    print(f"  mean z_gate activity={bal['z_gate'].mean():.4f}  "
          f"mean r_gate activity={bal['r_gate'].mean():.4f}")

    # Correlations
    sw_zmean  = sw['z_gate'].mean(axis=1)
    bal_zmean = bal['z_gate'].mean(axis=1)
    bal_rmean = bal['r_gate'].mean(axis=1)
    print(f"\n  Swingup  mean(z_t) vs KL:    r={np.corrcoef(sw_zmean,  sw['kl'])[0,1]:.4f}")
    print(f"  Balance  mean(z_t) vs KL:    r={np.corrcoef(bal_zmean, bal['kl'])[0,1]:.4f}")
    print(f"  Balance  mean(z_t) vs recon: r={np.corrcoef(bal_zmean, bal['recon'])[0,1]:.4f}")
    print(f"  Balance  mean(r_t) vs recon: r={np.corrcoef(bal_rmean, bal['recon'])[0,1]:.4f}")

    # Contrastive sets
    print("\nBuilding within-swingup contrastive set...")
    ws, ws_labels = build_contrastive_set(sw)
    print(f"  {(ws_labels==0).sum()} C1 + {(ws_labels==1).sum()} C2  |  "
          f"C1 KL={ws['kl'][ws_labels==0].mean():.2f}  C2 KL={ws['kl'][ws_labels==1].mean():.2f}  |  "
          f"C1 recon={ws['recon'][ws_labels==0].mean():.3f}  C2 recon={ws['recon'][ws_labels==1].mean():.3f}")

    print("\nBuilding within-balance contrastive set...")
    wb, wb_labels = build_contrastive_set(bal)
    print(f"  {(wb_labels==0).sum()} C1 + {(wb_labels==1).sum()} C2  |  "
          f"C1 KL={wb['kl'][wb_labels==0].mean():.2f}  C2 KL={wb['kl'][wb_labels==1].mean():.2f}  |  "
          f"C1 recon={wb['recon'][wb_labels==0].mean():.3f}  C2 recon={wb['recon'][wb_labels==1].mean():.3f}")

    # Train probes
    y_sw = binarise_by_median(sw['kl'])
    idx_all = np.arange(len(sw['h']))
    tr_idx, te_idx = train_test_split(idx_all, test_size=0.40, stratify=y_sw, random_state=0)

    print("\nTraining probes on swingup KL labels...")
    probes = {}
    for name, feat in [('z_gate', 'z_gate'), ('r_gate', 'r_gate'),
                       ('n_gate', 'n_gate'), ('zr_gate', 'zr_gate'),
                       ('dh', 'dh'), ('h', 'h')]:
        clf, sc = train_probe(sw[feat][tr_idx], y_sw[tr_idx])
        probes[name] = (clf, sc)

    # Evaluate
    def ev(name, feat, te_data, ws_data, wb_data):
        clf, sc = probes[name]
        return (
            auroc(clf, sc, te_data[feat][te_idx], y_sw[te_idx]),
            auroc(clf, sc, ws_data[feat], ws_labels),
            auroc(clf, sc, wb_data[feat], wb_labels),
        )

    results = {
        'h_t':         ev('h',       'h',       sw, ws, wb),
        'Δh_t':        ev('dh',      'dh',      sw, ws, wb),
        'z_gate':      ev('z_gate',  'z_gate',  sw, ws, wb),
        'r_gate':      ev('r_gate',  'r_gate',  sw, ws, wb),
        'n_gate':      ev('n_gate',  'n_gate',  sw, ws, wb),
        '[z;r]_gate':  ev('zr_gate', 'zr_gate', sw, ws, wb),
    }

    # Raw scalar signals (no probe)
    def raw_auc(scores_te, scores_ws, scores_wb):
        return (
            roc_auc_score(y_sw[te_idx], scores_te),
            roc_auc_score(ws_labels,    scores_ws),
            roc_auc_score(wb_labels,    scores_wb),
        )

    raw_z = raw_auc(
        sw['z_gate'][te_idx].mean(axis=1),
        ws['z_gate'].mean(axis=1),
        wb['z_gate'].mean(axis=1),
    )

    # Print
    print("\n" + "="*65)
    print("GRU GATE ANALYSIS — MECHANISTIC CONFUSION SIGNAL")
    print("="*65)
    print("\nProbes trained on swingup KL labels. KEY: within-balance.\n")

    headers = ['Signal', 'Dims', 'SW held-out', 'Within-SW', 'Within-BAL ←key']
    rows = [
        ['h_t (position)',      '256', f"{results['h_t'][0]:.4f}",        f"{results['h_t'][1]:.4f}",        f"{results['h_t'][2]:.4f}"],
        ['Δh_t (1st deriv)',    '256', f"{results['Δh_t'][0]:.4f}",       f"{results['Δh_t'][1]:.4f}",       f"{results['Δh_t'][2]:.4f}"],
        ['z_t gate',            '256', f"{results['z_gate'][0]:.4f}",     f"{results['z_gate'][1]:.4f}",     f"{results['z_gate'][2]:.4f}"],
        ['r_t gate',            '256', f"{results['r_gate'][0]:.4f}",     f"{results['r_gate'][1]:.4f}",     f"{results['r_gate'][2]:.4f}"],
        ['n_t gate (candidate)','256', f"{results['n_gate'][0]:.4f}",     f"{results['n_gate'][1]:.4f}",     f"{results['n_gate'][2]:.4f}"],
        ['[z_t; r_t]',          '512', f"{results['[z;r]_gate'][0]:.4f}", f"{results['[z;r]_gate'][1]:.4f}", f"{results['[z;r]_gate'][2]:.4f}"],
        ['mean(z_t) raw',       '1',   f"{raw_z[0]:.4f}",                 f"{raw_z[1]:.4f}",                 f"{raw_z[2]:.4f}"],
    ]

    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---']*len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    best_wb = max(results.items(), key=lambda x: x[1][2])
    print(f"\n  Best within-balance: {best_wb[0]} = {best_wb[1][2]:.4f}")
    print(f"  Δh_t within-balance (baseline): {results['Δh_t'][2]:.4f}")
    print(f"  mean(z_t) raw within-balance:   {raw_z[2]:.4f}")

    print(f"\n  Swingup  mean(z_t) vs KL:    r={np.corrcoef(sw_zmean, sw['kl'])[0,1]:.4f}")
    print(f"  Balance  mean(z_t) vs KL:    r={np.corrcoef(bal_zmean, bal['kl'])[0,1]:.4f}")
    print(f"  Balance  mean(z_t) vs recon: r={np.corrcoef(bal_zmean, bal['recon'])[0,1]:.4f}")


if __name__ == '__main__':
    main()
