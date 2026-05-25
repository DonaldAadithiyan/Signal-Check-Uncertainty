#!/usr/bin/env python3.11
"""
Trajectory curvature probe: is confusion encoded in the shape of h_t trajectories?

h_t direction (Δh_t) is task-specific — the confusion direction in h_t space differs
per task, so the Δh_t probe doesn't transfer (0.57 within-balance).

Hypothesis: trajectory curvature is task-agnostic.

  c_t = h_t - 2·h_{t-1} + h_{t-2}   (second derivative of h_t)

||c_t|| measures how sharply the trajectory bent at step t — how much the direction
of h_t movement changed. This is a scalar property of trajectory SHAPE, not position
or direction. If confused states consistently produce bent trajectories regardless of
task, this signal would transfer across tasks.

Mechanistic story:
  - Coping: consecutive observations are expected → small, consistent updates → smooth trajectory
  - Confused: consecutive observations are surprising in different ways → h_t keeps
    changing direction → high curvature

We test four signals:
  1. ||c_t||          — raw curvature norm (no probe, no training needed)
  2. c_t probe        — linear probe on full curvature vector (256-dim)
  3. [Δh_t; c_t] probe — first + second derivative together (512-dim)
  4. Δh_t probe       — baseline from previous experiment

KEY test: within-balance — both C1 and C2 from balance trajectories, same task identity.
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


def collect_with_episode_markers(model, env, n_episodes, cfg):
    device = next(model.parameters()).device
    model.eval()
    all_h, all_kl, all_recon = [], [], []
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
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_t   = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                embed = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                dec = model.decoder(torch.cat([h, z], dim=-1))
                kl_val    = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                recon_val = F.mse_loss(dec, obs_t, reduction='none').sum().item()

                all_h.append(h.squeeze(0).cpu().numpy().copy())
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
        'kl':         np.array(all_kl,       dtype=np.float32),
        'recon':      np.array(all_recon,    dtype=np.float32),
        'traj_id':    np.array(all_traj_id,  dtype=np.int32),
        'step_index': np.array(all_step_idx, dtype=np.int32),
    }


def compute_derivatives(data):
    """
    Compute Δh_t and c_t within each trajectory.
    Δh_t = h_t - h_{t-1}       (first derivative,  requires step >= 1)
    c_t  = h_t - 2h_{t-1} + h_{t-2}  (second derivative, requires step >= 2)

    Returns arrays aligned on steps with step_index >= 2 (both derivatives available).
    """
    h     = data['h']
    step  = data['step_index']

    # Valid: step >= 2, so we have h_{t-2}, h_{t-1}, h_t
    valid = step >= 2
    idx   = np.where(valid)[0]

    h_t   = h[idx]
    h_tm1 = h[idx - 1]
    h_tm2 = h[idx - 2]

    dh = h_t - h_tm1
    c  = h_t - 2 * h_tm1 + h_tm2

    return {
        'dh':      dh,
        'c':       c,
        'dh_c':    np.concatenate([dh, c], axis=1),  # 512-dim
        'h':       h_t,
        'kl':      data['kl'][idx],
        'recon':   data['recon'][idx],
        'traj_id': data['traj_id'][idx],
    }


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

    n1, n2  = len(c1_idx), len(c2_idx)
    all_idx = c1_idx + c2_idx
    labels  = np.array([0]*n1 + [1]*n2, dtype=np.int32)

    return {k: data[k][all_idx] for k in ['dh', 'c', 'dh_c', 'h', 'kl', 'recon']}, labels


def main():
    cfg = XS_CONFIG.copy()

    # ── Swingup: existing training states ──
    print("Loading swingup training states...")
    raw_sw = dict(np.load(cfg['training_data_path']))
    sw = compute_derivatives(raw_sw)
    print(f"  {len(sw['h'])} steps (step >= 2)")
    print(f"  mean KL={sw['kl'].mean():.2f}  "
          f"mean ||c_t||={np.linalg.norm(sw['c'], axis=1).mean():.4f}  "
          f"mean ||Δh_t||={np.linalg.norm(sw['dh'], axis=1).mean():.4f}")

    # ── Balance: collect fresh ──
    print("\nLoading model...")
    model = load_model(cfg, cfg['checkpoint_path'])

    print("\nCollecting balance states...")
    env_bal = CartpoleEnv(task='balance', noisy=False, seed=52)
    raw_bal = collect_with_episode_markers(model, env_bal, cfg['n_eval_episodes'], cfg)
    bal = compute_derivatives(raw_bal)
    print(f"  {len(bal['h'])} steps (step >= 2)")
    print(f"  mean KL={bal['kl'].mean():.2f}  "
          f"mean ||c_t||={np.linalg.norm(bal['c'], axis=1).mean():.4f}  "
          f"mean ||Δh_t||={np.linalg.norm(bal['dh'], axis=1).mean():.4f}")

    # ── Correlations ──
    sw_c_norm  = np.linalg.norm(sw['c'],  axis=1)
    bal_c_norm = np.linalg.norm(bal['c'], axis=1)
    bal_dh_norm = np.linalg.norm(bal['dh'], axis=1)

    print(f"\n  Swingup  ||c_t|| vs KL:   r={np.corrcoef(sw_c_norm,  sw['kl'])[0,1]:.4f}")
    print(f"  Balance  ||c_t|| vs KL:   r={np.corrcoef(bal_c_norm, bal['kl'])[0,1]:.4f}")
    print(f"  Balance  ||c_t|| vs recon: r={np.corrcoef(bal_c_norm, bal['recon'])[0,1]:.4f}")

    # ── Build contrastive sets ──
    print("\nBuilding within-swingup contrastive set...")
    ws_data, ws_labels = build_contrastive_set(sw)
    print(f"  {(ws_labels==0).sum()} C1 + {(ws_labels==1).sum()} C2  |  "
          f"C1 KL={ws_data['kl'][ws_labels==0].mean():.2f}  "
          f"C2 KL={ws_data['kl'][ws_labels==1].mean():.2f}  |  "
          f"C1 recon={ws_data['recon'][ws_labels==0].mean():.3f}  "
          f"C2 recon={ws_data['recon'][ws_labels==1].mean():.3f}")

    print("\nBuilding within-balance contrastive set...")
    wb_data, wb_labels = build_contrastive_set(bal)
    print(f"  {(wb_labels==0).sum()} C1 + {(wb_labels==1).sum()} C2  |  "
          f"C1 KL={wb_data['kl'][wb_labels==0].mean():.2f}  "
          f"C2 KL={wb_data['kl'][wb_labels==1].mean():.2f}  |  "
          f"C1 recon={wb_data['recon'][wb_labels==0].mean():.3f}  "
          f"C2 recon={wb_data['recon'][wb_labels==1].mean():.3f}")

    # ── Train probes on swingup ──
    y_sw = binarise_by_median(sw['kl'])
    idx_all = np.arange(len(sw['h']))
    tr_idx, te_idx = train_test_split(
        idx_all, test_size=0.40, stratify=y_sw, random_state=0)

    print("\nTraining probes on swingup KL labels...")
    clf_c,    sc_c    = train_probe(sw['c'][tr_idx],    y_sw[tr_idx])
    clf_dh,   sc_dh   = train_probe(sw['dh'][tr_idx],   y_sw[tr_idx])
    clf_dhc,  sc_dhc  = train_probe(sw['dh_c'][tr_idx], y_sw[tr_idx])
    clf_h,    sc_h    = train_probe(sw['h'][tr_idx],    y_sw[tr_idx])

    # ── Evaluate ──
    def eval_all(feat_key, clf, sc, te_feats, ws_feats, wb_feats):
        return (
            auroc(clf, sc, te_feats,  y_sw[te_idx]),
            auroc(clf, sc, ws_feats,  ws_labels),
            auroc(clf, sc, wb_feats,  wb_labels),
        )

    r_h   = eval_all('h',    clf_h,   sc_h,   sw['h'][te_idx],    ws_data['h'],    wb_data['h'])
    r_dh  = eval_all('dh',   clf_dh,  sc_dh,  sw['dh'][te_idx],   ws_data['dh'],   wb_data['dh'])
    r_c   = eval_all('c',    clf_c,   sc_c,   sw['c'][te_idx],    ws_data['c'],    wb_data['c'])
    r_dhc = eval_all('dh_c', clf_dhc, sc_dhc, sw['dh_c'][te_idx], ws_data['dh_c'], wb_data['dh_c'])

    # Raw norms — no probe needed
    def raw_auroc(norms_te, norms_ws, norms_wb):
        return (
            roc_auc_score(y_sw[te_idx], norms_te),
            roc_auc_score(ws_labels,    norms_ws),
            roc_auc_score(wb_labels,    norms_wb),
        )

    rn_c  = raw_auroc(
        np.linalg.norm(sw['c'][te_idx],    axis=1),
        np.linalg.norm(ws_data['c'],       axis=1),
        np.linalg.norm(wb_data['c'],       axis=1),
    )
    rn_dh = raw_auroc(
        np.linalg.norm(sw['dh'][te_idx],   axis=1),
        np.linalg.norm(ws_data['dh'],      axis=1),
        np.linalg.norm(wb_data['dh'],      axis=1),
    )

    # ── Print results ──
    print("\n" + "="*65)
    print("TRAJECTORY CURVATURE PROBE — IS CONFUSION GEOMETRIC?")
    print("="*65)
    print("\nProbes trained on swingup KL labels only.\n")
    print("KEY: within-balance = both C1+C2 from balance, no task identity signal.\n")

    headers = ['Signal', 'Dims', 'SW held-out', 'Within-SW', 'Within-BAL ←key']
    rows = [
        ['h_t probe (baseline)',        '256',  f'{r_h[0]:.4f}',   f'{r_h[1]:.4f}',   f'{r_h[2]:.4f}'],
        ['Δh_t probe',                  '256',  f'{r_dh[0]:.4f}',  f'{r_dh[1]:.4f}',  f'{r_dh[2]:.4f}'],
        ['c_t probe',                   '256',  f'{r_c[0]:.4f}',   f'{r_c[1]:.4f}',   f'{r_c[2]:.4f}'],
        ['[Δh_t; c_t] probe',           '512',  f'{r_dhc[0]:.4f}', f'{r_dhc[1]:.4f}', f'{r_dhc[2]:.4f}'],
        ['||c_t|| raw (no probe)',       '1',    f'{rn_c[0]:.4f}',  f'{rn_c[1]:.4f}',  f'{rn_c[2]:.4f}'],
        ['||Δh_t|| raw (no probe)',      '1',    f'{rn_dh[0]:.4f}', f'{rn_dh[1]:.4f}', f'{rn_dh[2]:.4f}'],
    ]

    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---']*len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    wb_c   = r_c[2]
    wb_raw = rn_c[2]

    print("\n--- Interpretation ---")
    print(f"\n  h_t within-balance:           {r_h[2]:.4f}  (task-specific position)")
    print(f"  Δh_t within-balance:          {r_dh[2]:.4f}  (task-specific direction)")
    print(f"  c_t within-balance:           {wb_c:.4f}  ← curvature probe")
    print(f"  ||c_t|| within-balance (raw): {wb_raw:.4f}  ← curvature magnitude, no training")

    if wb_c > 0.62:
        print(f"\n  CURVATURE TRANSFERS. Confusion leaves a task-agnostic geometric signature.")
        print(f"  The shape of the h_t trajectory encodes confusion, not its position or direction.")
    elif wb_c > 0.56:
        print(f"\n  WEAK curvature transfer. Partial geometric signal — not conclusive.")
    else:
        print(f"\n  CURVATURE DOES NOT TRANSFER. Confusion is not geometrically separable")
        print(f"  across tasks with a linear probe on c_t at this scale.")

    if wb_raw > 0.60:
        print(f"\n  Raw ||c_t|| transfers ({wb_raw:.4f}) — curvature magnitude alone is informative.")
        print(f"  Confused states produce more bent trajectories regardless of task.")

    # Correlation summary
    print(f"\n  Swingup  ||c_t|| vs KL:    r={np.corrcoef(sw_c_norm,  sw['kl'])[0,1]:.4f}")
    print(f"  Balance  ||c_t|| vs KL:    r={np.corrcoef(bal_c_norm, bal['kl'])[0,1]:.4f}")
    print(f"  Balance  ||c_t|| vs recon: r={np.corrcoef(bal_c_norm, bal['recon'])[0,1]:.4f}")


if __name__ == '__main__':
    main()
