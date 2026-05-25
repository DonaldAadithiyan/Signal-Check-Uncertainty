#!/usr/bin/env python3.11
"""
Δh_t probe: does confusion live in the dynamics of h_t, not its position?

The h_t probe failed to transfer across tasks (within-balance: 0.51).
But h_t values are task-specific. The GRU update rule is not.

Hypothesis: Δh_t = h_t - h_{t-1} encodes confusion task-agnostically.
When confused (high KL), the model received a surprising input and the GRU
made a large correction. When coping (low KL), small correction needed.
This is a property of GRU dynamics — same mechanism regardless of task.

We also test ||Δh_t|| (raw update magnitude, no probe needed) as the
simplest possible version of this hypothesis.

Three tests:
  1. Swingup held-out — sanity check
  2. Within-swingup contrastive (KL-matched) — clean baseline
  3. Within-balance contrastive (KL-matched) — KEY: same task identity, only confusion differs

If Δh_t transfers to within-balance where h_t didn't (0.51), confusion is
in the GRU dynamics, not the accumulated trajectory position.
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
    """Collect h_t per episode so Δh_t can be computed cleanly."""
    device = next(model.parameters()).device
    model.eval()
    all_h, all_kl, all_recon, all_obs = [], [], [], []
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
                all_obs.append(obs.copy())
                all_traj_id.append(ep)
                all_step_idx.append(step)

                obs, _, done = env.step(action)
                step += 1

        if (ep + 1) % 5 == 0:
            print(f"    episode {ep+1}/{n_episodes}")

    return {
        'h':          np.array(all_h,         dtype=np.float32),
        'kl':         np.array(all_kl,         dtype=np.float32),
        'recon':      np.array(all_recon,       dtype=np.float32),
        'obs':        np.array(all_obs,         dtype=np.float32),
        'traj_id':    np.array(all_traj_id,     dtype=np.int32),
        'step_index': np.array(all_step_idx,    dtype=np.int32),
    }


def compute_delta_ht(data):
    """
    Compute Δh_t = h_t - h_{t-1} within each trajectory.
    First step of each episode is excluded (no prior h_t to diff against).
    Returns a new dict aligned on the non-first steps.
    """
    traj_ids  = data['traj_id']
    step_idxs = data['step_index']
    h         = data['h']

    # Mask: step is valid for delta if it is NOT the first step of its trajectory
    is_first = step_idxs == 0
    valid    = ~is_first

    # For each valid step, the previous index is i-1 (guaranteed same traj because step > 0)
    prev_h  = h[np.where(valid)[0] - 1]
    curr_h  = h[valid]
    delta_h = curr_h - prev_h

    out = {
        'dh':      delta_h,
        'h':       curr_h,
        'kl':      data['kl'][valid],
        'recon':   data['recon'][valid],
        'traj_id': traj_ids[valid],
    }
    if 'obs' in data:
        out['obs'] = data['obs'][valid]
    return out


def build_contrastive_set(data, n_bins=10, per_bin=20, max_total=200, seed=42):
    """KL-matched contrastive set from a single task. Both C1 and C2 same source."""
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

    n1, n2   = len(c1_idx), len(c2_idx)
    labels   = np.array([0]*n1 + [1]*n2, dtype=np.int32)
    all_idx  = c1_idx + c2_idx

    return {
        'dh':     data['dh'][all_idx],
        'h':      data['h'][all_idx],
        'kl':     data['kl'][all_idx],
        'recon':  data['recon'][all_idx],
        'labels': labels,
    }


def main():
    cfg = XS_CONFIG.copy()

    # ── Swingup: use existing training states (already have traj_id/step_index) ──
    print("Loading swingup training states...")
    raw_sw = dict(np.load(cfg['training_data_path']))
    sw = compute_delta_ht(raw_sw)
    print(f"  Swingup: {len(sw['dh'])} steps after excluding episode starts")
    print(f"  mean KL={sw['kl'].mean():.2f}  "
          f"mean ||Δh||={np.linalg.norm(sw['dh'], axis=1).mean():.4f}")

    # ── Balance: collect fresh with episode markers ──
    print("\nLoading model...")
    model = load_model(cfg, cfg['checkpoint_path'])

    print("\nCollecting balance states with episode markers...")
    env_bal  = CartpoleEnv(task='balance', noisy=False, seed=52)
    raw_bal  = collect_with_episode_markers(model, env_bal, cfg['n_eval_episodes'], cfg)
    bal      = compute_delta_ht(raw_bal)
    print(f"  Balance: {len(bal['dh'])} steps after excluding episode starts")
    print(f"  mean KL={bal['kl'].mean():.2f}  "
          f"mean ||Δh||={np.linalg.norm(bal['dh'], axis=1).mean():.4f}")

    # ── Build contrastive sets ──
    print("\nBuilding within-swingup contrastive set...")
    ws = build_contrastive_set(sw)
    print(f"  {(ws['labels']==0).sum()} C1 + {(ws['labels']==1).sum()} C2  |  "
          f"C1 KL={ws['kl'][ws['labels']==0].mean():.2f}  "
          f"C2 KL={ws['kl'][ws['labels']==1].mean():.2f}  |  "
          f"C1 recon={ws['recon'][ws['labels']==0].mean():.3f}  "
          f"C2 recon={ws['recon'][ws['labels']==1].mean():.3f}")

    print("\nBuilding within-balance contrastive set...")
    wb = build_contrastive_set(bal)
    print(f"  {(wb['labels']==0).sum()} C1 + {(wb['labels']==1).sum()} C2  |  "
          f"C1 KL={wb['kl'][wb['labels']==0].mean():.2f}  "
          f"C2 KL={wb['kl'][wb['labels']==1].mean():.2f}  |  "
          f"C1 recon={wb['recon'][wb['labels']==0].mean():.3f}  "
          f"C2 recon={wb['recon'][wb['labels']==1].mean():.3f}")

    # ── Train probes on swingup ──
    y_sw    = binarise_by_median(sw['kl'])
    indices = np.arange(len(sw['dh']))
    tr_idx, te_idx = train_test_split(
        indices, test_size=0.40, stratify=y_sw, random_state=0)

    print("\nTraining Δh_t probe on swingup...")
    clf_dh, sc_dh = train_probe(sw['dh'][tr_idx], y_sw[tr_idx])

    print("Training h_t probe on swingup (baseline)...")
    clf_h,  sc_h  = train_probe(sw['h'][tr_idx],  y_sw[tr_idx])

    # ── Evaluate ──
    auroc_dh_id = auroc(clf_dh, sc_dh, sw['dh'][te_idx], y_sw[te_idx])
    auroc_h_id  = auroc(clf_h,  sc_h,  sw['h'][te_idx],  y_sw[te_idx])

    auroc_dh_ws = auroc(clf_dh, sc_dh, ws['dh'], ws['labels'])
    auroc_h_ws  = auroc(clf_h,  sc_h,  ws['h'],  ws['labels'])

    auroc_dh_wb = auroc(clf_dh, sc_dh, wb['dh'], wb['labels'])
    auroc_h_wb  = auroc(clf_h,  sc_h,  wb['h'],  wb['labels'])

    # Raw ||Δh_t|| norm as a signal — no probe, no training
    norm_sw_te  = np.linalg.norm(sw['dh'][te_idx], axis=1)
    norm_ws     = np.linalg.norm(ws['dh'],          axis=1)
    norm_wb     = np.linalg.norm(wb['dh'],          axis=1)

    auroc_norm_id = roc_auc_score(y_sw[te_idx], norm_sw_te)
    auroc_norm_ws = roc_auc_score(ws['labels'],  norm_ws)
    auroc_norm_wb = roc_auc_score(wb['labels'],  norm_wb)

    # ── Results ──
    print("\n" + "="*60)
    print("Δh_t PROBE — TASK-AGNOSTIC CONFUSION SIGNAL?")
    print("="*60)
    print("\nProbes trained on swingup only.\n")
    print("KEY: within-balance = both C1 and C2 from balance, no task identity signal.\n")

    headers = ['Test', 'h_t', 'Δh_t probe', '||Δh_t|| (raw)']
    rows = [
        ['Swingup held-out (ID)',
         f'{auroc_h_id:.4f}', f'{auroc_dh_id:.4f}', f'{auroc_norm_id:.4f}'],
        ['Within-swingup (KL-matched)',
         f'{auroc_h_ws:.4f}', f'{auroc_dh_ws:.4f}', f'{auroc_norm_ws:.4f}'],
        ['Within-balance ← KEY',
         f'{auroc_h_wb:.4f}', f'{auroc_dh_wb:.4f}', f'{auroc_norm_wb:.4f}'],
    ]
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---']*len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    print("\n--- Interpretation ---")
    print(f"\n  h_t within-balance:     {auroc_h_wb:.4f}  (expected ~0.51 — confirmed confound)")
    print(f"  Δh_t within-balance:    {auroc_dh_wb:.4f}")
    print(f"  ||Δh_t|| within-balance: {auroc_norm_wb:.4f}  (raw, no training)")

    if auroc_dh_wb > 0.62:
        print("\n  TRANSFERS. Confusion signal exists in GRU update dynamics.")
        print("  The signal is task-agnostic — not where h_t is, but how fast it moves.")
    elif auroc_dh_wb > 0.55:
        print("\n  WEAK transfer. Partial task-agnostic signal in Δh_t.")
    else:
        print("\n  NO transfer. Confusion is task-specific at this scale.")
        print("  Neither h_t position nor Δh_t dynamics generalise across tasks.")

    if auroc_norm_wb > 0.60:
        print(f"\n  Raw ||Δh_t|| transfers ({auroc_norm_wb:.4f}) — KL correlates with")
        print("  update magnitude even in balance. The signal is in the magnitude alone.")

    np.savez('outputs/data/balance_with_deltas.npz', **bal)
    print("\nSaved balance_with_deltas.npz")


if __name__ == '__main__':
    main()
