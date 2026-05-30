#!/usr/bin/env python3.11
"""
Task 2 — Probe-weighted return estimation: does the confusion signal improve
value estimation quality?

The full Task 2 (training an actor-critic with probe-weighted λ-returns) requires
a reward head and actor-critic not present in this world-model-only codebase.
This script implements the underlying mechanism test:

  Standard λ-return estimate (imagined):
    V̂(t) = sum_{k=1}^{K} γ^k * KL(t+k)  [proxy for future confusion]
    Computed from IMAGINATION rollouts — the model predicts its own future KL.

  Probe-weighted estimate:
    V̂_w(t) = sum_{k=1}^{K} γ^k * w_{t+k} * KL_imagined(t+k)
    where w_{t+k} = 1 − probe(h_{t+k})   (down-weight confused imagined states)

  Ground truth:
    V_real(t) = sum_{k=1}^{K} γ^k * KL_real(t+k)  [from the actual trajectory]

Metric: MSE(V̂ vs V_real) and correlation r(V̂, V_real) for both methods.

If probe-weighting reduces MSE / increases correlation: the signal would improve
value estimation in a full actor-critic training loop, even without actually
training one.

Note: KL(t+k) from imagination is systematically biased high (prior-only after the
first step). The probe-weighted estimator down-weights the most confused imagined
states, acting as a selective bias-corrector.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe


HORIZON  = 5      # imagination steps (DreamerV3 default is 15; 5 for speed)
GAMMA    = 0.995  # discount
N_STATES = 5_000  # starting states for imagination
BATCH    = 256


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def imagine_kl_sequence(model, h_start, z_start, horizon, cfg, seed=0):
    """
    Run imagination from N starting states.
    Returns imagined_kl: (N, horizon) array and imagined_h: (N, horizon, h_dim).
    """
    device = next(model.parameters()).device
    rng    = np.random.default_rng(seed)
    N      = h_start.shape[0]

    h = torch.tensor(h_start, dtype=torch.float32, device=device)
    z = torch.tensor(z_start, dtype=torch.float32, device=device)

    kl_seq = []
    h_seq  = []

    with torch.no_grad():
        for k in range(horizon):
            action = torch.tensor(
                rng.uniform(-1, 1, (N, cfg['act_dim'])).astype(np.float32), device=device)
            h, z, prior_l = model.rssm.imagine_step(h, z, action)
            # KL of next prior vs uniform (prior entropy proxy):
            # Use prior logits directly — no posterior available in imagination
            logits = prior_l.view(N, cfg['rssm_stoch'], cfg['rssm_classes'])
            log_p  = torch.log_softmax(logits, dim=-1)
            p      = torch.softmax(logits, dim=-1)
            H      = -(p * log_p).sum(dim=-1).mean(dim=-1)  # (N,) prior entropy as KL proxy
            kl_seq.append(H.cpu().numpy())
            h_seq.append(h.cpu().numpy().copy())

    return np.stack(kl_seq, axis=1), np.stack(h_seq, axis=1)  # (N, H), (N, H, d)


def lambda_return(kl_seq, gamma=0.995):
    """Simple discounted sum over KL sequence. Shape (N, H) → (N,)."""
    H = kl_seq.shape[1]
    gammas = gamma ** np.arange(1, H + 1)
    return (kl_seq * gammas[None, :]).sum(axis=1)


def probe_weighted_return(kl_seq, h_seq, clf, sc, gamma=0.995):
    """Down-weight each imagined step by (1 - probe(h_t))."""
    N, H, d = h_seq.shape
    weights = np.zeros((N, H), dtype=np.float32)
    for k in range(H):
        h_k = h_seq[:, k, :]
        probe_scores = clf.predict_proba(sc.transform(h_k))[:, 1]
        weights[:, k] = 1.0 - probe_scores  # down-weight confused imagined states
    gammas = gamma ** np.arange(1, H + 1)
    return (kl_seq * weights * gammas[None, :]).sum(axis=1)


def main():
    cfg = XS_CONFIG.copy()

    # ── Train Probe A ──
    print("Loading training states and training Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all  = tr['h']
    z_all  = tr['z']
    kl_all = tr['kl']
    traj   = tr['traj_id']
    N      = len(h_all)

    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(
        np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y_kl[tr_idx])
    print(f"  Trained on {len(tr_idx):,} states")

    # ── Build real K-step returns from trajectory data ──
    print(f"\nBuilding real {HORIZON}-step returns from trajectories...")
    # For each starting state i, find real KL(i+1..i+H) in same trajectory
    real_returns = {}
    for i in te_idx:
        kl_future = []
        ok = True
        for k in range(1, HORIZON + 1):
            j = i + k
            if j >= N or traj[j] != traj[i]:
                ok = False
                break
            kl_future.append(kl_all[j])
        if ok:
            real_returns[i] = np.array(kl_future, dtype=np.float32)

    valid_idx = np.array(list(real_returns.keys()), dtype=np.int64)
    print(f"  {len(valid_idx):,} valid starting states with {HORIZON}-step real trajectories")

    # Subsample for speed
    rng = np.random.default_rng(42)
    start_idx = valid_idx[rng.choice(len(valid_idx), min(N_STATES, len(valid_idx)), replace=False)]

    real_kl_matrix = np.stack([real_returns[i] for i in start_idx], axis=0)  # (N, H)
    real_V = lambda_return(real_kl_matrix, GAMMA)

    # ── Run imagination from starting states ──
    print(f"\nRunning {HORIZON}-step imagination from {len(start_idx):,} starting states...")
    model = load_model(cfg)
    imag_kl, imag_h = imagine_kl_sequence(
        model, h_all[start_idx], z_all[start_idx], HORIZON, cfg)

    # ── Compute standard vs probe-weighted imagined returns ──
    V_standard = lambda_return(imag_kl, GAMMA)
    V_weighted = probe_weighted_return(imag_kl, imag_h, clf, sc, GAMMA)

    # Normalise weighted returns to same scale as standard (for MSE comparison)
    V_weighted_norm = V_weighted * (V_standard.mean() / (V_weighted.mean() + 1e-8))

    # ── Evaluation ──
    r_standard = np.corrcoef(V_standard, real_V)[0, 1]
    r_weighted = np.corrcoef(V_weighted, real_V)[0, 1]

    mse_standard = np.mean((V_standard - real_V) ** 2)
    mse_weighted = np.mean((V_weighted_norm - real_V) ** 2)

    mae_standard = np.mean(np.abs(V_standard - real_V))
    mae_weighted = np.mean(np.abs(V_weighted_norm - real_V))

    # Bias: imagined KL is systematically higher than real
    imag_kl_mean = imag_kl.mean()
    real_kl_mean = real_kl_matrix.mean()

    # Per-horizon: correlation of imagined KL step k vs real KL step k
    print("\n" + "="*65)
    print("PROBE-WEIGHTED RETURN ESTIMATION")
    print("="*65)
    print(f"\n  Imagination horizon: {HORIZON} steps  |  γ={GAMMA}  |  N={len(start_idx):,}")
    print(f"\n  KL bias (imagination vs real):")
    print(f"    Imagined mean KL (prior entropy): {imag_kl_mean:.4f}")
    print(f"    Real mean KL:                     {real_kl_mean:.4f}")
    print(f"    Ratio (imag/real):                {imag_kl_mean/real_kl_mean:.2f}x")

    print(f"\n  Per-horizon step correlation r(imagined KL_k, real KL_k):")
    print(f"  {'k':>4}  {'r(imag, real)':>14}  {'mean_imag':>10}  {'mean_real':>10}")
    for k in range(HORIZON):
        r_k = np.corrcoef(imag_kl[:, k], real_kl_matrix[:, k])[0, 1]
        print(f"  {k+1:>4}  {r_k:>14.4f}  {imag_kl[:, k].mean():>10.4f}  "
              f"{real_kl_matrix[:, k].mean():>10.4f}")

    print(f"\n  Return estimate quality (λ-return over {HORIZON} steps):")
    print(f"  {'Method':<28}  {'r(V̂, V_real)':>14}  {'MSE':>10}  {'MAE':>10}")
    print(f"  {'-'*28}  {'-'*14}  {'-'*10}  {'-'*10}")
    print(f"  {'Standard imagined return':<28}  {r_standard:>14.4f}  "
          f"{mse_standard:>10.4f}  {mae_standard:>10.4f}")
    print(f"  {'Probe-weighted return':<28}  {r_weighted:>14.4f}  "
          f"{mse_weighted:>10.4f}  {mae_weighted:>10.4f}")
    print(f"\n  Δr (weighted − standard):  {r_weighted - r_standard:+.4f}")
    print(f"  ΔMSE (weighted − standard): {mse_weighted - mse_standard:+.4f} "
          f"({'better' if mse_weighted < mse_standard else 'worse'})")
    print(f"  ΔMAE (weighted − standard): {mae_weighted - mae_standard:+.4f} "
          f"({'better' if mae_weighted < mae_standard else 'worse'})")

    # ── Breakdown by confusion level ──
    probe_at_start = clf.predict_proba(sc.transform(h_all[start_idx]))[:, 1]
    conf_mask = probe_at_start > np.median(probe_at_start)
    cope_mask = ~conf_mask

    print(f"\n  Breakdown by starting confusion level:")
    print(f"  {'Group':<20}  {'N':>5}  {'r_std':>8}  {'r_wt':>8}  {'Δr':>7}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}")
    for name, mask in [('Coping (lo probe)', cope_mask), ('Confused (hi probe)', conf_mask)]:
        r_s = np.corrcoef(V_standard[mask], real_V[mask])[0, 1]
        r_w = np.corrcoef(V_weighted[mask], real_V[mask])[0, 1]
        print(f"  {name:<20}  {mask.sum():>5}  {r_s:>8.4f}  {r_w:>8.4f}  {r_w-r_s:>+7.4f}")

    if r_weighted > r_standard:
        print(f"\n  POSITIVE: probe-weighting improves return estimate correlation "
              f"(Δr={r_weighted-r_standard:+.4f}).")
        print("  In a full actor-critic, this would reduce value estimation error on")
        print("  confused imagined states, potentially improving sample efficiency.")
    elif abs(r_weighted - r_standard) < 0.005:
        print(f"\n  NULL: probe-weighting has negligible effect on return quality "
              f"(Δr={r_weighted-r_standard:+.4f}).")
        print("  The signal may be too small at XS scale / 5-step horizon to matter.")
    else:
        print(f"\n  NEGATIVE: probe-weighting degrades return estimates "
              f"(Δr={r_weighted-r_standard:+.4f}).")
        print("  Possible cause: the probe direction is more correlated with future KL")
        print("  than with imagination quality — weighting removes signal.")

    print("\n  Note: full Task 2 (actor-critic training) requires a reward head not present")
    print("  in this world-model-only codebase. This analysis tests the underlying")
    print("  mechanism: whether probe-weighting corrects imagination bias in KL estimates.")


if __name__ == '__main__':
    main()
