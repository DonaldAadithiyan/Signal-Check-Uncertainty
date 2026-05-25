#!/usr/bin/env python3.11
"""
Experiments 2 & 3: imagination depth as uncertainty ground truth.

Experiment 3 — Probe score vs rollout depth:
  Start from real swingup states. Run imagination for 15 steps with random
  actions. Record probe(h_t) and prior entropy at each depth.
  Hypothesis: both grow monotonically — probe reads genuine compounding
  uncertainty, not an artefact of the real-observation training distribution.

Experiment 2 — Clean Set C Strong (no task identity confound):
  C1 (confident): h_t at imagination depth 1–3
  C2 (confused):  h_t at imagination depth 13–15
  Both from swingup starting states — zero task identity signal available.
  AUROC of Probe A (trained on real KL labels) separating these groups.
  Reference: Set C (real obs, KL-matched) = 0.7227.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe, auroc


N_TRAJ  = 2000   # imagination trajectories (starting states)
HORIZON = 15     # imagination steps (DreamerV3 default)
DEPTHS  = [0, 1, 3, 6, 10, 15]   # depths to report in Experiment 3


def load_model(cfg, ck_path):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def prior_entropy(prior_logits, stoch, classes):
    """Mean entropy of categorical prior distribution across the 32 categoricals (nats)."""
    B = prior_logits.shape[0]
    logits_rs = prior_logits.view(B, stoch, classes)
    log_p = torch.log_softmax(logits_rs, dim=-1)
    p     = torch.softmax(logits_rs, dim=-1)
    H     = -(p * log_p).sum(dim=-1).mean(dim=-1)  # (B,)
    return H.cpu().numpy()


def run_imagination(model, h_start, z_start, horizon, cfg, seed=0):
    """
    Batch imagination rollout from N starting states.
    Returns:
      h_per_depth:   list[horizon+1] of (N, h_dim) arrays — depth 0 is real start
      ent_per_depth: list[horizon+1] of (N,) arrays    — prior entropy at each depth
    """
    device = next(model.parameters()).device
    rng    = np.random.default_rng(seed)
    N      = h_start.shape[0]

    h = torch.tensor(h_start, dtype=torch.float32, device=device)
    z = torch.tensor(z_start, dtype=torch.float32, device=device)

    h_per_depth, ent_per_depth = [], []

    with torch.no_grad():
        # depth 0: real posterior state — compute prior directly from h
        prior_l0 = model.rssm.prior_net(h)
        h_per_depth.append(h.cpu().numpy().copy())
        ent_per_depth.append(prior_entropy(prior_l0, cfg['rssm_stoch'], cfg['rssm_classes']))

        for _ in range(horizon):
            action = torch.tensor(
                rng.uniform(-1, 1, size=(N, cfg['act_dim'])).astype(np.float32),
                device=device,
            )
            h, z, prior_l = model.rssm.imagine_step(h, z, action)
            h_per_depth.append(h.cpu().numpy().copy())
            ent_per_depth.append(prior_entropy(prior_l, cfg['rssm_stoch'], cfg['rssm_classes']))

    return h_per_depth, ent_per_depth


def main():
    cfg = XS_CONFIG.copy()

    print("Loading model...")
    model = load_model(cfg, cfg['checkpoint_path'])

    print("\nLoading training states...")
    states = dict(np.load(cfg['training_data_path']))
    h_all, z_all, kl_all = states['h'], states['z'], states['kl']
    print(f"  {len(h_all)} states | mean KL={kl_all.mean():.2f}")

    # ── Train Probe A on real KL labels (same setup as all other experiments) ──
    print("\nTraining Probe A on KL labels...")
    y = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(
        np.arange(len(h_all)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y[tr_idx])
    auroc_id = auroc(clf, sc, h_all[te_idx], y[te_idx])
    print(f"  Held-out AUROC: {auroc_id:.4f}")

    # ── Sample starting states ──
    rng = np.random.default_rng(42)
    start_idx = rng.choice(len(h_all), N_TRAJ, replace=False)
    h_start   = h_all[start_idx]
    z_start   = z_all[start_idx]
    print(f"\nSampled {N_TRAJ} starting states | mean KL={kl_all[start_idx].mean():.2f}")

    # ── Run imagination ──
    print(f"\nRunning {HORIZON}-step imagination from {N_TRAJ} starting states...")
    h_per_depth, ent_per_depth = run_imagination(model, h_start, z_start, HORIZON, cfg)

    # ── Compute probe score at every depth ──
    probe_mean = []
    for d in range(HORIZON + 1):
        scores = clf.predict_proba(sc.transform(h_per_depth[d]))[:, 1]
        probe_mean.append(scores.mean())

    # ──────────────────────────────────────────────────────────────────
    # EXPERIMENT 3: probe score vs imagination depth
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 3 — Probe score vs imagination depth")
    print("="*60)
    print("\nProbe trained on real KL labels. Tested on imagination h_t.\n")
    print(f"  {'Depth':>6}  {'Probe score':>12}  {'Prior entropy':>14}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}")
    for d in DEPTHS:
        scores_d = clf.predict_proba(sc.transform(h_per_depth[d]))[:, 1]
        print(f"  {d:>6}  {scores_d.mean():>12.4f}  {ent_per_depth[d].mean():>14.4f}")

    depths_arr = np.arange(HORIZON + 1)
    r_probe_depth = np.corrcoef(depths_arr, probe_mean)[0, 1]
    ent_means     = [ent_per_depth[d].mean() for d in range(HORIZON + 1)]
    r_probe_ent   = np.corrcoef(ent_means, probe_mean)[0, 1]

    print(f"\n  Pearson r (depth vs probe score):         {r_probe_depth:+.4f}")
    print(f"  Pearson r (prior entropy vs probe score): {r_probe_ent:+.4f}")
    print(f"  Rise depth 0→15: {probe_mean[0]:.4f} → {probe_mean[15]:.4f}  "
          f"({probe_mean[15]-probe_mean[0]:+.4f})")

    if r_probe_depth > 0.90:
        print("\n  STRONG monotonic growth — probe tracks imagination uncertainty.")
    elif r_probe_depth > 0.70:
        print("\n  MODERATE growth — probe partially tracks imagination uncertainty.")
    else:
        print("\n  WEAK / NON-MONOTONIC — probe does not track imagination depth.")

    # ──────────────────────────────────────────────────────────────────
    # EXPERIMENT 2: shallow vs deep imagination (clean Set C Strong)
    # ──────────────────────────────────────────────────────────────────
    shallow_depths = [1, 2, 3]
    deep_depths    = [13, 14, 15]

    c1_h = np.concatenate([h_per_depth[d] for d in shallow_depths])
    c2_h = np.concatenate([h_per_depth[d] for d in deep_depths])

    c1_ent = np.concatenate([ent_per_depth[d] for d in shallow_depths])
    c2_ent = np.concatenate([ent_per_depth[d] for d in deep_depths])

    labels_2 = np.array([0]*len(c1_h) + [1]*len(c2_h), dtype=np.int32)
    scores_2  = clf.predict_proba(sc.transform(np.concatenate([c1_h, c2_h])))[:, 1]
    exp2_auroc = roc_auc_score(labels_2, scores_2)

    print("\n" + "="*60)
    print("EXPERIMENT 2 — Clean Set C Strong (imagination depth)")
    print("="*60)
    print("\nC1 (confident): depths 1–3  |  C2 (confused): depths 13–15")
    print("All from swingup starting states — no task identity signal.\n")
    print(f"  C1 mean prior entropy: {c1_ent.mean():.4f}")
    print(f"  C2 mean prior entropy: {c2_ent.mean():.4f}")
    print(f"  Entropy gap (C2-C1):   {c2_ent.mean()-c1_ent.mean():+.4f}")
    print(f"\n  Probe AUROC (C1 vs C2):          {exp2_auroc:.4f}")
    print(f"  Reference — Set C (real obs):    0.7227")
    print(f"  Reference — Within-balance:      0.5060")

    if exp2_auroc > 0.72:
        print(f"\n  MATCHES Set C. Probe generalises from real-KL to imagination depth.")
        print("  Three converging lines of evidence for the within-task confusion signal.")
    elif exp2_auroc > 0.60:
        print(f"\n  PARTIAL transfer. Signal exists but weaker than on real observations.")
    else:
        print(f"\n  WEAK. Probe does not generalise to imagination depth.")


if __name__ == '__main__':
    main()
