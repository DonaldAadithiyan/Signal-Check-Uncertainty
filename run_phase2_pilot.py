#!/usr/bin/env python3.11
"""
Phase 2 pilot — three experiments.

Experiment 1 — Real trajectory prediction:
  Does probe(h_t) predict KL(t+k) and recon(t+k) for k = 1, 3, 5, 10, 20?
  Compared against the KL(t) autocorrelation baseline.
  Key test: does probe add predictive information BEYOND current KL?
  (partial R² increase when probe is added to a model that already has KL_t)

Experiment 3 — Error at t+k regression:
  R² as a function of k: probe(h_t) → KL(t+k).
  Compared against R²: KL(t) → KL(t+k).
  If probe R² curve decays slower — probe carries longer-horizon information.

Experiment 2 — Observation-vs-imagination boundary:
  Does probe(h_t) predict imagination quality at t+1?
  For each real state, run one imagination step. Compute prior entropy of result.
  r(probe(h_t), prior_entropy_at_depth_1) vs r(KL(t), prior_entropy_at_depth_1).
  Positive result: the probe predicts how uncertain the model's next imagined
  step will be — a concrete early-warning signal.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe, auroc


LAGS      = [1, 3, 5, 10, 20]
N_IMAGINE = 10_000   # states for Experiment 2 imagination step


def load_model(cfg, ck_path):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def build_lag_pairs(te_idx_set, traj_id, k):
    """
    For each position i in te_idx, find i+k in the same trajectory.
    Returns: (i_arr, ik_arr) index arrays.
    """
    i_arr, ik_arr = [], []
    for i in te_idx_set:
        ik = i + k
        if ik < len(traj_id) and traj_id[ik] == traj_id[i]:
            i_arr.append(i)
            ik_arr.append(ik)
    return np.array(i_arr), np.array(ik_arr)


def partial_r2_increase(X_base, X_probe, y):
    """
    R² of base predictor, then R² when probe is added.
    Returns: r2_base, r2_full, delta_r2
    """
    X_b = X_base.reshape(-1, 1)
    X_f = np.column_stack([X_base, X_probe])
    r2_base = r2_score(y, LinearRegression().fit(X_b, y).predict(X_b))
    r2_full = r2_score(y, LinearRegression().fit(X_f, y).predict(X_f))
    return r2_base, r2_full, r2_full - r2_base


def prior_entropy_batch(model, h_arr, z_arr, cfg, batch=512, seed=0):
    """One imagination step from each (h, z). Returns prior entropy (N,)."""
    device = next(model.parameters()).device
    rng    = np.random.default_rng(seed)
    N      = len(h_arr)
    entropies = []

    with torch.no_grad():
        for start in range(0, N, batch):
            h_b = torch.tensor(h_arr[start:start+batch], dtype=torch.float32, device=device)
            z_b = torch.tensor(z_arr[start:start+batch], dtype=torch.float32, device=device)
            B   = h_b.shape[0]
            a_b = torch.tensor(
                rng.uniform(-1, 1, size=(B, cfg['act_dim'])).astype(np.float32),
                device=device,
            )
            _, _, prior_l = model.rssm.imagine_step(h_b, z_b, a_b)
            logits_rs = prior_l.view(B, cfg['rssm_stoch'], cfg['rssm_classes'])
            log_p = torch.log_softmax(logits_rs, dim=-1)
            p     = torch.softmax(logits_rs, dim=-1)
            H     = -(p * log_p).sum(dim=-1).mean(dim=-1)
            entropies.append(H.cpu().numpy())

    return np.concatenate(entropies)


def main():
    cfg = XS_CONFIG.copy()

    print("Loading model...")
    model = load_model(cfg, cfg['checkpoint_path'])

    print("\nLoading training states...")
    states  = dict(np.load(cfg['training_data_path']))
    h_all   = states['h']
    z_all   = states['z']
    kl_all  = states['kl']
    recon_all = states['recon']
    traj_id = states['traj_id']
    N = len(h_all)

    # ── Train Probe A — held-out split ──
    print("\nTraining Probe A (KL labels, 60/40 split)...")
    y = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(
        np.arange(N), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y[tr_idx])
    auroc_id = auroc(clf, sc, h_all[te_idx], y[te_idx])
    print(f"  Probe A held-out AUROC: {auroc_id:.4f}")

    # Probe scores on ALL states (for lag analysis and imagination step)
    print("  Computing probe scores for all states...")
    all_probe = clf.predict_proba(sc.transform(h_all))[:, 1]

    te_idx_set = set(te_idx.tolist())

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENTS 1 & 3 — real trajectory prediction
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("EXPERIMENTS 1 & 3 — Real trajectory prediction")
    print("="*65)
    print("\nProbe(h_t) and KL(t) as predictors of KL(t+k) and recon(t+k).")
    print("ΔR² = R²(KL_t + probe) − R²(KL_t alone): probe's extra contribution.\n")

    print(f"  {'k':>4}  {'N pairs':>8}  "
          f"{'r(probe, KL_tk)':>16}  {'r(KL_t, KL_tk)':>15}  "
          f"{'R²(KL_t)':>9}  {'R²(+probe)':>11}  {'ΔR²':>7}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*16}  {'-'*15}  {'-'*9}  {'-'*11}  {'-'*7}")

    lag_results = {}
    for k in LAGS:
        i_arr, ik_arr = build_lag_pairs(te_idx_set, traj_id, k)
        if len(i_arr) == 0:
            continue

        probe_t  = all_probe[i_arr]
        kl_t     = kl_all[i_arr]
        kl_tk    = kl_all[ik_arr]

        r_probe = np.corrcoef(probe_t, kl_tk)[0, 1]
        r_kl    = np.corrcoef(kl_t,   kl_tk)[0, 1]
        r2_base, r2_full, delta_r2 = partial_r2_increase(kl_t, probe_t, kl_tk)

        lag_results[k] = dict(r_probe=r_probe, r_kl=r_kl,
                               r2_base=r2_base, r2_full=r2_full, delta_r2=delta_r2,
                               n=len(i_arr))

        print(f"  {k:>4}  {len(i_arr):>8}  "
              f"{r_probe:>+16.4f}  {r_kl:>+15.4f}  "
              f"{r2_base:>9.4f}  {r2_full:>11.4f}  {delta_r2:>+7.4f}")

    # Also test against recon(t+k)
    print(f"\n  Probe vs recon(t+k):\n"
          f"  {'k':>4}  {'r(probe, recon_tk)':>19}  {'r(KL_t, recon_tk)':>18}")
    print(f"  {'-'*4}  {'-'*19}  {'-'*18}")
    for k in LAGS:
        i_arr, ik_arr = build_lag_pairs(te_idx_set, traj_id, k)
        if len(i_arr) == 0:
            continue
        probe_t   = all_probe[i_arr]
        kl_t      = kl_all[i_arr]
        recon_tk  = recon_all[ik_arr]
        r_probe_r = np.corrcoef(probe_t, recon_tk)[0, 1]
        r_kl_r    = np.corrcoef(kl_t,   recon_tk)[0, 1]
        print(f"  {k:>4}  {r_probe_r:>+19.4f}  {r_kl_r:>+18.4f}")

    # R² decay curve summary
    print(f"\n  R² decay curve (probe vs KL baseline):")
    print(f"  {'k':>4}  {'R²(probe)':>10}  {'R²(KL_t)':>10}  {'probe_edge':>11}")
    for k in LAGS:
        if k not in lag_results:
            continue
        r = lag_results[k]
        edge = r['r_probe']**2 - r['r_kl']**2
        print(f"  {k:>4}  {r['r_probe']**2:>10.4f}  {r['r_kl']**2:>10.4f}  {edge:>+11.4f}")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 2 — observation-vs-imagination boundary
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("EXPERIMENT 2 — Observation-vs-imagination boundary")
    print("="*65)
    print("\nDoes probe(h_t_real) predict prior entropy after one imagination step?")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(te_idx, N_IMAGINE, replace=False)

    print(f"\nRunning one imagination step from {N_IMAGINE} held-out states...")
    depth1_ent = prior_entropy_batch(
        model, h_all[sample_idx], z_all[sample_idx], cfg)

    probe_sample = all_probe[sample_idx]
    kl_sample    = kl_all[sample_idx]

    r_probe_ent = np.corrcoef(probe_sample, depth1_ent)[0, 1]
    r_kl_ent    = np.corrcoef(kl_sample,   depth1_ent)[0, 1]

    r2_base_e, r2_full_e, delta_r2_e = partial_r2_increase(
        kl_sample, probe_sample, depth1_ent)

    print(f"\n  r(probe(h_t),  prior_entropy_depth1): {r_probe_ent:+.4f}")
    print(f"  r(KL(t),       prior_entropy_depth1): {r_kl_ent:+.4f}")
    print(f"\n  R²(KL_t → entropy_depth1):            {r2_base_e:.4f}")
    print(f"  R²(KL_t + probe → entropy_depth1):    {r2_full_e:.4f}")
    print(f"  ΔR²:                                  {delta_r2_e:+.4f}")

    if r_probe_ent > 0.10:
        print(f"\n  Probe predicts imagination quality — early-warning signal exists.")
    elif r_probe_ent > 0.03:
        print(f"\n  Weak positive correlation — marginal early-warning signal.")
    else:
        print(f"\n  No early-warning signal from probe to imagination quality.")

    # ── Summary ──
    print("\n" + "="*65)
    print("PHASE 2 PILOT SUMMARY")
    print("="*65)
    k5 = lag_results.get(5, {})
    k10 = lag_results.get(10, {})
    print(f"\n  Probe AUROC (within-step):    {auroc_id:.4f}  [reference: Set C = 0.7227]")
    print(f"  r(probe, KL_t+5):             {k5.get('r_probe', float('nan')):+.4f}")
    print(f"  r(KL_t,  KL_t+5):             {k5.get('r_kl',    float('nan')):+.4f}")
    print(f"  ΔR² probe adds at k=5:        {k5.get('delta_r2', float('nan')):+.4f}")
    print(f"  r(probe, KL_t+10):            {k10.get('r_probe', float('nan')):+.4f}")
    print(f"  ΔR² probe adds at k=10:       {k10.get('delta_r2', float('nan')):+.4f}")
    print(f"  r(probe, depth1 entropy):     {r_probe_ent:+.4f}")
    print(f"  ΔR² probe adds (imagination): {delta_r2_e:+.4f}")

    if k5.get('delta_r2', 0) > 0.01:
        print("\n  PHASE 2 POSITIVE: probe adds predictive information beyond KL autocorrelation.")
    elif k5.get('delta_r2', 0) > 0.002:
        print("\n  MARGINAL: probe adds small but nonzero information beyond KL at k=5.")
    else:
        print("\n  NEGATIVE: probe adds no predictive information beyond KL autocorrelation.")


if __name__ == '__main__':
    main()
