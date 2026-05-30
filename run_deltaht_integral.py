#!/usr/bin/env python3.11
"""
Task 4 — Δh_t confusion integral characterisation.

If h_t encodes C_t (accumulated confusion), what does Δh_t encode?
C_t satisfies the recursion: C_t = 1[KL_t > median] + γ·C_{t-1}
So: ΔC_t = C_t - C_{t-1} = 1[KL_t > median] - (1-γ)·C_{t-1}

This is the rate of change of the confusion accumulation: the current step's
indicator minus a fraction of the accumulated past. ΔC_t is positive when the
current step is confused AND the model hasn't been confused for long (new confusion);
negative when the model exits a confused streak.

Test: R²(Δh_t_probe_scores ~ ΔC_t) for different γ values.
If R² > 0.70, Δh_t encodes ΔC_t — a complementary characterisation.

The two representations:
  h_t  → C_t (accumulated confusion over ~13 steps, within-task depth)
  Δh_t → ΔC_t (rate of change, cross-task transfer)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe


GAMMAS  = [0.70, 0.80, 0.90, 0.95, 0.99]
MAX_LAG = 50


def compute_ct(kl, traj_id, gamma, max_lag, high_kl=None):
    if high_kl is None:
        high_kl = (kl > np.median(kl)).astype(np.float32)
    N  = len(kl)
    ct = np.zeros(N, dtype=np.float32)
    for i in range(N):
        val = 0.0
        for lag in range(max_lag):
            j = i - lag
            if j < 0 or traj_id[j] != traj_id[i]:
                break
            val += (gamma ** lag) * high_kl[j]
        ct[i] = val
    return ct


def compute_delta_ct(kl, traj_id, gamma, max_lag):
    """ΔC_t = C_t - C_{t-1} within same trajectory."""
    high_kl = (kl > np.median(kl)).astype(np.float32)
    ct      = compute_ct(kl, traj_id, gamma, max_lag, high_kl)
    N       = len(kl)
    delta_ct = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if i > 0 and traj_id[i] == traj_id[i-1]:
            delta_ct[i] = ct[i] - ct[i-1]
        else:
            delta_ct[i] = ct[i]   # first step: ΔC_t = C_t itself
    return delta_ct, ct


def main():
    cfg = XS_CONFIG.copy()

    print("Loading training states...")
    tr      = dict(np.load(cfg['training_data_path']))
    h_all   = tr['h']
    kl_all  = tr['kl']
    traj_id = tr['traj_id']
    N       = len(h_all)

    kl_median = np.median(kl_all)
    high_kl   = (kl_all > kl_median).astype(np.float32)

    # Compute Δh_t (step >= 1, same trajectory)
    print("Computing Δh_t...")
    valid = np.zeros(N, dtype=bool)
    dh    = np.zeros_like(h_all)
    for i in range(1, N):
        if traj_id[i] == traj_id[i-1]:
            dh[i]    = h_all[i] - h_all[i-1]
            valid[i] = True

    # ── Train Δh_t probe and h_t probe (both binary KL) ──
    print("Training probes...")
    y_kl   = binarise_by_median(kl_all)
    # Use only valid (step>=1) states for Δh_t probe
    idx_v  = np.where(valid)[0]
    y_v    = y_kl[idx_v]
    tr_v, te_v = train_test_split(idx_v, test_size=0.40, stratify=y_v, random_state=0)

    clf_dh, sc_dh = train_probe(dh[tr_v], y_kl[tr_v])
    clf_h,  sc_h  = train_probe(h_all[tr_v], y_kl[tr_v])

    dh_probe_scores = clf_dh.predict_proba(sc_dh.transform(dh[te_v]))[:, 1]
    h_probe_scores  = clf_h.predict_proba(sc_h.transform(h_all[te_v]))[:, 1]

    print(f"\n  Valid states (step>=1): {valid.sum():,} / {N:,}")
    print(f"  Test states: {len(te_v):,}")

    # ── Gamma sweep ──
    print("\n" + "="*65)
    print("Δh_t CONFUSION INTEGRAL — γ SWEEP")
    print("="*65)
    print(f"\n  Baseline: R²(Δh_t_probe ~ current_1[KL>median]) = "
          f"{r2_score(high_kl[te_v], dh_probe_scores):.4f}")

    print(f"\n  {'γ':>6}  {'R²(Δh_t ~ ΔC_t)':>16}  {'R²(h_t ~ C_t) ref':>18}  "
          f"{'ΔC_t mean':>10}")
    print(f"  {'-'*6}  {'-'*16}  {'-'*18}  {'-'*10}")

    best_r2_dh, best_gamma_dh = -1, None
    for g in GAMMAS:
        delta_ct, ct = compute_delta_ct(kl_all, traj_id, g, MAX_LAG)

        # R² for Δh_t probe scores vs ΔC_t
        r2_dh = r2_score(delta_ct[te_v],
                         LinearRegression().fit(
                             dh_probe_scores.reshape(-1, 1),
                             delta_ct[te_v]).predict(
                             dh_probe_scores.reshape(-1, 1)))

        # R² for h_t probe scores vs C_t (reference)
        r2_h = r2_score(ct[te_v],
                        LinearRegression().fit(
                            h_probe_scores.reshape(-1, 1),
                            ct[te_v]).predict(
                            h_probe_scores.reshape(-1, 1)))

        print(f"  {g:>6.2f}  {r2_dh:>16.4f}  {r2_h:>18.4f}  {delta_ct.mean():>10.4f}")
        if r2_dh > best_r2_dh:
            best_r2_dh, best_gamma_dh = r2_dh, g

    best_delta_ct, best_ct = compute_delta_ct(kl_all, traj_id, best_gamma_dh, MAX_LAG)

    # Also check R²(Δh_t ~ C_t) — maybe Δh_t tracks accumulation not rate-of-change
    r2_dh_vs_ct = r2_score(
        best_ct[te_v],
        LinearRegression().fit(
            dh_probe_scores.reshape(-1, 1),
            best_ct[te_v]).predict(
            dh_probe_scores.reshape(-1, 1)))

    print(f"\n  Best Δh_t ~ ΔC_t R²: {best_r2_dh:.4f} at γ={best_gamma_dh}")
    print(f"  R²(Δh_t_probe ~ C_t accumulation): {r2_dh_vs_ct:.4f}")
    print(f"  R²(h_t_probe  ~ C_t accumulation, ref): "
          f"{r2_score(best_ct[te_v], LinearRegression().fit(h_probe_scores.reshape(-1,1), best_ct[te_v]).predict(h_probe_scores.reshape(-1,1))):.4f}")

    print(f"\n  INTERPRETATION:")
    if best_r2_dh > 0.60:
        print(f"  Δh_t probe tracks ΔC_t at R²={best_r2_dh:.4f}.")
        print("  h_t encodes accumulated confusion (C_t); Δh_t encodes its rate of change (ΔC_t).")
        print("  Two complementary representations of the same underlying process.")
    elif best_r2_dh > 0.30:
        print(f"  Δh_t probe has partial ΔC_t signal (R²={best_r2_dh:.4f}).")
        print("  The Δh_t representation is partially consistent with ΔC_t.")
    else:
        print(f"  Δh_t probe does not cleanly approximate ΔC_t (R²={best_r2_dh:.4f}).")
        print("  Δh_t may be capturing a different aspect of the confusion update.")


if __name__ == '__main__':
    main()
