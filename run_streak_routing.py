#!/usr/bin/env python3.11
"""
Mechanism analysis for why probe beats recon-threshold in active querying.

At 30% query budget, Probe A recall=0.818 > recon oracle recall=0.770.

Hypothesis: the probe's advantage is concentrated in states with high streak
length L_t (multi-step confused sequences). These states have high current
confusion (caught by probe via C_t) but may have moderate current recon error
(missed by recon-threshold because the confusion is sustained, not acute).

Analysis:
  For held-out states at 30% query budget:
    probe-only:  probe queries, recon does not  → these are the probe's advantage
    recon-only:  recon queries, probe does not  → these are recon's advantage
    both:        both query
    neither:     both miss

  Compare mean streak length L_t across these 4 groups.
  If probe-only states have substantially higher L_t than recon-only states,
  trajectory history (C_t) is the proven mechanism.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.probe.linear_probe import binarise_by_median, train_probe


QUERY_BUDGET = 0.30
MAX_LAG      = 50


def compute_streak(kl, traj_id, high_kl):
    N = len(kl)
    streak = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if high_kl[i]:
            if i > 0 and traj_id[i] == traj_id[i-1]:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        else:
            streak[i] = 0
    return streak


def main():
    cfg = XS_CONFIG.copy()

    print("Loading training states...")
    tr       = dict(np.load(cfg['training_data_path']))
    h_all    = tr['h']
    kl_all   = tr['kl']
    recon_all = tr['recon']
    traj_id  = tr['traj_id']
    N        = len(h_all)

    kl_median = np.median(kl_all)
    high_kl   = kl_all > kl_median

    print("Computing streak lengths...")
    streak_all = compute_streak(kl_all, traj_id, high_kl)

    print("Training probe...")
    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y_kl[tr_idx])

    h_te     = h_all[te_idx]
    kl_te    = kl_all[te_idx]
    recon_te = recon_all[te_idx]
    streak_te = streak_all[te_idx]
    N_te     = len(te_idx)

    # Rank-normalised scores
    probe_raw    = clf.predict_proba(sc.transform(h_te))[:, 1]
    probe_norm   = (rankdata(probe_raw)  / N_te).astype(np.float32)
    recon_norm   = (rankdata(recon_te)   / N_te).astype(np.float32)

    # Top-25% KL events
    kl_75th      = np.percentile(kl_te, 75)
    high_kl_mask = kl_te >= kl_75th

    # At 30% budget: which states each policy queries
    probe_thresh = np.percentile(probe_norm, 100 * (1 - QUERY_BUDGET))
    recon_thresh  = np.percentile(recon_norm, 100 * (1 - QUERY_BUDGET))

    probe_queries  = probe_norm >= probe_thresh
    recon_queries  = recon_norm  >= recon_thresh

    # 4 groups
    probe_only  = probe_queries & ~recon_queries
    recon_only  = ~probe_queries & recon_queries
    both        = probe_queries & recon_queries
    neither     = ~probe_queries & ~recon_queries

    print("\n" + "="*70)
    print("STREAK ROUTING MECHANISM ANALYSIS")
    print("="*70)
    print(f"\n  Query budget: {QUERY_BUDGET:.0%}  |  "
          f"Probe recall={high_kl_mask[probe_queries].mean():.3f}  "
          f"Recon recall={high_kl_mask[recon_queries].mean():.3f}")
    print(f"\n  {'Group':<20}  {'N':>6}  {'High-KL %':>10}  {'Mean streak L_t':>16}  "
          f"{'Mean KL':>8}  {'Mean recon':>10}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*10}  {'-'*16}  {'-'*8}  {'-'*10}")

    groups = {
        'probe-only':   probe_only,
        'recon-only':   recon_only,
        'both':         both,
        'neither':      neither,
    }

    streaks_by_group = {}
    for name, mask in groups.items():
        n        = mask.sum()
        if n == 0:
            print(f"  {name:<20}  {n:>6}  (empty)")
            continue
        hkl_pct  = high_kl_mask[mask].mean() * 100
        mean_str = streak_te[mask].mean()
        mean_kl  = kl_te[mask].mean()
        mean_rc  = recon_te[mask].mean()
        streaks_by_group[name] = streak_te[mask]
        print(f"  {name:<20}  {n:>6}  {hkl_pct:>9.1f}%  {mean_str:>16.2f}  "
              f"{mean_kl:>8.2f}  {mean_rc:>10.4f}")

    # ── Focus comparison: probe-only vs recon-only ──
    print(f"\n  Key comparison — probe-only vs recon-only states:")
    if 'probe-only' in streaks_by_group and 'recon-only' in streaks_by_group:
        sp = streaks_by_group['probe-only']
        sr = streaks_by_group['recon-only']
        print(f"    probe-only mean streak: {sp.mean():.2f}  median: {np.median(sp):.1f}  "
              f"(L_t > 5: {(sp>5).mean()*100:.0f}%)")
        print(f"    recon-only mean streak: {sr.mean():.2f}  median: {np.median(sr):.1f}  "
              f"(L_t > 5: {(sr>5).mean()*100:.0f}%)")
        print(f"    ratio: {sp.mean()/sr.mean():.2f}×  "
              f"({'probe catches longer streaks' if sp.mean() > sr.mean() else 'recon catches longer streaks'})")

        # What fraction of probe-only states have streak > threshold?
        for threshold in [3, 5, 10]:
            p_pct = (sp > threshold).mean() * 100
            r_pct = (sr > threshold).mean() * 100
            print(f"    L_t > {threshold}: probe-only {p_pct:.0f}%  vs  recon-only {r_pct:.0f}%")

    # ── Mean streak for all high-KL states (baseline) ──
    all_hkl_streak = streak_te[high_kl_mask]
    print(f"\n  All high-KL states (top-25%): mean streak = {all_hkl_streak.mean():.2f}  "
          f"(L_t > 5: {(all_hkl_streak>5).mean()*100:.0f}%)")

    # ── Interpretation ──
    if ('probe-only' in streaks_by_group and 'recon-only' in streaks_by_group and
            streaks_by_group['probe-only'].mean() > streaks_by_group['recon-only'].mean() * 1.3):
        print(f"\n  MECHANISM CONFIRMED: probe-only states have {streaks_by_group['probe-only'].mean():.1f}× "
              f"longer mean streak than recon-only states.")
        print("  The probe's advantage over recon-threshold is concentrated in multi-step")
        print("  confused sequences — exactly what C_t (trajectory history) captures.")
    elif ('probe-only' in streaks_by_group and 'recon-only' in streaks_by_group and
          streaks_by_group['probe-only'].mean() > streaks_by_group['recon-only'].mean()):
        print(f"\n  MECHANISM PARTIAL: probe-only states have higher mean streak "
              f"({streaks_by_group['probe-only'].mean():.1f} vs {streaks_by_group['recon-only'].mean():.1f}) "
              f"but difference is modest.")
    else:
        print("\n  MECHANISM WEAK: streak lengths not concentrated in probe-only states.")
        print("  The probe's advantage may be from overall better KL calibration,")
        print("  not specifically from trajectory-history context.")


if __name__ == '__main__':
    main()
