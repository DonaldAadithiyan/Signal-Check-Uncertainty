#!/usr/bin/env python3.11
"""
Task P — Confusion as an imagination STOPPING RULE (structurally different from Task F).

Task F tried confusion as a continuous REWEIGHTING term applied uniformly across an
imagined rollout (binary probe weight, then continuous C_t weight) — both failed cleanly.
That is one mechanism. This tries a structurally different one: confusion as a STOPPING
RULE that truncates imagination before it compounds error, rather than a weight that
discounts it after the fact.

Same setup as Task F: same 5,000 starting states (held-out), horizon-5 imagined rollouts,
γ=0.995, real-return ground truth. At each imagined step we monitor the confusion signal
(Probe A score on the imagined h_t) and truncate the rollout once it crosses a threshold,
using the discounted return accumulated up to truncation PLUS a terminal value estimate at
the truncation point (rather than continuing to the fixed horizon).

Thresholds selected on a held-out calibration subset (no cherry-picking). Compared against
Task F's standard (no-weight, fixed-horizon) baseline via r(V̂, V_real) with bootstrap CIs.

Reports whichever outcome is true, same honesty standard as Task F.
Runs on the existing frozen cartpole model. XS, CPU.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv  # noqa: F401 (kept for parity/reproducibility)
from src.probe.linear_probe import binarise_by_median, train_probe

HORIZON  = 5
GAMMA    = 0.995
N_STATES = 5000
N_BOOT   = 1000
OUT_DIR  = 'outputs/causal'


def load_model(cfg):
    ck = torch.load(cfg['checkpoint_path'], map_location='cpu')
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg'])
    m.load_state_dict(ck['model_state']); m.eval()
    return m


def imagine(model, cfg, h_start, z_start, horizon, seed=0):
    """Task F's imagination: returns per-step KL-proxy (prior entropy) and h, (N,H) and (N,H,D)."""
    rng = np.random.default_rng(seed)
    N = h_start.shape[0]
    h = torch.tensor(h_start, dtype=torch.float32)
    z = torch.tensor(z_start, dtype=torch.float32)
    kl_seq, h_seq = [], []
    with torch.no_grad():
        for _ in range(horizon):
            a = torch.tensor(rng.uniform(-1, 1, (N, cfg['act_dim'])).astype(np.float32))
            h, z, prior_l = model.rssm.imagine_step(h, z, a)
            logits = prior_l.view(N, cfg['rssm_stoch'], cfg['rssm_classes'])
            p = torch.softmax(logits, dim=-1)
            H = -(p * torch.log_softmax(logits, dim=-1)).sum(-1).mean(-1)
            kl_seq.append(H.numpy()); h_seq.append(h.numpy().copy())
    return np.stack(kl_seq, 1), np.stack(h_seq, 1)


def lambda_return(kl_seq, gamma=GAMMA):
    g = gamma ** np.arange(1, kl_seq.shape[1] + 1)
    return (kl_seq * g[None, :]).sum(1)


def stopped_return(kl_seq, probe_seq, threshold, gamma=GAMMA):
    """Truncate the discounted KL-proxy sum at the first imagined step whose probe score
    exceeds `threshold`; add a terminal value estimate = the accumulated mean per-step KL
    extrapolated over the remaining horizon (a simple bootstrap terminal value). Returns (N,)."""
    N, Hh = kl_seq.shape
    g = gamma ** np.arange(1, Hh + 1)
    out = np.zeros(N)
    for i in range(N):
        # find first step crossing threshold (1-indexed step k → array index k-1)
        crossed = np.where(probe_seq[i] > threshold)[0]
        t_stop = crossed[0] + 1 if len(crossed) > 0 else Hh   # truncate AFTER the crossing step
        acc = (kl_seq[i, :t_stop] * g[:t_stop]).sum()
        if t_stop < Hh:
            # terminal value: mean per-step KL so far, discounted over the remaining steps
            mean_kl = kl_seq[i, :t_stop].mean()
            acc += mean_kl * g[t_stop:].sum()
        out[i] = acc
    return out


def r_ci(a, b, n_boot=N_BOOT, seed=0):
    """Bootstrap CI of Pearson r between a and b."""
    rng = np.random.default_rng(seed)
    n = len(a)
    point = np.corrcoef(a, b)[0, 1]
    boots = [np.corrcoef(a[idx], b[idx])[0, 1]
             for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_dr_ci(V_a, V_b, real, n_boot=N_BOOT, seed=0):
    """Bootstrap CI of r(V_a,real) − r(V_b,real) (paired resample)."""
    rng = np.random.default_rng(seed)
    n = len(real)
    point = np.corrcoef(V_a, real)[0, 1] - np.corrcoef(V_b, real)[0, 1]
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(np.corrcoef(V_a[idx], real[idx])[0, 1] - np.corrcoef(V_b[idx], real[idx])[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading training states + Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, z_all, kl_all, traj = tr['h'], tr['z'], tr['kl'], tr['traj_id']
    N = len(h_all)
    y = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h_all[tr_idx], y[tr_idx])

    # real K-step returns (same as Task F)
    print("Building real returns...")
    real_returns = {}
    for i in te_idx:
        fut, ok = [], True
        for k in range(1, HORIZON + 1):
            j = i + k
            if j >= N or traj[j] != traj[i]:
                ok = False; break
            fut.append(kl_all[j])
        if ok:
            real_returns[i] = np.array(fut, np.float32)
    valid = np.array(list(real_returns.keys()))
    rng = np.random.default_rng(42)
    start_idx = valid[rng.choice(len(valid), min(N_STATES, len(valid)), replace=False)]
    real_kl = np.stack([real_returns[i] for i in start_idx], 0)
    real_V = lambda_return(real_kl)

    # imagination + online probe scores on imagined h
    print(f"Imagination rollouts from {len(start_idx):,} states...")
    model = load_model(cfg)
    imag_kl, imag_h = imagine(model, cfg, h_all[start_idx], z_all[start_idx], HORIZON)
    probe_seq = np.zeros_like(imag_kl)
    for k in range(HORIZON):
        probe_seq[:, k] = clf.predict_proba(sc.transform(imag_h[:, k, :]))[:, 1]

    # standard baseline (Task F)
    V_std = lambda_return(imag_kl)
    r_std = np.corrcoef(V_std, real_V)[0, 1]

    # ── calibration split: pick threshold on calibration half, evaluate on the other ──
    cal_idx, eval_idx = train_test_split(np.arange(len(start_idx)), test_size=0.50, random_state=0)
    print("\n" + "=" * 74)
    print("TASK P — CONFUSION AS AN IMAGINATION STOPPING RULE")
    print("=" * 74)
    print(f"\n  Horizon={HORIZON}  γ={GAMMA}  N={len(start_idx):,}  (calib={len(cal_idx)}, eval={len(eval_idx)})")

    # candidate thresholds = probe-score percentiles (top-tercile, top-quartile, top-third-ish)
    all_probe = probe_seq.flatten()
    cand_thresholds = {f'p{p}': float(np.percentile(all_probe, p)) for p in [50, 67, 75, 90]}
    print(f"\n  Selecting threshold on CALIBRATION subset by r(V̂_stop, V_real):")
    print(f"    {'threshold':<10}{'value':>8}{'calib r':>10}")
    best_name, best_r = None, -np.inf
    for name, thr in cand_thresholds.items():
        Vc = stopped_return(imag_kl[cal_idx], probe_seq[cal_idx], thr)
        rc = np.corrcoef(Vc, real_V[cal_idx])[0, 1]
        print(f"    {name:<10}{thr:>8.3f}{rc:>10.4f}")
        if rc > best_r:
            best_r, best_name = rc, name
    best_thr = cand_thresholds[best_name]
    print(f"  → selected: {best_name} (threshold {best_thr:.3f}), calib r={best_r:.4f}")

    # ── evaluate the selected threshold on the held-out EVAL subset ──
    V_stop_eval = stopped_return(imag_kl[eval_idx], probe_seq[eval_idx], best_thr)
    V_std_eval = V_std[eval_idx]
    real_eval = real_V[eval_idx]
    r_stop = r_ci(V_stop_eval, real_eval)
    r_std_eval = r_ci(V_std_eval, real_eval)
    dr = paired_dr_ci(V_stop_eval, V_std_eval, real_eval)

    # fraction of rollouts actually truncated (diagnostic)
    frac_trunc = float(np.mean([(probe_seq[eval_idx][i] > best_thr).any() for i in range(len(eval_idx))]))
    mean_stop_step = float(np.mean([
        (np.where(probe_seq[eval_idx][i] > best_thr)[0][0] + 1) if (probe_seq[eval_idx][i] > best_thr).any() else HORIZON
        for i in range(len(eval_idx))]))

    print(f"\n  Held-out EVAL subset (n={len(eval_idx)}), single split:")
    print(f"    {'method':<28}{'r(V̂, V_real)':>16}{'95% CI':>22}")
    print(f"    {'-'*28}{'-'*16}{'-'*22}")
    print(f"    {'standard (Task F, fixed-H)':<28}{r_std_eval[0]:>16.4f}   [{r_std_eval[1]:+.3f}, {r_std_eval[2]:+.3f}]")
    print(f"    {'stopping rule (confusion)':<28}{r_stop[0]:>16.4f}   [{r_stop[1]:+.3f}, {r_stop[2]:+.3f}]")
    print(f"    Δr (stopping − standard) = {dr[0]:+.4f}  95% CI [{dr[1]:+.4f}, {dr[2]:+.4f}]")
    print(f"    ({frac_trunc*100:.0f}% of rollouts truncated; mean stop step {mean_stop_step:.2f}/{HORIZON})")

    # ── cross-sample stability: the single-split Δr is tiny, so check whether its SIGN
    #    is stable across independent start samples (a per-split bootstrap CI can exclude 0
    #    while the effect flips sign across samples — the decisive robustness check). ──
    print(f"\n  Cross-sample stability — Δr across 5 independent start samples (threshold {best_name}):")
    drs = []
    for s in range(5):
        rngs = np.random.default_rng(100 + s)
        si = valid[rngs.choice(len(valid), min(N_STATES, len(valid)), replace=False)]
        rk = np.stack([real_returns[i] for i in si], 0); rv = lambda_return(rk)
        ik, ih = imagine(model, cfg, h_all[si], z_all[si], HORIZON, seed=100 + s)
        ps = np.zeros_like(ik)
        for k in range(HORIZON):
            ps[:, k] = clf.predict_proba(sc.transform(ih[:, k, :]))[:, 1]
        vst = lambda_return(ik); vsp = stopped_return(ik, ps, best_thr)
        drs.append(np.corrcoef(vsp, rv)[0, 1] - np.corrcoef(vst, rv)[0, 1])
    drs = np.array(drs)
    sign_stable = bool(np.all(drs > 0) or np.all(drs < 0))
    print(f"    Δr per sample: {[f'{d:+.4f}' for d in drs]}")
    print(f"    mean {drs.mean():+.4f} ± {drs.std():.4f}  |  sign stable across samples: {sign_stable}")

    # ── verdict (based on cross-sample stability, not a single split) ──
    print("\n" + "-" * 74)
    if sign_stable and drs.mean() > 0 and dr[1] > 0:
        verdict = (f"POSITIVE (small but stable): the confusion stopping rule improves value-estimate "
                   f"correlation with a sign-stable Δr = {drs.mean():+.4f} ± {drs.std():.4f} across 5 "
                   f"samples and a single-split CI excluding 0. Magnitude is small; truncating imagination "
                   f"on high confusion gives modestly better agreement with real returns than fixed-horizon.")
    elif sign_stable and drs.mean() < 0:
        verdict = (f"NEGATIVE: the confusion stopping rule DEGRADES value estimates (sign-stable Δr = "
                   f"{drs.mean():+.4f} ± {drs.std():.4f}, all 5 samples < 0). Fails in a STRUCTURALLY "
                   f"DIFFERENT way than Task F's weighting — confusion, however applied (weight OR stopping "
                   f"rule), does not help imagined rollouts in this setup.")
    elif not sign_stable:
        verdict = (f"NULL / negligible: the single-split Δr ({dr[0]:+.4f}, CI [{dr[1]:+.3f},{dr[2]:+.3f}]) "
                   f"is NOT robust — across 5 independent start samples Δr = {drs.mean():+.4f} ± {drs.std():.4f} "
                   f"and FLIPS SIGN ({(drs>0).sum()}/5 positive), so the per-split CI excluding 0 was "
                   f"sample-specific noise of the same magnitude as the effect. The stopping rule neither "
                   f"reliably helps nor hurts. Combined with Task F's clear weighting negative (Δr=−0.53), "
                   f"this makes the boundary statement general and defensible: confusion — as a uniform "
                   f"weight (Task F, actively harmful) OR as a truncation stopping rule (Task P, negligible) "
                   f"— does not improve value estimation from imagined rollouts in this setup. The one honest "
                   f"distinction: the stopping rule does not HURT the way weighting did.")
    else:
        verdict = (f"NULL: the stopping rule neither helps nor hurts significantly (Δr={dr[0]:+.3f}, "
                   f"95% CI [{dr[1]:+.3f},{dr[2]:+.3f}] straddles 0). A second, structurally different "
                   f"mechanism (truncation) also fails to improve on the fixed-horizon baseline — with "
                   f"Task F's weighting negatives, this makes 'confusion is diagnostic, not value-shaping "
                   f"in this setup' a more defensible boundary statement.")
    print(f"  {verdict}")

    results = dict(n_eval=len(eval_idx), n_calib=len(cal_idx),
                   selected_threshold=best_name, threshold_value=best_thr,
                   r_standard=list(r_std_eval), r_stopping=list(r_stop), delta_r=list(dr),
                   frac_truncated=frac_trunc, mean_stop_step=mean_stop_step,
                   r_standard_full=float(r_std), verdict=verdict,
                   cross_sample_dr=[float(x) for x in drs], cross_sample_dr_mean=float(drs.mean()),
                   cross_sample_dr_std=float(drs.std()), sign_stable=sign_stable,
                   n_positive_of_5=int((drs > 0).sum()),
                   thresholds_calib={k: float(v) for k, v in cand_thresholds.items()})
    with open(os.path.join(OUT_DIR, 'task_p_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_p_results.json')}")


if __name__ == '__main__':
    main()
