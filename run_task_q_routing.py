#!/usr/bin/env python3.11
"""
Task Q — Cross-environment validation of the observation-routing application.

The routing oracle (§4.5) is the paper's one demonstrated practical payoff, but it has
only been shown on cartpole. This runs it on all three environments with TWO router
variants and compares against the recon-error-oracle baseline:

  (A) Probe-A router — logistic probe trained on binary KL labels (same as cartpole §4.5).
  (B) C_t-direct router — Ridge probe trained directly on the continuous confusion integral
      C_t, bypassing the recon-based Set C labelling entirely.

KEY PREDICTION from Task N: on pendulum, Set C AUROC inverts (0.32) because the recon-based
labelling anti-aligns with confusion — but the underlying C_t signal is the STRONGEST of the
three (R²=0.89). This predicts the Probe-A router (trained against KL, correlated with the
same recon structure) may degrade on pendulum while the C_t-direct router stays strong there.

Routing metric (identical to §4.5 / Task C):
  events = top-25% KL steps (the states worth querying a real observation for).
  At query budget b, the router queries its top-b fraction by score; recall = fraction of
  events captured. Baseline = recon-error oracle (recon is a strong scalar, correlated w/ KL).

Reports recall @ 30%, the recall-vs-budget AUC (0–70%), and the gap vs recon-oracle, for
each env × router, with bootstrap CIs on recall@30%. Runs on the frozen models. XS, CPU.
"""

import os
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct

BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
FIXED_BUDGET = 0.30
GAMMA = {'cartpole': 0.95, 'reacher': 0.70, 'pendulum': 0.90}
N_BOOT = 1000
OUT_DIR = 'outputs/causal'

ENVS = {
    'cartpole': 'outputs/data/training_states.npz',
    'reacher':  'outputs/second_env/reacher_easy_training_states.npz',
    'pendulum': 'outputs/third_env/pendulum_swingup_training_states.npz',
}


def recall_curve(scores, high, budgets=BUDGETS):
    """Recall of `high` events at each query budget (query top-b fraction by score)."""
    n = len(scores)
    norm = rankdata(scores) / n
    out = {}
    for b in budgets:
        thr = np.percentile(norm, 100 * (1 - b))
        out[b] = float(high[norm >= thr].mean())
    return out


def recall_auc(curve, budgets=BUDGETS):
    """Trapezoidal AUC of recall-vs-budget over the budget range."""
    xs = np.array(budgets); ys = np.array([curve[b] for b in budgets])
    trap = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(trap(ys, xs) / (xs.max() - xs.min()))


def bootstrap_recall_ci(scores, high, budget=FIXED_BUDGET, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    def rec(idx):
        s, h = scores[idx], high[idx]
        norm = rankdata(s) / len(s)
        thr = np.percentile(norm, 100 * (1 - budget))
        return h[norm >= thr].mean()
    point = rec(np.arange(n))
    boots = [rec(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    print("=" * 84)
    print("TASK Q — CROSS-ENVIRONMENT ROUTING (Probe-A vs C_t-direct vs recon-oracle)")
    print("=" * 84)

    for env, path in ENVS.items():
        tr = dict(np.load(path))
        h, kl, recon, traj = tr['h'], tr['kl'], tr['recon'], tr['traj_id']
        N = len(h)
        y = binarise_by_median(kl); kl_median = float(np.median(kl))
        tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y, random_state=0)

        # events = top-25% KL steps (held-out)
        kl_te = kl[te_idx]; recon_te = recon[te_idx]
        high = kl_te >= np.percentile(kl_te, 75)

        # (A) Probe-A router (binary KL labels)
        clf, sc = train_probe(h[tr_idx], y[tr_idx])
        probe_scores = clf.predict_proba(sc.transform(h[te_idx]))[:, 1]

        # (B) C_t-direct router (Ridge on continuous C_t)
        ct = compute_ct(kl, traj, gamma=GAMMA[env], kl_median=kl_median)
        scaler = StandardScaler().fit(h[tr_idx])
        ridge = Ridge(alpha=1.0).fit(scaler.transform(h[tr_idx]), ct[tr_idx])
        ct_scores = ridge.predict(scaler.transform(h[te_idx]))

        # baseline: recon-error oracle
        recon_scores = recon_te

        curves = {'probeA': recall_curve(probe_scores, high),
                  'ct_direct': recall_curve(ct_scores, high),
                  'recon_oracle': recall_curve(recon_scores, high)}
        aucs = {k: recall_auc(v) for k, v in curves.items()}
        ci = {'probeA': bootstrap_recall_ci(probe_scores, high),
              'ct_direct': bootstrap_recall_ci(ct_scores, high),
              'recon_oracle': bootstrap_recall_ci(recon_scores, high)}

        results[env] = dict(
            recall30={k: v[FIXED_BUDGET] for k, v in curves.items()},
            recall30_ci={k: list(ci[k]) for k in ci},
            auc=aucs, curves={k: {str(b): v[b] for b in BUDGETS} for k, v in curves.items()},
            gap_probeA_vs_recon=float(curves['probeA'][FIXED_BUDGET] - curves['recon_oracle'][FIXED_BUDGET]),
            gap_ctdirect_vs_recon=float(curves['ct_direct'][FIXED_BUDGET] - curves['recon_oracle'][FIXED_BUDGET]),
            n_te=len(te_idx))

        print(f"\n  {env.upper()}  (held-out N={len(te_idx):,}, events=top-25% KL)")
        print(f"    {'router':<16}{'recall@30%':>12}{'  95% CI':>18}{'recall-AUC':>12}{'gap vs recon':>14}")
        for k, label in [('probeA', 'Probe-A'), ('ct_direct', 'C_t-direct'), ('recon_oracle', 'recon-oracle')]:
            g = curves[k][FIXED_BUDGET] - curves['recon_oracle'][FIXED_BUDGET]
            print(f"    {label:<16}{curves[k][FIXED_BUDGET]:>12.3f}"
                  f"   [{ci[k][1]:.3f}, {ci[k][2]:.3f}]{aucs[k]:>12.3f}"
                  f"{(g if k != 'recon_oracle' else 0):>+14.3f}")

    # ── side-by-side summary + dissociation test ──
    print("\n" + "=" * 84)
    print("SIDE-BY-SIDE  (recall@30%, and whether each router beats the recon oracle)")
    print("=" * 84)
    print(f"\n  {'env':<12}{'Probe-A':>10}{'C_t-direct':>12}{'recon':>9}"
          f"{'ProbeA>recon?':>15}{'C_t>recon?':>12}")
    for env in ENVS:
        r = results[env]['recall30']
        pa_win = r['probeA'] > r['recon_oracle']
        ct_win = r['ct_direct'] > r['recon_oracle']
        print(f"  {env:<12}{r['probeA']:>10.3f}{r['ct_direct']:>12.3f}{r['recon_oracle']:>9.3f}"
              f"{('YES' if pa_win else 'no'):>15}{('YES' if ct_win else 'no'):>12}")

    # dissociation verdict (the Task N prediction)
    print("\n" + "-" * 84)
    pend = results['pendulum']['recall30']
    pa_pend_gap = pend['probeA'] - pend['recon_oracle']
    ct_pend_gap = pend['ct_direct'] - pend['recon_oracle']
    cart = results['cartpole']['recall30']
    print(f"  TASK-N DISSOCIATION PREDICTION (pendulum): Probe-A routing may degrade while C_t-direct stays strong.")
    print(f"    pendulum Probe-A recall {pend['probeA']:.3f} (gap vs recon {pa_pend_gap:+.3f})")
    print(f"    pendulum C_t-direct recall {pend['ct_direct']:.3f} (gap vs recon {ct_pend_gap:+.3f})")
    if pa_pend_gap <= 0 < ct_pend_gap:
        verdict = ("DISSOCIATION HOLDS: on pendulum Probe-A routing fails to beat the recon oracle "
                   f"(gap {pa_pend_gap:+.3f}) while the C_t-direct router does (gap {ct_pend_gap:+.3f}). "
                   "The practical application survives exactly where the Set C diagnostic inverts — "
                   "bypassing the recon-based labelling (C_t-direct) recovers it. Elegant confirmation "
                   "that Set C's inversion is a property of that evaluation construction, not the signal.")
    elif pa_pend_gap > 0 and ct_pend_gap > 0:
        verdict = (f"PROBE-A ROBUST: both routers beat recon on pendulum (Probe-A {pa_pend_gap:+.3f}, "
                   f"C_t-direct {ct_pend_gap:+.3f}) despite Set C inverting. The routing application is "
                   "more robust to the labelling artefact than the Set C diagnostic — also a valuable, "
                   "reportable finding (routing survives the inversion).")
    elif pa_pend_gap <= 0 and ct_pend_gap <= 0:
        verdict = (f"BOTH UNDERPERFORM on pendulum (Probe-A {pa_pend_gap:+.3f}, C_t-direct {ct_pend_gap:+.3f}) "
                   "— a real limitation: the routing win is more cartpole-specific than currently claimed. "
                   "Reported plainly, not minimized.")
    else:
        verdict = (f"MIXED (Probe-A {pa_pend_gap:+.3f}, C_t-direct {ct_pend_gap:+.3f}) — reported as-is.")
    print(f"\n  {verdict}")
    results['_verdict'] = verdict
    results['_cartpole_probeA_gap'] = float(cart['probeA'] - cart['recon_oracle'])

    with open(os.path.join(OUT_DIR, 'task_q_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_q_results.json')}")


if __name__ == '__main__':
    main()
