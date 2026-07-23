#!/usr/bin/env python3.11
"""
Task R — KL-only routing baseline (Reviewer TAQ8).

Reviewer ask, verbatim:
  "The routing payoff (Table 4) is largest on pendulum (+0.30), the task where Set C fails,
   because routing uses KL events directly and bypasses the probe. A KL-only routing baseline
   would show what the probe adds over thresholding KL."

Task Q compared Probe-A and C_t-direct routers against a recon-error oracle. It never compared
against the simplest possible baseline: threshold raw KL_t directly — nothing learned. This adds
that baseline, on existing logged data (no new training).

────────────────────────────────────────────────────────────────────────────────────────────
A METHODOLOGICAL POINT THAT MUST BE REPORTED, NOT BURIED
────────────────────────────────────────────────────────────────────────────────────────────
Task Q defines routing *events* as the top-25% KL steps (§4.5 / Task C). A router that scores
states by raw KL_t at the SAME timestep is therefore scoring on the exact quantity that defines
the label. It is a perfect oracle by construction: at a 30% budget it captures ~100% of the
top-25%-KL events, not because KL thresholding is a good deployable router but because
score == label-generating variable. Reporting only that number would be meaningless — a
tautology, not a baseline.

So we report BOTH, clearly separated:

  (R1) KL-only, SAME-STEP  — the literal reading of the reviewer's ask. Included for
       completeness and explicitly labelled a tautological upper bound (score = label source).

  (R2) KL-only, CAUSAL/PRIOR-STEP — the honest, deployable version and the one that actually
       answers "what does the probe add over thresholding KL". At decision time t you know
       KL_{t-1} (you have observed it) but NOT KL_t — computing KL_t requires the very
       observation you are deciding whether to query. So the fair KL-only router thresholds
       the most recent AVAILABLE KL. This is exactly the information regime the probe operates
       in: Probe-A reads h_t, which is available before the step-t observation arrives.

R2 is the apples-to-apples comparison. R1 is reported so no one can claim it was omitted.

Everything else is identical to Task Q: same data, same 60/40 stratified split (random_state=0),
same events (top-25% KL, held-out), same budgets, same rank-normalised thresholding, same
bootstrap CI procedure. Runs on saved arrays. XS, CPU.
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

# ── identical constants to Task Q (apples-to-apples) ──
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

ROUTERS = ['probeA', 'ct_direct', 'kl_prior', 'kl_samestep', 'recon_oracle']
LABELS = {'probeA': 'Probe-A', 'ct_direct': 'C_t-direct', 'kl_prior': 'KL-only (prior step)',
          'kl_samestep': 'KL-only (same step)*', 'recon_oracle': 'recon-oracle'}


def recall_curve(scores, high, budgets=BUDGETS):
    """Recall of `high` events at each query budget (query top-b fraction by score).
    Identical to Task Q."""
    n = len(scores)
    norm = rankdata(scores) / n
    out = {}
    for b in budgets:
        thr = np.percentile(norm, 100 * (1 - b))
        out[b] = float(high[norm >= thr].mean())
    return out


def recall_auc(curve, budgets=BUDGETS):
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


def prior_step_kl(kl, traj_id):
    """KL_{t-1}, the most recent KL actually available at decision time t.

    At the start of each trajectory there is no prior step; we use the global KL median as the
    neutral prior (a router with no information ranks these in the middle rather than at an
    extreme, which would bias recall either way). Trajectory boundaries are respected so no KL
    leaks across episodes."""
    prior = np.empty_like(kl, dtype=np.float64)
    neutral = float(np.median(kl))
    starts = np.ones(len(kl), dtype=bool)
    starts[1:] = traj_id[1:] != traj_id[:-1]
    prior[1:] = kl[:-1]
    prior[0] = neutral
    prior[starts] = neutral
    return prior, int(starts.sum())


MULTISEED = {
    'cartpole': [f'outputs/multiseed/seed_{s}/training_states.npz' for s in range(5)],
    'reacher':  [f'outputs/multiseed_env/reacher_seed{s}/states.npz' for s in (1, 2, 3)],
    'pendulum': [f'outputs/multiseed_env/pendulum_seed{s}/states.npz' for s in (1, 2, 3)],
}


def route_one(path):
    """Probe-A and KL-only(prior) recall@30% for one saved collection. Same protocol as main()."""
    tr = dict(np.load(path))
    h, kl, recon, traj = tr['h'], tr['kl'], tr['recon'], tr['traj_id']
    N = len(h)
    y = binarise_by_median(kl)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y, random_state=0)
    kl_te = kl[te_idx]
    high = kl_te >= np.percentile(kl_te, 75)

    clf, sc = train_probe(h[tr_idx], y[tr_idx])
    probe_scores = clf.predict_proba(sc.transform(h[te_idx]))[:, 1]

    kl_prior_all, _ = prior_step_kl(kl, traj)
    kl_prior_scores = kl_prior_all[te_idx]

    pa = recall_curve(probe_scores, high)[FIXED_BUDGET]
    kp = recall_curve(kl_prior_scores, high)[FIXED_BUDGET]
    rc = recall_curve(recon[te_idx], high)[FIXED_BUDGET]
    return dict(probeA=pa, kl_prior=kp, recon_oracle=rc, delta=pa - kp)


def run_multiseed():
    """Replicate the Probe-A vs KL-only(prior) comparison across all existing seed collections."""
    out = {}
    for env, paths in MULTISEED.items():
        rows = []
        for p in paths:
            if not os.path.exists(p):
                continue
            r = route_one(p); r['path'] = p
            rows.append(r)
        if not rows:
            continue
        pa = np.array([r['probeA'] for r in rows])
        kp = np.array([r['kl_prior'] for r in rows])
        dl = np.array([r['delta'] for r in rows])
        out[env] = dict(
            n_seeds=len(rows), per_seed=rows,
            probeA_mean=float(pa.mean()), probeA_std=float(pa.std()),
            klprior_mean=float(kp.mean()), klprior_std=float(kp.std()),
            delta_mean=float(dl.mean()), delta_std=float(dl.std()),
            n_probe_wins=int((dl > 0).sum()),
            sign_stable=bool((dl > 0).all() or (dl <= 0).all()))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    print("=" * 94)
    print("TASK R — KL-ONLY ROUTING BASELINE (Reviewer TAQ8)")
    print("=" * 94)
    print("  * KL-only (same step) scores on the same variable that DEFINES the events")
    print("    (top-25% KL) — a tautological upper bound, not a deployable router.")
    print("    KL-only (prior step) is the fair, causally-available comparison.\n")

    for env, path in ENVS.items():
        tr = dict(np.load(path))
        h, kl, recon, traj = tr['h'], tr['kl'], tr['recon'], tr['traj_id']
        N = len(h)
        y = binarise_by_median(kl); kl_median = float(np.median(kl))
        tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y, random_state=0)

        # events = top-25% KL steps (held-out) — identical definition to Task Q
        kl_te = kl[te_idx]; recon_te = recon[te_idx]
        high = kl_te >= np.percentile(kl_te, 75)

        # (A) Probe-A router — unchanged from Task Q
        clf, sc = train_probe(h[tr_idx], y[tr_idx])
        probe_scores = clf.predict_proba(sc.transform(h[te_idx]))[:, 1]

        # (B) C_t-direct router — unchanged from Task Q
        ct = compute_ct(kl, traj, gamma=GAMMA[env], kl_median=kl_median)
        scaler = StandardScaler().fit(h[tr_idx])
        ridge = Ridge(alpha=1.0).fit(scaler.transform(h[tr_idx]), ct[tr_idx])
        ct_scores = ridge.predict(scaler.transform(h[te_idx]))

        # (R1) KL-only, same-step — tautological (score == label source)
        kl_same_scores = kl_te

        # (R2) KL-only, prior-step — the fair, deployable KL threshold
        kl_prior_all, n_starts = prior_step_kl(kl, traj)
        kl_prior_scores = kl_prior_all[te_idx]

        # baseline: recon-error oracle — unchanged
        recon_scores = recon_te

        score_map = {'probeA': probe_scores, 'ct_direct': ct_scores,
                     'kl_prior': kl_prior_scores, 'kl_samestep': kl_same_scores,
                     'recon_oracle': recon_scores}

        curves = {k: recall_curve(v, high) for k, v in score_map.items()}
        aucs = {k: recall_auc(v) for k, v in curves.items()}
        ci = {k: bootstrap_recall_ci(v, high) for k, v in score_map.items()}

        base = curves['recon_oracle'][FIXED_BUDGET]
        results[env] = dict(
            recall30={k: v[FIXED_BUDGET] for k, v in curves.items()},
            recall30_ci={k: list(ci[k]) for k in ci},
            auc=aucs,
            curves={k: {str(b): v[b] for b in BUDGETS} for k, v in curves.items()},
            gap_vs_recon={k: float(curves[k][FIXED_BUDGET] - base) for k in curves},
            probeA_minus_klprior=float(curves['probeA'][FIXED_BUDGET] - curves['kl_prior'][FIXED_BUDGET]),
            probeA_minus_klprior_auc=float(aucs['probeA'] - aucs['kl_prior']),
            n_te=len(te_idx), n_traj_starts=n_starts)

        print(f"\n  {env.upper()}  (held-out N={len(te_idx):,}, events = top-25% KL)")
        print(f"    {'router':<24}{'recall@30%':>12}{'  95% CI':>20}{'recall-AUC':>12}{'gap vs recon':>14}")
        for k in ROUTERS:
            g = curves[k][FIXED_BUDGET] - base
            gs = f"{g:+.3f}" if k != 'recon_oracle' else "   —"
            print(f"    {LABELS[k]:<24}{curves[k][FIXED_BUDGET]:>12.3f}"
                  f"   [{ci[k][1]:.3f}, {ci[k][2]:.3f}]{aucs[k]:>12.3f}{gs:>14}")
        print(f"    → Probe-A − KL-only(prior): {results[env]['probeA_minus_klprior']:+.3f} recall@30%, "
              f"{results[env]['probeA_minus_klprior_auc']:+.3f} AUC")

    # ── the reviewer's question, answered directly ──
    print("\n" + "=" * 94)
    print("WHAT DOES THE PROBE ADD OVER THRESHOLDING KL?  (Probe-A vs KL-only prior-step)")
    print("=" * 94)
    print(f"\n  {'env':<12}{'Probe-A':>10}{'KL-only(prior)':>16}{'Δ recall':>11}"
          f"{'Δ AUC':>9}{'probe wins?':>13}")
    deltas, auc_deltas = [], []
    for env in ENVS:
        r = results[env]
        d = r['probeA_minus_klprior']; da = r['probeA_minus_klprior_auc']
        deltas.append(d); auc_deltas.append(da)
        print(f"  {env:<12}{r['recall30']['probeA']:>10.3f}{r['recall30']['kl_prior']:>16.3f}"
              f"{d:>+11.3f}{da:>+9.3f}{('YES' if d > 0 else 'no'):>13}")

    n_win = sum(1 for d in deltas if d > 0)
    mean_d = float(np.mean(deltas))
    if n_win == 3:
        verdict = (f"PROBE ADDS VALUE ON ALL THREE TASKS: Probe-A beats the causally-fair KL-only "
                   f"router on cartpole, reacher and pendulum (mean Δ recall@30% {mean_d:+.3f}). "
                   f"The probe is reading more than a raw KL threshold — it is not simply "
                   f"re-expressing the world model's own training signal.")
    elif n_win == 0:
        verdict = (f"PROBE ADDS NOTHING OVER A KL THRESHOLD: the KL-only (prior-step) router matches "
                   f"or beats Probe-A on all three tasks (mean Δ {mean_d:+.3f}). This is a genuine "
                   f"threat to the operational-use contribution and must be reported plainly: the "
                   f"routing payoff does not require the probe.")
    else:
        won = [e for e, d in zip(ENVS, deltas) if d > 0]
        lost = [e for e, d in zip(ENVS, deltas) if d <= 0]
        verdict = (f"MIXED ({n_win}/3): Probe-A beats the fair KL-only router on {', '.join(won)} "
                   f"but not on {', '.join(lost)} (mean Δ {mean_d:+.3f}). Reported as-is — the probe's "
                   f"advantage over raw KL thresholding is task-dependent, not universal.")
    print(f"\n  {verdict}")

    # tautology note, recorded in the artifact so it travels with the numbers
    taut = {env: results[env]['recall30']['kl_samestep'] for env in ENVS}
    print(f"\n  Same-step KL-only recall@30%: " +
          ", ".join(f"{e} {v:.3f}" for e, v in taut.items()))
    print("  These are near-ceiling BY CONSTRUCTION (score = the variable defining the events),")
    print("  not evidence that KL thresholding is a good router. Do not quote them as a baseline.")

    results['_verdict'] = verdict
    results['_n_probe_wins_of_3'] = n_win
    results['_mean_delta_recall'] = mean_d
    results['_samestep_is_tautological'] = (
        "KL-only (same step) scores on KL_t, the same variable that defines the routing events "
        "(top-25% KL). Its recall is a construction artefact and an upper bound, not a baseline. "
        "The deployable comparison is KL-only (prior step), which uses KL_{t-1} — the most recent "
        "KL actually available at decision time, matching the information regime Probe-A operates in "
        "(h_t is available before the step-t observation).")

    # ── multi-seed replication: is the Probe-A vs KL-only(prior) gap stable? ──
    print("\n" + "=" * 94)
    print("MULTI-SEED REPLICATION (existing multi-seed collections, no new training)")
    print("=" * 94)
    ms = run_multiseed()
    results['_multiseed'] = ms
    print(f"\n  {'env':<12}{'seeds':>7}{'Probe-A':>18}{'KL-only(prior)':>20}{'Δ recall (mean±sd)':>22}{'probe wins':>12}")
    for env, m in ms.items():
        print(f"  {env:<12}{m['n_seeds']:>7}"
              f"{m['probeA_mean']:>11.3f}±{m['probeA_std']:.3f}"
              f"{m['klprior_mean']:>13.3f}±{m['klprior_std']:.3f}"
              f"{m['delta_mean']:>+15.3f}±{m['delta_std']:.3f}"
              f"{m['n_probe_wins']:>8}/{m['n_seeds']}")

    out = os.path.join(OUT_DIR, 'task_r_kl_routing.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {out}")

    # CSV of the headline table
    csv = os.path.join(OUT_DIR, 'task_r_kl_routing.csv')
    with open(csv, 'w') as f:
        f.write("env,router,recall30,ci_lo,ci_hi,recall_auc,gap_vs_recon\n")
        for env in ENVS:
            r = results[env]
            for k in ROUTERS:
                f.write(f"{env},{k},{r['recall30'][k]:.6f},{r['recall30_ci'][k][1]:.6f},"
                        f"{r['recall30_ci'][k][2]:.6f},{r['auc'][k]:.6f},{r['gap_vs_recon'][k]:+.6f}\n")
    print(f"  CSV saved:     {csv}")


if __name__ == '__main__':
    main()
