#!/usr/bin/env python3.11
"""
Task C — Multi-seed replication with proper uncertainty quantification.

For each of N_SEEDS independent seeds, from scratch:
  1. Train an XS world model (seed s) + a 3-member ensemble (seeds derived from s).
  2. Collect training_states, Set A/B/C (KL-matched), within-balance confound set.
  3. Fit Probe A / Probe C / z_t probe; block/quarter analysis.
  4. C_t characterisation (best-γ, R²).
  5. obs/imagination boundary probe AUROC.
  6. orthogonality (probe-direction · h_t-mean) sanity.
  7. Routing-oracle recall (Probe A vs recon-error oracle at 30% budget).
  8. Bootstrap CIs (1000×) for the three argument-carrying numbers.

Raw per-seed outputs are saved to outputs/multiseed/seed_<s>/ so aggregation
and paired tests can re-run without retraining. Resumable: seeds already having
metrics.json are skipped.

Aggregation (mean±std across seeds, paired bootstrap tests) is done by
aggregate_multiseed() at the end / on demand.

CPU-only, XS scale. Expect ~60-80 min per seed (train main + 3 ensemble members).
"""

import os
import gc
import json
import time
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.config import XS_CONFIG
from src.training.trainer import train_world_model
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.data.collect import collect_states, build_set_c
from src.probe.linear_probe import (
    binarise_by_median, train_probe, auroc,
)
from src.probe.intervention import (
    compute_ct, probe_direction, bootstrap_auroc_ci,
)

N_SEEDS   = 5
ROOT      = 'outputs/multiseed'
GAMMAS    = [0.70, 0.80, 0.90, 0.95, 0.99]
MAX_LAG   = 50
N_EP      = 20            # eval episodes per set (matches original n_eval_episodes)
N_BOOT    = 1000


# ─── model IO ────────────────────────────────────────────────────────────────

def load_wm(ck_path, cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def train_or_load(cfg, seed, ck_path, states_path=None):
    if os.path.exists(ck_path):
        m = load_wm(ck_path, cfg)
        states = dict(np.load(states_path)) if states_path and os.path.exists(states_path) else None
        return m, states
    cfg_s = {**cfg, 'checkpoint_path': ck_path}
    m, states = train_world_model(cfg_s, seed=seed)
    if states_path is not None:
        np.savez(states_path, **states)
    return m, states


# ─── per-seed pipeline ───────────────────────────────────────────────────────

def boundary_probe_auroc(model, h_all, z_all, kl_all, cfg, n_start=3000, horizon=15, seed=0):
    """obs/imagination boundary probe AUROC (Task E infra reused for replication)."""
    device = next(model.parameters()).device
    y_kl = binarise_by_median(kl_all)
    tr_idx, te_idx = train_test_split(np.arange(len(h_all)), test_size=0.40,
                                      stratify=y_kl, random_state=0)
    rng = np.random.default_rng(seed)
    start_idx = rng.choice(te_idx, min(n_start, len(te_idx)), replace=False)
    h = torch.tensor(h_all[start_idx], dtype=torch.float32, device=device)
    z = torch.tensor(z_all[start_idx], dtype=torch.float32, device=device)
    imagined = []
    with torch.no_grad():
        for _ in range(horizon):
            a = torch.tensor(rng.uniform(-1, 1, (h.shape[0], cfg['act_dim'])).astype(np.float32),
                             device=device)
            h, z, _ = model.rssm.imagine_step(h, z, a)
            imagined.append(h.cpu().numpy().copy())
    h_real = h_all[te_idx]
    h_imag = np.concatenate(imagined, axis=0)
    Xb = np.concatenate([h_real, h_imag], axis=0)
    yb = np.array([0] * len(h_real) + [1] * len(h_imag), dtype=np.int32)
    b_tr, b_te = train_test_split(np.arange(len(Xb)), test_size=0.30,
                                  stratify=yb, random_state=0)
    clf_b, sc_b = train_probe(Xb[b_tr], yb[b_tr])
    return auroc(clf_b, sc_b, Xb[b_te], yb[b_te])


def routing_recall(clf, sc, h_te, kl_te, recon_te, budget=0.30):
    """Recall of top-25% KL events at a given query budget, Probe A vs recon oracle."""
    from scipy.stats import rankdata
    n = len(h_te)
    probe_raw  = clf.predict_proba(sc.transform(h_te))[:, 1]
    probe_norm = rankdata(probe_raw) / n
    recon_norm = rankdata(recon_te) / n
    kl_75 = np.percentile(kl_te, 75)
    high  = kl_te >= kl_75
    p_thr = np.percentile(probe_norm, 100 * (1 - budget))
    r_thr = np.percentile(recon_norm, 100 * (1 - budget))
    probe_q = probe_norm >= p_thr
    recon_q = recon_norm >= r_thr
    return {
        'probe_recall': float(high[probe_q].mean()),
        'recon_recall': float(high[recon_q].mean()),
        'probe_scores': probe_norm, 'recon_scores': recon_norm, 'high': high,
    }


def run_seed(seed, cfg, force=False):
    sd_dir = os.path.join(ROOT, f'seed_{seed}')
    os.makedirs(sd_dir, exist_ok=True)
    metrics_path = os.path.join(sd_dir, 'metrics.json')
    if os.path.exists(metrics_path) and not force:
        print(f"[seed {seed}] metrics.json exists — skipping")
        with open(metrics_path) as f:
            return json.load(f)

    t0 = time.time()
    print(f"\n{'='*70}\n[seed {seed}] START\n{'='*70}", flush=True)

    # 1. Train main model + 3-member ensemble
    ck_main = os.path.join(sd_dir, 'world_model.pt')
    st_main = os.path.join(sd_dir, 'training_states.npz')
    print(f"[seed {seed}] training main model...", flush=True)
    model, states = train_or_load(cfg, seed, ck_main, st_main)

    ens_models = []
    for j in range(3):
        es = 1000 + seed * 10 + j          # distinct ensemble seeds per main seed
        ck_e = os.path.join(sd_dir, f'ensemble_{j}.pt')
        print(f"[seed {seed}] training ensemble member {j} (seed {es})...", flush=True)
        m_e, _ = train_or_load(cfg, es, ck_e, None)
        ens_models.append(m_e)

    h_all, z_all = states['h'], states['z']
    kl_all, recon_all, traj_id = states['kl'], states['recon'], states['traj_id']
    N = len(h_all)

    # 2. Collect Set A/B/C + within-balance confound set
    print(f"[seed {seed}] collecting eval sets...", flush=True)
    np.random.seed(42 + seed)
    env_a = CartpoleEnv(task='swingup', noisy=False, seed=100 + seed)
    set_a = collect_states(model, env_a, N_EP, cfg)
    env_b = CartpoleEnv(task='swingup', noisy=True, noise_std=cfg['noise_std'], seed=200 + seed)
    set_b = collect_states(model, env_b, N_EP, cfg)
    set_c = build_set_c(set_a, set_b)

    # within-balance confound: both C1/C2 from balance (different recon), same task id
    env_bal = CartpoleEnv(task='balance', noisy=False, seed=300 + seed)
    bal = collect_states(model, env_bal, N_EP, cfg)
    set_c_wb = _within_balance(bal)

    # 3. Probe A / Probe C / z_t
    y_kl = binarise_by_median(kl_all)
    kl_median = np.median(kl_all)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf_a, sc_a = train_probe(h_all[tr_idx], y_kl[tr_idx])

    def elabels(kl_arr):
        y = (kl_arr > kl_median).astype(np.int32)
        return y if len(np.unique(y)) == 2 else binarise_by_median(kl_arr)

    auroc_id = auroc(clf_a, sc_a, h_all[te_idx], y_kl[te_idx])
    auroc_a  = auroc(clf_a, sc_a, set_a['h'], elabels(set_a['kl']))
    auroc_b  = auroc(clf_a, sc_a, set_b['h'], elabels(set_b['kl']))
    auroc_c  = auroc(clf_a, sc_a, set_c['h'], set_c['labels'])
    auroc_wb = auroc(clf_a, sc_a, set_c_wb['h'], set_c_wb['labels'])

    # Probe C (recon)
    yc = binarise_by_median(recon_all)
    clf_c, sc_c = train_probe(h_all[tr_idx], yc[tr_idx])
    auroc_probec_c = auroc(clf_c, sc_c, set_c['h'], set_c['labels'])

    # z_t probe
    clf_z, sc_z = train_probe(z_all[tr_idx], y_kl[tr_idx])
    auroc_zt_c = auroc(clf_z, sc_z, set_c['z'], set_c['labels'])

    # block/quarter analysis (Set C AUROC per quarter of h_t)
    q = h_all.shape[1] // 4
    block_c = []
    for i in range(4):
        sl = slice(i * q, (i + 1) * q)
        cl_q, sc_q = train_probe(h_all[tr_idx][:, sl], y_kl[tr_idx])
        block_c.append(float(auroc(cl_q, sc_q, set_c['h'][:, sl], set_c['labels'])))

    # 4. C_t characterisation
    probe_scores_te = clf_a.predict_proba(sc_a.transform(h_all[te_idx]))[:, 1]
    kl_te = kl_all[te_idx]
    r2_kl = r2_score(probe_scores_te,
                     LinearRegression().fit(kl_te.reshape(-1, 1), probe_scores_te)
                     .predict(kl_te.reshape(-1, 1)))
    best_r2, best_g = -1, None
    ct_r2 = {}
    for g in GAMMAS:
        ct = compute_ct(kl_all, traj_id, gamma=g, max_lag=MAX_LAG, kl_median=kl_median)
        ct_te = ct[te_idx]
        r2 = r2_score(probe_scores_te,
                      LinearRegression().fit(ct_te.reshape(-1, 1), probe_scores_te)
                      .predict(ct_te.reshape(-1, 1)))
        ct_r2[g] = float(r2)
        if r2 > best_r2:
            best_r2, best_g = r2, g

    # 5. boundary probe AUROC
    auroc_boundary = boundary_probe_auroc(model, h_all, z_all, kl_all, cfg, seed=seed)

    # 6. orthogonality: correlation of probe direction with h_t mean direction
    v = probe_direction(clf_a, sc_a)
    h_mean_dir = h_all.mean(axis=0)
    h_mean_dir = h_mean_dir / (np.linalg.norm(h_mean_dir) + 1e-12)
    orthog = float(np.dot(v, h_mean_dir))

    # 7. routing recall
    route = routing_recall(clf_a, sc_a, h_all[te_idx], kl_te, recon_all[te_idx])

    # 8. ensemble disagreement on Set C
    from src.probe.linear_probe import ensemble_disagreement
    ens_dis, ens_auroc_c = ensemble_disagreement(ens_models, set_c, cfg)

    # ── bootstrap CIs for the three argument-carrying numbers ──
    scores_c = clf_a.predict_proba(sc_a.transform(set_c['h']))[:, 1]
    ci_setc = bootstrap_auroc_ci(set_c['labels'], scores_c, n_boot=N_BOOT, seed=seed)
    scores_wb = clf_a.predict_proba(sc_a.transform(set_c_wb['h']))[:, 1]
    ci_wb = bootstrap_auroc_ci(set_c_wb['labels'], scores_wb, n_boot=N_BOOT, seed=seed)
    # routing gap CI: bootstrap the per-event indicator difference
    ci_route_gap = _bootstrap_routing_gap(route, n_boot=N_BOOT, seed=seed)

    metrics = {
        'seed': seed,
        'n_states': int(N),
        'auroc_id': float(auroc_id),
        'auroc_a': float(auroc_a),
        'auroc_b': float(auroc_b),
        'auroc_c': float(auroc_c),
        'auroc_within_balance': float(auroc_wb),
        'auroc_probeC_setc': float(auroc_probec_c),
        'auroc_zt_setc': float(auroc_zt_c),
        'block_c_auroc': block_c,
        'r2_kl_baseline': float(r2_kl),
        'ct_r2_by_gamma': ct_r2,
        'best_gamma': float(best_g),
        'best_ct_r2': float(best_r2),
        'auroc_boundary': float(auroc_boundary),
        'orthogonality': orthog,
        'routing_probe_recall': route['probe_recall'],
        'routing_recon_recall': route['recon_recall'],
        'ens_auroc_setc': float(ens_auroc_c),
        # CIs (point, lo, hi)
        'ci_setc_auroc': list(ci_setc),
        'ci_within_balance_auroc': list(ci_wb),
        'ci_routing_gap': list(ci_route_gap),
        'elapsed_min': (time.time() - t0) / 60,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # save the argument-carrying raw scores so paired tests can re-run
    np.savez(os.path.join(sd_dir, 'setc_scores.npz'),
             labels=set_c['labels'], probe=scores_c, ens=ens_dis)
    np.savez(os.path.join(sd_dir, 'routing_scores.npz'),
             high=route['high'], probe=route['probe_scores'], recon=route['recon_scores'])

    print(f"[seed {seed}] DONE in {metrics['elapsed_min']:.1f} min | "
          f"SetC={auroc_c:.4f} WB={auroc_wb:.4f} route(p/r)="
          f"{route['probe_recall']:.3f}/{route['recon_recall']:.3f} "
          f"best_γ={best_g} R²={best_r2:.3f}", flush=True)

    # free memory before next seed
    del model, ens_models, states, h_all, z_all
    gc.collect()
    return metrics


def _within_balance(bal, n_bins=10, per_bin=20, max_total=200, seed=42):
    """C1/C2 both from balance task: low vs high recon within KL bins.
    Same task identity → tests the confound (should be ~chance)."""
    all_h, all_z = bal['h'], bal['z']
    all_kl, all_recon = bal['kl'], bal['recon']
    bin_edges = np.percentile(all_kl, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(all_kl, bin_edges[1:-1])
    rng = np.random.default_rng(seed)
    c1, c2 = [], []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) < 4:
            continue
        rb = all_recon[idx]
        c1c = idx[rb <= np.percentile(rb, 25)]
        c2c = idx[rb >= np.percentile(rb, 75)]
        n_pick = min(per_bin, len(c1c), len(c2c))
        if n_pick == 0:
            continue
        c1.extend(rng.choice(c1c, n_pick, replace=False).tolist())
        c2.extend(rng.choice(c2c, n_pick, replace=False).tolist())
    if len(c1) > max_total:
        c1 = rng.choice(c1, max_total, replace=False).tolist()
    if len(c2) > max_total:
        c2 = rng.choice(c2, max_total, replace=False).tolist()
    return {
        'h': np.concatenate([all_h[c1], all_h[c2]]),
        'z': np.concatenate([all_z[c1], all_z[c2]]),
        'labels': np.array([0] * len(c1) + [1] * len(c2), dtype=np.int32),
    }


def _bootstrap_routing_gap(route, budget=0.30, n_boot=1000, seed=0):
    """Bootstrap CI of (probe_recall - recon_recall) at fixed budget by
    resampling held-out events, recomputing thresholds each resample."""
    rng = np.random.default_rng(seed)
    high = route['high']
    p = route['probe_scores']
    r = route['recon_scores']
    n = len(high)
    point = high[p >= np.percentile(p, 100 * (1 - budget))].mean() - \
            high[r >= np.percentile(r, 100 * (1 - budget))].mean()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        hi = high[idx]; pi = p[idx]; ri = r[idx]
        if hi.sum() == 0:
            continue
        pr = hi[pi >= np.percentile(pi, 100 * (1 - budget))].mean()
        rr = hi[ri >= np.percentile(ri, 100 * (1 - budget))].mean()
        boots.append(pr - rr)
    boots = np.array(boots)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ─── aggregation across seeds ────────────────────────────────────────────────

def aggregate():
    from src.probe.intervention import paired_bootstrap_diff
    from scipy.stats import wilcoxon
    seeds = sorted(int(d.split('_')[1]) for d in os.listdir(ROOT)
                   if d.startswith('seed_') and
                   os.path.exists(os.path.join(ROOT, d, 'metrics.json')))
    allm = []
    for s in seeds:
        with open(os.path.join(ROOT, f'seed_{s}', 'metrics.json')) as f:
            allm.append(json.load(f))
    if not allm:
        print("No completed seeds yet.")
        return

    def col(key):
        return np.array([m[key] for m in allm], dtype=float)

    print(f"\n{'='*74}\nTASK C — MULTI-SEED SUMMARY  (n_seeds={len(allm)}: {seeds})\n{'='*74}")
    print(f"\n  {'Metric':<34}{'mean ± std':>22}{'median [min,max]':>18}")
    print(f"  {'-'*34}{'-'*22}{'-'*18}")
    report_keys = [
        ('auroc_a', 'Probe A Set A AUROC'),
        ('auroc_c', 'Probe A Set C AUROC (headline)'),
        ('auroc_within_balance', 'Within-balance AUROC (≈0.5)'),
        ('auroc_zt_setc', 'z_t probe Set C AUROC'),
        ('auroc_boundary', 'Boundary probe AUROC'),
        ('best_ct_r2', 'C_t R² (best γ)'),
        ('routing_probe_recall', 'Routing recall — Probe A'),
        ('routing_recon_recall', 'Routing recall — recon oracle'),
        ('ens_auroc_setc', 'Ensemble Set C AUROC'),
        ('orthogonality', 'Probe⊥h_mean (cos)'),
    ]
    for k, label in report_keys:
        v = col(k)
        print(f"  {label:<34}{v.mean():>10.4f} ± {v.std():<8.4f}"
              f"{np.median(v):>8.4f} [{v.min():.3f},{v.max():.3f}]")

    # best gamma distribution
    bg = col('best_gamma')
    print(f"\n  best γ across seeds: {sorted(bg.tolist())}")

    # paired tests: (a) Probe A vs ensemble on Set C  (b) Probe A vs recon in routing
    pa_c = col('auroc_c'); ens_c = col('ens_auroc_setc')
    pr = col('routing_probe_recall'); rr = col('routing_recon_recall')
    print(f"\n  Paired comparisons across seeds (Wilcoxon signed-rank):")
    for name, a, b in [('Probe A vs Ensemble (Set C AUROC)', pa_c, ens_c),
                       ('Probe A vs Recon oracle (routing recall)', pr, rr)]:
        diff = a - b
        try:
            stat, p = wilcoxon(a, b)
            pstr = f"p={p:.4f}"
        except Exception as e:
            pstr = f"(wilcoxon n/a: {e})"
        print(f"    {name}: Δ mean={diff.mean():+.4f} ± {diff.std():.4f}  {pstr}")

    # pooled paired bootstrap on concatenated held-out scores (routing + setc)
    print(f"\n  Pooled paired-bootstrap (all seeds' held-out events concatenated):")
    for tag, fname, akey, bkey in [
        ('Probe A vs Ensemble, Set C AUROC', 'setc_scores.npz', 'probe', 'ens'),
    ]:
        Y, A, B = [], [], []
        for s in seeds:
            d = np.load(os.path.join(ROOT, f'seed_{s}', fname))
            Y.append(d['labels']); A.append(d['probe']); B.append(d['ens'])
        Y = np.concatenate(Y); A = np.concatenate(A); B = np.concatenate(B)
        pt, lo, hi, p = paired_bootstrap_diff(Y, A, B, n_boot=N_BOOT, seed=0)
        print(f"    {tag}: Δ={pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p≈{p:.4f}")

    # mean CI widths for the three argument-carrying numbers
    print(f"\n  Bootstrap 95% CIs (mean across seeds of per-seed CI):")
    for k, label in [('ci_setc_auroc', 'Set C AUROC'),
                     ('ci_within_balance_auroc', 'Within-balance AUROC'),
                     ('ci_routing_gap', 'Routing gap (probe-recon recall)')]:
        arr = np.array([m[k] for m in allm])  # (n_seeds, 3) point,lo,hi
        print(f"    {label:<34} point={arr[:,0].mean():+.4f}  "
              f"CI≈[{arr[:,1].mean():+.4f}, {arr[:,2].mean():+.4f}]")

    out = os.path.join(ROOT, 'aggregate.json')
    with open(out, 'w') as f:
        json.dump({'seeds': seeds, 'per_seed': allm}, f, indent=2)
    print(f"\n  Saved aggregate → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(N_SEEDS)))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    os.makedirs(ROOT, exist_ok=True)
    cfg = XS_CONFIG.copy()

    if not args.aggregate_only:
        for s in args.seeds:
            run_seed(s, cfg, force=args.force)

    aggregate()


if __name__ == '__main__':
    main()
