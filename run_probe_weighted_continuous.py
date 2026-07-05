#!/usr/bin/env python3.11
"""
Task F — Retry probe-weighted returns with a CONTINUOUS confusion signal.

The existing negative result (Δr = −0.53, kept and honest) weighted imagined
returns by the output of a probe TRAINED on binarised KL labels. Task F checks
whether that binarisation of the supervision — not the underlying confusion
signal — caused the degradation, by re-running the identical experiment with
continuous weighting signals:

  (w0) standard (no weighting)                         [reference]
  (w1) binary-KL-probe probability   1 − p_probe(h)    [the existing method]
  (w2) raw probe decision function   1 − σ(scaled df)   [same probe, un-binarised score]
  (w3) direct C_t regression         1 − norm(Ĉ_t(h))   [Ridge probe trained on continuous C_t]

Reported either way. Still-negative ⇒ stronger, more specific negative result
(rules out "it was just the binarisation"). Positive ⇒ new contribution.
Ships with N and bootstrap CIs on every Δr.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct

HORIZON  = 5
GAMMA    = 0.995
N_STATES = 5_000
CT_GAMMA = 0.95
N_BOOT   = 1000
OUT_DIR  = 'outputs/causal'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def imagine_kl_sequence(model, h_start, z_start, horizon, cfg, seed=0):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    N = h_start.shape[0]
    h = torch.tensor(h_start, dtype=torch.float32, device=device)
    z = torch.tensor(z_start, dtype=torch.float32, device=device)
    kl_seq, h_seq = [], []
    with torch.no_grad():
        for k in range(horizon):
            action = torch.tensor(rng.uniform(-1, 1, (N, cfg['act_dim'])).astype(np.float32),
                                  device=device)
            h, z, prior_l = model.rssm.imagine_step(h, z, action)
            logits = prior_l.view(N, cfg['rssm_stoch'], cfg['rssm_classes'])
            log_p = torch.log_softmax(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            H = -(p * log_p).sum(dim=-1).mean(dim=-1)
            kl_seq.append(H.cpu().numpy())
            h_seq.append(h.cpu().numpy().copy())
    return np.stack(kl_seq, axis=1), np.stack(h_seq, axis=1)


def lambda_return(kl_seq, gamma=GAMMA):
    H = kl_seq.shape[1]
    g = gamma ** np.arange(1, H + 1)
    return (kl_seq * g[None, :]).sum(axis=1)


def weighted_return(kl_seq, weights, gamma=GAMMA):
    H = kl_seq.shape[1]
    g = gamma ** np.arange(1, H + 1)
    return (kl_seq * weights * g[None, :]).sum(axis=1)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading training states...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, z_all, kl_all, traj = tr['h'], tr['z'], tr['kl'], tr['traj_id']
    N = len(h_all)
    y_kl = binarise_by_median(kl_all)
    kl_median = float(np.median(kl_all))
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)

    # (w1/w2) binary-KL probe
    clf, sc = train_probe(h_all[tr_idx], y_kl[tr_idx])
    # (w3) continuous C_t Ridge probe
    ct_all = compute_ct(kl_all, traj, gamma=CT_GAMMA, kl_median=kl_median)
    sc_ct = StandardScaler().fit(h_all[tr_idx])
    ridge = Ridge(alpha=1.0).fit(sc_ct.transform(h_all[tr_idx]), ct_all[tr_idx])
    ct_lo, ct_hi = np.percentile(ct_all[tr_idx], [1, 99])

    # raw decision-function range for min-max normalisation of the un-squashed logit
    df_tr = clf.decision_function(sc.transform(h_all[tr_idx]))
    df_lo, df_hi = np.percentile(df_tr, [1, 99])

    def w_binary(h):     # existing method: 1 - probe probability (sigmoid-squashed)
        return 1.0 - clf.predict_proba(sc.transform(h))[:, 1]

    def w_rawlogit(h):   # un-squashed decision function, min-max normalised to [0,1]
        df = clf.decision_function(sc.transform(h))
        s = np.clip((df - df_lo) / (df_hi - df_lo + 1e-9), 0.0, 1.0)
        return 1.0 - s

    def w_ct(h):         # continuous C_t regression, min-max normalised to [0,1]
        chat = ridge.predict(sc_ct.transform(h))
        chat = np.clip((chat - ct_lo) / (ct_hi - ct_lo + 1e-9), 0.0, 1.0)
        return 1.0 - chat

    # ── real K-step returns ──
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
            real_returns[i] = np.array(fut, dtype=np.float32)
    valid = np.array(list(real_returns.keys()), dtype=np.int64)
    rng = np.random.default_rng(42)
    start_idx = valid[rng.choice(len(valid), min(N_STATES, len(valid)), replace=False)]
    real_kl = np.stack([real_returns[i] for i in start_idx], axis=0)
    real_V = lambda_return(real_kl)

    # ── imagination ──
    print(f"Imagination rollouts from {len(start_idx):,} states...")
    model = load_model(cfg)
    imag_kl, imag_h = imagine_kl_sequence(model, h_all[start_idx], z_all[start_idx], HORIZON, cfg)

    # weights per method
    def build_weights(wfn):
        W = np.zeros((len(start_idx), HORIZON), dtype=np.float32)
        for k in range(HORIZON):
            W[:, k] = wfn(imag_h[:, k, :])
        return W

    V_std = lambda_return(imag_kl)
    methods = {
        'standard':          None,
        'binary-KL-probe':   build_weights(w_binary),
        'raw-logit-probe':   build_weights(w_rawlogit),
        'continuous-C_t':    build_weights(w_ct),
    }

    print("\n" + "=" * 70)
    print("TASK F — CONTINUOUS-SIGNAL PROBE-WEIGHTED RETURNS")
    print("=" * 70)
    print(f"\n  Horizon={HORIZON}  γ={GAMMA}  C_t γ={CT_GAMMA}  N={len(start_idx):,}")
    print(f"\n  {'Method':<20}  {'r(V̂,V_real)':>13}  {'Δr vs std [95% CI]':>30}")
    print(f"  {'-'*20}  {'-'*13}  {'-'*30}")

    r_std = np.corrcoef(V_std, real_V)[0, 1]
    results = {'N': int(len(start_idx)), 'horizon': HORIZON, 'r_standard': float(r_std),
               'methods': {}}
    print(f"  {'standard':<20}  {r_std:>13.4f}  {'—':>30}")

    # per-sample correlation contribution via bootstrap of Δr
    def r_of(V):
        return np.corrcoef(V, real_V)[0, 1]

    for name, W in methods.items():
        if W is None:
            continue
        V_w = weighted_return(imag_kl, W)
        r_w = r_of(V_w)
        # bootstrap Δr
        boots = []
        rb = np.random.default_rng(7)
        n = len(V_w)
        for _ in range(N_BOOT):
            idx = rb.integers(0, n, n)
            boots.append(np.corrcoef(V_w[idx], real_V[idx])[0, 1] -
                         np.corrcoef(V_std[idx], real_V[idx])[0, 1])
        boots = np.array(boots)
        dr = r_w - r_std
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results['methods'][name] = dict(r=float(r_w), dr=float(dr), dr_lo=float(lo), dr_hi=float(hi))
        print(f"  {name:<20}  {r_w:>13.4f}  {dr:>+8.4f} [{lo:>+7.4f}, {hi:>+7.4f}]")

    # verdict
    best = max(results['methods'].items(), key=lambda kv: kv[1]['dr'])
    print("\n" + "-" * 70)
    ct = results['methods']['continuous-C_t']
    if ct['dr_lo'] > 0:
        print(f"  POSITIVE (continuous C_t): Δr={ct['dr']:+.4f}, 95% CI excludes 0 "
              f"[{ct['dr_lo']:+.4f},{ct['dr_hi']:+.4f}].")
        print("  Continuous confusion weighting improves return estimates — the previous")
        print("  negative was driven by the binarised supervision, not the signal itself.")
    elif ct['dr_hi'] < 0:
        print(f"  NEGATIVE confirmed (continuous C_t): Δr={ct['dr']:+.4f}, 95% CI "
              f"[{ct['dr_lo']:+.4f},{ct['dr_hi']:+.4f}] entirely below 0.")
        print("  Stronger, more specific negative: it was NOT just the binarisation —")
        print("  confusion-weighting of imagined returns degrades value estimates even")
        print("  with a continuous C_t signal. (Reported alongside the original.)")
    else:
        print(f"  NULL (continuous C_t): Δr={ct['dr']:+.4f}, 95% CI straddles 0 "
              f"[{ct['dr_lo']:+.4f},{ct['dr_hi']:+.4f}].")
        print("  Continuous weighting neither helps nor hurts significantly.")
    print(f"  (best method by Δr: {best[0]}, Δr={best[1]['dr']:+.4f})")

    with open(os.path.join(OUT_DIR, 'task_f_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_f_results.json')}")


if __name__ == '__main__':
    main()
