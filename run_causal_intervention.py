#!/usr/bin/env python3.11
"""
Task A — Causal intervention on the confusion direction (highest priority).

A linear probe found a direction v in h_t-space that CORRELATES with KL history.
This script tests whether that direction is CAUSALLY load-bearing by intervening
on h_t at inference time on the frozen world model and continuing the rollout.

Conditions (all applied at a chosen site t inside a held-out trajectory):
  - Ablation:      h' = h - (h·v) v
  - Amplification: h' = h + α v, α ∈ {±1,±2,±3}·std(proj)
  - Random control: same ablation/amplification with a random unit direction,
                    matched in projection scale.

Measured, per condition, over ≥500 intervention sites:
  1. Δ probe score at t, t+1, t+5, t+10  (does ablation erase confusion and does
     it recover at the γ=0.95 rate predicted by the closed-form C_t model?)
  2. Δ model's own next-step predicted KL(posterior‖prior)
  3. Δ observation-routing decision (would the 30%-budget policy query here?)
  4. Effect sizes with bootstrap 95% CIs, confusion-direction vs random-direction.

Success: confusion-direction interventions produce a statistically distinguishable
effect from the random-direction control (non-overlapping / well-separated CIs)
on at least the probe-score-decay and routing-decision measures. A null here is
itself a publishable "representationally present but not causally load-bearing"
finding — reported honestly either way.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import (
    probe_direction, regression_direction, random_matched_direction,
    compute_ct, bootstrap_ci,
)

N_TRAJ       = 60          # held-out trajectories to collect for intervention sites
MIN_SITE_T   = 12          # need history before t and >=10 steps after
LOOKAHEAD    = [0, 1, 5, 10]
GAMMA        = 0.95
ALPHAS_SD    = [-3, -2, -1, 1, 2, 3]   # amplification sweep in units of std(proj)
QUERY_BUDGET = 0.30
N_BOOT       = 1000
OUT_DIR      = 'outputs/causal'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def collect_trajectories(model, cfg, n_traj, seed=777):
    """Collect full trajectories keeping obs, actions, and per-step h/z/kl/recon."""
    device = next(model.parameters()).device
    env = CartpoleEnv(task='swingup', noisy=False, seed=seed)
    np.random.seed(seed)
    trajs = []
    for ep in range(n_traj):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        obs_l, act_l, h_l, z_l, kl_l = [], [], [], [], []
        done, step = False, 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = np.random.uniform(-1, 1, (cfg['act_dim'],)).astype(np.float32)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_t   = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                embed = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                obs_l.append(obs.copy()); act_l.append(a.copy())
                h_l.append(h.squeeze(0).cpu().numpy().copy())
                z_l.append(post_l.squeeze(0).cpu().numpy().copy())
                kl_l.append(kl)
                obs, _, done = env.step(a)
                step += 1
        trajs.append(dict(
            obs=np.array(obs_l, dtype=np.float32),
            act=np.array(act_l, dtype=np.float32),
            h=np.array(h_l, dtype=np.float32),
            z=np.array(z_l, dtype=np.float32),
            kl=np.array(kl_l, dtype=np.float32),
        ))
    return trajs


@torch.no_grad()
def continue_rollout(model, cfg, traj, t, h_intervened):
    """Re-run the observed trajectory from step t with h at t replaced by
    h_intervened, using the trajectory's actual observations & actions.
    Returns per-step h and next-step KL for steps t, t+1, ... to end.
    h_intervened: numpy (deter,). Uses the true prev z at t (posterior computed
    from the intervened h and the real observation at t)."""
    device = next(model.parameters()).device
    T = len(traj['obs'])
    # Replace h_t; recompute posterior z_t from intervened h_t + real obs_t so the
    # downstream trajectory is internally consistent.
    h = torch.tensor(h_intervened, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    embed = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, embed], dim=-1))
    prior_l = model.rssm.prior_net(h)
    z = model.rssm._straight_through_sample(post_l)
    kl_t = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()

    hs = [h.squeeze(0).cpu().numpy().copy()]
    next_kls = [kl_t]
    for k in range(t + 1, T):
        a_t = torch.tensor(traj['act'][k - 1], dtype=torch.float32, device=device).unsqueeze(0)
        obs_k = torch.tensor(traj['obs'][k], dtype=torch.float32, device=device).unsqueeze(0)
        embed = model.encoder(obs_k)
        h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
        kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
        hs.append(h.squeeze(0).cpu().numpy().copy())
        next_kls.append(kl)
    return np.array(hs, dtype=np.float32), np.array(next_kls, dtype=np.float32)


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ── Train Probe A on the ORIGINAL training states (frozen convention) ──
    print("Loading training states + training Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, kl_all, traj_id = tr['h'], tr['kl'], tr['traj_id']
    y = binarise_by_median(kl_all)
    kl_median = float(np.median(kl_all))
    idx_tr, idx_te = train_test_split(np.arange(len(h_all)), test_size=0.40,
                                      stratify=y, random_state=0)
    clf, sc = train_probe(h_all[idx_tr], y[idx_tr])

    # confusion direction (probe) and consistency check vs C_t-regression direction
    v = probe_direction(clf, sc)
    ct_all = compute_ct(kl_all, traj_id, gamma=GAMMA, kl_median=kl_median)
    sc_ct = StandardScaler().fit(h_all[idx_tr])
    ridge = Ridge(alpha=1.0).fit(sc_ct.transform(h_all[idx_tr]), ct_all[idx_tr])
    v_ct = regression_direction(ridge, sc_ct)
    cos_consistency = float(abs(np.dot(v, v_ct)))
    print(f"  Confusion direction consistency (probe vs C_t-regression): "
          f"cos={cos_consistency:.4f}  angle={np.degrees(np.arccos(min(cos_consistency,1))):.1f}°")

    # natural projection statistics of h onto v (empirical, not guessed)
    proj_all = h_all @ v
    std_proj = float(proj_all.std())
    print(f"  Natural std(h·v) = {std_proj:.4f}")

    def probe_score(h_arr):
        return clf.predict_proba(sc.transform(np.atleast_2d(h_arr)))[:, 1]

    # routing threshold on held-out probe scores (30% budget)
    probe_te = probe_score(h_all[idx_te])
    route_thresh = float(np.percentile(probe_te, 100 * (1 - QUERY_BUDGET)))

    # ── Collect held-out intervention trajectories ──
    print(f"\nCollecting {N_TRAJ} held-out trajectories for intervention sites...")
    model = load_model(cfg)
    trajs = collect_trajectories(model, cfg, N_TRAJ)

    # choose intervention sites: one per trajectory-window, prefer high-C_t
    sites = []   # (traj_idx, t)
    rng = np.random.default_rng(0)
    for ti, tr_j in enumerate(trajs):
        T = len(tr_j['obs'])
        if T < MIN_SITE_T + max(LOOKAHEAD) + 1:
            continue
        # per-traj C_t to bias toward confused sites (where signal should matter)
        hk = (tr_j['kl'] > kl_median).astype(np.float32)
        ctj = np.zeros(T)
        for i in range(T):
            val = 0.0
            for lag in range(50):
                j = i - lag
                if j < 0:
                    break
                val += (GAMMA ** lag) * hk[j]
            ctj[i] = val
        valid = np.arange(MIN_SITE_T, T - max(LOOKAHEAD) - 1)
        # pick up to ~10 sites per traj, weighted toward higher C_t
        w = ctj[valid] + 0.1
        n_pick = min(10, len(valid))
        chosen = rng.choice(valid, size=n_pick, replace=False, p=w / w.sum())
        for t in chosen:
            sites.append((ti, int(t)))
    print(f"  {len(sites)} intervention sites across {len(trajs)} trajectories")

    # ── Run interventions ──
    # Precompute per-site unperturbed continuation once.
    rngdir = np.random.default_rng(123)
    records = []  # each: dict of measures for ablation/amplify/random per site
    for (ti, t) in sites:
        trj = trajs[ti]
        h_t = trj['h'][t]

        # unperturbed continuation (baseline)
        h_base, kl_base = continue_rollout(model, cfg, trj, t, h_t)

        # a fresh random unit direction per site, scaled to match v's projection magnitude
        v_rand = random_matched_direction(rngdir, h_t.shape[0])

        def run_condition(h_new):
            h_seq, kl_seq = continue_rollout(model, cfg, trj, t, h_new)
            ps = probe_score(h_seq)                    # probe score along continuation
            ps_base = probe_score(h_base)
            # align lookahead offsets that exist
            out = {}
            for k in LOOKAHEAD:
                if k < len(ps):
                    out[f'dprobe_{k}'] = float(ps[k] - ps_base[k])
                    out[f'dkl_{k}']    = float(kl_seq[k] - kl_base[k])
            # routing decision at t (before vs after)
            out['route_base'] = int(ps_base[0] >= route_thresh)
            out['route_new']  = int(ps[0] >= route_thresh)
            out['route_flip'] = int((ps[0] >= route_thresh) != (ps_base[0] >= route_thresh))
            return out

        proj_t = float(h_t @ v)
        # ablation (confusion dir) and its random control
        rec = {'ti': ti, 't': t, 'proj_t': proj_t}
        rec['ablate_conf'] = run_condition(h_t - proj_t * v)
        rec['ablate_rand'] = run_condition(h_t - float(h_t @ v_rand) * v_rand)
        # amplification sweeps
        rec['amp_conf'] = {a: run_condition(h_t + a * std_proj * v) for a in ALPHAS_SD}
        rec['amp_rand'] = {a: run_condition(h_t + a * std_proj * v_rand) for a in ALPHAS_SD}
        records.append(rec)

    print(f"  Completed interventions on {len(records)} sites")

    # ── Aggregate ──
    def collect(cond_key, field):
        return np.array([r[cond_key][field] for r in records if field in r[cond_key]])

    results = {'n_sites': len(records),
               'consistency_cos': cos_consistency,
               'std_proj': std_proj,
               'gamma': GAMMA}

    print("\n" + "=" * 74)
    print("TASK A — CAUSAL INTERVENTION ON THE CONFUSION DIRECTION")
    print("=" * 74)

    # 1. Ablation probe-score decay: confusion vs random
    print(f"\n[1] ABLATION — Δ probe score by look-ahead k (mean [95% CI]), n={len(records)}")
    print(f"    {'k':>4}  {'confusion-dir':>28}  {'random-dir':>28}  {'separated?':>10}")
    decay_conf, decay_rand = {}, {}
    for k in LOOKAHEAD:
        dc = collect('ablate_conf', f'dprobe_{k}')
        dr = collect('ablate_rand', f'dprobe_{k}')
        pc, lc, hc = bootstrap_ci(dc, n_boot=N_BOOT, seed=1)
        pr, lr, hr = bootstrap_ci(dr, n_boot=N_BOOT, seed=2)
        sep = (hc < lr) or (hr < lc)
        decay_conf[k] = (pc, lc, hc); decay_rand[k] = (pr, lr, hr)
        print(f"    {k:>4}  {pc:>+8.4f} [{lc:>+7.4f},{hc:>+7.4f}]  "
              f"{pr:>+8.4f} [{lr:>+7.4f},{hr:>+7.4f}]  {'YES' if sep else 'no':>10}")
    results['ablation_dprobe_conf'] = {str(k): decay_conf[k] for k in LOOKAHEAD}
    results['ablation_dprobe_rand'] = {str(k): decay_rand[k] for k in LOOKAHEAD}

    # γ-decay prediction: does |Δprobe| at t+k / |Δprobe| at t ≈ γ^k for confusion dir?
    base_effect = abs(decay_conf[0][0])
    if base_effect > 1e-6:
        print(f"\n    γ-decay test (confusion dir): predicted ratio γ^k vs observed |Δ_k|/|Δ_0|")
        print(f"    {'k':>4}  {'predicted γ^k':>14}  {'observed':>10}")
        gamma_fit = {}
        for k in LOOKAHEAD:
            pred = GAMMA ** k
            obs = abs(decay_conf[k][0]) / base_effect
            gamma_fit[k] = (pred, obs)
            print(f"    {k:>4}  {pred:>14.4f}  {obs:>10.4f}")
        results['gamma_decay_fit'] = {str(k): gamma_fit[k] for k in LOOKAHEAD}

    # 2. Next-step KL change at t (model's own surprise)
    print(f"\n[2] Δ model next-step KL at t (ablation):")
    dkl_c = collect('ablate_conf', 'dkl_0'); dkl_r = collect('ablate_rand', 'dkl_0')
    pc, lc, hc = bootstrap_ci(dkl_c, n_boot=N_BOOT, seed=3)
    pr, lr, hr = bootstrap_ci(dkl_r, n_boot=N_BOOT, seed=4)
    print(f"    confusion-dir: {pc:>+8.4f} [{lc:+.4f},{hc:+.4f}]   "
          f"random-dir: {pr:>+8.4f} [{lr:+.4f},{hr:+.4f}]")
    results['ablation_dkl_conf'] = (pc, lc, hc)
    results['ablation_dkl_rand'] = (pr, lr, hr)

    # 3. Routing flip rate
    print(f"\n[3] Routing-decision flip rate at t (fraction of sites where the "
          f"{QUERY_BUDGET:.0%}-budget query decision changes):")
    flip_c = collect('ablate_conf', 'route_flip')
    flip_r = collect('ablate_rand', 'route_flip')
    pc, lc, hc = bootstrap_ci(flip_c.astype(float), n_boot=N_BOOT, seed=5)
    pr, lr, hr = bootstrap_ci(flip_r.astype(float), n_boot=N_BOOT, seed=6)
    sep_route = (hc < lr) or (hr < lc)
    print(f"    confusion-dir: {pc:.3f} [{lc:.3f},{hc:.3f}]   "
          f"random-dir: {pr:.3f} [{lr:.3f},{hr:.3f}]   separated: {'YES' if sep_route else 'no'}")
    results['ablation_routeflip_conf'] = (pc, lc, hc)
    results['ablation_routeflip_rand'] = (pr, lr, hr)

    # 4. Amplification sweep: mean Δprobe at t vs α
    print(f"\n[4] AMPLIFICATION — Δ probe score at t vs α (× std proj):")
    print(f"    {'α':>5}  {'confusion Δprobe':>22}  {'random Δprobe':>22}")
    amp_conf_curve, amp_rand_curve = {}, {}
    for a in ALPHAS_SD:
        dc = np.array([r['amp_conf'][a]['dprobe_0'] for r in records if 'dprobe_0' in r['amp_conf'][a]])
        dr = np.array([r['amp_rand'][a]['dprobe_0'] for r in records if 'dprobe_0' in r['amp_rand'][a]])
        pc, lc, hc = bootstrap_ci(dc, n_boot=N_BOOT, seed=7)
        pr, lr, hr = bootstrap_ci(dr, n_boot=N_BOOT, seed=8)
        amp_conf_curve[a] = (pc, lc, hc); amp_rand_curve[a] = (pr, lr, hr)
        print(f"    {a:>+5}  {pc:>+8.4f} [{lc:+.4f},{hc:+.4f}]  {pr:>+8.4f} [{lr:+.4f},{hr:+.4f}]")
    results['amp_conf_curve'] = {str(a): amp_conf_curve[a] for a in ALPHAS_SD}
    results['amp_rand_curve'] = {str(a): amp_rand_curve[a] for a in ALPHAS_SD}

    # ── Verdict ──
    sep_decay = any((decay_conf[k][2] < decay_rand[k][0] or decay_rand[k][2] < decay_conf[k][0])
                    for k in LOOKAHEAD)
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  Confusion-direction vs random-direction, well-separated CIs:")
    print(f"    probe-score decay:   {'YES' if sep_decay else 'NO'}")
    print(f"    routing flip rate:   {'YES' if sep_route else 'NO'}")
    if sep_decay and sep_route:
        print("\n  CAUSAL: the confusion direction is load-bearing — ablating it specifically")
        print("  erases the probe-read confusion signal AND changes routing decisions, well")
        print("  beyond a norm-matched random perturbation.")
    elif sep_decay or sep_route:
        print("\n  PARTIAL CAUSAL: distinguishable from random on at least one measure.")
    else:
        print("\n  NULL: confusion direction is representationally present but NOT causally")
        print("  distinguishable from a random perturbation on these measures. (Honest")
        print("  negative — relevant to the 'sufficiency' theme.)")

    with open(os.path.join(OUT_DIR, 'task_a_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle('Task A — Causal Intervention on the Confusion Direction',
                 fontweight='bold', fontsize=12)

    ax = axes[0]
    ks = LOOKAHEAD
    cc = [decay_conf[k][0] for k in ks]; cl = [decay_conf[k][1] for k in ks]; ch = [decay_conf[k][2] for k in ks]
    rc = [decay_rand[k][0] for k in ks]; rl = [decay_rand[k][1] for k in ks]; rh = [decay_rand[k][2] for k in ks]
    ax.errorbar(ks, cc, yerr=[np.array(cc)-np.array(cl), np.array(ch)-np.array(cc)],
                fmt='b-o', capsize=3, label='confusion dir')
    ax.errorbar(ks, rc, yerr=[np.array(rc)-np.array(rl), np.array(rh)-np.array(rc)],
                fmt='r-s', capsize=3, label='random dir')
    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('look-ahead k'); ax.set_ylabel('Δ probe score (ablation)')
    ax.set_title('Ablation effect vs look-ahead\n(specific > random ⇒ causal)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    if base_effect > 1e-6:
        obs = [abs(decay_conf[k][0]) / base_effect for k in ks]
        pred = [GAMMA ** k for k in ks]
        ax.plot(ks, pred, 'k--', label=f'predicted γ^k (γ={GAMMA})')
        ax.plot(ks, obs, 'b-o', label='observed |Δ_k|/|Δ_0|')
        ax.set_xlabel('look-ahead k'); ax.set_ylabel('normalised effect')
        ax.set_title('γ-decay prediction test'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    al = ALPHAS_SD
    ac = [amp_conf_curve[a][0] for a in al]; ar = [amp_rand_curve[a][0] for a in al]
    ax.plot(al, ac, 'b-o', label='confusion dir')
    ax.plot(al, ar, 'r-s', label='random dir')
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_xlabel('α (× std proj)'); ax.set_ylabel('Δ probe score at t')
    ax.set_title('Amplification dose-response'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'causal_intervention.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {fig_path}")
    print(f"  Results saved: {os.path.join(OUT_DIR, 'task_a_results.json')}")


if __name__ == '__main__':
    main()
