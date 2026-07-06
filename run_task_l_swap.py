#!/usr/bin/env python3.11
"""
Task L — Swap-based natural intervention (the mitigation Task G substituted away).

Task G's robustness check ended up testing direction-rotation instead of the
originally-envisioned natural/substitution intervention (the noise version was
degenerate). Rotation is a real check but still a synthetic edit to h_t. A
substitution intervention never pushes h_t anywhere synthetic — it only recombines
REAL content that occurred somewhere in the data — which is a strictly stronger
defense against the "dormant parallel pathway" illusion.

Construction: for a recipient site with confusion-direction projection p_r, find a
DONOR real held-out state matched on the component ORTHOGONAL to v (nearest neighbour
in h⊥v space) but with a substantially DIFFERENT v-projection (low-confusion donor for
a high-confusion recipient and vice versa). The spliced state keeps the recipient's
⊥v component and replaces ONLY its projection onto v with the donor's:

    h_spliced = (h_r − (h_r·v) v) + (h_d·v) v

Then continue the trajectory exactly as Task A/G and measure the same primary outcomes
(probe-score decay at the same look-aheads, routing-flip rate) against the same
50-direction empirical null. Compare the swap effect size directly to the ablation
effect size from Task A/G on the same sites.

Agreement (same direction, comparable magnitude) ⇒ two structurally different
intervention types — synthetic edit and real-content substitution — agree, a stronger
illusion defense than rotation alone. Disagreement ⇒ reported honestly.

Runs on the existing frozen cartpole model. XS, CPU.
"""

import os
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import probe_direction, random_matched_direction

N_TRAJ     = 60
MIN_SITE_T = 12
LOOKAHEAD  = [0, 1, 5, 10]
GAMMA      = 0.95
N_NULL     = 50
QUERY_BUDGET = 0.30
N_DONOR_POOL = 8000       # real held-out states to draw donors from
OUT_DIR    = 'outputs/causal'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state']); m.eval()
    return m


def collect_trajectories(model, cfg, n_traj, seed=777):
    device = next(model.parameters()).device
    env = CartpoleEnv(task='swingup', noisy=False, seed=seed)
    np.random.seed(seed)
    trajs = []
    for ep in range(n_traj):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        obs_l, act_l, h_l, kl_l = [], [], [], []
        done, step = False, 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = np.random.uniform(-1, 1, (cfg['act_dim'],)).astype(np.float32)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                emb = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, emb)
                kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                obs_l.append(obs.copy()); act_l.append(a.copy())
                h_l.append(h.squeeze(0).cpu().numpy().copy()); kl_l.append(kl)
                obs, _, done = env.step(a); step += 1
        trajs.append(dict(obs=np.array(obs_l, np.float32), act=np.array(act_l, np.float32),
                          h=np.array(h_l, np.float32), kl=np.array(kl_l, np.float32)))
    return trajs


@torch.no_grad()
def continue_probe(model, cfg, traj, t, h_new, clf, sc):
    device = next(model.parameters()).device
    T = len(traj['obs'])
    t_end = min(T, t + max(LOOKAHEAD) + 1)
    h = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    emb = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    z = model.rssm._straight_through_sample(post_l)
    hs = [h.squeeze(0).cpu().numpy().copy()]
    for k in range(t + 1, t_end):
        a = torch.tensor(traj['act'][k - 1], dtype=torch.float32, device=device).unsqueeze(0)
        obs_k = torch.tensor(traj['obs'][k], dtype=torch.float32, device=device).unsqueeze(0)
        emb = model.encoder(obs_k)
        h, z, _, _ = model.rssm.observe_step(h, z, a, emb)
        hs.append(h.squeeze(0).cpu().numpy().copy())
    return clf.predict_proba(sc.transform(np.array(hs, np.float32)))[:, 1]


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading training states + Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, kl_all, traj_id = tr['h'], tr['kl'], tr['traj_id']
    y = binarise_by_median(kl_all); kl_median = float(np.median(kl_all))
    idx_tr, idx_te = train_test_split(np.arange(len(h_all)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h_all[idx_tr], y[idx_tr])
    v = probe_direction(clf, sc)
    std_proj = float((h_all @ v).std())
    probe_te = clf.predict_proba(sc.transform(h_all[idx_te]))[:, 1]
    route_thresh = float(np.percentile(probe_te, 100 * (1 - QUERY_BUDGET)))

    # ── donor pool: real held-out states, precompute ⊥v component + v-projection ──
    rng = np.random.default_rng(11)
    donor_idx = rng.choice(idx_te, min(N_DONOR_POOL, len(idx_te)), replace=False)
    H_donor = h_all[donor_idx]                    # (P, D)
    proj_donor = H_donor @ v                       # (P,)
    perp_donor = H_donor - np.outer(proj_donor, v) # (P, D) orthogonal-to-v component
    # normalise ⊥ component for nearest-neighbour matching (scale-free)
    perp_donor_n = perp_donor / (np.linalg.norm(perp_donor, axis=1, keepdims=True) + 1e-9)
    print(f"  donor pool: {len(donor_idx):,} real held-out states")

    print(f"\nCollecting {N_TRAJ} held-out trajectories (seed 777, same sites as Task G/A)...")
    model = load_model(cfg)
    trajs = collect_trajectories(model, cfg, N_TRAJ)

    sites, rng2 = [], np.random.default_rng(0)
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        if T < MIN_SITE_T + max(LOOKAHEAD) + 1:
            continue
        hk = (trj['kl'] > kl_median).astype(np.float32)
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
        w = ctj[valid] + 0.1
        chosen = rng2.choice(valid, size=min(10, len(valid)), replace=False, p=w / w.sum())
        for t in chosen:
            sites.append((ti, int(t)))
    for (ti, t) in sites:
        trajs[ti].setdefault('_ps_base', {})
        trajs[ti]['_ps_base'][t] = continue_probe(model, cfg, trajs[ti], t, trajs[ti]['h'][t], clf, sc)
    print(f"  {len(sites)} sites")

    # median v-projection to define low/high donor targets
    proj_median = float(np.median(proj_donor))
    proj_lo, proj_hi = np.percentile(proj_donor, [15, 85])

    def find_donor(h_r):
        """Nearest donor in ⊥v space whose v-projection is on the OPPOSITE side of the
        median from the recipient (low↔high confusion swap)."""
        p_r = float(h_r @ v)
        perp_r = h_r - p_r * v
        perp_r_n = perp_r / (np.linalg.norm(perp_r) + 1e-9)
        # candidate mask: opposite side of median, substantially different
        if p_r >= proj_median:
            mask = proj_donor <= proj_lo       # recipient high → low-confusion donor
        else:
            mask = proj_donor >= proj_hi       # recipient low → high-confusion donor
        cand = np.where(mask)[0]
        if len(cand) == 0:
            cand = np.arange(len(proj_donor))
        # nearest in normalised ⊥v space (cosine ~ dot)
        sims = perp_donor_n[cand] @ perp_r_n
        best = cand[int(np.argmax(sims))]
        return float(proj_donor[best]), float(sims.max())

    # ── swap effect + ablation effect (for direct comparison) ──
    def swap_state(h_r):
        p_d, _ = find_donor(h_r)
        return (h_r - float(h_r @ v) * v) + p_d * v

    def ablate_state(h_r):
        return h_r - float(h_r @ v) * v

    def effect(state_fn, is_random=False, rng_r=None):
        dprobe = {k: [] for k in LOOKAHEAD}; flips = []
        for (ti, t) in sites:
            trj = trajs[ti]; h_r = trj['h'][t]
            ps_base = trj['_ps_base'][t]
            if is_random:
                vr = rng_r
                h_new = h_r - float(h_r @ vr) * vr
            else:
                h_new = state_fn(h_r)
            ps = continue_probe(model, cfg, trj, t, h_new, clf, sc)
            for k in LOOKAHEAD:
                if k < len(ps) and k < len(ps_base):
                    dprobe[k].append(ps[k] - ps_base[k])
            flips.append(int((ps[0] >= route_thresh) != (ps_base[0] >= route_thresh)))
        return {**{f'dprobe_{k}': float(np.mean(dprobe[k])) for k in LOOKAHEAD},
                'routeflip': float(np.mean(flips))}

    print("\n  Computing swap effect and ablation effect on the same sites...")
    swap = effect(swap_state)
    ablate = effect(ablate_state)
    # match quality (mean cosine of ⊥v match)
    match_sims = [find_donor(trajs[ti]['h'][t])[1] for (ti, t) in sites]

    # empirical null (random-direction ablation, same as Task G)
    print(f"  Building 50-direction empirical null...")
    rng_null = np.random.default_rng(2024)
    null = {f'dprobe_{k}': [] for k in LOOKAHEAD}; null['routeflip'] = []
    for i in range(N_NULL):
        vr = random_matched_direction(rng_null, v.shape[0])
        e = effect(None, is_random=True, rng_r=vr)
        for key in null:
            null[key].append(e[key])
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N_NULL}", flush=True)
    null = {k: np.array(v_) for k, v_ in null.items()}

    print("\n" + "=" * 74)
    print("TASK L — SWAP-BASED NATURAL INTERVENTION vs ABLATION")
    print("=" * 74)
    print(f"\n  {len(sites)} sites, {N_NULL}-dir null. Mean ⊥v donor-match cosine = {np.mean(match_sims):.3f}.")
    print(f"\n  {'Measure':<12}{'swap':>10}{'ablation':>11}{'null mean±std':>20}{'swap pct':>10}{'abl pct':>9}")
    print(f"  {'-'*12}{'-'*10}{'-'*11}{'-'*20}{'-'*10}{'-'*9}")
    results = {'n_sites': len(sites), 'n_null': N_NULL, 'mean_match_cos': float(np.mean(match_sims)),
               'swap': swap, 'ablation': ablate, 'null_summary': {}}
    for key in [f'dprobe_{k}' for k in LOOKAHEAD] + ['routeflip']:
        s, a, d = swap[key], ablate[key], null[key]
        if key == 'routeflip':
            s_pct = float((d < s).mean() * 100); a_pct = float((d < a).mean() * 100)
        else:
            s_pct = float((d > s).mean() * 100); a_pct = float((d > a).mean() * 100)
        results['null_summary'][key] = dict(swap=float(s), ablation=float(a),
                                            null_mean=float(d.mean()), null_std=float(d.std()),
                                            swap_pct=s_pct, ablation_pct=a_pct)
        print(f"  {key:<12}{s:>+10.4f}{a:>+11.4f}{d.mean():>+11.4f}±{d.std():<7.4f}{s_pct:>9.0f}%{a_pct:>8.0f}%")

    # agreement check on the primary measure (probe-decay @ t)
    s0, a0 = swap['dprobe_0'], ablate['dprobe_0']
    same_dir = (np.sign(s0) == np.sign(a0))
    ratio = abs(s0) / (abs(a0) + 1e-9)
    swap_sep = results['null_summary']['dprobe_0']['swap_pct'] >= 98
    route_sep = results['null_summary']['routeflip']['swap_pct'] >= 98
    print("\n" + "-" * 74)
    print(f"  Swap Δprobe@t = {s0:+.4f}  vs  ablation Δprobe@t = {a0:+.4f}  "
          f"(same direction: {same_dir}, |swap|/|abl| = {ratio:.2f})")
    if same_dir and swap_sep and 0.4 < ratio < 2.5:
        print("\n  AGREEMENT: the swap-based real-content substitution reproduces the ablation")
        print("  effect in direction and rough magnitude, at the extreme of the empirical null.")
        print("  Two structurally different intervention types — synthetic edit (ablation) and")
        print("  real-content substitution (swap) — agree. This is a stronger defense against the")
        print("  dormant-parallel-pathway illusion than direction-rotation alone provided.")
    elif same_dir:
        print(f"\n  PARTIAL AGREEMENT: same direction, but magnitude differs (ratio {ratio:.2f}) or")
        print(f"  null-separation weaker (swap pct {results['null_summary']['dprobe_0']['swap_pct']:.0f}). "
              f"Reported honestly.")
    else:
        print("\n  DISAGREEMENT: swap and ablation point in different directions — the ablation")
        print("  result is more fragile to intervention type than assumed. Important to report.")

    with open(os.path.join(OUT_DIR, 'task_l_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_l_results.json')}")


if __name__ == '__main__':
    main()
