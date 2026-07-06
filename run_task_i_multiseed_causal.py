#!/usr/bin/env python3.11
"""
Task I — Replicate Task A's causal intervention (Task-G-upgraded) across the 5
multiseed models.

For each of the 5 independently-trained seed world models (from run_multiseed.py):
  * fit Probe A on that seed's own training_states (60% split),
  * extract the confusion direction,
  * collect held-out intervention trajectories with that seed's model,
  * ablate the confusion direction vs an empirical null of N_NULL random directions,
  * record Δprobe@{0,1,5,10}, routing-flip rate, next-step-KL change, and each's
    percentile / z against that seed's own null.

Aggregate across seeds: mean ± std of the confusion effect and its null percentile,
plus a check that the qualitative Task A finding replicates in the same direction on
every seed (confusion dir separates cleanly on probe-decay + routing; next-step KL
does not cleanly separate).

Blocked on run_multiseed.py finishing all 5 seeds. Uses a reduced site/null count
per seed (still ample) since it runs 5×.
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

ROOT       = 'outputs/multiseed'
N_TRAJ     = 40
MIN_SITE_T = 12
LOOKAHEAD  = [0, 1, 5, 10]
N_NULL     = 50
GAMMA      = 0.95
QUERY_BUDGET = 0.30
OUT_DIR    = 'outputs/causal'


def load_seed_model(seed, cfg):
    ck = torch.load(os.path.join(ROOT, f'seed_{seed}', 'world_model.pt'), map_location='cpu')
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg'])
    m.load_state_dict(ck['model_state']); m.eval()
    return m


def collect_trajectories(model, cfg, n_traj, seed):
    device = next(model.parameters()).device
    env = CartpoleEnv(task='swingup', noisy=False, seed=1000 + seed)
    np.random.seed(1000 + seed)
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
def continue_probe_kl(model, cfg, traj, t, h_new, clf, sc):
    """Return probe scores at t..t+max(LOOKAHEAD) and next-step KL at t."""
    device = next(model.parameters()).device
    T = len(traj['obs'])
    t_end = min(T, t + max(LOOKAHEAD) + 1)
    h = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    emb = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    prior_l = model.rssm.prior_net(h)
    kl_t = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
    z = model.rssm._straight_through_sample(post_l)
    hs = [h.squeeze(0).cpu().numpy().copy()]
    for k in range(t + 1, t_end):
        a = torch.tensor(traj['act'][k - 1], dtype=torch.float32, device=device).unsqueeze(0)
        obs_k = torch.tensor(traj['obs'][k], dtype=torch.float32, device=device).unsqueeze(0)
        emb = model.encoder(obs_k)
        h, z, _, _ = model.rssm.observe_step(h, z, a, emb)
        hs.append(h.squeeze(0).cpu().numpy().copy())
    ps = clf.predict_proba(sc.transform(np.array(hs, np.float32)))[:, 1]
    return ps, kl_t


def effect_for_direction(model, cfg, trajs, sites, v, clf, sc, route_thresh):
    dprobe = {k: [] for k in LOOKAHEAD}
    flips, dkls = [], []
    for (ti, t) in sites:
        trj = trajs[ti]
        h_t = trj['h'][t]
        ps_base, kl_base = trj['_base']
        proj = float(h_t @ v)
        ps, kl_new = continue_probe_kl(model, cfg, trj, t, h_t - proj * v, clf, sc)
        for k in LOOKAHEAD:
            if k < len(ps) and k < len(ps_base):
                dprobe[k].append(ps[k] - ps_base[k])
        flips.append(int((ps[0] >= route_thresh) != (ps_base[0] >= route_thresh)))
        dkls.append(kl_new - kl_base)
    return {**{f'dprobe_{k}': float(np.mean(dprobe[k])) for k in LOOKAHEAD},
            'routeflip': float(np.mean(flips)), 'dkl_0': float(np.mean(dkls))}


def run_seed(seed, cfg):
    print(f"\n{'='*60}\n[Task I] seed {seed}\n{'='*60}", flush=True)
    st = dict(np.load(os.path.join(ROOT, f'seed_{seed}', 'training_states.npz')))
    h_all, kl_all = st['h'], st['kl']
    y = binarise_by_median(kl_all)
    idx_tr, idx_te = train_test_split(np.arange(len(h_all)), test_size=0.40,
                                      stratify=y, random_state=0)
    clf, sc = train_probe(h_all[idx_tr], y[idx_tr])
    v = probe_direction(clf, sc)
    probe_te = clf.predict_proba(sc.transform(h_all[idx_te]))[:, 1]
    route_thresh = float(np.percentile(probe_te, 100 * (1 - QUERY_BUDGET)))

    model = load_seed_model(seed, cfg)
    trajs = collect_trajectories(model, cfg, N_TRAJ, seed)
    kl_median = float(np.median(kl_all))
    sites, rng = [], np.random.default_rng(seed)
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        if T < MIN_SITE_T + max(LOOKAHEAD) + 1:
            continue
        valid = np.arange(MIN_SITE_T, T - max(LOOKAHEAD) - 1)
        chosen = rng.choice(valid, size=min(10, len(valid)), replace=False)
        for t in chosen:
            sites.append((ti, int(t)))
    for (ti, t) in sites:
        trj = trajs[ti]
        trj['_base'] = continue_probe_kl(model, cfg, trj, t, trj['h'][t], clf, sc)

    conf = effect_for_direction(model, cfg, trajs, sites, v, clf, sc, route_thresh)
    rng_null = np.random.default_rng(2024 + seed)
    null = {f'dprobe_{k}': [] for k in LOOKAHEAD}
    null['routeflip'] = []; null['dkl_0'] = []
    for i in range(N_NULL):
        vr = random_matched_direction(rng_null, v.shape[0])
        e = effect_for_direction(model, cfg, trajs, sites, vr, clf, sc, route_thresh)
        for key in null:
            null[key].append(e[key])
    null = {k: np.array(v_) for k, v_ in null.items()}

    out = {'seed': seed, 'n_sites': len(sites), 'confusion': conf, 'null_pct': {}, 'null_z': {}}
    for key in [f'dprobe_{k}' for k in LOOKAHEAD] + ['routeflip', 'dkl_0']:
        c, d = conf[key], null[key]
        z = (c - d.mean()) / (d.std() + 1e-12)
        pct = float((d < c).mean() * 100) if key in ('routeflip',) else float((d > c).mean() * 100)
        out['null_pct'][key] = pct
        out['null_z'][key] = float(z)
    print(f"  seed {seed}: dprobe_0={conf['dprobe_0']:+.3f} (pct {out['null_pct']['dprobe_0']:.0f}, "
          f"z={out['null_z']['dprobe_0']:+.0f})  routeflip={conf['routeflip']:.3f} "
          f"(pct {out['null_pct']['routeflip']:.0f})  dkl_0 z={out['null_z']['dkl_0']:+.1f}", flush=True)
    return out


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = sorted(int(d.split('_')[1]) for d in os.listdir(ROOT)
                   if d.startswith('seed_') and
                   os.path.exists(os.path.join(ROOT, d, 'world_model.pt')) and
                   os.path.exists(os.path.join(ROOT, d, 'training_states.npz')))
    print(f"Task I over seeds: {seeds}")
    results = [run_seed(s, cfg) for s in seeds]

    print("\n" + "=" * 74)
    print("TASK I — CAUSAL INTERVENTION REPLICATED ACROSS SEEDS")
    print("=" * 74)
    def col(key, sub='confusion'):
        return np.array([r[sub][key] for r in results])
    print(f"\n  n_seeds={len(results)}  (each: {results[0]['n_sites']} sites, {N_NULL}-dir null)")
    print(f"\n  {'Metric':<14}{'confusion mean±std':>22}{'null-pct mean':>16}{'all-seeds sep?':>16}")
    print(f"  {'-'*14}{'-'*22}{'-'*16}{'-'*16}")
    summary = {}
    for key in [f'dprobe_{k}' for k in LOOKAHEAD] + ['routeflip', 'dkl_0']:
        c = col(key)
        pct = np.array([r['null_pct'][key] for r in results])
        if key == 'dkl_0':
            allsep = bool(np.all(pct >= 95) or np.all(pct <= 5))
        elif key == 'routeflip':
            allsep = bool(np.all(pct >= 95))
        else:
            allsep = bool(np.all(pct >= 95))
        summary[key] = dict(mean=float(c.mean()), std=float(c.std()),
                            pct_mean=float(pct.mean()), all_sep=allsep)
        print(f"  {key:<14}{c.mean():>+12.4f} ± {c.std():<7.4f}{pct.mean():>15.0f}%{('YES' if allsep else 'no'):>16}")

    probe_ok = summary['dprobe_0']['all_sep']
    route_ok = summary['routeflip']['all_sep']
    kl_clean = summary['dkl_0']['all_sep']
    print(f"\n  Qualitative Task A finding replication across all {len(results)} seeds:")
    print(f"    confusion dir separates on probe-decay:  {'YES' if probe_ok else 'NO'}")
    print(f"    confusion dir separates on routing:      {'YES' if route_ok else 'NO'}")
    print(f"    next-step KL does NOT cleanly separate:  {'YES (as in Task A)' if not kl_clean else 'NO — it does separate'}")
    if probe_ok and route_ok:
        print(f"\n  REPLICATED: the causal confusion-direction effect holds on all "
              f"{len(results)} independently-trained seeds, at the extreme of each seed's "
              f"own empirical null on the primary measures.")
    else:
        print(f"\n  PARTIAL: see per-metric separation above; reported honestly.")

    with open(os.path.join(OUT_DIR, 'task_i_results.json'), 'w') as f:
        json.dump({'seeds': [r['seed'] for r in results], 'per_seed': results,
                   'summary': summary}, f, indent=2)
    print(f"\n  Results: {os.path.join(OUT_DIR, 'task_i_results.json')}")


if __name__ == '__main__':
    main()
