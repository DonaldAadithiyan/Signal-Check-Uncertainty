#!/usr/bin/env python3.11
"""
Task K — Complete Task G's illusion-mitigation distinct-measure check.

Task G's spec named TWO candidate mechanistically-distinct downstream measures and
only tested one (imagined-vs-real latent divergence — came back a non-separating
partial). This is the other one: after ablating the confusion direction, does the
frozen decoder — conditioned on the post-ablation h_t — reconstruct the ACTUAL NEXT
REAL observation any worse (or better) than under a random-direction ablation?

Same 600 held-out sites as Task G (env seed 777, disjoint from Probe A's training
split), same 50-direction empirical null, same percentile/z reporting format.

Two decoder-recon variants measured (both on the REAL next observation, not the
model's own imagined prediction):
  * recon@t   — decode from the post-ablation posterior at t; error vs real obs_t
  * recon@t+1 — step one real observation forward from the ablated state; error vs real obs_{t+1}

Expected (consistent with Task G's divergence partial): the confusion direction is
NOT the causal lever for forward-dynamics accuracy, so this should ALSO fail to
separate from the null — a second, independent confirmation that "confusion =
accumulated history, not a dynamics-reliability mechanism." If it DOES separate, that
is a new, more surprising finding, reported honestly.

Runs on the existing frozen cartpole model. XS, CPU.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import probe_direction, random_matched_direction, compute_ct

N_TRAJ     = 60
MIN_SITE_T = 12
LOOKAHEAD  = [0, 1, 5, 10]     # kept identical to Task G for site-selection parity
GAMMA      = 0.95
N_NULL     = 50
OUT_DIR    = 'outputs/causal'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state']); m.eval()
    return m


def collect_trajectories(model, cfg, n_traj, seed=777):
    """Identical collection to Task G (same seed) → same trajectories/sites."""
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
                a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                emb = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, emb)
                kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                obs_l.append(obs.copy()); act_l.append(a.copy())
                h_l.append(h.squeeze(0).cpu().numpy().copy())
                z_l.append(post_l.squeeze(0).cpu().numpy().copy()); kl_l.append(kl)
                obs, _, done = env.step(a); step += 1
        trajs.append(dict(obs=np.array(obs_l, np.float32), act=np.array(act_l, np.float32),
                          h=np.array(h_l, np.float32), z=np.array(z_l, np.float32),
                          kl=np.array(kl_l, np.float32)))
    return trajs


@torch.no_grad()
def recon_after_ablation(model, cfg, traj, t, h_new):
    """From site t with h replaced by h_new: form the posterior at t (using the
    real obs_t), decode → recon error vs REAL obs_t; then step ONE real observation
    forward (real action, real obs_{t+1}) and decode → recon error vs REAL obs_{t+1}.
    Returns (recon_t, recon_t1). recon_t1 is nan if t is the last step."""
    device = next(model.parameters()).device
    T = len(traj['obs'])
    h = torch.tensor(h_new, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t = torch.tensor(traj['obs'][t], dtype=torch.float32, device=device).unsqueeze(0)
    emb = model.encoder(obs_t)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    z = model.rssm._straight_through_sample(post_l)
    dec_t = model.decoder(torch.cat([h, z], dim=-1))
    recon_t = F.mse_loss(dec_t, obs_t, reduction='none').sum().item()

    recon_t1 = float('nan')
    if t + 1 < T:
        a = torch.tensor(traj['act'][t], dtype=torch.float32, device=device).unsqueeze(0)
        obs_t1 = torch.tensor(traj['obs'][t + 1], dtype=torch.float32, device=device).unsqueeze(0)
        emb1 = model.encoder(obs_t1)
        h1, z1, _, _ = model.rssm.observe_step(h, z, a, emb1)
        dec_t1 = model.decoder(torch.cat([h1, z1], dim=-1))
        recon_t1 = F.mse_loss(dec_t1, obs_t1, reduction='none').sum().item()
    return recon_t, recon_t1


def effect_for_direction(model, cfg, trajs, sites, v):
    """Mean Δ recon (ablated − baseline) across sites, for recon@t and recon@t+1."""
    d_rt, d_rt1 = [], []
    for (ti, t) in sites:
        trj = trajs[ti]
        h_t = trj['h'][t]
        base_rt, base_rt1 = trj['_recon_base']
        proj = float(h_t @ v)
        rt, rt1 = recon_after_ablation(model, cfg, trj, t, h_t - proj * v)
        d_rt.append(rt - base_rt)
        if not (np.isnan(rt1) or np.isnan(base_rt1)):
            d_rt1.append(rt1 - base_rt1)
    return {'d_recon_t': float(np.mean(d_rt)), 'd_recon_t1': float(np.mean(d_rt1))}


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading training states + Probe A (60% train split)...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, kl_all, traj_id = tr['h'], tr['kl'], tr['traj_id']
    y = binarise_by_median(kl_all); kl_median = float(np.median(kl_all))
    idx_tr, _ = train_test_split(np.arange(len(h_all)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h_all[idx_tr], y[idx_tr])
    v = probe_direction(clf, sc)
    std_proj = float((h_all @ v).std())
    print(f"  std(h·v)={std_proj:.4f}")

    print(f"\nCollecting {N_TRAJ} held-out trajectories (seed 777, same as Task G)...")
    model = load_model(cfg)
    trajs = collect_trajectories(model, cfg, N_TRAJ)

    # identical site selection to Task G
    sites, rng = [], np.random.default_rng(0)
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
        chosen = rng.choice(valid, size=min(10, len(valid)), replace=False, p=w / w.sum())
        for t in chosen:
            sites.append((ti, int(t)))
    # baseline (unperturbed) recon per site
    for (ti, t) in sites:
        trj = trajs[ti]
        trj['_recon_base'] = recon_after_ablation(model, cfg, trj, t, trj['h'][t])
    print(f"  {len(sites)} intervention sites (same scheme as Task G)")

    # confusion direction effect
    conf = effect_for_direction(model, cfg, trajs, sites, v)
    print(f"\n  Confusion-direction Δrecon: @t={conf['d_recon_t']:+.5f}  @t+1={conf['d_recon_t1']:+.5f}")

    # empirical null
    print(f"  Building 50-direction empirical null...")
    rng_null = np.random.default_rng(2024)
    null = {'d_recon_t': [], 'd_recon_t1': []}
    for i in range(N_NULL):
        vr = random_matched_direction(rng_null, v.shape[0])
        e = effect_for_direction(model, cfg, trajs, sites, vr)
        null['d_recon_t'].append(e['d_recon_t'])
        null['d_recon_t1'].append(e['d_recon_t1'])
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N_NULL}", flush=True)
    null = {k: np.array(v_) for k, v_ in null.items()}

    print("\n" + "=" * 74)
    print("TASK K — DECODER RECON ON NEXT REAL OBS vs EMPIRICAL NULL")
    print("=" * 74)
    print(f"\n  N_null={N_NULL}, {len(sites)} sites. Positive Δrecon = ablation HURTS reconstruction.")
    print(f"\n  {'Measure':<14}{'confusion':>12}{'null mean±std':>22}{'z':>8}{'|percentile|':>14}")
    print(f"  {'-'*14}{'-'*12}{'-'*22}{'-'*8}{'-'*14}")
    results = {'n_sites': len(sites), 'n_null': N_NULL, 'std_proj': std_proj,
               'confusion': conf, 'null_summary': {}}
    for key in ['d_recon_t', 'd_recon_t1']:
        c = conf[key]; d = null[key]
        z = (c - d.mean()) / (d.std() + 1e-12)
        # two-sided extremity percentile: fraction of |null| less extreme than |confusion|
        pct = float((np.abs(d - d.mean()) < abs(c - d.mean())).mean() * 100)
        results['null_summary'][key] = dict(confusion=float(c), null_mean=float(d.mean()),
                                            null_std=float(d.std()), z=float(z), pct_extreme=pct)
        print(f"  {key:<14}{c:>+12.5f}{d.mean():>+13.5f}±{d.std():<8.5f}{z:>+8.1f}{pct:>13.0f}%")

    sep_t = abs(results['null_summary']['d_recon_t']['z']) >= 2.0
    sep_t1 = abs(results['null_summary']['d_recon_t1']['z']) >= 2.0
    print("\n" + "-" * 74)
    if not sep_t and not sep_t1:
        print("  DOES NOT SEPARATE (both measures |z|<2) — SECOND INDEPENDENT CONFIRMATION.")
        print("  Ablating the confusion direction changes the decoder's reconstruction of the")
        print("  actual next observation no more than a random direction does. Together with")
        print("  Task G's imagined-vs-real divergence result, two structurally different")
        print("  forward-dynamics measures now AGREE: the confusion direction is not the causal")
        print("  lever for the model's dynamics accuracy — it encodes accumulated history, not")
        print("  a dynamics-reliability mechanism. (Contrast: it DOES causally drive the probe")
        print("  readout and routing at the 100th percentile, Tasks A/G/I.)")
    else:
        print(f"  SEPARATES (recon@t z={results['null_summary']['d_recon_t']['z']:+.1f}, "
              f"recon@t+1 z={results['null_summary']['d_recon_t1']['z']:+.1f}) — SURPRISING.")
        print("  The confusion direction DOES affect reconstruction of the next real observation,")
        print("  even though (Task G) it does not affect imagined-vs-real latent drift. These two")
        print("  measures diverge and this needs honest discussion of why.")

    with open(os.path.join(OUT_DIR, 'task_k_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_k_results.json')}")


if __name__ == '__main__':
    main()
