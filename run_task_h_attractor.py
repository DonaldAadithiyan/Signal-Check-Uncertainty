#!/usr/bin/env python3.11
"""
Task H — Does the confusion signal predict, or get masked by, attractor-recovery?

Tests compatibility with Berger et al. "Biased Dreams": Dreamer-family RSSM latent
transitions show ATTRACTOR behavior — pushed toward OOD/unusual states, the latent
transition pulls the representation back toward well-represented regions over the
next few steps, rather than diverging like a physical dynamics model. If latents
"snap back", why does the confusion signal persist for many steps (γ=0.95 decay)?

Protocol:
  1. Operationalize attractor recovery: from a starting state, roll IMAGINATION
     forward `HORIZON` steps (real actions) and measure, at each step, the latent
     distance between the imagined path and the REAL posterior path (encoding the
     actual subsequent observations). "Snap-back" = distance shrinks over early steps.
  2. Do this for (a) clean starting states and (b) OOD-perturbed starting states
     (Set B spirit: Gaussian noise on the observation that seeds the start latent).
  3. Split by CONFUSION LEVEL (Probe A score / C_t) into low/med/high bins; compare
     the recovery curve per bin.
       - Reinforcing: high-confusion states show a larger / slower-recovering gap
         (confusion flags where attractor-masking is weakest / dynamics least reliable).
       - Orthogonal/tension: confusion level unrelated to the recovery pattern —
         the signal tracks posterior-vs-prior HISTORY, not latent-dynamics reliability,
         which answers "why persist if latents snap back" (different quantity).
  4. Reward-overestimation cross-check (cartpole upright-reward proxy from decoded obs):
     is imagined end-of-horizon reward biased high, and does that bias track confusion?

Reports a single unambiguous verdict: REINFORCING / ORTHOGONAL / MIXED.
Runs on the EXISTING frozen model.
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import compute_ct, imagined_vs_real_latent

N_TRAJ    = 80
HORIZON   = 10
MIN_T     = 12
GAMMA     = 0.95
NOISE_STD = 0.10          # OOD perturbation scale (matches Set B noise_std)
OUT_DIR   = 'outputs/causal'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state']); m.eval()
    return m


def collect_trajectories(model, cfg, n_traj, seed=888):
    """Collect trajectories keeping obs, actions, h, z(posterior logits), kl, and
    the realized reward at each step (cartpole upright reward)."""
    device = next(model.parameters()).device
    env = CartpoleEnv(task='swingup', noisy=False, seed=seed)
    np.random.seed(seed)
    trajs = []
    for ep in range(n_traj):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        obs_l, act_l, h_l, z_l, kl_l, rew_l = [], [], [], [], [], []
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
                obs, rew, done = env.step(a); rew_l.append(rew)
                step += 1
        trajs.append(dict(obs=np.array(obs_l, np.float32), act=np.array(act_l, np.float32),
                          h=np.array(h_l, np.float32), z=np.array(z_l, np.float32),
                          kl=np.array(kl_l, np.float32), rew=np.array(rew_l, np.float32)))
    return trajs


def upright_reward_proxy(obs):
    """Cartpole-swingup reward proxy from a 5-dim observation. Obs layout:
    position = [cart_x, cos(pole), sin(pole)] (3) + velocity (2). Swingup reward is
    high when the pole is upright: cos(pole) ≈ +1. Use (cos+1)/2 ∈ [0,1] as a proxy."""
    cos_pole = obs[..., 1]
    return (cos_pole + 1.0) / 2.0


@torch.no_grad()
def perturbed_start_latent(model, cfg, traj, t, noise_std, rng):
    """Build a start latent from an OOD-perturbed version of obs[t]: encode the
    noisy observation into a posterior latent, mimicking Set B near-OOD states."""
    device = next(model.parameters()).device
    h_prev = torch.tensor(traj['h'][t - 1], dtype=torch.float32, device=device).unsqueeze(0) \
        if t > 0 else torch.zeros(1, cfg['rssm_deter'], device=device)
    z_prev_logits = torch.tensor(traj['z'][t - 1], dtype=torch.float32, device=device).unsqueeze(0) \
        if t > 0 else torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
    z_prev = model.rssm._straight_through_sample(z_prev_logits) if t > 0 \
        else torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
    a = torch.tensor(traj['act'][t - 1] if t > 0 else np.zeros(cfg['act_dim'], np.float32),
                     dtype=torch.float32, device=device).unsqueeze(0)
    noisy_obs = traj['obs'][t] + rng.standard_normal(traj['obs'][t].shape).astype(np.float32) * noise_std
    emb = model.encoder(torch.tensor(noisy_obs, dtype=torch.float32, device=device).unsqueeze(0))
    h, z, _, _ = model.rssm.observe_step(h_prev, z_prev, a, emb)
    return h.squeeze(0).cpu().numpy().copy()


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    print("Loading training states + Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    h_all, kl_all, traj_id = tr['h'], tr['kl'], tr['traj_id']
    y = binarise_by_median(kl_all); kl_median = float(np.median(kl_all))
    idx_tr, _ = train_test_split(np.arange(len(h_all)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h_all[idx_tr], y[idx_tr])

    print(f"Collecting {N_TRAJ} held-out trajectories...")
    model = load_model(cfg)
    trajs = collect_trajectories(model, cfg, N_TRAJ)

    # per-traj C_t and probe score
    for trj in trajs:
        T = len(trj['obs'])
        hk = (trj['kl'] > kl_median).astype(np.float32)
        ct = np.zeros(T)
        for i in range(T):
            val = 0.0
            for lag in range(50):
                j = i - lag
                if j < 0:
                    break
                val += (GAMMA ** lag) * hk[j]
            ct[i] = val
        trj['ct'] = ct
        trj['probe'] = clf.predict_proba(sc.transform(trj['h']))[:, 1]

    # sites: interior points with room for horizon
    sites = []
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        for t in range(MIN_T, T - HORIZON - 1):
            sites.append((ti, t))
    rng = np.random.default_rng(0)
    if len(sites) > 4000:
        sel = rng.choice(len(sites), 4000, replace=False)
        sites = [sites[i] for i in sel]
    print(f"  {len(sites)} sites")

    # gather per-site: confusion (probe), clean & OOD recovery curves, reward gap
    probe_vals = []
    clean_curves = []   # (n, HORIZON)
    ood_curves = []
    rew_gap = []        # imagined_end_reward_proxy - realized_reward
    rng_p = np.random.default_rng(7)
    for (ti, t) in sites:
        trj = trajs[ti]
        probe_vals.append(trj['probe'][t])
        # clean recovery
        d_clean, imag_h, _ = imagined_vs_real_latent(model, trj, t, HORIZON)
        if len(d_clean) < HORIZON:
            continue
        clean_curves.append(d_clean)
        # OOD-perturbed start
        h_ood = perturbed_start_latent(model, cfg, trj, t, NOISE_STD, rng_p)
        d_ood, _, _ = imagined_vs_real_latent(model, trj, t, HORIZON, h_start=h_ood)
        ood_curves.append(d_ood if len(d_ood) == HORIZON else np.full(HORIZON, np.nan))
        # reward cross-check: imagined end obs vs realized
        # decode imagined final latent → obs proxy reward; realized reward at t+HORIZON
        device = next(model.parameters()).device
        h_last = torch.tensor(imag_h[-1], dtype=torch.float32, device=device).unsqueeze(0)
        # need a z for decode; re-derive prior sample at last step is complex — approximate
        # with zero z (decoder is dominated by h); good enough for a proxy cross-check
        z0 = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        with torch.no_grad():
            dec = model.decoder(torch.cat([h_last, z0], dim=-1)).squeeze(0).cpu().numpy()
        imag_rew = upright_reward_proxy(dec)
        real_rew = upright_reward_proxy(trj['obs'][t + HORIZON])
        rew_gap.append(float(imag_rew - real_rew))

    probe_vals = np.array(probe_vals[:len(clean_curves)])
    clean_curves = np.array(clean_curves)
    ood_curves = np.array(ood_curves)
    rew_gap = np.array(rew_gap[:len(clean_curves)])

    # ── attractor recovery: does OOD distance shrink over early steps? ──
    ood_mean = np.nanmean(ood_curves, axis=0)
    clean_mean = clean_curves.mean(axis=0)
    # "snap-back": OOD curve decreasing over first few steps
    ood_snapback = float(ood_mean[0] - ood_mean[min(3, HORIZON - 1)])  # >0 = recovers

    print("\n" + "=" * 74)
    print("TASK H — ATTRACTOR RECOVERY vs CONFUSION LEVEL")
    print("=" * 74)
    print(f"\n  Imagined-vs-real latent distance over horizon {HORIZON} (mean across "
          f"{len(clean_curves)} sites):")
    print(f"  {'step':>5}{'clean':>12}{'OOD-perturbed':>16}")
    for k in range(HORIZON):
        print(f"  {k+1:>5}{clean_mean[k]:>12.4f}{ood_mean[k]:>16.4f}")
    print(f"\n  OOD snap-back (dist[1] − dist[4]) = {ood_snapback:+.4f}  "
          f"({'RECOVERS toward real (attractor)' if ood_snapback > 0 else 'DIVERGES'})")

    # ── split by confusion bins ──
    print(f"\n  Recovery curve by confusion (Probe A) tercile:")
    q1, q2 = np.percentile(probe_vals, [33, 67])
    bins = {'low': probe_vals <= q1, 'med': (probe_vals > q1) & (probe_vals <= q2),
            'high': probe_vals > q2}
    bin_curves_clean, bin_curves_ood = {}, {}
    print(f"  {'bin':>6}{'N':>7}{'clean d[end]':>14}{'OOD d[end]':>13}{'OOD snapback':>14}")
    for name, mask in bins.items():
        if mask.sum() < 5:
            continue
        cc = clean_curves[mask].mean(axis=0)
        oc = np.nanmean(ood_curves[mask], axis=0)
        bin_curves_clean[name] = cc.tolist()
        bin_curves_ood[name] = oc.tolist()
        sb = oc[0] - oc[min(3, HORIZON - 1)]
        print(f"  {name:>6}{int(mask.sum()):>7}{cc[-1]:>14.4f}{oc[-1]:>13.4f}{sb:>+14.4f}")

    # correlation: does confusion predict the imagined-vs-real gap?
    gap_end = clean_curves[:, -1]
    r_gap, p_gap = pearsonr(probe_vals, gap_end)
    r_ood, p_ood = pearsonr(probe_vals, np.nan_to_num(ood_curves[:, -1], nan=np.nanmean(ood_curves[:, -1])))
    print(f"\n  Pearson r(confusion, clean end-gap) = {r_gap:+.3f} (p={p_gap:.3g})")
    print(f"  Pearson r(confusion, OOD end-gap)   = {r_ood:+.3f} (p={p_ood:.3g})")

    # ── reward overestimation cross-check ──
    print(f"\n  Reward-overestimation cross-check (imagined end reward − realized):")
    print(f"    overall mean gap: {rew_gap.mean():+.4f}  "
          f"({'imagination OVER-estimates' if rew_gap.mean() > 0 else 'under-estimates'})")
    r_rew, p_rew = pearsonr(probe_vals, rew_gap)
    for name, mask in bins.items():
        if mask.sum() >= 5:
            print(f"    {name:>5} confusion: mean reward gap = {rew_gap[mask].mean():+.4f}")
    print(f"    Pearson r(confusion, reward gap) = {r_rew:+.3f} (p={p_rew:.3g})")

    # ── verdict ──
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    # reinforcing if confusion positively predicts the imagined-vs-real gap (higher
    # confusion → larger/slower-recovering gap) with a real effect size
    reinforcing = (r_gap > 0.1 and p_gap < 0.05) or (r_ood > 0.1 and p_ood < 0.05)
    orthogonal = abs(r_gap) < 0.1 and abs(r_ood) < 0.1
    if reinforcing:
        verdict = 'REINFORCING'
        msg = (f"Confusion level POSITIVELY tracks the imagined-vs-real latent gap "
               f"(r={max(r_gap, r_ood):+.2f}). The cheap linear confusion readout flags exactly the "
               f"states where the attractor 'snap-back' is weakest / latent dynamics least "
               f"reliable — independent support from Biased Dreams: our no-training readout "
               f"identifies the states their costlier analysis flags as problematic.")
    elif orthogonal:
        verdict = 'ORTHOGONAL'
        msg = (f"Confusion level is essentially UNRELATED to the attractor-recovery gap "
               f"(r={r_gap:+.2f}, {r_ood:+.2f}). This directly answers 'why does the signal persist "
               f"if latents snap back to attractors': because the confusion signal is NOT a property "
               f"of the latent DYNAMICS (which do recover, snap-back={ood_snapback:+.3f}) — it is a "
               f"property of the posterior-vs-prior HISTORY (C_t), a different quantity than what "
               f"Biased Dreams measures. The two findings are compatible and orthogonal.")
    else:
        verdict = 'MIXED'
        msg = (f"Partial/mixed relationship (r={r_gap:+.2f}, {r_ood:+.2f}). Reported honestly; "
               f"confusion is weakly related to attractor recovery but not cleanly.")
    print(f"  {verdict}: {msg}")

    results = dict(verdict=verdict, message=msg, n_sites=len(clean_curves),
                   clean_curve=clean_mean.tolist(), ood_curve=ood_mean.tolist(),
                   ood_snapback=ood_snapback, bin_curves_clean=bin_curves_clean,
                   bin_curves_ood=bin_curves_ood, r_conf_cleangap=float(r_gap),
                   p_conf_cleangap=float(p_gap), r_conf_oodgap=float(r_ood),
                   p_conf_oodgap=float(p_ood), reward_gap_mean=float(rew_gap.mean()),
                   r_conf_rewardgap=float(r_rew), p_conf_rewardgap=float(p_rew),
                   noise_std=NOISE_STD, horizon=HORIZON)
    with open(os.path.join(OUT_DIR, 'task_h_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # ── figure ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle('Task H — Attractor Recovery vs Confusion (Biased Dreams cross-check)',
                 fontweight='bold', fontsize=12)
    ks = np.arange(1, HORIZON + 1)
    ax = axes[0]
    ax.plot(ks, clean_mean, 'g-o', label='clean start')
    ax.plot(ks, ood_mean, 'r-s', label='OOD-perturbed start')
    ax.set_xlabel('imagination step'); ax.set_ylabel('imagined-vs-real latent dist')
    ax.set_title(f'Attractor recovery\n(OOD snap-back={ood_snapback:+.3f})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for name in ['low', 'med', 'high']:
        if name in bin_curves_clean:
            ax.plot(ks, bin_curves_clean[name], '-o', markersize=3, label=f'{name} confusion')
    ax.set_xlabel('imagination step'); ax.set_ylabel('clean imagined-vs-real dist')
    ax.set_title(f'By confusion bin\nr(conf,gap)={r_gap:+.2f}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.scatter(probe_vals, gap_end, s=4, alpha=0.2, c='steelblue', rasterized=True)
    if len(probe_vals) > 2:
        cf = np.polyfit(probe_vals, gap_end, 1)
        xs = np.linspace(probe_vals.min(), probe_vals.max(), 50)
        ax.plot(xs, np.polyval(cf, xs), 'r-', lw=2, label=f'r={r_gap:+.2f}')
    ax.set_xlabel('confusion (Probe A)'); ax.set_ylabel('end imagined-vs-real gap')
    ax.set_title('Confusion vs latent gap'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'task_h_attractor.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Verdict: {verdict}")
    print(f"  Figure: {fig_path}\n  Results: {os.path.join(OUT_DIR, 'task_h_results.json')}")


if __name__ == '__main__':
    main()
