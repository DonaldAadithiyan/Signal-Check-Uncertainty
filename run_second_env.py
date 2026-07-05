#!/usr/bin/env python3.11
"""
Task D — Generalise the whole pipeline to a structurally different environment.

Second environment: dm_control reacher/easy — a 2-link arm reaching a target.
Genuinely different dynamics and a different observation dimensionality (6 vs
cartpole's 5) and action dimensionality (2 vs 1). This bounds the paper's
generality claim beyond the cartpole family, which the target workshop cares about.

Runs the ENTIRE Phase-1 pipeline independently on reacher:
  * train an XS world model from scratch (obs_dim=6, act_dim=2)
  * Set A (ID), Set B (near-OOD noisy obs), Set C (KL-matched contrastive)
  * within-task confound control (reacher-only C1/C2, should be ~chance)
  * Probe A / Probe C / z_t probe
  * C_t characterisation (does γ≈0.95 replicate? report the best γ either way)
  * null-space PCA-angle geometry (is the ~88°/tiny-frac finding cartpole-specific?)
  * block/quarter analysis
Produces a side-by-side comparison table cartpole-swingup vs reacher-easy.

CPU-only, XS scale. Resumable via saved checkpoint/states.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.training.replay_buffer import EpisodeReplayBuffer
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import compute_ct, bootstrap_auroc_ci

DOMAIN, TASK      = 'reacher', 'easy'
OOD_TASK          = 'hard'          # different-task states for the strong contrastive
ENV_LABEL         = f'{DOMAIN}_{TASK}'
GAMMAS            = [0.70, 0.80, 0.90, 0.95, 0.99]
MAX_LAG           = 50
N_EP              = 20
OUT_DIR           = 'outputs/second_env'
CKPT              = os.path.join(OUT_DIR, f'{ENV_LABEL}_world_model.pt')
STATES            = os.path.join(OUT_DIR, f'{ENV_LABEL}_training_states.npz')


# ─── training (env-parametrised copy of the cartpole trainer) ────────────────

def train_on_env(cfg, env_factory, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(cfg.get('device', 'cpu'))
    env = env_factory(seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len, batch, warmup, max_steps = cfg['seq_len'], cfg['batch_size'], cfg['warmup_steps'], cfg['total_env_steps']

    log_h, log_z, log_kl, log_recon, log_traj = [], [], [], [], []
    step_count, traj_id, t0 = 0, 0, time.time()
    ep_obs, ep_act = [], []
    h_inf = torch.zeros(1, cfg['rssm_deter'], device=device)
    z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
    obs = env.reset(); ep_obs.append(obs.copy())
    print(f"[train {ENV_LABEL}] obs_dim={env.obs_dim} act_dim={env.act_dim} steps={max_steps:,}", flush=True)

    while step_count < max_steps:
        action = np.random.uniform(-1, 1, size=(env.act_dim,)).astype(np.float32)
        model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            embed = model.encoder(obs_t)
            h_inf, z_inf, prior_l, post_l = model.rssm.observe_step(h_inf, z_inf, a_t, embed)
            decoded = model.decoder(torch.cat([h_inf, z_inf], dim=-1))
            kl_val = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            recon_val = F.mse_loss(decoded, obs_t, reduction='none').sum().item()
        log_h.append(h_inf.squeeze(0).cpu().numpy().copy())
        log_z.append(post_l.squeeze(0).cpu().numpy().copy())
        log_kl.append(kl_val); log_recon.append(recon_val); log_traj.append(traj_id)

        obs_new, _, done = env.step(action)
        ep_act.append(action.copy()); step_count += 1
        if done or (len(ep_act) >= cfg['episode_max_steps']):
            ep_obs.append(obs_new.copy())
            buffer.add_episode(ep_obs[:-1], ep_act)
            traj_id += 1; ep_obs, ep_act = [], []
            h_inf = torch.zeros(1, cfg['rssm_deter'], device=device)
            z_inf = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
            obs = env.reset(); ep_obs.append(obs.copy())
        else:
            obs = obs_new; ep_obs.append(obs.copy())

        if step_count >= warmup and len(buffer) >= seq_len * batch:
            model.train()
            obs_b, act_b = buffer.sample(batch, seq_len, device=str(device))
            loss, _, _ = model.compute_loss(obs_b, act_b, kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip']); optim.step()
        if step_count % 10000 == 0:
            print(f"  step {step_count:,}/{max_steps:,}  kl={np.mean(log_kl[-500:]):.3f}  "
                  f"elapsed={(time.time()-t0)/60:.1f}m  traj={traj_id}", flush=True)

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'cfg': cfg,
                'obs_dim': env.obs_dim, 'act_dim': env.act_dim}, CKPT)
    states = dict(h=np.array(log_h, np.float32), z=np.array(log_z, np.float32),
                  kl=np.array(log_kl, np.float32), recon=np.array(log_recon, np.float32),
                  traj_id=np.array(log_traj, np.int64))
    np.savez(STATES, **states)
    print(f"[train {ENV_LABEL}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return model, states


def collect_env(model, env, n_ep, cfg):
    device = next(model.parameters()).device; model.eval()
    H, Z, KL, RC, OB = [], [], [], [], []
    for ep in range(n_ep):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        done, step = False, 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = np.random.uniform(-1, 1, size=(env.act_dim,)).astype(np.float32)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
                embed = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                dec = model.decoder(torch.cat([h, z], dim=-1))
                kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                rc = F.mse_loss(dec, obs_t, reduction='none').sum().item()
                H.append(h.squeeze(0).cpu().numpy().copy()); Z.append(post_l.squeeze(0).cpu().numpy().copy())
                KL.append(kl); RC.append(rc); OB.append(obs.copy())
                obs, _, done = env.step(a); step += 1
    return dict(h=np.array(H, np.float32), z=np.array(Z, np.float32), kl=np.array(KL, np.float32),
                recon=np.array(RC, np.float32), obs=np.array(OB, np.float32))


def build_set_c(set_a, set_b, n_bins=10, per_bin=20, max_total=200, seed=42):
    ah, az = np.concatenate([set_a['h'], set_b['h']]), np.concatenate([set_a['z'], set_b['z']])
    akl, arc = np.concatenate([set_a['kl'], set_b['kl']]), np.concatenate([set_a['recon'], set_b['recon']])
    edges = np.percentile(akl, np.linspace(0, 100, n_bins + 1)); bi = np.digitize(akl, edges[1:-1])
    rng = np.random.default_rng(seed); c1, c2 = [], []
    for b in range(n_bins):
        idx = np.where(bi == b)[0]
        if len(idx) < 4: continue
        rb = arc[idx]; c1c = idx[rb <= np.percentile(rb, 25)]; c2c = idx[rb >= np.percentile(rb, 75)]
        n = min(per_bin, len(c1c), len(c2c))
        if n == 0: continue
        c1.extend(rng.choice(c1c, n, replace=False).tolist()); c2.extend(rng.choice(c2c, n, replace=False).tolist())
    if len(c1) > max_total: c1 = rng.choice(c1, max_total, replace=False).tolist()
    if len(c2) > max_total: c2 = rng.choice(c2, max_total, replace=False).tolist()
    return dict(h=np.concatenate([ah[c1], ah[c2]]), z=np.concatenate([az[c1], az[c2]]),
                labels=np.array([0]*len(c1)+[1]*len(c2), np.int32))


def null_space_geometry(h, kl, top_k=10):
    y = binarise_by_median(kl)
    tr_idx, _ = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h[tr_idx], y[tr_idx])
    h_sc = sc.transform(h)
    pca = PCA(n_components=top_k, random_state=0).fit(h_sc)
    w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    angles = [np.degrees(np.arccos(np.clip(abs(np.dot(w, pca.components_[k])), 0, 1))) for k in range(top_k)]
    frac_in_topk = float(np.sum((pca.components_ @ w) ** 2))
    return float(np.mean(angles)), frac_in_topk


def main():
    cfg = XS_CONFIG.copy()
    # reacher has 6-dim obs / 2-dim act — override the cartpole defaults
    cfg['obs_dim'], cfg['act_dim'] = 6, 2
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── train or load ──
    if os.path.exists(CKPT) and os.path.exists(STATES):
        print(f"[{ENV_LABEL}] loading existing checkpoint + states")
        ck = torch.load(CKPT, map_location='cpu')
        model = WorldModel(ck['obs_dim'], ck['act_dim'], ck['cfg'])
        model.load_state_dict(ck['model_state']); model.eval()
        states = dict(np.load(STATES))
    else:
        model, states = train_on_env(cfg, lambda s: DMCEnv(DOMAIN, TASK, seed=s), seed=0)

    h_all, z_all, kl_all, recon_all, traj_id = (states['h'], states['z'], states['kl'],
                                                states['recon'], states['traj_id'])
    N = len(h_all)
    print(f"[{ENV_LABEL}] {N:,} training states  mean KL={kl_all.mean():.3f}")

    # ── eval sets ──
    print(f"[{ENV_LABEL}] collecting eval sets...")
    np.random.seed(42)
    set_a = collect_env(model, DMCEnv(DOMAIN, TASK, noisy=False, seed=100), N_EP, cfg)
    set_b = collect_env(model, DMCEnv(DOMAIN, TASK, noisy=True, noise_std=cfg['noise_std'], seed=200), N_EP, cfg)
    set_c = build_set_c(set_a, set_b)
    # within-task confound control: split set_a itself by recon (same task identity)
    set_c_within = _within_task(set_a)

    # ── probes ──
    y_kl = binarise_by_median(kl_all); kl_median = float(np.median(kl_all))
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y_kl, random_state=0)
    clf_a, sc_a = train_probe(h_all[tr_idx], y_kl[tr_idx])

    def elab(kl_arr):
        y = (kl_arr > kl_median).astype(np.int32)
        return y if len(np.unique(y)) == 2 else binarise_by_median(kl_arr)

    auroc_id = auroc(clf_a, sc_a, h_all[te_idx], y_kl[te_idx])
    auroc_a = auroc(clf_a, sc_a, set_a['h'], elab(set_a['kl']))
    auroc_b = auroc(clf_a, sc_a, set_b['h'], elab(set_b['kl']))
    auroc_c = auroc(clf_a, sc_a, set_c['h'], set_c['labels'])
    auroc_within = auroc(clf_a, sc_a, set_c_within['h'], set_c_within['labels'])

    yc = binarise_by_median(recon_all)
    clf_c, sc_c = train_probe(h_all[tr_idx], yc[tr_idx])
    auroc_probec = auroc(clf_c, sc_c, set_c['h'], set_c['labels'])
    clf_z, sc_z = train_probe(z_all[tr_idx], y_kl[tr_idx])
    auroc_zt = auroc(clf_z, sc_z, set_c['z'], set_c['labels'])

    # block/quarter Set C AUROC
    q = h_all.shape[1] // 4; block_c = []
    for i in range(4):
        sl = slice(i * q, (i + 1) * q)
        cq, scq = train_probe(h_all[tr_idx][:, sl], y_kl[tr_idx])
        block_c.append(float(auroc(cq, scq, set_c['h'][:, sl], set_c['labels'])))

    # C_t characterisation
    probe_te = clf_a.predict_proba(sc_a.transform(h_all[te_idx]))[:, 1]
    kl_te = kl_all[te_idx]
    r2_kl = r2_score(probe_te, LinearRegression().fit(kl_te.reshape(-1,1), probe_te).predict(kl_te.reshape(-1,1)))
    best_r2, best_g, ct_r2 = -1, None, {}
    for g in GAMMAS:
        ct = compute_ct(kl_all, traj_id, gamma=g, max_lag=MAX_LAG, kl_median=kl_median)[te_idx]
        r2 = r2_score(probe_te, LinearRegression().fit(ct.reshape(-1,1), probe_te).predict(ct.reshape(-1,1)))
        ct_r2[g] = float(r2)
        if r2 > best_r2: best_r2, best_g = r2, g

    # null-space geometry
    angle, frac_topk = null_space_geometry(h_all, kl_all)

    # bootstrap CI for headline Set C
    scores_c = clf_a.predict_proba(sc_a.transform(set_c['h']))[:, 1]
    ci_c = bootstrap_auroc_ci(set_c['labels'], scores_c, seed=0)

    res = dict(env=ENV_LABEL, obs_dim=int(cfg['obs_dim']), act_dim=int(cfg['act_dim']),
               n_states=int(N), auroc_id=float(auroc_id), auroc_a=float(auroc_a),
               auroc_b=float(auroc_b), auroc_c=float(auroc_c), ci_setc=list(ci_c),
               auroc_within_task=float(auroc_within), auroc_probeC=float(auroc_probec),
               auroc_zt=float(auroc_zt), block_c=block_c, r2_kl_baseline=float(r2_kl),
               ct_r2_by_gamma=ct_r2, best_gamma=float(best_g), best_ct_r2=float(best_r2),
               nullspace_angle=float(angle), frac_in_top10=float(frac_topk))
    with open(os.path.join(OUT_DIR, f'{ENV_LABEL}_results.json'), 'w') as f:
        json.dump(res, f, indent=2)

    # ── side-by-side table vs cartpole (from DEV_LOG headline numbers) ──
    cart = dict(auroc_id=0.9019, auroc_a=0.8632, auroc_b=0.8464, auroc_c=0.7227,
                auroc_within=0.5060, auroc_zt=None, best_gamma=0.95, best_ct_r2=0.798,
                nullspace_angle=88.0, frac_in_top10=0.09, obs_dim=5, act_dim=1)
    print("\n" + "=" * 78)
    print("TASK D — SECOND-ENVIRONMENT GENERALISATION (side-by-side)")
    print("=" * 78)
    print(f"\n  {'Metric':<32}{'cartpole-swingup':>20}{'reacher-easy':>20}")
    print(f"  {'-'*32}{'-'*20}{'-'*20}")
    def row(label, ck, rv, fmt='{:.4f}'):
        cs = fmt.format(ck) if ck is not None else 'n/a'
        rs = fmt.format(rv) if rv is not None else 'n/a'
        print(f"  {label:<32}{cs:>20}{rs:>20}")
    row('obs_dim', cart['obs_dim'], res['obs_dim'], '{:d}')
    row('act_dim', cart['act_dim'], res['act_dim'], '{:d}')
    row('Probe A held-out AUROC', cart['auroc_id'], res['auroc_id'])
    row('Probe A Set A AUROC', cart['auroc_a'], res['auroc_a'])
    row('Probe A Set B AUROC', cart['auroc_b'], res['auroc_b'])
    row('Probe A Set C AUROC (headline)', cart['auroc_c'], res['auroc_c'])
    print(f"  {'  Set C 95% CI':<32}{'—':>20}{'[%.3f,%.3f]'%(ci_c[1],ci_c[2]):>20}")
    row('Within-task confound AUROC', cart['auroc_within'], res['auroc_within_task'])
    row('C_t best γ', cart['best_gamma'], res['best_gamma'], '{:.2f}')
    row('C_t best R²', cart['best_ct_r2'], res['best_ct_r2'])
    row('Null-space angle (°)', cart['nullspace_angle'], res['nullspace_angle'], '{:.1f}')
    row('Frac probe dir in top-10 PC', cart['frac_in_top10'], res['frac_in_top10'])

    print("\n  Interpretation:")
    replicates = (res['auroc_c'] > 0.60 and res['nullspace_angle'] > 80 and res['frac_in_top10'] < 0.2)
    if replicates:
        print(f"    Confusion signal REPLICATES on a structurally different env: Set C AUROC="
              f"{res['auroc_c']:.3f} (CI [{ci_c[1]:.3f},{ci_c[2]:.3f}]), null-space geometry "
              f"present (angle {res['nullspace_angle']:.1f}°, {res['frac_in_top10']*100:.1f}% in top-10 PC).")
        if abs(res['best_gamma'] - 0.95) < 1e-6:
            print(f"    Effective memory γ={res['best_gamma']} matches cartpole (0.95).")
        else:
            print(f"    Effective memory γ={res['best_gamma']} (cartpole 0.95) — memory length differs, "
                  f"which is itself informative.")
    else:
        print(f"    PARTIAL / NON-replication (Set C AUROC={res['auroc_c']:.3f}, angle="
              f"{res['nullspace_angle']:.1f}°). Reported honestly — sharpens the claim: the finding")
        print(f"    is architecture-general on cartpole but does not fully transfer to reacher dynamics.")

    print(f"\n  Results saved: {os.path.join(OUT_DIR, ENV_LABEL+'_results.json')}")


def _within_task(set_a, n_bins=10, per_bin=20, max_total=200, seed=13):
    """C1/C2 both from the SAME reacher task (low vs high recon within KL bins).
    Same task identity → the confound control; should be ~chance if the probe
    reads genuine confusion rather than task identity."""
    h, z, kl, rc = set_a['h'], set_a['z'], set_a['kl'], set_a['recon']
    edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1)); bi = np.digitize(kl, edges[1:-1])
    rng = np.random.default_rng(seed); c1, c2 = [], []
    for b in range(n_bins):
        idx = np.where(bi == b)[0]
        if len(idx) < 4: continue
        rb = rc[idx]; c1c = idx[rb <= np.percentile(rb, 25)]; c2c = idx[rb >= np.percentile(rb, 75)]
        n = min(per_bin, len(c1c), len(c2c))
        if n == 0: continue
        c1.extend(rng.choice(c1c, n, replace=False).tolist()); c2.extend(rng.choice(c2c, n, replace=False).tolist())
    if len(c1) > max_total: c1 = rng.choice(c1, max_total, replace=False).tolist()
    if len(c2) > max_total: c2 = rng.choice(c2, max_total, replace=False).tolist()
    return dict(h=np.concatenate([h[c1], h[c2]]), z=np.concatenate([z[c1], z[c2]]),
                labels=np.array([0]*len(c1)+[1]*len(c2), np.int32))


if __name__ == '__main__':
    main()
