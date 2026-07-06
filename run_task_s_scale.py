#!/usr/bin/env python3.11
"""
Task S (optional) — A partial scale check.

"The scale question is open" is the paper's most-repeated caveat and the only one with
zero empirical evidence. This cannot resolve it (200M params is out of scope), but one
intermediate data point beats none: train a single wider-GRU cartpole model (deter=512,
2× the XS 256) at the same step budget, and check whether the two most load-bearing
findings survive:
  1. Null-space geometry — is the confusion direction still near-orthogonal to the top PCs?
  2. Core causal ablation — does ablating the confusion direction still collapse the probe
     readout relative to a random-direction control? (LIGHT version: single random control
     over held-out sites, not the full 50-direction null / cross-seed protocol of Task A/G.)

Reported as ONE additional data point, explicitly scoped as partial evidence, not a
resolution of the scale question.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.training.replay_buffer import EpisodeReplayBuffer
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import probe_direction, random_matched_direction

DETER = 512                 # 2× the XS 256-dim GRU
OUT_DIR = 'outputs/scale'
CKPT = os.path.join(OUT_DIR, f'cartpole_deter{DETER}.pt')
STATES = os.path.join(OUT_DIR, f'cartpole_deter{DETER}_states.npz')
LOOKAHEAD = [0, 1, 5, 10]


def train(cfg, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    env = CartpoleEnv(task='swingup', seed=seed)
    model = WorldModel(env.obs_dim, env.act_dim, cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len, batch, warmup, max_steps = cfg['seq_len'], cfg['batch_size'], cfg['warmup_steps'], cfg['total_env_steps']
    log_h, log_z, log_kl, log_recon, log_traj = [], [], [], [], []
    step, traj, t0 = 0, 0, time.time()
    ep_obs, ep_act = [], []
    h = torch.zeros(1, cfg['rssm_deter']); z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
    obs = env.reset(); ep_obs.append(obs.copy())
    print(f"[scale deter={cfg['rssm_deter']}] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    while step < max_steps:
        a = np.random.uniform(-1, 1, (env.act_dim,)).astype(np.float32)
        model.eval()
        with torch.no_grad():
            ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0); at = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            emb = model.encoder(ot)
            h, z, prior_l, post_l = model.rssm.observe_step(h, z, at, emb)
            dec = model.decoder(torch.cat([h, z], dim=-1))
            klv = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            rcv = F.mse_loss(dec, ot, reduction='none').sum().item()
        log_h.append(h.squeeze(0).numpy().copy()); log_z.append(post_l.squeeze(0).numpy().copy())
        log_kl.append(klv); log_recon.append(rcv); log_traj.append(traj)
        obs_new, _, done = env.step(a); ep_act.append(a.copy()); step += 1
        if done or len(ep_act) >= cfg['episode_max_steps']:
            ep_obs.append(obs_new.copy()); buffer.add_episode(ep_obs[:-1], ep_act)
            traj += 1; ep_obs, ep_act = [], []
            h = torch.zeros(1, cfg['rssm_deter']); z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
            obs = env.reset(); ep_obs.append(obs.copy())
        else:
            obs = obs_new; ep_obs.append(obs.copy())
        if step >= warmup and len(buffer) >= seq_len * batch:
            model.train()
            ob, ab = buffer.sample(batch, seq_len, device='cpu')
            loss, _, _ = model.compute_loss(ob, ab, kl_free=cfg['kl_free'], kl_scale=cfg['kl_scale'])
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip']); optim.step()
        if step % 20000 == 0:
            print(f"  step {step:,}/{max_steps:,} kl={np.mean(log_kl[-500:]):.2f} {(time.time()-t0)/60:.0f}m", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'obs_dim': env.obs_dim, 'act_dim': env.act_dim}, CKPT)
    states = dict(h=np.array(log_h, np.float32), z=np.array(log_z, np.float32),
                  kl=np.array(log_kl, np.float32), recon=np.array(log_recon, np.float32),
                  traj_id=np.array(log_traj, np.int64))
    np.savez(STATES, **states)
    return model, states


def collect_traj(model, cfg, n_traj=40, seed=777):
    env = CartpoleEnv(task='swingup', noisy=False, seed=seed); np.random.seed(seed)
    trajs = []
    for ep in range(n_traj):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter']); z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
        O, A, H, kl_l = [], [], [], []
        done, step = False, 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = np.random.uniform(-1, 1, (cfg['act_dim'],)).astype(np.float32)
                ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0); at = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
                emb = model.encoder(ot)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, at, emb)
                kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                O.append(obs.copy()); A.append(a.copy()); H.append(h.squeeze(0).numpy().copy()); kl_l.append(kl)
                obs, _, done = env.step(a); step += 1
        trajs.append(dict(obs=np.array(O, np.float32), act=np.array(A, np.float32),
                          h=np.array(H, np.float32), kl=np.array(kl_l, np.float32)))
    return trajs


@torch.no_grad()
def continue_probe(model, cfg, traj, t, h_new, clf, sc):
    T = len(traj['obs']); t_end = min(T, t + max(LOOKAHEAD) + 1)
    h = torch.tensor(h_new, dtype=torch.float32).unsqueeze(0)
    ot = torch.tensor(traj['obs'][t], dtype=torch.float32).unsqueeze(0)
    emb = model.encoder(ot)
    post_l = model.rssm.post_net(torch.cat([h, emb], dim=-1))
    z = model.rssm._straight_through_sample(post_l)
    hs = [h.squeeze(0).numpy().copy()]
    for k in range(t + 1, t_end):
        at = torch.tensor(traj['act'][k - 1], dtype=torch.float32).unsqueeze(0)
        ok = torch.tensor(traj['obs'][k], dtype=torch.float32).unsqueeze(0)
        emb = model.encoder(ok)
        h, z, _, _ = model.rssm.observe_step(h, z, at, emb)
        hs.append(h.squeeze(0).numpy().copy())
    return clf.predict_proba(sc.transform(np.array(hs, np.float32)))[:, 1]


def main():
    cfg = XS_CONFIG.copy()
    cfg['rssm_deter'] = DETER
    cfg['rssm_hidden'] = DETER
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(CKPT) and os.path.exists(STATES):
        ck = torch.load(CKPT, map_location='cpu')
        model = WorldModel(ck['obs_dim'], ck['act_dim'], ck['cfg']); model.load_state_dict(ck['model_state']); model.eval()
        states = dict(np.load(STATES))
    else:
        model, states = train(cfg, seed=0)

    h, kl = states['h'], states['kl']
    N = len(h); y = binarise_by_median(kl)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h[tr_idx], y[tr_idx])
    auroc_id = auroc(clf, sc, h[te_idx], y[te_idx])

    print("\n" + "=" * 74)
    print(f"TASK S — PARTIAL SCALE CHECK (cartpole, GRU deter={DETER} vs XS 256)")
    print("=" * 74)
    print(f"\n  Probe A held-out AUROC: {auroc_id:.4f}  (XS: 0.9019)")

    # ── (1) null-space geometry ──
    v = probe_direction(clf, sc)
    h_sc = sc.transform(h)
    pca = PCA(n_components=10, random_state=0).fit(h_sc)
    w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    angles = [np.degrees(np.arccos(np.clip(abs(np.dot(w, pca.components_[k])), 0, 1))) for k in range(10)]
    frac_top10 = float(np.sum((pca.components_ @ w) ** 2))
    mean_angle = float(np.mean(angles))
    print(f"\n  [1] Null-space geometry:")
    print(f"      mean angle to top-10 PCs: {mean_angle:.1f}°  (XS: ~88°)")
    print(f"      frac probe variance in top-10 PC: {frac_top10:.4f}  (XS: ~0.005 top-10)")
    geom_holds = mean_angle > 80 and frac_top10 < 0.1

    # ── (2) light causal ablation (confusion dir vs 1 random control) ──
    print(f"\n  [2] Causal ablation (confusion dir vs single random control), 300 sites:")
    trajs = collect_traj(model, cfg)
    kl_median = float(np.median(kl))
    sites, rng = [], np.random.default_rng(0)
    for ti, trj in enumerate(trajs):
        T = len(trj['obs'])
        if T < 12 + max(LOOKAHEAD) + 1:
            continue
        valid = np.arange(12, T - max(LOOKAHEAD) - 1)
        for t in rng.choice(valid, size=min(8, len(valid)), replace=False):
            sites.append((ti, int(t)))
    v_rand = random_matched_direction(rng, v.shape[0])
    dconf = {k: [] for k in LOOKAHEAD}; drand = {k: [] for k in LOOKAHEAD}
    for (ti, t) in sites:
        trj = trajs[ti]; h_t = trj['h'][t]
        base = continue_probe(model, cfg, trj, t, h_t, clf, sc)
        ac = continue_probe(model, cfg, trj, t, h_t - float(h_t @ v) * v, clf, sc)
        ar = continue_probe(model, cfg, trj, t, h_t - float(h_t @ v_rand) * v_rand, clf, sc)
        for k in LOOKAHEAD:
            if k < len(base):
                dconf[k].append(ac[k] - base[k]); drand[k].append(ar[k] - base[k])
    print(f"      {'k':>4}{'Δprobe confusion':>20}{'Δprobe random':>18}")
    abl = {}
    for k in LOOKAHEAD:
        mc, mr = float(np.mean(dconf[k])), float(np.mean(drand[k]))
        abl[k] = dict(confusion=mc, random=mr)
        print(f"      {k:>4}{mc:>+20.4f}{mr:>+18.4f}")
    causal_holds = abs(abl[0]['confusion']) > 5 * abs(abl[0]['random']) and abl[0]['confusion'] < -0.1

    # ── verdict ──
    print("\n" + "-" * 74)
    print(f"  Null-space geometry survives at deter={DETER}: {'YES' if geom_holds else 'NO'} "
          f"(angle {mean_angle:.1f}°, {frac_top10*100:.1f}% in top-10 PC)")
    print(f"  Causal ablation survives at deter={DETER}: {'YES' if causal_holds else 'NO'} "
          f"(confusion Δ{abl[0]['confusion']:+.3f} vs random Δ{abl[0]['random']:+.3f} at t)")
    if geom_holds and causal_holds:
        verdict = (f"BOTH load-bearing findings HOLD at deter={DETER} (2× the XS width): the confusion "
                   f"direction remains near-orthogonal to the top PCs ({mean_angle:.1f}°) and ablating it "
                   f"still collapses the probe readout ({abl[0]['confusion']:+.3f}) far beyond a random "
                   f"control ({abl[0]['random']:+.3f}). ONE intermediate data point — partial evidence the "
                   f"core findings are not XS-specific, NOT a resolution of the scale question.")
    else:
        verdict = (f"PARTIAL/NEGATIVE at deter={DETER}: geometry_holds={geom_holds}, causal_holds={causal_holds}. "
                   f"Reported honestly as one data point.")
    print(f"\n  {verdict}")

    results = dict(deter=DETER, auroc_id=float(auroc_id), mean_angle=mean_angle,
                   frac_top10=frac_top10, ablation={str(k): abl[k] for k in LOOKAHEAD},
                   geom_holds=geom_holds, causal_holds=causal_holds, verdict=verdict, n_sites=len(sites))
    with open(os.path.join(OUT_DIR, 'task_s_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_s_results.json')}")


if __name__ == '__main__':
    main()
