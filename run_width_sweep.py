#!/usr/bin/env python3.11
"""
Tier 1 — Width-scaling sweep (the scale trend, on existing infrastructure).

Turns the paper's single deter=512 data point (Appendix D / Task S) into a TREND across an
8x GRU-width range: 256 -> 512 -> 1024 -> 2048. Same tasks, same hyperparameters, same
measurements. The only variable is GRU width (rssm_deter / rssm_hidden). No new architecture,
no images -- this runs entirely on the existing hand-rolled RSSM, on CPU, and is guaranteed to
finish, so it is the safe backbone of the scale story regardless of how the image-scale run goes.

At each (task, width) it measures the two load-bearing findings plus two context metrics:
  [1] Null-space geometry  -- mean angle of the confusion direction to the top-10 PCs, and the
      fraction of probe variance living in that top-10 subspace. (Should stay ~88-89 deg, <1%.)
  [2] Causal ablation      -- Delta-probe from removing the confusion direction vs a random
      matched direction (single-random light control, as in Task S). (Confusion should stay far
      more negative than random.)
  [3] C_t R^2              -- best-gamma fit of the closed-form confusion integral. (Context.)
  [4] Set C AUROC          -- the KL-matched contrastive score. (Context; may vary by task.)

Reuses Task S's exact train() / geometry / ablation code so the 512 column reproduces Task S.

USAGE (each run is independent and checkpointed -- safe to launch a subset, resume, or parallelise):
  python3.11 run_width_sweep.py --task cartpole --width 256
  python3.11 run_width_sweep.py --task cartpole --width 512 1024 2048
  python3.11 run_width_sweep.py --task all --width all           # everything (slow on CPU)
  python3.11 run_width_sweep.py --summarize                      # collate finished cells -> table

Widths 1024/2048 are SLOW on CPU (the RSSM MLPs grow with width). Expect the 2048 cartpole run
to take on the order of a day. Launch overnight / over a weekend; each cell writes its own JSON
so partial progress is never lost. --summarize collates whatever has finished so far.
"""

import os
import re
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.training.replay_buffer import EpisodeReplayBuffer
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import probe_direction, random_matched_direction, compute_ct

OUT_DIR = 'outputs/width_sweep'
WIDTHS = [256, 512, 1024, 2048]
LOOKAHEAD = [0, 1, 5, 10]
GAMMAS = [0.70, 0.80, 0.90, 0.95, 0.99]

# per-task env factory + gamma default (matches Task N / paper); cartpole uses the native wrapper
TASKS = {
    'cartpole': dict(factory=lambda s, noisy=False: CartpoleEnv(task='swingup', noisy=noisy, seed=s), gamma=0.95),
    'reacher':  dict(factory=lambda s, noisy=False: DMCEnv('reacher', 'easy', noisy=noisy, seed=s), gamma=0.70),
    'pendulum': dict(factory=lambda s, noisy=False: DMCEnv('pendulum', 'swingup', noisy=noisy, seed=s), gamma=0.90),
}


def cell_paths(task, width):
    base = os.path.join(OUT_DIR, f'{task}_deter{width}')
    return base + '.pt', base + '_states.npz', base + '_result.json'


def make_cfg(task, width):
    cfg = XS_CONFIG.copy()
    cfg['rssm_deter'] = width
    cfg['rssm_hidden'] = width
    # obs/act dims are set from the env at train time
    return cfg


# ── training (generalised from Task S train(), task-parametric) ──
def train(task, cfg, ckpt, states_path, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    env = TASKS[task]['factory'](seed)
    cfg = cfg.copy(); cfg['obs_dim'] = env.obs_dim; cfg['act_dim'] = env.act_dim
    model = WorldModel(env.obs_dim, env.act_dim, cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    buffer = EpisodeReplayBuffer(capacity=cfg['replay_capacity'])
    seq_len, batch, warmup, max_steps = cfg['seq_len'], cfg['batch_size'], cfg['warmup_steps'], cfg['total_env_steps']
    log_h, log_z, log_kl, log_recon, log_traj = [], [], [], [], []
    step, traj, t0 = 0, 0, time.time()
    ep_obs, ep_act = [], []
    h = torch.zeros(1, cfg['rssm_deter']); z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
    obs = env.reset(); ep_obs.append(obs.copy())
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{task} deter={cfg['rssm_deter']}] params={nparams:.1f}M  obs={env.obs_dim} act={env.act_dim}", flush=True)
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
            print(f"  {task} d{cfg['rssm_deter']} step {step:,}/{max_steps:,} "
                  f"kl={np.mean(log_kl[-500:]):.2f} {(time.time()-t0)/60:.0f}m", flush=True)
    torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'obs_dim': env.obs_dim,
                'act_dim': env.act_dim, 'nparams_m': nparams}, ckpt)
    states = dict(h=np.array(log_h, np.float32), z=np.array(log_z, np.float32),
                  kl=np.array(log_kl, np.float32), recon=np.array(log_recon, np.float32),
                  traj_id=np.array(log_traj, np.int64))
    np.savez(states_path, **states)
    return model, states, cfg, nparams


def collect_traj(task, model, cfg, n_traj=40, seed=777):
    env = TASKS[task]['factory'](seed); np.random.seed(seed)
    trajs = []
    for ep in range(n_traj):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter']); z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
        O, A, H = [], [], []
        done, step = False, 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                a = np.random.uniform(-1, 1, (cfg['act_dim'],)).astype(np.float32)
                ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0); at = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
                emb = model.encoder(ot)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, at, emb)
                O.append(obs.copy()); A.append(a.copy()); H.append(h.squeeze(0).numpy().copy())
                obs, _, done = env.step(a); step += 1
        trajs.append(dict(obs=np.array(O, np.float32), act=np.array(A, np.float32), h=np.array(H, np.float32)))
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


def build_set_c(states, seed=42, n_bins=10, per_bin=20, max_total=200):
    """KL-matched contrastive set (paper's Set C, single-model version): within each KL bin,
    bottom-30% recon = group 0, top-30% = group 1."""
    kl, rc, h = states['kl'], states['recon'], states['h']
    rng = np.random.default_rng(seed)
    edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1)); kb = np.digitize(kl, edges[1:-1])
    c1, c2 = [], []
    for b in range(n_bins):
        idx = np.where(kb == b)[0]
        if len(idx) < 4:
            continue
        rb = rc[idx]
        lo = idx[rb <= np.percentile(rb, 30)]; hi = idx[rb >= np.percentile(rb, 70)]
        n = min(per_bin, len(lo), len(hi))
        if n == 0:
            continue
        c1.extend(rng.choice(lo, n, replace=False).tolist())
        c2.extend(rng.choice(hi, n, replace=False).tolist())
    if len(c1) > max_total: c1 = rng.choice(c1, max_total, replace=False).tolist()
    if len(c2) > max_total: c2 = rng.choice(c2, max_total, replace=False).tolist()
    return np.concatenate([h[c1], h[c2]]), np.array([0]*len(c1) + [1]*len(c2), np.int32)


def measure_cell(task, width):
    """Train (or load) one (task,width) model and measure all four quantities."""
    ckpt, states_path, result_path = cell_paths(task, width)
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = make_cfg(task, width)

    if os.path.exists(ckpt) and os.path.exists(states_path):
        ck = torch.load(ckpt, map_location='cpu')
        cfg = ck['cfg']
        model = WorldModel(ck['obs_dim'], ck['act_dim'], cfg); model.load_state_dict(ck['model_state']); model.eval()
        states = dict(np.load(states_path)); nparams = ck.get('nparams_m', float('nan'))
        print(f"[{task} d{width}] loaded existing checkpoint")
    else:
        model, states, cfg, nparams = train(task, cfg, ckpt, states_path, seed=0)

    h, kl, recon, traj = states['h'], states['kl'], states['recon'], states['traj_id']
    N = len(h); y = binarise_by_median(kl)
    tr_idx, te_idx = train_test_split(np.arange(N), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h[tr_idx], y[tr_idx])
    auroc_id = float(auroc(clf, sc, h[te_idx], y[te_idx]))

    # [1] null-space geometry (top-10 PCs), identical to Task S
    w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    pca = PCA(n_components=10, random_state=0).fit(sc.transform(h))
    angles = [np.degrees(np.arccos(np.clip(abs(np.dot(w, pca.components_[k])), 0, 1))) for k in range(10)]
    mean_angle = float(np.mean(angles))
    frac_top10 = float(np.sum((pca.components_ @ w) ** 2))

    # [3] C_t R^2 best gamma (ridge of h -> C_t, on test split)
    best_r2, best_gamma = -1.0, None
    for g in GAMMAS:
        ct = compute_ct(kl, traj, gamma=g)
        scaler = StandardScaler().fit(h[tr_idx])
        rg = Ridge(alpha=1.0).fit(scaler.transform(h[tr_idx]), ct[tr_idx])
        r2 = float(rg.score(scaler.transform(h[te_idx]), ct[te_idx]))
        if r2 > best_r2:
            best_r2, best_gamma = r2, g

    # [4] Set C AUROC (single-model)
    Xc, yc = build_set_c(states)
    auroc_setc = float(auroc(clf, sc, Xc, yc))

    # [2] causal ablation vs single random control (light, as Task S)
    v = probe_direction(clf, sc)
    trajs = collect_traj(task, model, cfg)
    rng = np.random.default_rng(0)
    sites = []
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
    abl = {str(k): dict(confusion=float(np.mean(dconf[k])), random=float(np.mean(drand[k]))) for k in LOOKAHEAD}

    geom_holds = mean_angle > 80 and frac_top10 < 0.1
    causal_holds = abs(abl['0']['confusion']) > 5 * abs(abl['0']['random']) and abl['0']['confusion'] < -0.1

    result = dict(
        task=task, width=width, nparams_m=float(nparams), n_sites=len(sites),
        auroc_id=auroc_id, mean_angle=mean_angle, frac_top10=frac_top10,
        best_gamma=best_gamma, ct_r2=best_r2, auroc_setc=auroc_setc,
        ablation=abl, geom_holds=bool(geom_holds), causal_holds=bool(causal_holds))
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  ── {task} deter={width} ({nparams:.1f}M) ──")
    print(f"     Probe A held-out AUROC : {auroc_id:.4f}")
    print(f"     null-space angle (t10) : {mean_angle:.1f}°   frac in top-10 PC: {frac_top10:.4f}   [{'OK' if geom_holds else 'FAIL'}]")
    print(f"     C_t R² (best γ={best_gamma}) : {best_r2:.3f}      Set C AUROC: {auroc_setc:.3f}")
    print(f"     ablation Δ@0 confusion  : {abl['0']['confusion']:+.4f}  vs random {abl['0']['random']:+.4f}   [{'OK' if causal_holds else 'FAIL'}]")
    print(f"     → saved {result_path}")
    return result


def summarize():
    rows = []
    for task in TASKS:
        for width in WIDTHS:
            _, _, rp = cell_paths(task, width)
            if os.path.exists(rp):
                rows.append(json.load(open(rp)))
    if not rows:
        print("No finished cells yet. Run some (task,width) combinations first.")
        return
    rows.sort(key=lambda r: (r['task'], r['width']))
    print("\n" + "=" * 100)
    print("WIDTH-SCALING TREND  (the scale story: does each finding hold as width grows?)")
    print("=" * 100)
    hdr = f"{'task':<10}{'width':>6}{'params':>9}{'AUROC-ID':>10}{'angle°':>8}{'top10%':>9}{'γ':>6}{'C_t R²':>8}{'SetC':>7}{'abl Δ0':>9}{'rand':>9}{'geom':>6}{'causal':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['task']:<10}{r['width']:>6}{r['nparams_m']:>8.1f}M{r['auroc_id']:>10.3f}"
              f"{r['mean_angle']:>8.1f}{r['frac_top10']*100:>8.2f}%{str(r['best_gamma']):>6}"
              f"{r['ct_r2']:>8.3f}{r['auroc_setc']:>7.3f}{r['ablation']['0']['confusion']:>+9.3f}"
              f"{r['ablation']['0']['random']:>+9.3f}{('OK' if r['geom_holds'] else 'FAIL'):>6}"
              f"{('OK' if r['causal_holds'] else 'FAIL'):>7}")
    # per-task trend verdict on the two load-bearing findings
    print("\n  Load-bearing findings across width (per task):")
    for task in TASKS:
        tr = [r for r in rows if r['task'] == task]
        if len(tr) < 2:
            continue
        widths = [r['width'] for r in tr]
        angles = [r['mean_angle'] for r in tr]
        abls = [r['ablation']['0']['confusion'] for r in tr]
        geom_all = all(r['geom_holds'] for r in tr)
        causal_all = all(r['causal_holds'] for r in tr)
        print(f"    {task:<10} widths {widths}: angle {min(angles):.1f}–{max(angles):.1f}° "
              f"[geometry {'HOLDS' if geom_all else 'BREAKS'}], "
              f"ablation Δ0 {min(abls):+.2f}…{max(abls):+.2f} [causal {'HOLDS' if causal_all else 'BREAKS'}]")
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Collated → {os.path.join(OUT_DIR, 'summary.json')}")


def parse_widths(vals):
    if vals == ['all'] or vals is None:
        return WIDTHS
    return [int(v) for v in vals]


def parse_tasks(val):
    if val == 'all':
        return list(TASKS)
    if val not in TASKS:
        raise SystemExit(f"unknown task '{val}'; choose from {list(TASKS)} or 'all'")
    return [val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='cartpole', help="cartpole|reacher|pendulum|all")
    ap.add_argument('--width', nargs='+', default=None, help="256 512 1024 2048 | all")
    ap.add_argument('--summarize', action='store_true', help="collate finished cells into the trend table")
    args = ap.parse_args()

    if args.summarize:
        summarize(); return

    tasks = parse_tasks(args.task)
    widths = parse_widths(args.width)
    print(f"Running cells: tasks={tasks} widths={widths}")
    print("(each cell trains one model then measures; slow widths take many hours on CPU)\n")
    for task in tasks:
        for width in widths:
            t0 = time.time()
            measure_cell(task, width)
            print(f"  [{task} d{width}] cell done in {(time.time()-t0)/60:.1f} min\n")
    summarize()


if __name__ == '__main__':
    main()
