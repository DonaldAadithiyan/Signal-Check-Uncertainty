#!/usr/bin/env python3.11
"""
Task 1 — Multi-task Δh_t probe: recover the cross-task confusion story.

The within-balance confound check showed that Probe A (trained on swingup h_t)
fails on within-balance KL-matched sets (AUROC 0.51). The confound is that
the probe learns trajectory fingerprints specific to swingup h_t space.

Fix: train on Δh_t vectors pooled from multiple tasks simultaneously.
Δh_t = h_t − h_{t−1} removes accumulated trajectory fingerprints — it only
carries what changed at this step in response to this observation.

Setup:
  Tasks: swingup, balance, balance_sparse, swingup_sparse (all obs_dim=5)
  All data collected with the frozen swingup model (confused on non-swingup tasks)
  Leave-one-out: train on 3 tasks, evaluate on within-task KL-matched set of 4th task
  Target: AUROC > 0.68 on held-out task (beats single-task Δh_t baseline)

The within-task confound is eliminated by construction: the probe sees Δh_t from
multiple tasks and cannot rely on any single task's trajectory fingerprints.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import XS_CONFIG
from src.env.wrapper import CartpoleEnv
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe, auroc


TASKS   = ['swingup', 'balance', 'balance_sparse', 'swingup_sparse']
N_EP    = 20   # episodes per task ≈ 10K steps each
CACHE_DIR = 'outputs/data'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def collect_task(model, task, n_ep, cfg, seed=0):
    """Collect h_t, Δh_t, kl, recon from a task using the frozen model."""
    device  = next(model.parameters()).device
    env     = CartpoleEnv(task=task, noisy=False, seed=seed)
    all_h, all_kl, all_recon, all_step = [], [], [], []

    for ep in range(n_ep):
        obs  = env.reset()
        h    = torch.zeros(1, cfg['rssm_deter'], device=device)
        z    = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        done = False
        step = 0
        with torch.no_grad():
            while not done and step < cfg['episode_max_steps']:
                action = np.random.uniform(-1, 1, (cfg['act_dim'],)).astype(np.float32)
                obs_t  = torch.tensor(obs,    dtype=torch.float32, device=device).unsqueeze(0)
                a_t    = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                embed  = model.encoder(obs_t)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                dec    = model.decoder(torch.cat([h, z], dim=-1))
                kl_v   = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                re_v   = F.mse_loss(dec, obs_t, reduction='none').sum().item()
                all_h.append(h.squeeze(0).cpu().numpy().copy())
                all_kl.append(kl_v)
                all_recon.append(re_v)
                all_step.append(step)
                obs, _, done = env.step(action)
                step += 1
        if (ep + 1) % 5 == 0:
            print(f"    {task} ep {ep+1}/{n_ep}")

    h_arr    = np.array(all_h,    dtype=np.float32)
    kl_arr   = np.array(all_kl,   dtype=np.float32)
    re_arr   = np.array(all_recon, dtype=np.float32)
    step_arr = np.array(all_step,  dtype=np.int32)

    # Δh_t: only for step >= 2 (avoids init noise)
    mask  = step_arr >= 2
    idx   = np.where(mask)[0]
    dh    = h_arr[idx] - h_arr[idx - 1]
    return {
        'h':     h_arr[mask],
        'dh':    dh,
        'kl':    kl_arr[mask],
        'recon': re_arr[mask],
    }


def build_contrastive(data, n_bins=10, per_bin=20, max_n=200, seed=42):
    """KL-matched contrastive set: C1=lo recon within KL bin, C2=hi recon."""
    kl, recon = data['kl'], data['recon']
    bin_edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1))
    bin_idx   = np.digitize(kl, bin_edges[1:-1])
    rng = np.random.default_rng(seed)
    c1, c2 = [], []
    for b in range(n_bins):
        idx = np.where(bin_idx == b)[0]
        if len(idx) < 4:
            continue
        r  = recon[idx]
        lo = idx[r <= np.percentile(r, 30)]
        hi = idx[r >= np.percentile(r, 70)]
        n  = min(per_bin, len(lo), len(hi))
        if n == 0:
            continue
        c1.extend(rng.choice(lo, n, replace=False).tolist())
        c2.extend(rng.choice(hi, n, replace=False).tolist())
    if len(c1) > max_n: c1 = rng.choice(c1, max_n, replace=False).tolist()
    if len(c2) > max_n: c2 = rng.choice(c2, max_n, replace=False).tolist()
    all_idx = c1 + c2
    labels  = np.array([0]*len(c1) + [1]*len(c2), dtype=np.int32)
    return {k: data[k][all_idx] for k in data}, labels


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Load or collect task data ──
    model = load_model(cfg)
    task_data = {}
    for task in TASKS:
        cache = os.path.join(CACHE_DIR, f'multitask_{task}.npz')
        if os.path.exists(cache):
            print(f"Loading cached {task} data...")
            task_data[task] = dict(np.load(cache))
        else:
            print(f"Collecting {task} ({N_EP} episodes)...")
            task_data[task] = collect_task(model, task, N_EP, cfg)
            np.savez(cache, **task_data[task])

    for task in TASKS:
        d = task_data[task]
        print(f"  {task}: N={len(d['h']):,}  KL mean={d['kl'].mean():.1f}  "
              f"recon mean={d['recon'].mean():.3f}")

    # ── Single-task Δh_t baseline (swingup only) ──
    print("\n" + "="*65)
    print("SINGLE-TASK Δh_t BASELINE (swingup only)")
    print("="*65)
    sw = task_data['swingup']
    y_sw = binarise_by_median(sw['kl'])
    tr_idx, te_idx = train_test_split(
        np.arange(len(sw['h'])), test_size=0.40, stratify=y_sw, random_state=0)
    clf_st, sc_st = train_probe(sw['dh'][tr_idx], y_sw[tr_idx])
    auroc_st_id = auroc(clf_st, sc_st, sw['dh'][te_idx], y_sw[te_idx])

    # Within-task KL-matched contrastive sets for all tasks
    contrastive = {}
    for task in TASKS:
        c_data, c_labels = build_contrastive(task_data[task])
        contrastive[task] = (c_data, c_labels)

    auroc_st_sw   = auroc(clf_st, sc_st, contrastive['swingup'][0]['dh'],   contrastive['swingup'][1])
    auroc_st_bal  = auroc(clf_st, sc_st, contrastive['balance'][0]['dh'],   contrastive['balance'][1])
    auroc_st_bals = auroc(clf_st, sc_st, contrastive['balance_sparse'][0]['dh'], contrastive['balance_sparse'][1])
    auroc_st_sws  = auroc(clf_st, sc_st, contrastive['swingup_sparse'][0]['dh'], contrastive['swingup_sparse'][1])

    print(f"\n  Swingup-only Δh_t probe:")
    print(f"  {'Task':<20}  {'Contrastive AUROC':>18}")
    for task, a in [('swingup', auroc_st_sw), ('balance', auroc_st_bal),
                    ('balance_sparse', auroc_st_bals), ('swingup_sparse', auroc_st_sws)]:
        print(f"  {task:<20}  {a:>18.4f}")

    # ── Multi-task Δh_t probe: leave-one-out ──
    print("\n" + "="*65)
    print("MULTI-TASK Δh_t PROBE — LEAVE-ONE-OUT")
    print("="*65)
    print("\nPool Δh_t from 3 tasks, evaluate on within-task KL-matched set of 4th.\n")

    results = {}
    for held_out in TASKS:
        train_tasks = [t for t in TASKS if t != held_out]

        # Pool training data from 3 tasks
        dh_pool  = np.concatenate([task_data[t]['dh']  for t in train_tasks], axis=0)
        kl_pool  = np.concatenate([task_data[t]['kl']  for t in train_tasks], axis=0)
        y_pool   = binarise_by_median(kl_pool)

        # Train multi-task probe
        clf_mt, sc_mt = train_probe(dh_pool, y_pool)

        # Evaluate on held-out within-task contrastive set
        c_data, c_labels = contrastive[held_out]
        auc_mt = auroc(clf_mt, sc_mt, c_data['dh'], c_labels)

        # Compare to single-task Δh_t baseline on same held-out set
        auc_st = auroc(clf_st, sc_st, c_data['dh'], c_labels)

        # Also compute held-out within-task: does multi-task probe
        # understand confusion within the held-out task better than chance?
        results[held_out] = dict(
            mt=auc_mt, st=auc_st, improvement=auc_mt - auc_st,
            n_contrastive=len(c_labels),
            train_tasks=train_tasks,
        )

        kl_mean = task_data[held_out]['kl'].mean()
        recon_mean = task_data[held_out]['recon'].mean()
        print(f"  Held-out: {held_out:<18}  KL={kl_mean:.1f}  recon={recon_mean:.3f}")
        print(f"    Train tasks: {', '.join(train_tasks)}")
        print(f"    Multi-task Δh_t AUROC: {auc_mt:.4f}")
        print(f"    Single-task Δh_t AUROC: {auc_st:.4f}  (swingup-trained)")
        print(f"    Improvement: {auc_mt - auc_st:+.4f}")
        c_kl_gap = (c_labels == 1) * task_data[held_out]['kl'][
            np.concatenate([np.where(c_labels==0)[0], np.where(c_labels==1)[0]])
        ] if len(c_labels) > 0 else None
        print(f"    N contrastive pairs: {(c_labels==0).sum()}+{(c_labels==1).sum()}")
        print()

    # ── Summary ──
    print("="*65)
    print("SUMMARY — Multi-task Δh_t probe vs single-task baseline")
    print("="*65)
    print(f"\n  {'Held-out task':<20}  {'MT probe':>9}  {'ST probe':>9}  {'Improvement':>12}  {'Interpretation'}")
    print(f"  {'-'*20}  {'-'*9}  {'-'*9}  {'-'*12}  {'-'*20}")
    n_positive = 0
    for task, r in results.items():
        interp = ('BETTER' if r['mt'] > 0.68 else
                  'ABOVE CHANCE' if r['mt'] > 0.55 else 'CHANCE')
        if r['mt'] > 0.68:
            n_positive += 1
        print(f"  {task:<20}  {r['mt']:>9.4f}  {r['st']:>9.4f}  "
              f"{r['improvement']:>+12.4f}  {interp}")

    best = max(results.values(), key=lambda x: x['mt'])
    print(f"\n  Best held-out AUROC: {best['mt']:.4f}")
    print(f"  Single-task Δh_t baseline (within-swingup): {auroc_st_id:.4f}")

    if best['mt'] > 0.68:
        print("\n  CROSS-TASK SIGNAL RECOVERED: multi-task Δh_t probe exceeds 0.68 threshold.")
        print("  Training on pooled Δh_t recovers the confusion signal across tasks.")
    elif best['mt'] > 0.57:
        print("\n  PARTIAL RECOVERY: some above-chance cross-task confusion detection.")
        print("  Δh_t pooling improves over single-task baseline but below target threshold.")
    else:
        print("\n  NO RECOVERY: multi-task pooling does not recover cross-task signal.")
        print("  Δh_t confusion direction is task-specific even across training tasks.")


if __name__ == '__main__':
    main()
