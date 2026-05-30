#!/usr/bin/env python3.11
"""
Task 4 — Gate analysis formalisation: mechanistic account of the confusion signal.

4a — Confusion subspace analysis (PCA + probe direction):
  Does the probe's decision boundary align with dominant variance directions in h_t?
  Expected: probe direction is roughly orthogonal to top PCA components — the
  confusion signal lives in a low-variance subspace that PCA would discard.

4b — Update gate saturation curve:
  Distribution of mean(z_gate) by KL quartile. Expected: near-constant at ~0.94
  (GRU learned "always overwrite") with tiny negative correlation with KL.

4c — Candidate gate direction analysis:
  Project n_t onto the probe direction. Does n_t·w_probe correlate with confusion
  (recon error) and KL? Expected: positive — confused states push the candidate
  in the direction the probe reads as uncertain.
  Cross-task version: same projection on balance data.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.config import XS_CONFIG
from src.env.wrapper import CartpoleEnv
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe


GATE_CACHE_SW  = 'outputs/data/gate_geom_swingup.npz'
GATE_CACHE_BAL = 'outputs/data/gate_geom_balance.npz'
N_COLLECT_EP   = 20   # 20 episodes ≈ 10K steps — sufficient for all analyses


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


def compute_gru_gates(gru_cell, inp, h):
    W_ih, W_hh = gru_cell.weight_ih, gru_cell.weight_hh
    b_ih, b_hh = gru_cell.bias_ih,   gru_cell.bias_hh
    deter = h.shape[-1]
    pre_ih = inp @ W_ih.T + b_ih
    pre_hh = h   @ W_hh.T + b_hh
    r_t = torch.sigmoid(pre_ih[:, :deter]        + pre_hh[:, :deter])
    z_t = torch.sigmoid(pre_ih[:, deter:2*deter] + pre_hh[:, deter:2*deter])
    n_t = torch.tanh(   pre_ih[:, 2*deter:]      + r_t * pre_hh[:, 2*deter:])
    return z_t, r_t, n_t


def collect_gates(model, env, n_episodes, cfg, label='?'):
    device = next(model.parameters()).device
    gru    = model.rssm.gru
    all_h, all_z_g, all_r_g, all_n_g = [], [], [], []
    all_kl, all_recon = [], []
    all_step = []

    for ep in range(n_episodes):
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
                inp    = torch.cat([z, a_t], dim=-1)
                zg, rg, ng = compute_gru_gates(gru, inp, h)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                dec  = model.decoder(torch.cat([h, z], dim=-1))
                kl_v = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                re_v = F.mse_loss(dec, obs_t, reduction='none').sum().item()
                all_h.append(h.squeeze(0).cpu().numpy().copy())
                all_z_g.append(zg.squeeze(0).cpu().numpy().copy())
                all_r_g.append(rg.squeeze(0).cpu().numpy().copy())
                all_n_g.append(ng.squeeze(0).cpu().numpy().copy())
                all_kl.append(kl_v)
                all_recon.append(re_v)
                all_step.append(step)
                obs, _, done = env.step(action)
                step += 1
        if (ep + 1) % 5 == 0:
            print(f"    {label} ep {ep+1}/{n_episodes}")

    return {
        'h':      np.array(all_h,     dtype=np.float32),
        'z_gate': np.array(all_z_g,   dtype=np.float32),
        'r_gate': np.array(all_r_g,   dtype=np.float32),
        'n_gate': np.array(all_n_g,   dtype=np.float32),
        'kl':     np.array(all_kl,    dtype=np.float32),
        'recon':  np.array(all_recon, dtype=np.float32),
        'step':   np.array(all_step,  dtype=np.int32),
    }


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    # ── Train Probe A ──
    print("Loading training states and training Probe A...")
    tr = dict(np.load(cfg['training_data_path']))
    y_tr = binarise_by_median(tr['kl'])
    tr_idx, te_idx = train_test_split(
        np.arange(len(tr['h'])), test_size=0.40, stratify=y_tr, random_state=0)
    clf, sc = train_probe(tr['h'][tr_idx], y_tr[tr_idx])
    probe_w_scaled = clf.coef_[0]                           # (256,) in scaled space
    probe_w_orig   = probe_w_scaled / sc.scale_             # (256,) in original h_t space
    probe_w_norm   = probe_w_orig / np.linalg.norm(probe_w_orig)
    print(f"  Probe trained on {len(tr_idx):,} states")

    # ── Collect / load gate data ──
    model = load_model(cfg)

    if os.path.exists(GATE_CACHE_SW):
        print(f"\nLoading cached swingup gate data from {GATE_CACHE_SW}...")
        sw = dict(np.load(GATE_CACHE_SW))
    else:
        print(f"\nCollecting swingup gate data ({N_COLLECT_EP} episodes)...")
        env_sw = CartpoleEnv(task='swingup', noisy=False, seed=42)
        sw = collect_gates(model, env_sw, N_COLLECT_EP, cfg, 'swingup')
        np.savez(GATE_CACHE_SW, **sw)
        print(f"  Saved to {GATE_CACHE_SW}")

    if os.path.exists(GATE_CACHE_BAL):
        print(f"Loading cached balance gate data from {GATE_CACHE_BAL}...")
        bal = dict(np.load(GATE_CACHE_BAL))
    else:
        print(f"Collecting balance gate data ({N_COLLECT_EP} episodes)...")
        env_bal = CartpoleEnv(task='balance', noisy=False, seed=52)
        bal = collect_gates(model, env_bal, N_COLLECT_EP, cfg, 'balance')
        np.savez(GATE_CACHE_BAL, **bal)
        print(f"  Saved to {GATE_CACHE_BAL}")

    # Filter step >= 2
    sw_mask  = sw['step']  >= 2
    bal_mask = bal['step'] >= 2
    sw  = {k: v[sw_mask]  for k, v in sw.items()}
    bal = {k: v[bal_mask] for k, v in bal.items()}
    print(f"\n  Swingup: {len(sw['h']):,} steps  KL mean={sw['kl'].mean():.1f}  "
          f"recon mean={sw['recon'].mean():.3f}")
    print(f"  Balance: {len(bal['h']):,} steps  KL mean={bal['kl'].mean():.1f}  "
          f"recon mean={bal['recon'].mean():.3f}")

    # ════════════════════════════════════════════════════════════════════
    # 4a — Confusion subspace: PCA on h_t vs probe direction
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("4a — CONFUSION SUBSPACE ANALYSIS (PCA + PROBE DIRECTION)")
    print("="*65)

    print("\nFitting PCA on scaled h_t (training states, 50 components)...")
    h_scaled = sc.transform(tr['h'])
    pca = PCA(n_components=50, random_state=0).fit(h_scaled)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Angle between probe direction (scaled space) and each PC
    probe_w_s_norm = probe_w_scaled / np.linalg.norm(probe_w_scaled)
    angles = []
    for k in range(50):
        pc = pca.components_[k]
        cos_val = abs(np.dot(probe_w_s_norm, pc))  # already unit norm
        angle_deg = np.degrees(np.arccos(np.clip(cos_val, 0.0, 1.0)))
        angles.append(angle_deg)

    angles_arr = np.array(angles)
    print(f"\n  PCs 1-10 angles with probe direction (degrees from orthogonal=90°):")
    print(f"  {'PC':>4}  {'Expl var %':>10}  {'Cum var %':>10}  {'Angle (°)':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")
    for k in range(10):
        print(f"  {k+1:>4}  {pca.explained_variance_ratio_[k]*100:>10.2f}  "
              f"{cum_var[k]*100:>10.2f}  {angles_arr[k]:>10.2f}")

    print(f"\n  Mean angle (PCs 1-10): {angles_arr[:10].mean():.1f}°  "
          f"(90°=orthogonal, 0°=aligned)")
    print(f"  Mean angle (PCs 1-50): {angles_arr.mean():.1f}°")

    # Project probe weight onto top-k PCs: how much of the probe is captured?
    probe_pca_proj = pca.transform(probe_w_scaled.reshape(1, -1))[0]  # (50,)
    probe_pca_var  = np.cumsum(probe_pca_proj**2) / np.sum(probe_w_scaled**2)
    print(f"\n  Fraction of ||probe||² captured by top PCA components:")
    for k in [1, 5, 10, 20, 50]:
        idx_k = min(k, len(probe_pca_var)) - 1
        print(f"    top {k:>2}: {probe_pca_var[idx_k]*100:.1f}%")

    if angles_arr[:10].mean() > 75:
        print("\n  CONCLUSION: Probe direction is roughly orthogonal to top PCA components.")
        print("  Confusion signal lives in a low-variance subspace — PCA would discard it.")
    elif angles_arr[:10].mean() > 60:
        print("\n  CONCLUSION: Probe partially aligns with some high-variance PCs.")
    else:
        print("\n  CONCLUSION: Probe is aligned with dominant variance directions.")

    # ════════════════════════════════════════════════════════════════════
    # 4b — Update gate saturation curve
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("4b — UPDATE GATE SATURATION BY KL QUARTILE")
    print("="*65)

    kl_sw = sw['kl']
    zg_sw = sw['z_gate']
    q_bounds = np.percentile(kl_sw, [0, 25, 50, 75, 100])

    print(f"\n  KL range: [{kl_sw.min():.1f}, {kl_sw.max():.1f}]  mean={kl_sw.mean():.1f}")
    print(f"\n  {'KL quartile':>20}  {'KL range':>18}  {'N':>6}  "
          f"{'mean(z_gate)':>12}  {'std(z_gate)':>11}")
    print(f"  {'-'*20}  {'-'*18}  {'-'*6}  {'-'*12}  {'-'*11}")

    q_zmeans = []
    for q in range(4):
        lo, hi = q_bounds[q], q_bounds[q+1]
        mask = (kl_sw >= lo) & (kl_sw < (hi + 1e-6))
        z_mean_per_state = zg_sw[mask].mean(axis=1)
        q_zmeans.append(z_mean_per_state.mean())
        print(f"  {'Q'+str(q+1)+' (KL '+f'{lo:.0f}–{hi:.0f})':>20}  "
              f"{'['+f'{lo:.1f}'+', '+f'{hi:.1f}'+']':>18}  "
              f"{mask.sum():>6}  "
              f"{z_mean_per_state.mean():>12.4f}  "
              f"{z_mean_per_state.std():>11.4f}")

    r_zmean_kl = np.corrcoef(zg_sw.mean(axis=1), kl_sw)[0, 1]
    print(f"\n  Overall mean(z_gate): {zg_sw.mean():.4f}  "
          f"(std across states: {zg_sw.mean(axis=1).std():.4f})")
    print(f"  Pearson r (mean(z_gate), KL): {r_zmean_kl:+.4f}")
    print(f"  Q4 vs Q1 mean(z_gate) gap: {q_zmeans[3]-q_zmeans[0]:+.4f}")

    if abs(q_zmeans[3] - q_zmeans[0]) < 0.02:
        print("\n  SATURATION CONFIRMED: z_gate nearly constant across all KL levels.")
        print("  GRU learned an 'always overwrite' policy — confusion is not in update frequency.")
    else:
        print("\n  z_gate shows meaningful variation across KL quartiles.")

    # ════════════════════════════════════════════════════════════════════
    # 4c — Candidate gate direction analysis
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("4c — CANDIDATE GATE DIRECTION ANALYSIS (n_t PROJECTION)")
    print("="*65)
    print("\nProjecting n_t onto probe direction (original space).")
    print("If confused states push n_t along the probe direction, this explains")
    print("why Δh_t partially transfers: the confusion signal is in the candidate content.\n")

    for label, data in [('Swingup', sw), ('Balance', bal)]:
        ng = data['n_gate']    # (N, 256)
        kl_d = data['kl']
        recon_d = data['recon']

        n_proj = ng @ probe_w_norm   # scalar per state

        r_kl   = np.corrcoef(n_proj, kl_d)[0, 1]
        r_recon = np.corrcoef(n_proj, recon_d)[0, 1]

        # Coping vs confused split (by recon median within this dataset)
        recon_med = np.median(recon_d)
        coping_proj   = n_proj[recon_d <  recon_med]
        confused_proj = n_proj[recon_d >= recon_med]

        print(f"  {label}:")
        print(f"    r(n_t·w_probe, KL):    {r_kl:+.4f}")
        print(f"    r(n_t·w_probe, recon): {r_recon:+.4f}")
        print(f"    Coping mean projection:   {coping_proj.mean():+.4f}  "
              f"(N={len(coping_proj):,})")
        print(f"    Confused mean projection: {confused_proj.mean():+.4f}  "
              f"(N={len(confused_proj):,})")
        print(f"    Group gap (confused - coping): {confused_proj.mean() - coping_proj.mean():+.4f}")

        if r_recon > 0.1:
            print(f"    → Confused states push n_t in the probe direction.")
        elif r_recon > 0.03:
            print(f"    → Weak positive: marginal directional signal.")
        else:
            print(f"    → No directional signal in n_t.")
        print()

    # ════════════════════════════════════════════════════════════════════
    # Figures
    # ════════════════════════════════════════════════════════════════════
    print("Generating gate geometry figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('GRU Gate Geometry — Mechanistic Account', fontsize=12, fontweight='bold')

    # 4a: Angle vs PC rank
    ax = axes[0]
    ax.plot(range(1, 51), angles_arr, 'b-o', markersize=3, linewidth=1.2)
    ax.axhline(90, color='gray', linestyle='--', linewidth=0.8, label='Orthogonal (90°)')
    ax.set_xlabel('PCA component rank')
    ax.set_ylabel('Angle with probe direction (°)')
    ax.set_title('4a — Probe vs PCA directions\n(90° = orthogonal, confusion in low-var subspace)')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4b: z_gate distribution by KL quartile (violin or box)
    ax = axes[1]
    quartile_data = []
    quartile_labels = []
    for q in range(4):
        lo, hi = q_bounds[q], q_bounds[q+1]
        mask = (kl_sw >= lo) & (kl_sw < (hi + 1e-6))
        quartile_data.append(zg_sw[mask].mean(axis=1))
        quartile_labels.append(f'Q{q+1}\nKL∈[{lo:.0f},{hi:.0f}]')
    parts = ax.violinplot(quartile_data, positions=range(1, 5), showmedians=True)
    for pc in parts['bodies']:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(quartile_labels, fontsize=8)
    ax.set_ylabel('mean(z_gate) per state')
    ax.set_title('4b — Update gate saturation by KL quartile\n(expect ~0.94 flat = always-overwrite)')
    ax.set_ylim(0.85, 1.0)
    ax.axhline(0.94, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='0.94 reference')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 4c: n_t projection scatter (swingup)
    ax = axes[2]
    ng_sw = sw['n_gate']
    n_proj_sw = ng_sw @ probe_w_norm
    recon_sw  = sw['recon']
    # Subsample for readability
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(n_proj_sw), min(3000, len(n_proj_sw)), replace=False)
    ax.scatter(recon_sw[idx_sub], n_proj_sw[idx_sub],
               alpha=0.15, s=5, c='steelblue', rasterized=True)
    # Trend line
    coeffs = np.polyfit(recon_sw[idx_sub], n_proj_sw[idx_sub], 1)
    x_range = np.linspace(recon_sw.min(), np.percentile(recon_sw, 95), 100)
    ax.plot(x_range, np.polyval(coeffs, x_range), 'r-', linewidth=2,
            label=f'r={np.corrcoef(n_proj_sw, recon_sw)[0,1]:+.3f}')
    ax.set_xlabel('Reconstruction error (log scale)')
    ax.set_ylabel('n_t · w_probe (projection)')
    ax.set_title('4c — Candidate gate aligned with probe\n(swingup, step≥2)')
    ax.set_xscale('symlog', linthresh=0.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'gate_geometry.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    main()
