#!/usr/bin/env python3.11
"""
Seed verification of the z_gate saturation theory (§6.2).

Theoretical prediction: as mean(z_gate) increases (more saturation), the angle
between the probe direction and the top PCA components should increase (probe
becomes more orthogonal to the high-variance subspace).

Tests across 4 checkpoints (3 distinct random seeds). For each model:
  1. Collect 20 episodes → h_t, gate activations, kl, recon
  2. Train probe on h_t with KL binary labels
  3. Fit PCA on scaled h_t
  4. Measure mean angle between probe direction and top-10 PCA components
  5. Measure mean(z_gate)

If the (mean z_gate, mean angle) relationship is monotonic across seeds,
the theoretical prediction is partially verified at XS scale.
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
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.env.wrapper import CartpoleEnv
from src.model.world_model import WorldModel
from src.probe.linear_probe import binarise_by_median, train_probe


CHECKPOINTS = {
    'main (seed 0)':      'outputs/checkpoints/world_model.pt',
    'ensemble seed 0':    'outputs/checkpoints/ensemble_seed0.pt',
    'ensemble seed 1':    'outputs/checkpoints/ensemble_seed1.pt',
    'ensemble seed 2':    'outputs/checkpoints/ensemble_seed2.pt',
}
N_EP      = 20
CACHE_DIR = 'outputs/data'


def load_model(ck_path, cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(ck_path, map_location=device)
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


def collect_seed(model, cfg, n_ep, seed):
    device = next(model.parameters()).device
    gru    = model.rssm.gru
    env    = CartpoleEnv(task='swingup', noisy=False, seed=seed)
    all_h, all_z_g, all_kl = [], [], []

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
                inp    = torch.cat([z, a_t], dim=-1)
                z_gate, _, _ = compute_gru_gates(gru, inp, h)
                h, z, prior_l, post_l = model.rssm.observe_step(h, z, a_t, embed)
                kl_v = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
                all_h.append(h.squeeze(0).cpu().numpy().copy())
                all_z_g.append(z_gate.squeeze(0).cpu().numpy().copy())
                all_kl.append(kl_v)
                obs, _, done = env.step(action)
                step += 1

    return {
        'h':      np.array(all_h,   dtype=np.float32),
        'z_gate': np.array(all_z_g, dtype=np.float32),
        'kl':     np.array(all_kl,  dtype=np.float32),
    }


def probe_pc_angle(h, kl):
    """Train probe on h with KL labels. Return (mean z_gate already done externally),
    mean angle to top-10 PCs of h in scaled space."""
    y = binarise_by_median(kl)
    tr_idx, _ = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=0)
    clf, sc   = train_probe(h[tr_idx], y[tr_idx])

    h_sc = sc.transform(h)
    pca  = PCA(n_components=10, random_state=0).fit(h_sc)
    w_sc = clf.coef_[0] / np.linalg.norm(clf.coef_[0])

    angles = []
    for k in range(10):
        cos = abs(np.dot(w_sc, pca.components_[k]))
        angles.append(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))

    return np.mean(angles), clf.coef_[0], sc, pca


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    results = {}
    for name, ck_path in CHECKPOINTS.items():
        cache = os.path.join(CACHE_DIR, f'seed_verif_{name.replace(" ", "_")}.npz')
        if os.path.exists(cache):
            print(f"Loading cached data for {name}...")
            data = dict(np.load(cache))
        else:
            print(f"Collecting {name} ({N_EP} episodes)...")
            model = load_model(ck_path, cfg)
            # Use different env seed per checkpoint to avoid identical trajectories
            env_seed = hash(name) % 1000 + 100
            data = collect_seed(model, cfg, N_EP, env_seed)
            np.savez(cache, **data)

        mean_z  = data['z_gate'].mean()
        std_z   = data['z_gate'].mean(axis=1).std()
        mean_kl = data['kl'].mean()

        mean_angle, w, sc, pca = probe_pc_angle(data['h'], data['kl'])

        results[name] = dict(
            mean_z=mean_z, std_z=std_z, mean_kl=mean_kl,
            mean_angle=mean_angle, n=len(data['h'])
        )
        print(f"  {name}: mean(z_gate)={mean_z:.4f}  mean_angle(probe,PC1-10)={mean_angle:.1f}°"
              f"  KL={mean_kl:.1f}  N={len(data['h']):,}")

    # ── Print summary ──
    print("\n" + "="*70)
    print("SEED VERIFICATION — z_gate vs PROBE-PC ANGLE")
    print("="*70)
    print(f"\n  Theoretical prediction: z_gate↑ → angle↑ (more saturation → probe more orthogonal)")
    print(f"\n  {'Model':<22}  {'mean(z_gate)':>13}  {'std(z_mean)':>12}  {'angle (°)':>10}  {'KL mean':>8}")
    print(f"  {'-'*22}  {'-'*13}  {'-'*12}  {'-'*10}  {'-'*8}")

    z_vals = []
    a_vals = []
    for name, r in results.items():
        print(f"  {name:<22}  {r['mean_z']:>13.4f}  {r['std_z']:>12.4f}  "
              f"{r['mean_angle']:>10.1f}  {r['mean_kl']:>8.1f}")
        z_vals.append(r['mean_z'])
        a_vals.append(r['mean_angle'])

    z_arr = np.array(z_vals)
    a_arr = np.array(a_vals)
    r_corr, p_corr = pearsonr(z_arr, a_arr) if len(z_arr) >= 3 else (float('nan'), float('nan'))

    print(f"\n  Pearson r(mean z_gate, mean probe-PC angle): {r_corr:+.3f}  (p={p_corr:.3f})")
    print(f"  z_gate range across seeds: [{z_arr.min():.4f}, {z_arr.max():.4f}]  "
          f"span={z_arr.max()-z_arr.min():.4f}")
    print(f"  angle range across seeds: [{a_arr.min():.1f}°, {a_arr.max():.1f}°]  "
          f"span={a_arr.max()-a_arr.min():.1f}°")

    if r_corr > 0.7:
        print(f"\n  PREDICTION VERIFIED (r={r_corr:+.3f}): z_gate saturation increases → "
              f"probe becomes more orthogonal to top PCs.")
    elif r_corr > 0.3:
        print(f"\n  PARTIAL VERIFICATION (r={r_corr:+.3f}): positive trend, but weak across {len(z_vals)} seeds.")
    elif abs(r_corr) < 0.3:
        print(f"\n  INCONCLUSIVE (r={r_corr:+.3f}): no clear trend. Seeds may not vary enough in z_gate.")
        print(f"  z_gate span ({z_arr.max()-z_arr.min():.4f}) may be too small to see the effect.")
    else:
        print(f"\n  NEGATIVE (r={r_corr:+.3f}): inverse trend — inconsistent with prediction.")

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Seed Verification: z_gate Saturation vs Probe-PC Orthogonality',
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    colors = ['blue', 'green', 'orange', 'red']
    for i, (name, r) in enumerate(results.items()):
        ax.scatter(r['mean_z'], r['mean_angle'], s=100, color=colors[i],
                   zorder=5, label=name)
    if len(z_arr) >= 2:
        xs = np.linspace(z_arr.min()-0.001, z_arr.max()+0.001, 100)
        coeffs = np.polyfit(z_arr, a_arr, 1)
        ax.plot(xs, np.polyval(coeffs, xs), 'k--', linewidth=1, alpha=0.5,
                label=f'r={r_corr:+.3f}')
    ax.set_xlabel('mean(z_gate) across states')
    ax.set_ylabel('Mean angle to top-10 PCs (°)')
    ax.set_title('Theoretical prediction: more saturation → more orthogonality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, (name, r) in enumerate(results.items()):
        ax.scatter(r['mean_z'], r['mean_kl'], s=80, color=colors[i], zorder=5, label=name)
    ax.set_xlabel('mean(z_gate)')
    ax.set_ylabel('Mean KL (confusedness level)')
    ax.set_title('z_gate vs KL across seeds\n(sanity: z_gate negatively correlated with KL)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'seed_verification.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {fig_path}")


if __name__ == '__main__':
    main()
