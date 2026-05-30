#!/usr/bin/env python3.11
"""
Checkpoint verification of the z_gate saturation theory (§6.2 fix).

Loads checkpoints at 5K, 10K, 20K, 40K, 70K, 100K training steps.
For each checkpoint:
  1. Collect 5000 states (h_t, z_gate, kl) via the model at that training stage
  2. Train probe on h_t with KL binary labels
  3. Fit PCA on scaled h_t
  4. Measure mean(z_gate) and mean angle between probe direction and top-10 PCs

Plots (mean z_gate vs mean probe-PC angle) across checkpoints.
Expected: z_gate grows from ~0.5 toward 0.94 during training;
angle grows from <90° toward ~88° as saturation increases.
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


CHECKPOINT_STEPS = [5_000, 10_000, 20_000, 40_000, 70_000, 100_000]
CKPT_DIR         = 'outputs/checkpoints'
N_COLLECT        = 5_000
CACHE_DIR        = 'outputs/data'


def load_checkpoint(step, cfg):
    path   = os.path.join(CKPT_DIR, f'ckpt_{step:06d}.pt')
    device = torch.device(cfg.get('device', 'cpu'))
    ck     = torch.load(path, map_location=device)
    m      = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
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


def collect_at_checkpoint(model, cfg, n, seed):
    device = next(model.parameters()).device
    gru    = model.rssm.gru
    env    = CartpoleEnv(task='swingup', noisy=False, seed=seed)
    all_h, all_z_g, all_kl = [], [], []
    step = 0
    obs  = env.reset()
    h    = torch.zeros(1, cfg['rssm_deter'], device=device)
    z    = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
    done = False
    with torch.no_grad():
        while len(all_h) < n:
            if done:
                obs = env.reset()
                h   = torch.zeros(1, cfg['rssm_deter'], device=device)
                z   = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
                step = 0
                done = False
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
        'h':      np.array(all_h[:n],   dtype=np.float32),
        'z_gate': np.array(all_z_g[:n], dtype=np.float32),
        'kl':     np.array(all_kl[:n],  dtype=np.float32),
    }


def measure_angle(h, kl):
    y = binarise_by_median(kl)
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan'), float('nan')
    tr, _ = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=0)
    clf, sc = train_probe(h[tr], y[tr])
    h_sc = sc.transform(h)
    pca  = PCA(n_components=10, random_state=0).fit(h_sc)
    w    = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    angles = [np.degrees(np.arccos(np.clip(abs(np.dot(w, pca.components_[k])), 0, 1)))
              for k in range(10)]
    return np.mean(angles), w, pca


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    results = []
    for step in CHECKPOINT_STEPS:
        ckpt_path = os.path.join(CKPT_DIR, f'ckpt_{step:06d}.pt')
        if not os.path.exists(ckpt_path):
            print(f"  Missing checkpoint at step {step} — run run_checkpoint_training.py first.")
            continue

        cache = os.path.join(CACHE_DIR, f'ckpt_verif_{step:06d}.npz')
        if os.path.exists(cache):
            print(f"  Loading cached states for step {step}...")
            data = dict(np.load(cache))
        else:
            print(f"  Collecting states at step {step}...")
            model = load_checkpoint(step, cfg)
            data  = collect_at_checkpoint(model, cfg, N_COLLECT, seed=99)
            np.savez(cache, **data)

        mean_z  = data['z_gate'].mean()
        std_z   = data['z_gate'].mean(axis=1).std()
        mean_kl = data['kl'].mean()
        angle, _, _ = measure_angle(data['h'], data['kl'])

        results.append(dict(step=step, mean_z=mean_z, std_z=std_z,
                            mean_kl=mean_kl, angle=angle))
        print(f"    step={step:>7,}  z_gate={mean_z:.4f}  angle={angle:.2f}°  KL={mean_kl:.1f}")

    if not results:
        print("\nNo checkpoints found. Run run_checkpoint_training.py first.")
        return

    print("\n" + "="*70)
    print("CHECKPOINT VERIFICATION — z_gate SATURATION VS PROBE-PC ANGLE")
    print("="*70)
    print(f"\n  {'Step':>8}  {'mean(z_gate)':>13}  {'Probe-PC angle':>15}  {'KL mean':>8}")
    print(f"  {'-'*8}  {'-'*13}  {'-'*15}  {'-'*8}")
    for r in results:
        print(f"  {r['step']:>8,}  {r['mean_z']:>13.4f}  {r['angle']:>15.2f}°  {r['mean_kl']:>8.1f}")

    z_arr = np.array([r['mean_z']  for r in results])
    a_arr = np.array([r['angle']   for r in results])
    s_arr = np.array([r['step']    for r in results])

    if len(results) >= 3:
        r_corr, p_corr = pearsonr(z_arr, a_arr)
        r_step_z, _    = pearsonr(s_arr, z_arr)
        r_step_a, _    = pearsonr(s_arr, a_arr)

        print(f"\n  Pearson r(mean z_gate, probe-PC angle): {r_corr:+.3f}  (p={p_corr:.4f})")
        print(f"  Pearson r(step, mean z_gate):           {r_step_z:+.3f}")
        print(f"  Pearson r(step, probe-PC angle):        {r_step_a:+.3f}")
        print(f"\n  z_gate range: [{z_arr.min():.4f}, {z_arr.max():.4f}]  "
              f"span={z_arr.max()-z_arr.min():.4f}")
        print(f"  angle range:  [{a_arr.min():.2f}°, {a_arr.max():.2f}°]  "
              f"span={a_arr.max()-a_arr.min():.2f}°")

        if z_arr.max() - z_arr.min() > 0.10:
            print("\n  Real z_gate variance across checkpoints — theory testable.")
            if r_corr > 0.80:
                print(f"  PREDICTION VERIFIED: r={r_corr:+.3f}, span={z_arr.max()-z_arr.min():.3f}.")
                print("  z_gate saturation develops during training; confusion signal")
                print("  progressively migrates to the null space as the GRU learns")
                print("  its always-overwrite policy.")
            elif r_corr > 0.50:
                print(f"  PARTIAL VERIFICATION: r={r_corr:+.3f}. Positive trend.")
            else:
                print(f"  WEAK: r={r_corr:+.3f}. Theory not strongly supported.")
        else:
            print(f"\n  z_gate span ({z_arr.max()-z_arr.min():.4f}) small — early checkpoints")
            print("  may not differ enough from later ones at this scale.")

    # ── Figure ──
    if len(results) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle('Checkpoint Verification: z_gate Saturation During Training',
                     fontsize=11, fontweight='bold')

        ax = axes[0]
        ax.plot(s_arr / 1000, z_arr, 'b-o', markersize=7, linewidth=2)
        ax.set_xlabel('Training steps (K)')
        ax.set_ylabel('mean(z_gate)')
        ax.set_title('z_gate saturation over training\n(expected: grows toward 0.94)')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(s_arr / 1000, a_arr, 'r-o', markersize=7, linewidth=2)
        ax.set_xlabel('Training steps (K)')
        ax.set_ylabel('Mean probe-PC angle (°)')
        ax.set_title('Confusion signal migrates to null-space\n(expected: grows toward 88°)')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.scatter(z_arr, a_arr, c=s_arr / 1000, cmap='viridis', s=80, zorder=5)
        for r in results:
            ax.annotate(f"{r['step']//1000}K", (r['mean_z'], r['angle']),
                        textcoords='offset points', xytext=(5, 3), fontsize=8)
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=plt.Normalize(s_arr.min()/1000, s_arr.max()/1000))
        plt.colorbar(sm, ax=ax, label='Training steps (K)')
        if len(results) >= 2:
            coeffs = np.polyfit(z_arr, a_arr, 1)
            xs = np.linspace(z_arr.min(), z_arr.max(), 100)
            ax.plot(xs, np.polyval(coeffs, xs), 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('mean(z_gate)')
        ax.set_ylabel('Mean probe-PC angle (°)')
        ax.set_title('z_gate vs orthogonality\n(theory: positive relationship)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(cfg['figures_dir'], 'checkpoint_verification.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n  Figure saved: {fig_path}")


if __name__ == '__main__':
    main()
