#!/usr/bin/env python3.11
"""
Task B — Causal (not observational) test of the z_gate mechanism.

The original z_gate story: as the GRU update gate z_t saturates (→0.94), the
confusion direction moves into the near-null space of h_t's PCA (large angle to
top PCs). The observational test used 6 checkpoints (r=-0.889, n=6) but is
confounded: checkpoint number co-varies with z_gate AND training progress AND
representation quality.

This is a PURE INFERENCE-TIME CAUSAL PROBE on ONE frozen fully-trained model:
we intercept the GRU update-gate output and force z_gate to a fixed scalar for
every step, holding everything else identical, then recompute the h_t
distribution, re-fit PCA, and re-measure the confusion-direction / top-PC angle.

Sweep z ∈ {0.5,0.7,0.8,0.9,0.94(natural),0.97,0.99}. The 0.94 value should
reproduce baseline (sanity). If the angle grows monotonically with forced z,
the causal claim is supported; if not, that further supports the revised
structural (content-based) account. Reported honestly either way.
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.probe.linear_probe import binarise_by_median, train_probe
from src.probe.intervention import rssm_observe_with_override

Z_SWEEP  = [0.50, 0.70, 0.80, 0.90, 0.94, 0.97, 0.99]  # 0.94 ≈ natural
NATURAL  = 0.94
N_EP     = 20
TOP_K    = 10
OUT_DIR  = 'outputs/causal'


def load_model(cfg):
    device = torch.device(cfg.get('device', 'cpu'))
    ck = torch.load(cfg['checkpoint_path'], map_location=device)
    m = WorldModel(cfg['obs_dim'], cfg['act_dim'], ck['cfg']).to(device)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m


@torch.no_grad()
def collect_with_zoverride(model, cfg, z_override, n_ep, seed=555):
    """Roll trajectories with the GRU update gate forced to z_override (or None
    for natural). Returns h, kl, and realised mean z_gate."""
    device = next(model.parameters()).device
    env = CartpoleEnv(task='swingup', noisy=False, seed=seed)
    np.random.seed(seed)
    all_h, all_kl, all_zg = [], [], []
    for ep in range(n_ep):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'], device=device)
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'], device=device)
        done, step = False, 0
        while not done and step < cfg['episode_max_steps']:
            a = np.random.uniform(-1, 1, (cfg['act_dim'],)).astype(np.float32)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
            embed = model.encoder(obs_t)
            h, z, prior_l, post_l, zg = rssm_observe_with_override(
                model.rssm, h, z, a_t, embed, z_override=z_override)
            kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            all_h.append(h.squeeze(0).cpu().numpy().copy())
            all_kl.append(kl)
            all_zg.append(float(zg.mean()))
            obs, _, done = env.step(a)
            step += 1
    return (np.array(all_h, dtype=np.float32),
            np.array(all_kl, dtype=np.float32),
            float(np.mean(all_zg)))


def probe_pc_angle(h, kl, top_k=TOP_K):
    """Train a KL-label probe on h, fit PCA on scaled h, return mean angle (deg)
    between the probe direction and the top-k PCs (in scaled space)."""
    y = binarise_by_median(kl)
    if len(np.unique(y)) < 2:
        return float('nan')
    tr_idx, _ = train_test_split(np.arange(len(h)), test_size=0.40,
                                 stratify=y, random_state=0)
    clf, sc = train_probe(h[tr_idx], y[tr_idx])
    h_sc = sc.transform(h)
    pca = PCA(n_components=top_k, random_state=0).fit(h_sc)
    w = clf.coef_[0] / np.linalg.norm(clf.coef_[0])
    angles = []
    for k in range(top_k):
        cos = abs(np.dot(w, pca.components_[k]))
        angles.append(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
    # also fraction of probe direction variance explained by top-k PCs
    proj = pca.components_ @ w
    frac_in_topk = float(np.sum(proj ** 2))
    return float(np.mean(angles)), frac_in_topk


def main():
    cfg = XS_CONFIG.copy()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(cfg['figures_dir'], exist_ok=True)

    print("Loading frozen fully-trained model...")
    model = load_model(cfg)

    rows = []
    print(f"\nSweeping forced z_gate over {Z_SWEEP} (natural≈{NATURAL})...")
    print(f"  {'forced z':>9}  {'realised z':>11}  {'mean angle°':>11}  "
          f"{'frac in top-{}PC'.format(TOP_K):>15}  {'mean KL':>8}  {'N':>6}")
    print(f"  {'-'*9}  {'-'*11}  {'-'*11}  {'-'*15}  {'-'*8}  {'-'*6}")

    # natural baseline first (z_override=None) as the sanity anchor
    h_nat, kl_nat, zg_nat = collect_with_zoverride(model, cfg, None, N_EP)
    ang_nat, frac_nat = probe_pc_angle(h_nat, kl_nat)
    print(f"  {'natural':>9}  {zg_nat:>11.4f}  {ang_nat:>11.2f}  {frac_nat:>15.4f}  "
          f"{kl_nat.mean():>8.2f}  {len(h_nat):>6}")

    for zf in Z_SWEEP:
        h, kl, zg = collect_with_zoverride(model, cfg, zf, N_EP)
        ang, frac = probe_pc_angle(h, kl)
        rows.append(dict(forced_z=zf, realised_z=zg, angle=ang,
                         frac_in_topk=frac, mean_kl=float(kl.mean()), n=len(h)))
        print(f"  {zf:>9.2f}  {zg:>11.4f}  {ang:>11.2f}  {frac:>15.4f}  "
              f"{kl.mean():>8.2f}  {len(h):>6}")

    # ── Causal trend: forced z vs angle ──
    zf_arr = np.array([r['forced_z'] for r in rows])
    ang_arr = np.array([r['angle'] for r in rows])
    frac_arr = np.array([r['frac_in_topk'] for r in rows])
    valid = ~np.isnan(ang_arr)
    r_ang, p_ang = pearsonr(zf_arr[valid], ang_arr[valid])
    r_frac, p_frac = pearsonr(zf_arr[valid], frac_arr[valid])

    print("\n" + "=" * 74)
    print("TASK B — CAUSAL z_gate OVERRIDE RESULT")
    print("=" * 74)
    print(f"\n  Original observational finding: r(z_gate, angle) = -0.889 (n=6, confounded)")
    print(f"  Original theory prediction:     forced z ↑  ⇒  angle ↑  (more orthogonality)")
    print(f"\n  CAUSAL (single frozen model, inference-time override, n={valid.sum()} z-values):")
    print(f"    Pearson r(forced z, mean angle)        = {r_ang:+.3f}  (p={p_ang:.3f})")
    print(f"    Pearson r(forced z, frac in top-{TOP_K} PC) = {r_frac:+.3f}  (p={p_frac:.3f})")
    print(f"    Natural-z sanity: forced 0.94 angle={ang_arr[zf_arr==0.94][0]:.2f}° "
          f"vs natural {ang_nat:.2f}° (should be close)")

    angle_span = float(ang_arr[valid].max() - ang_arr[valid].min())
    min_angle  = float(ang_arr[valid].min())
    max_frac   = float(frac_arr[valid].max())
    print(f"    Angle span across ALL forced z: {angle_span:.2f}°  "
          f"(min {min_angle:.2f}°)  |  max frac in top-{TOP_K} PC: {max_frac:.4f}")

    # The scientifically decisive facts are (a) the magnitude of the angle move and
    # (b) whether the near-null-space geometry survives across the WHOLE sweep.
    # A statistically-significant Pearson r over a <1° span, while the direction stays
    # ~89° (frac<<1) at every z including z=0.5, does NOT support "saturation CREATES
    # the orthogonality" — it shows the geometry is largely z-independent.
    near_null_everywhere = (min_angle > 85.0) and (max_frac < 0.02)
    if near_null_everywhere:
        verdict = (f"z-INDEPENDENT NULL-SPACE GEOMETRY: the confusion direction stays near-"
                   f"orthogonal to the top PCs at EVERY forced z (angle {min_angle:.1f}–"
                   f"{ang_arr[valid].max():.1f}°, ≤{max_frac*100:.1f}% variance in top-{TOP_K} "
                   f"PCs even at z=0.5). Forced z moves the angle by only {angle_span:.2f}° "
                   f"(Pearson r={r_ang:+.2f} over that tiny span). Gate saturation is therefore "
                   f"NOT the causal driver of the null-space geometry — this CAUSALLY supports "
                   f"the revised structural (content-based) account over the original saturation "
                   f"story, and does so on a single frozen model free of the checkpoint confound.")
    elif r_ang > 0.5 and p_ang < 0.1 and angle_span > 3.0:
        verdict = ("CAUSAL SUPPORT for saturation story: forcing higher z_gate substantially "
                   "increases the probe-to-top-PC angle.")
    elif r_ang < -0.5 and p_ang < 0.1 and angle_span > 3.0:
        verdict = ("CAUSAL REFUTATION: forcing higher z_gate substantially DECREASES the angle.")
    else:
        verdict = ("NO SUBSTANTIVE CAUSAL EFFECT: forced z_gate barely moves the angle "
                   f"({angle_span:.2f}° span). The gate is not the causal driver of the "
                   "null-space geometry — consistent with the revised structural account.")
    print(f"\n  {verdict}")

    results = dict(rows=rows, natural=dict(realised_z=zg_nat, angle=ang_nat,
                   frac_in_topk=frac_nat, mean_kl=float(kl_nat.mean())),
                   r_forcedz_angle=float(r_ang), p_forcedz_angle=float(p_ang),
                   r_forcedz_frac=float(r_frac), p_forcedz_frac=float(p_frac),
                   angle_span=angle_span, min_angle=min_angle, max_frac_in_topk=max_frac,
                   verdict=verdict)
    with open(os.path.join(OUT_DIR, 'task_b_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Figure ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.suptitle('Task B — Causal z_gate Override (single frozen model)',
                 fontweight='bold', fontsize=12)
    ax = axes[0]
    ax.plot(zf_arr, ang_arr, 'b-o', markersize=7)
    ax.axhline(ang_nat, color='green', ls='--', lw=1, label=f'natural angle ({ang_nat:.1f}°)')
    ax.axvline(NATURAL, color='gray', ls=':', lw=1, label=f'natural z≈{NATURAL}')
    ax.set_xlabel('forced z_gate'); ax.set_ylabel(f'mean angle to top-{TOP_K} PCs (°)')
    ax.set_title(f'Causal: forced z vs angle\nr={r_ang:+.3f} (p={p_ang:.3f})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(zf_arr, frac_arr, 'r-s', markersize=7)
    ax.set_xlabel('forced z_gate'); ax.set_ylabel(f'frac of probe dir in top-{TOP_K} PCs')
    ax.set_title(f'Probe-direction variance in top PCs\nr={r_frac:+.3f} (p={p_frac:.3f})')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(cfg['figures_dir'], 'zgate_causal.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {fig_path}")
    print(f"  Results saved: {os.path.join(OUT_DIR, 'task_b_results.json')}")


if __name__ == '__main__':
    main()
