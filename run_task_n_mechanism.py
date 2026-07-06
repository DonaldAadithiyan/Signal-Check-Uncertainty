#!/usr/bin/env python3.11
"""
Task N — Explain the mechanism behind pendulum's Set C inversion.

Task J found pendulum has the strongest C_t encoding (R²=0.886) yet an inverted Set C
AUROC (0.322) — diagnosed as a labelling artefact. This task goes one level deeper:
WHY does the recon-based C1/C2 split invert on pendulum specifically?

Hypothesis: Set C assumes per-step reconstruction error proxies "is the model currently
unreliable" (what C1/C2 are built around). If, on pendulum, recon error is driven mostly
by dynamically hard *instants* (near the unstable upright equilibrium, or at high angular
velocity) rather than by accumulated confusion, then the high-recon C2 group is selecting
"hard instants" not "confused trajectories" — corrupting the split's assumption.

Step 1 (cheap, all 3 envs): compare corr(recon, C_t) vs corr(recon, instantaneous
kinematic feature). If pendulum's recon correlates far more with the kinematic feature
than with C_t, relative to cartpole/reacher, that is the mechanism.

Step 2 (confirming, pendulum only, if step 1 finds a candidate): build a modified Set C
for pendulum that matches C1/C2 on KL AND the kinematic feature; check whether AUROC
un-inverts toward alignment with the strong C_t signal.

Step 3 (fallback): if no kinematic explanation, test whether the percentile-binning
Set-C procedure interacts with pendulum's KL-distribution shape differently.

Per-env kinematic features (from the observation):
  cartpole (obs [cart_x, cosθ, sinθ, cart_v, poleω]): upright = cosθ (obs[1]); |poleω|=|obs[4]|
  reacher  (obs [q0,q1, tt0,tt1, v0,v1]):            dist-to-target ‖obs[2:4]‖; |vel|=‖obs[4:6]‖
  pendulum (obs [cosθ, sinθ, ω]):                    upright = cosθ (obs[0]); |ω|=|obs[2]|

Runs on the existing frozen models. XS, CPU.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from src.config import XS_CONFIG
from src.model.world_model import WorldModel
from src.env.wrapper import CartpoleEnv
from src.env.dmc_wrapper import DMCEnv
from src.probe.linear_probe import binarise_by_median, train_probe, auroc
from src.probe.intervention import compute_ct

N_EP    = 40
GAMMA   = {'cartpole': 0.95, 'reacher': 0.70, 'pendulum': 0.90}
OUT_DIR = 'outputs/causal'

ENVS = {
    'cartpole': dict(ck='outputs/checkpoints/world_model.pt', obs_dim=5, act_dim=1,
                     factory=lambda s: CartpoleEnv(task='swingup', noisy=False, seed=s)),
    'reacher':  dict(ck='outputs/second_env/reacher_easy_world_model.pt', obs_dim=6, act_dim=2,
                     factory=lambda s: DMCEnv('reacher', 'easy', seed=s)),
    'pendulum': dict(ck='outputs/third_env/pendulum_swingup_world_model.pt', obs_dim=3, act_dim=1,
                     factory=lambda s: DMCEnv('pendulum', 'swingup', seed=s)),
}


def kinematic_features(env_name, obs):
    """Return dict of instantaneous kinematic features from the observation array (N, D)."""
    if env_name == 'cartpole':
        return {'upright(cosθ)': obs[:, 1], '|poleω|': np.abs(obs[:, 4])}
    if env_name == 'reacher':
        return {'dist_to_target': np.linalg.norm(obs[:, 2:4], axis=1),
                '|vel|': np.linalg.norm(obs[:, 4:6], axis=1)}
    if env_name == 'pendulum':
        return {'upright(cosθ)': obs[:, 0], '|ω|': np.abs(obs[:, 2])}
    return {}


def load_model(ck, obs_dim, act_dim):
    d = torch.load(ck, map_location='cpu')
    cfg = d['cfg']
    m = WorldModel(obs_dim, act_dim, cfg)
    m.load_state_dict(d['model_state']); m.eval()
    return m


@torch.no_grad()
def collect(model, env_factory, act_dim, n_ep, seed=555):
    """Collect obs, kl, recon, traj_id, h, z with a random policy (frozen model)."""
    cfg = XS_CONFIG.copy()
    env = env_factory(seed)
    np.random.seed(seed)
    O, KL, RC, TR, H, Z = [], [], [], [], [], []
    for ep in range(n_ep):
        obs = env.reset()
        h = torch.zeros(1, cfg['rssm_deter'])
        z = torch.zeros(1, cfg['rssm_stoch'] * cfg['rssm_classes'])
        done, step = False, 0
        while not done and step < cfg['episode_max_steps']:
            a = np.random.uniform(-1, 1, (act_dim,)).astype(np.float32)
            ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            at = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
            emb = model.encoder(ot)
            h, z, prior_l, post_l = model.rssm.observe_step(h, z, at, emb)
            dec = model.decoder(torch.cat([h, z], dim=-1))
            kl = model.rssm.kl_divergence(post_l, prior_l, free_bits=0.0).item()
            rc = F.mse_loss(dec, ot, reduction='none').sum().item()
            O.append(obs.copy()); KL.append(kl); RC.append(rc); TR.append(ep)
            H.append(h.squeeze(0).numpy().copy()); Z.append(post_l.squeeze(0).numpy().copy())
            obs, _, done = env.step(a); step += 1
    return dict(obs=np.array(O, np.float32), kl=np.array(KL, np.float32),
                recon=np.array(RC, np.float32), traj_id=np.array(TR, np.int64),
                h=np.array(H, np.float32), z=np.array(Z, np.float32))


def build_set_c(data, extra_match=None, n_bins=10, per_bin=20, max_total=200, seed=42):
    """KL-matched (and optionally extra-feature-matched) contrastive set from a single
    env's states. C1 = low recon, C2 = high recon within each (KL[,extra]) bin.
    extra_match: optional 1-D feature array to additionally bin on (2-D binning)."""
    kl, rc = data['kl'], data['recon']
    rng = np.random.default_rng(seed)
    kl_edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1))
    kl_bin = np.digitize(kl, kl_edges[1:-1])
    if extra_match is not None:
        n_ebins = 5
        e_edges = np.percentile(extra_match, np.linspace(0, 100, n_ebins + 1))
        e_bin = np.digitize(extra_match, e_edges[1:-1])
        cells = [(b, e) for b in range(n_bins) for e in range(n_ebins)]
        keyfn = lambda: (kl_bin, e_bin)
    else:
        cells = [(b,) for b in range(n_bins)]
        keyfn = lambda: (kl_bin,)
    keys = keyfn()
    c1, c2 = [], []
    for cell in cells:
        mask = np.ones(len(kl), bool)
        for kv, cv in zip(keys, cell):
            mask &= (kv == cv)
        idx = np.where(mask)[0]
        if len(idx) < 4:
            continue
        rb = rc[idx]
        c1c = idx[rb <= np.percentile(rb, 25)]
        c2c = idx[rb >= np.percentile(rb, 75)]
        n = min(per_bin, len(c1c), len(c2c))
        if n == 0:
            continue
        c1.extend(rng.choice(c1c, n, replace=False).tolist())
        c2.extend(rng.choice(c2c, n, replace=False).tolist())
    if len(c1) > max_total:
        c1 = rng.choice(c1, max_total, replace=False).tolist()
    if len(c2) > max_total:
        c2 = rng.choice(c2, max_total, replace=False).tolist()
    return dict(h=np.concatenate([data['h'][c1], data['h'][c2]]),
                labels=np.array([0]*len(c1) + [1]*len(c2), np.int32),
                idx_c1=c1, idx_c2=c2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {'step1': {}, 'step2': {}, 'step3': {}}

    print("=" * 74)
    print("TASK N — MECHANISM OF PENDULUM'S SET C INVERSION")
    print("=" * 74)

    # ── Step 1: corr(recon, C_t) vs corr(recon, kinematic) across all 3 envs ──
    print("\n[Step 1] corr(recon, C_t) vs corr(recon, kinematic feature), per environment:")
    env_data = {}
    for name, spec in ENVS.items():
        model = load_model(spec['ck'], spec['obs_dim'], spec['act_dim'])
        data = collect(model, spec['factory'], spec['act_dim'], N_EP)
        ct = compute_ct(data['kl'], data['traj_id'], gamma=GAMMA[name])
        data['ct'] = ct
        env_data[name] = data
        r_recon_ct, _ = pearsonr(data['recon'], ct)
        feats = kinematic_features(name, data['obs'])
        feat_corrs = {fn: pearsonr(data['recon'], fv)[0] for fn, fv in feats.items()}
        best_feat = max(feat_corrs, key=lambda k: abs(feat_corrs[k]))
        results['step1'][name] = dict(r_recon_ct=float(r_recon_ct),
                                      feat_corrs={k: float(v) for k, v in feat_corrs.items()},
                                      best_feat=best_feat, best_feat_r=float(feat_corrs[best_feat]),
                                      n=len(data['recon']))
        print(f"\n  {name} (N={len(data['recon']):,}):")
        print(f"    r(recon, C_t)                 = {r_recon_ct:+.3f}")
        for fn, fv in feat_corrs.items():
            print(f"    r(recon, {fn:<16})    = {fv:+.3f}")
        print(f"    → dominant driver of recon: "
              f"{'kinematic ('+best_feat+')' if abs(feat_corrs[best_feat]) > abs(r_recon_ct) else 'C_t (accumulated confusion)'}")

    # ── Diagnosis ──
    print("\n" + "-" * 74)
    pend = results['step1']['pendulum']
    pend_kin_dominates = abs(pend['best_feat_r']) > abs(pend['r_recon_ct'])
    # relative: how much more does kinematic dominate on pendulum vs others?
    def ratio(nm):
        s = results['step1'][nm]
        return abs(s['best_feat_r']) / (abs(s['r_recon_ct']) + 1e-6)
    print(f"  Kinematic-vs-C_t recon-correlation ratio (|r_kin|/|r_ct|):")
    for nm in ENVS:
        print(f"    {nm:<10}: {ratio(nm):.2f}  (kin r={results['step1'][nm]['best_feat_r']:+.2f}, "
              f"ct r={results['step1'][nm]['r_recon_ct']:+.2f})")
    candidate = pend['best_feat'] if pend_kin_dominates else None
    if candidate:
        print(f"\n  CANDIDATE MECHANISM: on pendulum, recon error is driven more by the instantaneous")
        print(f"  kinematic feature '{candidate}' (r={pend['best_feat_r']:+.2f}) than by accumulated")
        print(f"  confusion C_t (r={pend['r_recon_ct']:+.2f}). The high-recon C2 group is selecting")
        print(f"  dynamically hard *instants*, not confused *trajectories*. → run Step 2.")

    # ── Step 2: modified Set C for pendulum controlling for the candidate feature ──
    if candidate:
        print(f"\n[Step 2] Modified Set C for pendulum — match C1/C2 on KL AND '{candidate}':")
        pdata = env_data['pendulum']
        # train probe A on pendulum states (same protocol as pipeline)
        y = binarise_by_median(pdata['kl'])
        tr_idx, _ = train_test_split(np.arange(len(pdata['h'])), test_size=0.40,
                                     stratify=y, random_state=0)
        clf, sc = train_probe(pdata['h'][tr_idx], y[tr_idx])

        feat_arr = kinematic_features('pendulum', pdata['obs'])[candidate]
        sc_orig = build_set_c(pdata, extra_match=None, seed=42)
        sc_mod = build_set_c(pdata, extra_match=feat_arr, seed=42)
        auroc_orig = auroc(clf, sc, sc_orig['h'], sc_orig['labels'])
        auroc_mod = auroc(clf, sc, sc_mod['h'], sc_mod['labels'])

        # verify the modification actually balanced the feature across C1/C2
        def feat_gap(setc):
            f1 = feat_arr[setc['idx_c1']].mean(); f2 = feat_arr[setc['idx_c2']].mean()
            return f1, f2
        o1, o2 = feat_gap(sc_orig); m1, m2 = feat_gap(sc_mod)
        print(f"    original Set C:  AUROC={auroc_orig:.4f}  |  "
              f"{candidate} C1={o1:.3f} C2={o2:.3f} (gap {o2-o1:+.3f})")
        print(f"    modified Set C:  AUROC={auroc_mod:.4f}  |  "
              f"{candidate} C1={m1:.3f} C2={m2:.3f} (gap {m2-m1:+.3f})")
        results['step2'] = dict(candidate=candidate, auroc_orig=float(auroc_orig),
                                auroc_mod=float(auroc_mod),
                                feat_gap_orig=float(o2 - o1), feat_gap_mod=float(m2 - m1),
                                n_c1=len(sc_mod['idx_c1']), n_c2=len(sc_mod['idx_c2']))
        moved_up = auroc_mod - auroc_orig
        print(f"\n    Δ AUROC (modified − original) = {moved_up:+.4f}")
        if auroc_mod > 0.5 and moved_up > 0.1:
            print(f"    MECHANISM CONFIRMED: controlling for '{candidate}' un-inverts Set C")
            print(f"    ({auroc_orig:.3f} → {auroc_mod:.3f}), moving it back toward the strong C_t signal.")
            print(f"    The inversion was caused by recon error tracking hard kinematic instants,")
            print(f"    not accumulated confusion — a demonstrated, tested mechanism.")
        elif moved_up > 0.05:
            print(f"    PARTIAL: controlling for '{candidate}' moves AUROC up by {moved_up:+.3f} but")
            print(f"    does not fully un-invert. The feature explains part of the inversion.")
        else:
            print(f"    NOT CONFIRMED: controlling for '{candidate}' does not un-invert Set C")
            print(f"    (Δ={moved_up:+.3f}). The candidate is correlated with recon but is not the")
            print(f"    driver of the inversion → fall through to Step 3.")
            candidate = None if moved_up <= 0.05 else candidate

    # ── Step 3: fallback — labelling-procedure / KL-distribution-shape hypothesis ──
    if not candidate or (results.get('step2') and results['step2'].get('auroc_mod', 0) <= 0.55):
        print(f"\n[Step 3] Fallback — is the inversion a property of the recon→C_t alignment sign?")
        # The decisive check: within each env, does higher recon (C2) correspond to higher or
        # LOWER C_t? If C2 (high recon) has LOWER C_t on pendulum, the recon-based label is
        # anti-aligned with confusion → probe (which reads C_t) inverts on that label.
        for name, data in env_data.items():
            sc_set = build_set_c(data, seed=42)
            ct = data['ct']
            ct_c1 = ct[sc_set['idx_c1']].mean()   # low recon group
            ct_c2 = ct[sc_set['idx_c2']].mean()   # high recon group
            results['step3'][name] = dict(ct_c1_lowrecon=float(ct_c1), ct_c2_highrecon=float(ct_c2),
                                          ct_gap=float(ct_c2 - ct_c1))
            aligned = ct_c2 > ct_c1
            print(f"    {name:<10}: mean C_t  low-recon(C1)={ct_c1:.3f}  high-recon(C2)={ct_c2:.3f}  "
                  f"gap={ct_c2-ct_c1:+.3f}  → recon {'ALIGNED' if aligned else 'ANTI-ALIGNED'} with C_t")
        pg = results['step3']['pendulum']['ct_gap']
        print(f"\n    On pendulum the high-recon C2 group has {'HIGHER' if pg>0 else 'LOWER'} C_t than")
        print(f"    low-recon C1 (gap {pg:+.3f}). If anti-aligned, the recon-based Set C label runs")
        print(f"    opposite to the confusion the probe reads — exactly producing an inverted AUROC.")

    # ── Step 4 (confirming): C_t-labelled Set C on pendulum should NOT invert ──
    # The mechanism above predicts: if we label C1/C2 by C_t (accumulated confusion) instead
    # of by recon — KL-matched, same construction otherwise — the pendulum probe should score
    # WELL ABOVE 0.5, because the probe reads C_t and now the label agrees with C_t. This is
    # the confirming experiment: it un-inverts the result by fixing the labelling target.
    print(f"\n[Step 4 — CONFIRMING] C_t-labelled Set C on pendulum (C1=low C_t, C2=high C_t, KL-matched):")
    from src.probe.intervention import bootstrap_auroc_ci
    pdata = env_data['pendulum']
    y = binarise_by_median(pdata['kl'])
    tr_idx, _ = train_test_split(np.arange(len(pdata['h'])), test_size=0.40,
                                 stratify=y, random_state=0)
    clf, sc = train_probe(pdata['h'][tr_idx], y[tr_idx])

    def build_set_c_by_ct(data, n_bins=10, per_bin=20, max_total=200, seed=42):
        kl, ct = data['kl'], data['ct']
        rng = np.random.default_rng(seed)
        edges = np.percentile(kl, np.linspace(0, 100, n_bins + 1)); kb = np.digitize(kl, edges[1:-1])
        c1, c2 = [], []
        for b in range(n_bins):
            idx = np.where(kb == b)[0]
            if len(idx) < 4:
                continue
            cb = ct[idx]
            c1c = idx[cb <= np.percentile(cb, 25)]   # low C_t = coping
            c2c = idx[cb >= np.percentile(cb, 75)]   # high C_t = confused
            n = min(per_bin, len(c1c), len(c2c))
            if n == 0:
                continue
            c1.extend(rng.choice(c1c, n, replace=False).tolist())
            c2.extend(rng.choice(c2c, n, replace=False).tolist())
        if len(c1) > max_total: c1 = rng.choice(c1, max_total, replace=False).tolist()
        if len(c2) > max_total: c2 = rng.choice(c2, max_total, replace=False).tolist()
        return dict(h=np.concatenate([data['h'][c1], data['h'][c2]]),
                    labels=np.array([0]*len(c1) + [1]*len(c2), np.int32))

    sc_recon = build_set_c(pdata, seed=42)                 # original recon-labelled
    sc_ct = build_set_c_by_ct(pdata, seed=42)              # C_t-labelled
    auroc_recon = auroc(clf, sc, sc_recon['h'], sc_recon['labels'])
    scores_ct = clf.predict_proba(sc.transform(sc_ct['h']))[:, 1]
    ci_ct = bootstrap_auroc_ci(sc_ct['labels'], scores_ct, seed=0)
    results['step4'] = dict(auroc_recon_labelled=float(auroc_recon),
                            auroc_ct_labelled=float(ci_ct[0]), ci_ct=list(ci_ct))
    print(f"    recon-labelled Set C AUROC (original):  {auroc_recon:.4f}  (inverted, <0.5)")
    print(f"    C_t-labelled Set C AUROC (confirming):  {ci_ct[0]:.4f}  95% CI [{ci_ct[1]:.3f}, {ci_ct[2]:.3f}]")
    if ci_ct[1] > 0.5:
        print(f"\n    MECHANISM CONFIRMED: relabelling Set C by C_t (instead of recon) un-inverts the")
        print(f"    result — AUROC {auroc_recon:.3f} → {ci_ct[0]:.3f} (CI above 0.5). The probe reads")
        print(f"    confusion correctly; the pendulum inversion is caused specifically by the")
        print(f"    recon-based labelling running opposite to C_t on pendulum's dynamics, NOT by any")
        print(f"    failure of the confusion signal. Demonstrated, tested mechanism.")
    else:
        print(f"\n    NOT fully un-inverted — the C_t-labelled set also does not separate; the mechanism")
        print(f"    is not purely the recon↔C_t sign. Reported honestly.")

    with open(os.path.join(OUT_DIR, 'task_n_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved: {os.path.join(OUT_DIR, 'task_n_results.json')}")


if __name__ == '__main__':
    main()
