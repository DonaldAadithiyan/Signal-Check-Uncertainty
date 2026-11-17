#!/usr/bin/env python3.11
"""Camera-ready Part A, step 4: SAE results (paper Tables 7-9) on corrected states.

Two variants, both reported:
  'published' : the three SAE dictionaries used in the paper
                (outputs/phase3_sae/sae_seed{0,1,2}.pt, trained on the old-convention
                training log), applied to corrected states and the corrected v.
                The published atoms (#156 cartpole, #612 reacher, #310 pendulum; SAE
                run 0) are re-measured and re-ablated.
  'retrained' : the paper's SAE recipe (TopK, 256->2048, k=48, 40 epochs, batch 1024,
                seeds 0,1,2) retrained on corrected probe-fit states pooled over the
                three tasks' primary models; atoms re-selected with the paper's rule.
Paper rules reused:
  share of atom i = |c_i| / sum_j |c_j|, c = ridge (lambda=1e-2) reconstruction of v
  from the unit-norm decoder atoms; a random unit vector is decomposed the same way.
  Selection: on one half of the held-out sites (evaluation episodes 0-49), among the
  20 atoms with the largest activation variance, the atom with the largest |r(act, E10)|;
  evaluated on the other half (episodes 50-99).
  Ablation: h' = h - (h.d)d along the atom's unit decoder column, at 800 evaluation
  sites; z of mean change in E (K=10) and readout (k=0) vs the canonical 50-direction
  null on the same sites.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr

from src.camera_ready import core as C
from src.camera_ready import causal as CZ
from src.probe.sae import TopKSAE, train_topk_sae, get_all_activations, sae_metrics
from src.rerun import pipeline as P
from run_cr_A_causal import fit_probe

OUT = os.environ.get('CR_RAW_DIR', os.path.join(C.ROOT, 'A_raw'))
SAE_DIR = os.path.join(C.ROOT, 'sae')
TASKS = ['cartpole', 'reacher', 'pendulum']
PUBLISHED_ATOMS = {'cartpole': 156, 'reacher': 612, 'pendulum': 310}
N_ATOMS, K_SPARSE, N_EPOCHS, BATCH = 2048, 48, 40, 1024
N_SITES_ABL = 800
SITE_SEED = 33


def load_sae(path):
    s = TopKSAE(d_in=256, n_atoms=N_ATOMS, k=K_SPARSE)
    s.load_state_dict(torch.load(path, map_location='cpu'))
    return s.eval()


def decompose(sae, v):
    D = sae.decoder.weight.detach().numpy().astype(np.float64)
    c = np.linalg.solve(D.T @ D + 1e-2 * np.eye(D.shape[1]), D.T @ v)
    share = np.abs(c) / np.abs(c).sum()
    o = np.argsort(-share)
    recon_cos = float((D @ c) @ v / (np.linalg.norm(D @ c) * np.linalg.norm(v)))
    return dict(top1=float(share[o[0]]), top4=float(share[o[:4]].sum()), top10=float(share[o[:10]].sum()),
                top10_atoms=o[:10].tolist(), recon_cos=recon_cos)


def random_decomp(sae, n=20, seed=5):
    rng = np.random.default_rng(seed)
    rows = [decompose(sae, (r := rng.standard_normal(256)) / np.linalg.norm(r)) for _ in range(n)]
    return {k: dict(mean=float(np.mean([r[k] for r in rows])), min=float(np.min([r[k] for r in rows])),
                    max=float(np.max([r[k] for r in rows]))) for k in ('top1', 'top4', 'top10', 'recon_cos')}


def site_data(task):
    ev = C.load_states(task, 'primary', 'evaluation')
    m = P.site_mask(ev['kl'].shape, C.KMAX)
    e, t = np.nonzero(m)
    return dict(h=ev['h'][e, t], kl=ev['kl'][e, t], E=ev['estate_10'][e, t], ep=e)


def select_and_eval(sae, sd):
    A = get_all_activations(sae, sd['h'])
    half_a, half_b = sd['ep'] < 50, sd['ep'] >= 50
    cand = np.argsort(-A[half_a].var(0))[:20]
    rs = []
    for j in cand:
        a = A[half_a, j]
        rs.append(abs(pearsonr(a, sd['E'][half_a])[0]) if a.std() > 1e-8 else -1)
    best = int(cand[int(np.argmax(rs))])
    return best, atom_corrs(A[:, best], sd, half_b)


def atom_corrs(a, sd, sel):
    if a[sel].std() < 1e-8:
        return dict(r_kl=None, r_E=None, active_frac=0.0)
    return dict(r_kl=float(pearsonr(a[sel], sd['kl'][sel])[0]), r_E=float(pearsonr(a[sel], sd['E'][sel])[0]),
                active_frac=float((a[sel] > 0).mean()))


def ablate_atom(task, d):
    reg = dict((l, (p, k)) for l, p, k in C.registry()[task])
    model = C.load_any(*reg['primary'])
    clf, sc, v, sigma_v, _ = fit_probe(task, 'primary')
    ev = C.load_states(task, 'primary', 'evaluation')
    sites = CZ.Sites(model, clf, sc, ev['h'], C.episodes(task, 'evaluation'), N_SITES_ABL, SITE_SEED, v=v)
    eff = sites.effects(d.astype(np.float64), steer=False)
    null = [sites.effects(u, steer=False) for u in CZ.canonical_null()]
    return dict(effect=eff, null50=CZ.score(eff, null), n_sites=N_SITES_ABL,
                cos_atom_v=float(d @ v / np.linalg.norm(d)))


def run_variant(name, saes, sae_meta):
    res = dict(variant=name, sae_meta=sae_meta, tasks={})
    for task in TASKS:
        _, _, v, _, _ = fit_probe(task, 'primary')
        sd = site_data(task)
        r = dict(decomposition=[decompose(s, v) for s in saes],
                 random_decomposition=[random_decomp(s) for s in saes], selection=[])
        for i, s in enumerate(saes):
            best, corr = select_and_eval(s, sd)
            r['selection'].append(dict(sae_run=i, atom=best, heldout=corr))
        if name == 'published':
            atom = PUBLISHED_ATOMS[task]
            A0 = get_all_activations(saes[0], sd['h'])[:, atom]
            r['published_atom'] = dict(atom=atom, heldout=atom_corrs(A0, sd, sd['ep'] >= 50),
                                       all_sites=atom_corrs(A0, sd, np.ones(len(A0), bool)),
                                       selected_again_run0=r['selection'][0]['atom'] == atom)
            target = atom
        else:
            target = r['selection'][0]['atom']
        d = saes[0].decoder.weight.detach().numpy()[:, target].astype(np.float64)
        r['ablation'] = dict(atom=target, **ablate_atom(task, d / np.linalg.norm(d)))
        s = r['ablation']['null50']
        print(f"[SAE {name}] {task}: atom #{target}  E z={s['ablate_es']['z']:+.2f}  "
              f"readout k0 prob z={s['ablate_r0']['z']:+.2f} proj z={s['ablate_p0']['z']:+.2f}  | selected per run: "
              f"{[x['atom'] for x in r['selection']]}  shares top1 "
              f"{[round(x['top1'] * 100, 2) for x in r['decomposition']]}%", flush=True)
        res['tasks'][task] = r
    with open(os.path.join(OUT, f'sae_{name}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=P._json_default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['published', 'retrained'], required=True)
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    P.seed_everything(0)
    torch.set_num_threads(a.threads)
    os.makedirs(SAE_DIR, exist_ok=True)
    if a.variant == 'published':
        saes = [load_sae(f'outputs/phase3_sae/sae_seed{i}.pt') for i in range(3)]
        run_variant('published', saes, dict(source='outputs/phase3_sae/sae_seed{0,1,2}.pt'))
        return
    X = np.concatenate([C.load_states(t, 'primary', 'probe')['h'][:, C.MIN_T:].reshape(-1, 256) for t in TASKS])
    saes, meta = [], []
    for seed in range(3):
        p = os.path.join(SAE_DIR, f'sae_corrected_seed{seed}.pt')
        if os.path.exists(p):
            s = load_sae(p)
        else:
            s, _ = train_topk_sae(X, n_atoms=N_ATOMS, k=K_SPARSE, n_epochs=N_EPOCHS, batch_size=BATCH,
                                  seed=seed, verbose=False)
            s.eval()
            torch.save(s.state_dict(), p)
        saes.append(s)
        meta.append(dict(seed=seed, n_train=int(len(X)), metrics=sae_metrics(s, X)))
        print(f'  SAE seed {seed}: {meta[-1]["metrics"]}', flush=True)
    run_variant('retrained', saes, meta)


if __name__ == '__main__':
    main()
