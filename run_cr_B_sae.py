#!/usr/bin/env python3.11
"""SAEs retrained on corrected states (paper recipe), all 3 runs.

Fixed choices: corrected logging; the paper's probe v (60% of the model's training
log); readout = projection v^T h_t; canonical 50-direction null with CRN.
For each task x SAE run: the atom chosen by the paper's rule (top-20-variance atoms;
largest |r(act, E10)| on evaluation episodes 0-49), held-out r(act, KL) and
r(act, E10) on episodes 50-99, ablation z on E (K=10) and on the projection readout
(k=0) at 800 evaluation sites (seed 33) vs the 50 canonical directions on the same
sites, cos(atom decoder, v). Plus Table-7 shares of v (top-1/4/10) per task x run.

Run with CR_PROBE=paper.
"""
import json
import os

import numpy as np
import torch

from src.camera_ready import core as C
from src.camera_ready import causal as CZ
from src.probe.sae import get_all_activations
from run_cr_A_sae import load_sae, decompose, random_decomp, site_data, select_and_eval, N_SITES_ABL, SITE_SEED
from run_cr_A_causal import fit_probe

OUT = os.path.join(C.ROOT, 'B_sae_corrected.json' if C.PROBE_MODE == 'paper' else 'B_sae_corrected_refit.json')
TASKS = ['cartpole', 'reacher', 'pendulum']


def main():
    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(6)
    saes = [load_sae(os.path.join(C.ROOT, 'sae', f'sae_corrected_seed{s}.pt')) for s in range(3)]
    res = {}
    for task in TASKS:
        clf, sc, v, sigma_v, _ = fit_probe(task, 'primary')
        sd = site_data(task)
        reg = dict((l, (p, k)) for l, p, k in C.registry()[task])
        model = C.load_any(*reg['primary'])
        ev = C.load_states(task, 'primary', 'evaluation')
        sites = CZ.Sites(model, clf, sc, ev['h'], C.episodes(task, 'evaluation'), N_SITES_ABL, SITE_SEED, v=v)
        null = [sites.effects(u, steer=False) for u in CZ.canonical_null()]
        rows = []
        for s, sae in enumerate(saes):
            atom, corr = select_and_eval(sae, sd)
            d = sae.decoder.weight.detach().numpy()[:, atom].astype(np.float64)
            d /= np.linalg.norm(d)
            eff = sites.effects(d, steer=False)
            z = CZ.score(eff, null)
            rows.append(dict(sae_run=s, atom=atom, heldout_r_kl=corr['r_kl'], heldout_r_E=corr['r_E'],
                             active_frac=corr['active_frac'], cos_atom_v=float(d @ v),
                             ablate_E=z['ablate_es'], ablate_readout_prob=z['ablate_r0'],
                             ablate_readout_logit=z['ablate_p0'],   # logit is affine in v.h: z identical to projection z
                             ablate_readout_proj=z['ablate_p0'],
                             ablate_readout_proj_k10=z['ablate_p10'],
                             shares=decompose(sae, v), random_shares=random_decomp(sae)))
            print(f"[B] {task} run {s}: atom #{atom} r(KL) {corr['r_kl']:+.3f} r(E) {corr['r_E']:+.3f} "
                  f"E z {z['ablate_es']['z']:+.2f} readout prob z {z['ablate_r0']['z']:+.2f} logit z {z['ablate_p0']['z']:+.2f} "
                  f"cos {d @ v:+.3f} top1 {rows[-1]['shares']['top1']*100:.2f}%", flush=True)
        res[task] = dict(sites=N_SITES_ABL, null='canonical 50', probe_mode=C.PROBE_MODE, runs=rows,
                         null_readout_prob_sd=float(np.std([e['ablate_r0'] for e in null])))
        with open(OUT, 'w') as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    main()
