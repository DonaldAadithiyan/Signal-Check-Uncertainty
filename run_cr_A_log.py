#!/usr/bin/env python3.11
"""Camera-ready Part A, step 1: re-log hidden states with the corrected pipeline.

For every model (13 standard + 10 partial-observation cartpole), splits
probe (100 eps) / calibration (50) / evaluation (100) [+ evaluation_noisy, sigma=0.1,
standard models only]. Saved per step: h, z_idx, KL, Rec, prior/posterior entropy;
on calibration + evaluation also E_{t,K} (K=1,5,10,20) and imagined/real reward at
k=10. Observations, actions, rewards, true env state, traj_id (= episode seed) and t
live in outputs/camera_ready/episodes/{task}_{split}.npz.

Usage:
  python run_cr_A_log.py --check            # episode identity vs sprint cache + byte repro
  python run_cr_A_log.py --tasks cartpole   # full logging
"""
import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np
import torch

from src.camera_ready import core as C
from src.rerun import pipeline as P

SPLITS_WM = ('probe', 'calibration', 'evaluation', 'evaluation_noisy')
SPLITS_POMDP = ('probe', 'calibration', 'evaluation')
ES_SPLITS = ('calibration', 'evaluation')


def log_model(task, label, path, kind, splits=None, n_eps=None, out_dir=C.ST_DIR):
    model = C.load_any(path, kind)
    os.makedirs(out_dir, exist_ok=True)
    for split in splits or (SPLITS_WM if kind == 'wm' else SPLITS_POMDP):
        f = os.path.join(out_dir, f'{task}_{label}_{split}.npz')
        if os.path.exists(f) and n_eps is None:
            continue
        ep = C.episodes(task, split)
        if n_eps is not None:
            ep = {k: v[:n_eps] for k, v in ep.items()}
        st = C.log_states_full(model, ep, P.TORCH_SEEDS[split])
        C.assert_finite(st, f'{task}/{label}/{split}')
        if split in ES_SPLITS:
            st.update(C.estate_all_sites(model, ep, st, P.TORCH_SEEDS[split], task))
            v = st['estate_10'][:, C.MIN_T:ep['act'].shape[1] - C.KMAX - 1]
            if not np.all(np.isfinite(v)):
                raise FloatingPointError(f'non-finite E at valid sites {task}/{label}/{split}')
        st['traj_id'] = ep['seeds']
        np.savez(f, **st)
        print(f'  saved {f}  KL mean {st["kl"].mean():.3f}', flush=True)


def check():
    # 1) episodes identical to the sprint cache (same seeds, same generator)
    for task in P.TASKS:
        for split in ('probe', 'calibration', 'evaluation'):
            a = C.episodes(task, split)
            f = f'outputs/sprint/episodes/{task}_{split}.npz'
            if os.path.exists(f):
                b = dict(np.load(f))
                for k in ('obs', 'act', 'rew', 'seeds'):
                    if a[k].tobytes() != b[k].tobytes():
                        raise AssertionError(f'episode mismatch vs sprint cache: {task}/{split}/{k}')
                print(f'  episodes {task}/{split}: identical to sprint cache', flush=True)
            else:
                print(f'  episodes {task}/{split}: generated (no sprint cache to compare)', flush=True)
    # 2) byte-reproducibility across two independent processes (5 eval episodes)
    hashes = []
    for i in range(2):
        d = os.path.join(C.ROOT, f'repro_{i}')
        subprocess.run([sys.executable, __file__, '--_repro', d], check=True)
        h = hashlib.sha256()
        for fn in sorted(os.listdir(d)):
            z = np.load(os.path.join(d, fn))
            for k in sorted(z.files):
                h.update(k.encode()); h.update(z[k].tobytes())
        hashes.append(h.hexdigest())
    if hashes[0] != hashes[1]:
        raise AssertionError(f'byte-reproducibility FAILED: {hashes}')
    print(f'  byte-reproducible across processes: sha256 {hashes[0][:16]}', flush=True)
    return hashes[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--_repro')
    ap.add_argument('--tasks', nargs='+', default=['cartpole', 'reacher', 'pendulum', 'cartpole_pomdp'])
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    P.seed_everything(0)
    torch.set_num_threads(a.threads)
    if a._repro:
        reg = C.registry()
        for task in ('cartpole', 'reacher', 'cartpole_pomdp'):
            lab, path, kind = reg[task][0] if task != 'cartpole_pomdp' else reg[task][1]
            log_model(task, lab, path, kind, splits=('evaluation',), n_eps=5, out_dir=a._repro)
        return
    if a.check:
        check()
        return
    reg = C.registry()
    for task in a.tasks:
        for label, path, kind in reg[task]:
            if not os.path.exists(path):
                raise FileNotFoundError(f'{task}/{label}: {path}')
            print(f'[log] {task}/{label}', flush=True)
            log_model(task, label, path, kind)


if __name__ == '__main__':
    main()
