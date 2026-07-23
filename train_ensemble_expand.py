#!/usr/bin/env python3.11
"""
Task T (part 1) — Train additional ensemble members at the SAME XS configuration.

Reviewer R3TF: "Ensemble baseline uses only 2–3 models (cost-reduced for multi-seed runs) —
a stronger ensemble (5+ models, as is more typical in the cited baseline literature) might
narrow the gap reported against the probe."

This is NOT a scale check. Same XS config (256-dim GRU, ~12M params), same hyperparameters
(XS_CONFIG), same trainer — just more members. Existing: ensemble_seed{0,1,2}.pt.
This adds seeds 3 and 4, bringing cartpole's ensemble to 5 total, matching the paper's
existing 5-seed probe replication count for a clean apples-to-apples framing.

Trains sequentially (each run is CPU-bound; the machine is an M4 laptop and CPU beats MPS at
batch_size=8). Skips any member already on disk, so this is safely resumable.
"""

import os
import sys
import time

sys.path.insert(0, '.')

from src.config import XS_CONFIG

NEW_SEEDS = [3, 4]
CK = 'outputs/checkpoints/ensemble_seed{}.pt'


def main():
    from src.training.trainer import train_world_model

    todo = [s for s in NEW_SEEDS if not os.path.exists(CK.format(s))]
    have = [s for s in NEW_SEEDS if os.path.exists(CK.format(s))]
    if have:
        print(f"[skip] already trained: seeds {have}", flush=True)
    if not todo:
        print("[done] nothing to train", flush=True)
        return

    print(f"[start] training ensemble seeds {todo} at XS config "
          f"(deter={XS_CONFIG['rssm_deter']}, {XS_CONFIG['total_env_steps']:,} env steps each)",
          flush=True)

    for s in todo:
        t0 = time.time()
        cfg = {**XS_CONFIG, 'checkpoint_path': CK.format(s)}
        print(f"[seed {s}] starting", flush=True)
        train_world_model(cfg, seed=s)
        print(f"[seed {s}] done in {(time.time()-t0)/60:.1f} min → {CK.format(s)}", flush=True)

    print("[done] all requested ensemble members trained", flush=True)


if __name__ == '__main__':
    main()
