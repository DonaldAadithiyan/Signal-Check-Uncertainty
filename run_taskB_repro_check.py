#!/usr/bin/env python3.11
"""
Task B pilot reproducibility check (§3.6.3): rerun the first 15,000 steps of
condition B (cartpole, seed 3101) TWICE, in scratch directories, and confirm
byte-identical h/kl/recon logs. Standalone so it can run independently of
the full pilot's 100,000-step cell.
"""
import shutil
import numpy as np
import run_taskB_training_necessity as B

REPRO_STEPS = 15_000

def main():
    cfg = B.XS_CONFIG.copy()
    dir_a = '/tmp/taskB_repro_a'
    dir_b = '/tmp/taskB_repro_b'
    shutil.rmtree(dir_a, ignore_errors=True)
    shutil.rmtree(dir_b, ignore_errors=True)

    cfg_a = cfg.copy(); cfg_a['total_env_steps'] = REPRO_STEPS
    res_a = B.train_condition('cartpole', B.TASKS['cartpole'], cfg_a, 'B_vconstrained', 3101, dir_a)
    cfg_b = cfg.copy(); cfg_b['total_env_steps'] = REPRO_STEPS
    res_b = B.train_condition('cartpole', B.TASKS['cartpole'], cfg_b, 'B_vconstrained', 3101, dir_b)

    sa = dict(np.load(res_a['states_path']))
    sb = dict(np.load(res_b['states_path']))
    print("\n=== Reproducibility check (first 15,000 steps, condition B) ===")
    all_identical = True
    for key in ['h', 'kl', 'recon']:
        identical = np.array_equal(sa[key], sb[key])
        all_identical = all_identical and identical
        print(f"  {key}: byte-identical = {identical}")
    print(f"\nOVERALL: {'PASS' if all_identical else 'FAIL'}")


if __name__ == '__main__':
    main()
