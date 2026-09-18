import numpy as np
from src.config import XS_CONFIG
from run_task11_onset_persistence import TASKS, CALIB_SEED, build_dataset, fit_ct_residual_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

cfg = XS_CONFIG.copy()

for label, spec_override in [
    ('seed_0 (original)', None),
    ('seed_1', dict(TASKS['cartpole'], env_cls='cartpole', domain='cartpole', task='swingup',
                     checkpoint='outputs/multiseed/seed_1/world_model.pt',
                     training_states='outputs/multiseed/seed_1/training_states.npz')),
]:
    calib_rows, _ = build_dataset('cartpole', cfg, CALIB_SEED, 60, spec_override=spec_override)
    kl = np.array([r['kl'] for r in calib_rows])
    recon = np.array([r['recon'] for r in calib_rows])
    ct = np.array([r['ct'] for r in calib_rows])

    r_kl = np.corrcoef(ct, kl)[0, 1]
    r_recon = np.corrcoef(ct, recon)[0, 1]
    r_kl_recon = np.corrcoef(kl, recon)[0, 1]

    X = np.column_stack([kl, recon])
    scaler = StandardScaler().fit(X)
    reg = LinearRegression().fit(scaler.transform(X), ct)
    r2_linear = reg.score(scaler.transform(X), ct)

    print(f"--- {label} ---")
    print(f"  raw corr(C_t, KL)    = {r_kl:+.4f}")
    print(f"  raw corr(C_t, Recon) = {r_recon:+.4f}")
    print(f"  raw corr(KL, Recon)  = {r_kl_recon:+.4f}")
    print(f"  linear R^2 of C_t ~ KL+Recon (in-sample, calib) = {r2_linear:.4f}")
    print(f"  C_t range: [{ct.min():.3f}, {ct.max():.3f}]  mean={ct.mean():.3f}  std={ct.std():.3f}")
    print()
