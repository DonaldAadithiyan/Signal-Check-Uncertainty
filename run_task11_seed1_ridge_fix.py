import numpy as np
from src.config import XS_CONFIG
from run_task11_onset_persistence import (
    TASKS, CALIB_SEED, EVAL_SEED, build_dataset, check_balance, assert_balance_ok,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

cfg = XS_CONFIG.copy()
spec_seed1 = dict(TASKS['cartpole'], env_cls='cartpole', domain='cartpole', task='swingup',
                   checkpoint='outputs/multiseed/seed_1/world_model.pt',
                   training_states='outputs/multiseed/seed_1/training_states.npz')

calib_rows, _ = build_dataset('cartpole', cfg, CALIB_SEED, 60, spec_override=spec_seed1)
eval_rows, _ = build_dataset('cartpole', cfg, EVAL_SEED, 60, spec_override=spec_seed1)

kl_c = np.array([r['kl'] for r in calib_rows]); recon_c = np.array([r['recon'] for r in calib_rows]); ct_c = np.array([r['ct'] for r in calib_rows])
kl_e = np.array([r['kl'] for r in eval_rows]); recon_e = np.array([r['recon'] for r in eval_rows]); ct_e = np.array([r['ct'] for r in eval_rows])

X_c = np.column_stack([kl_c, recon_c])
X_e = np.column_stack([kl_e, recon_e])
scaler = StandardScaler().fit(X_c)

for alpha in [0.1, 1.0, 5.0, 10.0, 50.0]:
    reg = Ridge(alpha=alpha).fit(scaler.transform(X_c), ct_c)
    pred_e = reg.predict(scaler.transform(X_e))
    ct_matched_eval = ct_e - pred_e
    balance = check_balance(ct_matched_eval, kl_e, recon_e)
    r2_calib = reg.score(scaler.transform(X_c), ct_c)
    print(f"alpha={alpha}: coef={reg.coef_}  calib_R2={r2_calib:.4f}  "
          f"balance: KL={balance['corr_with_kl']:+.4f} Recon={balance['corr_with_recon']:+.4f}")
