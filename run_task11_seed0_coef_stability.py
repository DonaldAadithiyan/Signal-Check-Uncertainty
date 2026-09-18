import numpy as np
from src.config import XS_CONFIG
from run_task11_onset_persistence import CALIB_SEED, EVAL_SEED, build_dataset, fit_ct_residual_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

cfg = XS_CONFIG.copy()
calib_rows, _ = build_dataset('cartpole', cfg, CALIB_SEED, 60)
eval_rows, _ = build_dataset('cartpole', cfg, EVAL_SEED, 60)

kl_c = np.array([r['kl'] for r in calib_rows]); recon_c = np.array([r['recon'] for r in calib_rows]); ct_c = np.array([r['ct'] for r in calib_rows])
kl_e = np.array([r['kl'] for r in eval_rows]); recon_e = np.array([r['recon'] for r in eval_rows]); ct_e = np.array([r['ct'] for r in eval_rows])

scaler_c, reg_c = fit_ct_residual_model(calib_rows)
print('seed_0 calib-fit coef:', reg_c.coef_, 'intercept:', reg_c.intercept_)

X_e = np.column_stack([kl_e, recon_e])
scaler_e = StandardScaler().fit(X_e)
reg_e = LinearRegression().fit(scaler_e.transform(X_e), ct_e)
print('seed_0 eval-refit coef:', reg_e.coef_, 'intercept:', reg_e.intercept_)

print('Recon calib: mean=', recon_c.mean(), 'std=', recon_c.std())
print('Recon eval:  mean=', recon_e.mean(), 'std=', recon_e.std())
