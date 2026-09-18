import numpy as np
from src.config import XS_CONFIG
from run_task11_onset_persistence import TASKS, CALIB_SEED, EVAL_SEED, build_dataset, fit_ct_residual_model, apply_ct_residual
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

cfg = XS_CONFIG.copy()
spec_seed1 = dict(TASKS['cartpole'], env_cls='cartpole', domain='cartpole', task='swingup',
                   checkpoint='outputs/multiseed/seed_1/world_model.pt',
                   training_states='outputs/multiseed/seed_1/training_states.npz')

calib_rows, _ = build_dataset('cartpole', cfg, CALIB_SEED, 60, spec_override=spec_seed1)
eval_rows, _ = build_dataset('cartpole', cfg, EVAL_SEED, 60, spec_override=spec_seed1)

kl_c = np.array([r['kl'] for r in calib_rows]); recon_c = np.array([r['recon'] for r in calib_rows]); ct_c = np.array([r['ct'] for r in calib_rows])
kl_e = np.array([r['kl'] for r in eval_rows]); recon_e = np.array([r['recon'] for r in eval_rows]); ct_e = np.array([r['ct'] for r in eval_rows])

# in-sample (calib) fit quality
scaler, reg = fit_ct_residual_model(calib_rows)
X_c = np.column_stack([kl_c, recon_c])
r2_calib = reg.score(scaler.transform(X_c), ct_c)
print('in-sample calib R^2:', r2_calib)
print('regression coef (standardized):', reg.coef_, 'intercept:', reg.intercept_)

# out-of-sample (eval) fit quality using the SAME calib-fit model
X_e = np.column_stack([kl_e, recon_e])
r2_eval_direct = reg.score(scaler.transform(X_e), ct_e)
print('out-of-sample eval R^2 (using calib-fit model):', r2_eval_direct)

# compare distributions of KL/Recon/Ct between calib and eval splits
for name, c, e in [('KL', kl_c, kl_e), ('Recon', recon_c, recon_e), ('Ct', ct_c, ct_e)]:
    print(f'{name}: calib mean={c.mean():.4f} std={c.std():.4f}  |  eval mean={e.mean():.4f} std={e.std():.4f}')

# what if we refit directly on eval (in-sample on eval itself)?
scaler_e = StandardScaler().fit(X_e)
reg_e = LinearRegression().fit(scaler_e.transform(X_e), ct_e)
r2_eval_refit = reg_e.score(scaler_e.transform(X_e), ct_e)
print('in-sample eval-refit R^2 (sanity, not used downstream):', r2_eval_refit)
print('eval-refit coef:', reg_e.coef_, 'intercept:', reg_e.intercept_)

ct_matched_eval = apply_ct_residual(eval_rows, scaler, reg)
print('final balance: corr with KL=', np.corrcoef(ct_matched_eval, kl_e)[0,1], ' corr with Recon=', np.corrcoef(ct_matched_eval, recon_e)[0,1])
