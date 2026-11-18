# Out-of-sample ΔR² (grouped 5-fold CV by trajectory), camera-ready, final rule

Script: `run_cv_delta_r2.py`. Raw: `CV_delta_r2.json`.

**Protocol:**
- Data: the same 46,700 evaluation sites per task (100 episodes), target E_{t,10}.
- Folds: the 100 episodes are split into 5 folds of 20 by a seeded permutation (seed 55). The same folds are used for every regression.
- Fitting: OLS with intercept, fit on 4 folds, predicting the held-out fold.
- ΔR²_cv = R²_cv(base + signal) − R²_cv(base), computed on the pooled out-of-fold predictions.
- CI: trajectory bootstrap, 1,000 resamples (seed 7), over the fixed out-of-fold predictions. ΔR²_cv can be negative.
- In-sample ΔR² (same columns, GramSet bootstrap) is shown alongside.
- Definitions follow the final rule: refit probe; readout = probability, logit also shown; published γ and α; Section-6 C_t; Task 2's EMA-KL α. For Task 4 (walker, cheetah), γ and α come from the paper's procedure applied on probe-fit episodes.

**Pre-registered reading:** a signal "adds out-of-sample predictive value" on a task only if its ΔR²_cv CI excludes zero (✓).

## cartpole (46700 sites, 100 episodes)

| Row | in-sample ΔR² [95% CI] | out-of-sample ΔR²_cv [95% CI] | R²_cv base → full | adds OOS value |
|---|---|---|---|---|
| (a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ens. on cartpole) | +0.01685 [+0.00785, +0.02881] | +0.01683 [+0.00649, +0.02712] | 0.6999 → 0.7168 | ✓ |
| (b) Task 1: readout (probability) beyond the same controls | +0.04196 [+0.02782, +0.06030] | +0.04290 [+0.02746, +0.06026] | 0.6999 → 0.7428 | ✓ |
| (b) Task 1: readout (logit) beyond the same controls | +0.04277 [+0.03084, +0.05673] | +0.04309 [+0.03043, +0.05712] | 0.6999 → 0.7430 | ✓ |
| (c) Task 2: C_t beyond all others (6 standard + readout prob.) | +0.00039 [+0.00000, +0.00285] | -0.00007 [-0.00141, +0.00128] | 0.7633 → 0.7632 | ✗ |
| (c) Task 2: C_t beyond all others (6 standard + readout logit) | +0.00078 [+0.00000, +0.00420] | +0.00020 [-0.00177, +0.00213] | 0.7675 → 0.7677 | ✗ |
| (c) Task 2: C_t beyond the 6 standard signals | +0.00159 [+0.00003, +0.00569] | +0.00076 [-0.00218, +0.00367] | 0.7529 → 0.7536 | ✗ |
| (c) Task 2: readout (prob.) beyond all others (6 standard + C_t) | +0.00917 [+0.00402, +0.01723] | +0.00960 [+0.00339, +0.01711] | 0.7536 → 0.7632 | ✓ |
| (c) Task 2: readout (logit) beyond all others (6 standard + C_t) | +0.01429 [+0.00758, +0.02335] | +0.01409 [+0.00630, +0.02276] | 0.7536 → 0.7677 | ✓ |
| (c) Task 2: readout (prob.) beyond the 6 standard signals | +0.01037 [+0.00486, +0.01892] | +0.01043 [+0.00405, +0.01833] | 0.7529 → 0.7633 | ✓ |
| (c) Task 2: readout (logit) beyond the 6 standard signals | +0.01511 [+0.00824, +0.02402] | +0.01465 [+0.00695, +0.02317] | 0.7529 → 0.7675 | ✓ |

## reacher (46700 sites, 100 episodes)

| Row | in-sample ΔR² [95% CI] | out-of-sample ΔR²_cv [95% CI] | R²_cv base → full | adds OOS value |
|---|---|---|---|---|
| (a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ens. on cartpole) | +0.00528 [+0.00146, +0.00842] | +0.01010 [+0.00154, +0.01996] | 0.5315 → 0.5416 | ✓ |
| (b) Task 1: readout (probability) beyond the same controls | +0.00300 [+0.00098, +0.00623] | +0.00208 [-0.00053, +0.00544] | 0.5315 → 0.5336 | ✗ |
| (b) Task 1: readout (logit) beyond the same controls | +0.00469 [+0.00203, +0.00919] | +0.00273 [-0.00192, +0.00818] | 0.5315 → 0.5343 | ✗ |
| (c) Task 2: C_t beyond all others (6 standard + readout prob.) | +0.00046 [+0.00001, +0.00168] | +0.00054 [-0.00087, +0.00163] | 0.6296 → 0.6301 | ✗ |
| (c) Task 2: C_t beyond all others (6 standard + readout logit) | +0.00035 [+0.00000, +0.00144] | +0.00035 [-0.00098, +0.00134] | 0.6322 → 0.6325 | ✗ |
| (c) Task 2: C_t beyond the 6 standard signals | +0.00010 [+0.00000, +0.00095] | -0.00013 [-0.00097, +0.00047] | 0.6282 → 0.6281 | ✗ |
| (c) Task 2: readout (prob.) beyond all others (6 standard + C_t) | +0.00288 [+0.00098, +0.00623] | +0.00206 [-0.00063, +0.00543] | 0.6281 → 0.6301 | ✗ |
| (c) Task 2: readout (logit) beyond all others (6 standard + C_t) | +0.00562 [+0.00260, +0.01064] | +0.00447 [+0.00003, +0.00994] | 0.6281 → 0.6325 | ✓ |
| (c) Task 2: readout (prob.) beyond the 6 standard signals | +0.00252 [+0.00079, +0.00588] | +0.00139 [-0.00142, +0.00493] | 0.6282 → 0.6296 | ✗ |
| (c) Task 2: readout (logit) beyond the 6 standard signals | +0.00537 [+0.00250, +0.01044] | +0.00399 [-0.00040, +0.00957] | 0.6282 → 0.6322 | ✗ |

## pendulum (46700 sites, 100 episodes)

| Row | in-sample ΔR² [95% CI] | out-of-sample ΔR²_cv [95% CI] | R²_cv base → full | adds OOS value |
|---|---|---|---|---|
| (a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ens. on cartpole) | +0.00217 [+0.00091, +0.00393] | +0.00209 [+0.00056, +0.00360] | 0.4901 → 0.4922 | ✓ |
| (b) Task 1: readout (probability) beyond the same controls | +0.02502 [+0.01967, +0.02933] | +0.02545 [+0.02077, +0.03042] | 0.4901 → 0.5156 | ✓ |
| (b) Task 1: readout (logit) beyond the same controls | +0.01701 [+0.01141, +0.02312] | +0.01700 [+0.01137, +0.02340] | 0.4901 → 0.5071 | ✓ |
| (c) Task 2: C_t beyond all others (6 standard + readout prob.) | +0.00280 [+0.00144, +0.00475] | +0.00273 [+0.00107, +0.00430] | 0.6111 → 0.6139 | ✓ |
| (c) Task 2: C_t beyond all others (6 standard + readout logit) | +0.00137 [+0.00048, +0.00268] | +0.00128 [+0.00014, +0.00232] | 0.6149 → 0.6162 | ✓ |
| (c) Task 2: C_t beyond the 6 standard signals | +0.00303 [+0.00163, +0.00497] | +0.00297 [+0.00128, +0.00458] | 0.6101 → 0.6130 | ✓ |
| (c) Task 2: readout (prob.) beyond all others (6 standard + C_t) | +0.00084 [+0.00019, +0.00194] | +0.00082 [-0.00014, +0.00167] | 0.6130 → 0.6139 | ✗ |
| (c) Task 2: readout (logit) beyond all others (6 standard + C_t) | +0.00318 [+0.00120, +0.00604] | +0.00315 [+0.00047, +0.00556] | 0.6130 → 0.6162 | ✓ |
| (c) Task 2: readout (prob.) beyond the 6 standard signals | +0.00106 [+0.00034, +0.00222] | +0.00106 [+0.00004, +0.00196] | 0.6101 → 0.6111 | ✓ |
| (c) Task 2: readout (logit) beyond the 6 standard signals | +0.00484 [+0.00217, +0.00841] | +0.00484 [+0.00146, +0.00787] | 0.6101 → 0.6149 | ✓ |

## walker (46700 sites, 100 episodes, γ = 0.7, EMARec α = 0.5)

| Row | in-sample ΔR² [95% CI] | out-of-sample ΔR²_cv [95% CI] | R²_cv base → full | adds OOS value |
|---|---|---|---|---|
| (a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ens. on cartpole) | +0.00207 [+0.00088, +0.00368] | +0.00192 [+0.00055, +0.00336] | 0.0576 → 0.0596 | ✓ |

## cheetah (46700 sites, 100 episodes, γ = 0.7, EMARec α = 0.5)

| Row | in-sample ΔR² [95% CI] | out-of-sample ΔR²_cv [95% CI] | R²_cv base → full | adds OOS value |
|---|---|---|---|---|
| (a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ens. on cartpole) | +0.01135 [+0.00826, +0.01525] | +0.01129 [+0.00795, +0.01497] | 0.0165 → 0.0278 | ✓ |

## Summary

| Row | cartpole | reacher | pendulum | walker | cheetah |
|---|---|---|---|---|---|
| (a) Table 4: C_t beyond {KL, Rec, EMARec} (+ ens. on cartpole) | ✓ | ✓ | ✓ | ✓ | ✓ |
| (b) Task 1: readout (probability) beyond the same controls | ✓ | ✗ | ✓ | — | — |
| (b) Task 1: readout (logit) beyond the same controls | ✓ | ✗ | ✓ | — | — |
| (c) Task 2: C_t beyond all others (6 standard + readout prob.) | ✗ | ✗ | ✓ | — | — |
| (c) Task 2: C_t beyond all others (6 standard + readout logit) | ✗ | ✗ | ✓ | — | — |
| (c) Task 2: C_t beyond the 6 standard signals | ✗ | ✗ | ✓ | — | — |
| (c) Task 2: readout (prob.) beyond all others (6 standard + C_t) | ✓ | ✗ | ✗ | — | — |
| (c) Task 2: readout (logit) beyond all others (6 standard + C_t) | ✓ | ✓ | ✓ | — | — |
| (c) Task 2: readout (prob.) beyond the 6 standard signals | ✓ | ✗ | ✓ | — | — |
| (c) Task 2: readout (logit) beyond the 6 standard signals | ✓ | ✗ | ✓ | — | — |
