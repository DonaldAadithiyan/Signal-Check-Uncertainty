# Task 4: generality on two additional DeepMind Control tasks (Reviewer 2)

**Choice recorded before training** (`outputs/t4_generality/TASK_CHOICE.md`): walker-walk and cheetah-run. Both trained, so the fallback (ball_in_cup-catch) was not used.

**Training** (`run_t4_train.py`): the exact recipe of the existing models.
- XS RSSM, 100,000 environment steps, uniform-random policy, seed 0, 111 min each on CPU.
- Two corrections relative to the old training scripts:
  - the online log pairs o_t with a_{t−1};
  - Python's `random` is seeded as well.
- Final reconstruction error: walker about 5.5, cheetah about 0.17.

**Analysis** (`run_t4_analyze.py`), under the final rule: corrected logging, the same episode seeds as the other tasks, and the probe refit on probe-fit episodes (paper recipe). Readout = probability; the logit is stored.
- γ: the paper's selection rule (grid {0.70, 0.80, 0.90, 0.95, 0.99}, maximising R²(readout, C_t)) applied on probe-fit episodes.
- EMARec α: the paper's rule (maximise |corr(EMA(Rec), KL)|) applied on probe-fit episodes.
- ΔR²: the Table 4 protocol (Section-6 C_t; controls {KL, Rec, EMARec}; no ensemble exists for these tasks), plus the out-of-sample row (grouped 5-fold CV).
- Steering: the Table 6 protocol (500 sites, canonical 50-direction null, CRN).

## Flags

- **Walker: steering moves imagination error beyond the null, E z = +2.30** (98th percentile of the 50 directions). On the 13 original models the largest |E z| was 0.59. The "steering moves the readout but not E" dissociation does **not** hold on walker. It holds on cheetah (E z +0.04).
- **Cheetah: C_t has no standalone correlation with E_{t,10}** (r −0.007 [−0.028, +0.011]), yet its ΔR² beyond the controls is +0.011. C_t helps only together with the controls, a suppression effect. On both tasks the controls themselves explain very little of E (base R² 0.058 and 0.017).
- **γ selected at the lower edge of the paper's grid (0.70) on both tasks.** R² decreases monotonically with γ on both, so the best value may lie below the grid.

## Table 3 / §5 format (emergence)

| Quantity | walker-walk | cheetah-run |
|---|---|---|
| probe AUROC (1[KL_t > median]) | 0.756 | 0.828 |
| selected γ (R² at each grid value) | 0.7 (0.7: 0.281, 0.8: 0.260, 0.9: 0.202, 0.95: 0.112, 0.99: 0.018) | 0.7 (0.7: 0.265, 0.8: 0.191, 0.9: 0.097, 0.95: 0.043, 0.99: 0.002) |
| R²(readout, C_t), probability | 0.266 [0.248, 0.282] | 0.264 [0.255, 0.274] |
| R²(readout, KL_t), probability | 0.288 [0.274, 0.301] | 0.436 [0.429, 0.444] |
| R²(readout, C_t) / R²(readout, KL_t), logit | 0.266 / 0.280 | 0.257 / 0.432 |
| C_t ahead of KL? | no (-0.023) | no (-0.172) |
| fraction of sites with saturated probability | 0.001 | 0.017 |

## Table 4 format (E_{t,10}, 46,700 sites / 100 episodes)

| Quantity | walker-walk | cheetah-run |
|---|---|---|
| EMARec α | 0.5 | 0.5 |
| baseline R² of {KL, Rec, EMARec} | 0.058 | 0.017 |
| r(C_t, E_{t,10}) | +0.167 [+0.147, +0.187] | -0.007 [-0.028, +0.011] |
| ΔR² of C_t, in-sample [95% CI] | +0.0021 [+0.0009, +0.0037] | +0.0113 [+0.0083, +0.0153] |
| ΔR²_cv of C_t, out-of-sample [95% CI] | +0.0019 [+0.0006, +0.0034] | +0.0113 [+0.0080, +0.0150] |
| adds out-of-sample value | yes | yes |

## Table 6 format (steering z vs the 50-direction null, 1 model per task)

| Task | readout k=0 | readout k=10 | E | logit k=0 / k=10 |
|---|---|---|---|---|
| walker | +12.8 | +4.5 | **+2.30** | +15.5 / +4.5 |
| cheetah | +11.9 | +5.0 | **+0.04** | +15.6 / +4.7 |

## Reading

Every task trained is reported (both). Against the paper's three tasks:
- **Emergence:** holds in reduced form. The probe detects high-KL states (AUROC 0.76 and 0.83), and the readout tracks C_t (R² 0.27 and 0.26). But C_t is **not** ahead of current KL on either task (walker −0.02; cheetah −0.17).
- **External (Table 4):** C_t adds variance beyond the controls on both tasks, in-sample and out of sample (walker +0.0019, cheetah +0.0113 cross-validated, CIs > 0). On cheetah this happens with no standalone correlation.
- **Steering:** the readout moves at k=0 (+12.8, +11.9) and still at k=10 (+4.5, +5.0) on both tasks. On walker, imagination error also moves beyond the null (+2.30), so the dissociation is task-dependent rather than general.
