# Task 1 — Direct predictive value of the learned readout (Reviewer 2)

Script: `run_t1_readout.py` · numbers: `T1_readout.json` · run 2026-11-16, CPU, local M4.

## Flags (read first)

1. **The logging pipeline behind these numbers has a known action-pairing issue.** As instructed, Task 1 uses the paper's original data path. Phase 1's `collect_trajectories` feeds `obs_t` together with `a_t`. Training and the paper's Eq. 1 use `a_{t-1}`. Before the scope change, a corrected-pipeline consistency check (old Part A) was running. It was stopped part-way, on primary models only, and its numbers are not deliverables. What it had found:
   - Table 4 / Task-1-style ΔR² of C_t still excludes zero on all three tasks.
   - Some published intervention results did change:
     - reacher atom #612 ablation: E z +1.61 (published +2.67)
     - cartpole atom #156 ablation: readout z −1.35 (published −5.0)
     - pendulum steering, readout at k=10: z −0.0 (published +11.4)
     - reacher steering, readout at k=10: z +2.0 (published +5.9)

   Partial outputs are in `outputs/camera_ready/A_raw/`. This is your call. I have not acted on it.
2. **The readout's ΔR² does not clear the random-direction null on cartpole or pendulum.** Projecting h_t onto random directions often adds as much variance in E beyond the controls as v does. C_t's own ΔR² also sits inside that null on all three tasks (reference rows below).
3. **The readout definition matters on pendulum.** The handoff defines the readout as v^T h_t, the projection. The probe *probability* (the paper's code convention) saturates on pendulum, where probe AUROC is ≈1. Its r there is +0.197, against +0.563 for the projection. Both are reported. The projection is the primary row.

## Protocol (same data and definitions as Table 4)

- Frozen primary checkpoint per task.
- Probe: fit on 60% of the training log, `v` = unit probe direction in raw h space.
- Evaluation set rebuilt with Phase 1's own code and seed: 100 trajectories, 6,000 sites per task.
- C_t: per-trajectory median, γ = 0.95 / 0.70 / 0.90. EMARec α = 0.02 / 0.5 / 0.02.
- Controls: {KL, Rec, EMARec}, plus ensemble disagreement on cartpole (as in Table 4).
- ΔR² CI: 1,000 trajectory resamples, seed 0, using the addendum's bootstrap code.
- **Check passed:** the rebuilt C_t ΔR² equals the published Table 4 to 4 decimals (point and CI) on all three tasks.
- Null: the 50 canonical directions `u` (`outputs/canonical_null/random_directions_50_dim256.npz`), each used as u^T h_t in place of v^T h_t.
  - z = (v − null mean) / null sd. Percentile = % of the 50 directions below v.
  - The null for r uses |r|, because a random direction's sign is arbitrary.
- Per-model numbers: the paper reports ΔR² only for the primary model of each task, so there are no further models to add.

## Results (E_{t,10}, 6,000 sites / 100 trajectories per task)

| Task | Signal | r [95% CI] | R² | ΔR² beyond controls [95% CI] | null z (R²) / pct | null z (ΔR²) / pct |
|---|---|---|---|---|---|---|
| cartpole | **readout v^T h** | +0.719 [+0.687, +0.753] | 0.517 | **+0.0577 [+0.0392, +0.0769]** | +4.47 / 100 | **+0.76 / 84** |
| cartpole | C_t (Table 4) | +0.608 [+0.569, +0.651] | 0.369 | +0.0280 [+0.0154, +0.0433] | +2.93 / 98 | −0.05 / 66 |
| cartpole | probe probability | +0.542 [+0.504, +0.584] | 0.293 | +0.0368 [+0.0240, +0.0555] | — | — |
| reacher | **readout v^T h** | +0.308 [+0.267, +0.345] | 0.095 | **+0.0375 [+0.0240, +0.0528]** | +8.84 / 100 | **+4.03 / 100** |
| reacher | C_t (Table 4) | +0.256 [+0.214, +0.299] | 0.066 | +0.0182 [+0.0083, +0.0309] | +5.82 / 100 | +1.47 / 88 |
| reacher | probe probability | +0.301 [+0.260, +0.338] | 0.090 | +0.0328 [+0.0200, +0.0474] | — | — |
| pendulum | **readout v^T h** | +0.563 [+0.518, +0.603] | 0.317 | **+0.0444 [+0.0334, +0.0555]** | +2.02 / 94 | **+0.21 / 80** |
| pendulum | C_t (Table 4) | +0.174 [+0.145, +0.204] | 0.030 | +0.0019 [+0.0001, +0.0051] | −0.55 / 50 | −0.76 / 10 |
| pendulum | probe probability | +0.197 [+0.157, +0.234] | 0.039 | +0.0020 [+0.0003, +0.0045] | — | — |

Null distribution of the random-direction projections (mean / sd / max over 50):

| Task | R² | ΔR² |
|---|---|---|
| cartpole | 0.090 / 0.095 / 0.425 | 0.030 / 0.037 / 0.129 |
| reacher | 0.010 / 0.010 / 0.037 | 0.007 / 0.008 / 0.024 |
| pendulum | 0.091 / 0.112 / 0.413 | 0.035 / 0.044 / 0.167 |

The |r| null gives readout z = +2.80 on cartpole, +4.44 on reacher and +1.78 on pendulum. The baseline R² of the controls alone is 0.666 (cartpole), 0.132 (reacher) and 0.321 (pendulum).

## Reading (no pre-registered rule for Task 1; descriptive)

- **Without the null:** on all three tasks the readout predicts E_{t,10} and adds variance beyond {KL, Rec, EMARec} (+ ensemble), with a CI that excludes zero. Its ΔR² is larger than C_t's on every task: about 2× on cartpole and reacher, about 23× on pendulum.
- **Against the random-direction null:**
  - The readout's *standalone* association clears the null on cartpole and reacher (R² z ≥ 4.5). It is borderline on pendulum (z = +2.0, 94th percentile).
  - Its *incremental* value clears the null **only on reacher** (z = +4.03, above all 50 directions).
  - On cartpole and pendulum, h_t carries information about later imagination error that many directions expose. The share that v adds beyond the controls is not distinguishable from what a random direction adds (z = +0.76 and +0.21).
- C_t's incremental value is also not above the random-direction null on any task (z −0.05, +1.47, −0.76). These reference rows are not in the handoff's table. They are included because they bear directly on how the comparison should be read.
