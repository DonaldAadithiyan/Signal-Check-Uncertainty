# Endogenous Predictive-Difficulty Representations in Recurrent World Models

Code, results and figures for the ICAART 2027 paper *Endogenous Predictive-Difficulty Representations in Recurrent World Models* (full paper, #388).

> **In one sentence.** A small DreamerV3-style world model, trained only to predict its observations, keeps a readable memory of how hard prediction has *recently* been. That memory predicts how far the model's imagined future will drift from reality. Pushing the model's hidden state along it changes the model's own "difficulty report", but leaves the quality of its predictions unchanged.

Everything in this repository runs on a laptop CPU. No GPU is needed.

---

## Contents

1. [The question](#1-the-question)
2. [Key concepts](#2-key-concepts-read-this-first)
3. [Central claims and where to find the evidence](#3-central-claims-and-where-to-find-the-evidence)
4. [Results in detail](#4-results-in-detail)
5. [What the results do not show](#5-what-the-results-do-not-show)
6. [Experimental setup](#6-experimental-setup)
7. [Installation](#7-installation)
8. [Reproducing every number and figure](#8-reproducing-every-number-and-figure)
9. [Repository map](#9-repository-map)
10. [Result files reference](#10-result-files-reference)
11. [Earlier experiments in this repository](#11-earlier-experiments-in-this-repository)

---

## 1. The question

A world model lets an agent imagine the consequences of its actions without acting. Recurrent world models such as DreamerV3 compress the past into a hidden state `h_t`, and they are trained only to predict what comes next.

They already expose *instantaneous* signals of difficulty:
- the KL divergence between their prior and posterior (how surprised they are right now);
- the reconstruction error (how badly they decode the current observation).

A recurrent state is a running summary of the past, so it *could* also carry a **history** of difficulty. That would tell it not just "this step is surprising" but "predictions have been hard for a while".

**We ask whether such a history representation emerges on its own, without any training signal for it, and what it means if it does.**

To show that, three things are needed:
1. that the history is readable from `h_t` beyond what the current step explains (emergence);
2. that it predicts something *outside* the signal used to find it (external meaning);
3. that intervening on it tells us what role it plays (causal test).

---

## 2. Key concepts (read this first)

| Symbol | Meaning |
|---|---|
| `h_t` | The world model's 256-dimensional deterministic recurrent state (GRU) at time `t`. |
| `KL_t` | Per-step KL divergence between the model's posterior and prior over its stochastic latent. Its instantaneous surprise. |
| `Rec_t` | Observation reconstruction error at step `t`. |
| `EMARec_t`, `EMAKL_t` | Exponential moving averages of `Rec_t` and `KL_t`: the "smoothed error" baselines a designer would build by hand. |
| **Probe / readout** | A logistic-regression probe on `h_t` that predicts whether `KL_t` is above its median. Its weight vector `v` defines the model's **internal difficulty readout**, the probe's output probability `σ(vᵀh_t + b)`. The logit `vᵀh_t + b` is also reported throughout. |
| **`C_t`** | **Accumulated predictive difficulty**: a discounted count of recent high-KL steps, `C_t = Σ_{i≥0} γ^i · 1[KL_{t−i} > τ]`. It resets at episode start; `τ` is the median KL on calibration episodes. |
| `C_t^past` | The same sum without the current step (`i ≥ 1`). It cannot share the probe's label. |
| **`E_{t,K}`** | **Imagination error**, the external target. From the state at `t`, the model imagines `K` steps ahead with the real actions. `E_{t,K}` is the mean L2 distance between the decoded imagined observations and the real ones. It is defined without any reference to KL or to the probe. |
| **ΔR²** | Extra variance in `E_{t,10}` explained by adding a signal to a regression that already contains the baselines. ΔR²_cv is its out-of-sample version (grouped 5-fold cross-validation by episode). |
| **Random-direction null** | Every intervention on `v` is repeated along 50 random unit directions in `h`-space with the same step size. Effects are reported as `z` against that null. |

---

## 3. Central claims and where to find the evidence

| # | Claim | Status | Key evidence | Figure | Result file |
|---|---|---|---|---|---|
| 1 | **Emergence.** A linear readout of `h_t` tracks accumulated difficulty `C_t`, beyond current KL, and temporal order matters. | **Supported** (strength depends on readout scale, see §4.1) | R²(readout, `C_t`) = 0.84 / 0.50 / 0.89; order-scramble z = +22.8 / +36.3 / +2.4 | [fig01](outputs/camera_ready/figures_final/fig01_readout_vs_Ct.pdf), [fig02](outputs/camera_ready/figures_final/fig02_R2_Ct_vs_KL.pdf) | [camera_ready_numbers.md](outputs/camera_ready/camera_ready_numbers.md) |
| 2 | **External meaning.** `C_t` predicts future imagination error beyond KL, reconstruction error and their moving averages, in-sample and out of sample. | **Supported on all 5 tasks** | ΔR²_cv = +0.017 / +0.010 / +0.002 / +0.002 / +0.011, all CIs > 0 | [fig03](outputs/camera_ready/figures_final/fig03_horizon.pdf), [fig04](outputs/camera_ready/figures_final/fig04_cv_deltaR2_Ct.pdf), [fig05](outputs/camera_ready/figures_final/fig05_order_scramble.pdf) | [CV_delta_r2.md](outputs/camera_ready/CV_delta_r2.md) |
| 3 | **Causal dissociation.** Steering `h_t` along `v` strongly moves the internal readout but not imagination error. | **Supported at k=0 on 13/13 models.** The E null holds on the original 3 tasks but not on walker. Persistence at k=10 holds on cartpole only. | Readout z ≈ +11 at k=0; max \|z\| on E = 0.59 over 13 models | [fig08](outputs/camera_ready/figures_final/fig08_steering.pdf) | [camera_ready_numbers.md](outputs/camera_ready/camera_ready_numbers.md) |

Values in claim rows are in task order cartpole / reacher / pendulum (/ walker / cheetah).

**Supporting and qualifying analyses:**

| Question | Answer | Figure | Result file |
|---|---|---|---|
| Does the readout itself predict imagination error, and is that special to `v`? | It predicts E beyond the baselines, but projections onto random directions of `h_t` add about as much | — | [T1_readout.md](outputs/camera_ready/T1_readout.md) |
| How do the readout and `C_t` compare with six standard uncertainty signals? | The readout adds a small, reliable increment on all 3 tasks. `C_t` adds clearly only on pendulum. | [fig06](outputs/camera_ready/figures_final/fig06_comparison_forest.pdf) | [T2_comparison.md](outputs/camera_ready/T2_comparison.md) |
| Is it useful for deciding when to trust imagination? | **No.** A pre-registered risk–coverage test fails on 3/3 tasks; lagged reconstruction error does better. | [fig07a–c](outputs/camera_ready/figures_final/) | [T3_decision.md](outputs/camera_ready/T3_decision.md) |
| Does it generalise to new tasks? | Partly (walker-walk, cheetah-run) | [fig04](outputs/camera_ready/figures_final/fig04_cv_deltaR2_Ct.pdf), [fig08](outputs/camera_ready/figures_final/fig08_steering.pdf) | [T4_generality.md](outputs/camera_ready/T4_generality.md) |
| Is it a single sparse feature? | No. It is spread over many sparse-autoencoder atoms. | [fig09](outputs/camera_ready/figures_final/fig09_atoms.pdf) | [B_sae_corrected_refit.json](outputs/camera_ready/B_sae_corrected_refit.json) |

**Bottom line.** Recurrent world models carry a history-of-difficulty signal that is real (claim 1) and externally meaningful (claim 2). It is causally separable from predictive quality (claim 3): the direction that carries the model's report of its own difficulty can be moved without changing how well it imagines. The signal is small in practical terms and does not beat simple baselines for decision-making.

---

## 4. Results in detail

Numbers come from the evaluation episodes: 100 per task, 46,700 time steps ("sites") per task. Intervals are 95% bootstrap CIs with episodes as the resampling unit (1,000 resamples). Multi-model numbers are mean ± sd over independently trained world models. Every number in this section is generated by a script into a JSON file; none is typed by hand.

### 4.1 Emergence: the hidden state tracks a history of difficulty

**The probe works.** On cartpole it separates high- from low-KL states with AUROC 0.983 (0.832 with σ = 0.1 observation noise).

**It tracks history, not just the present.** The readout's variance is explained better by `C_t` than by current KL:

| | cartpole | reacher | pendulum |
|---|---|---|---|
| R²(readout, `C_t`), primary model | **0.84** [0.80, 0.87] | **0.50** [0.47, 0.54] | **0.89** [0.87, 0.91] |
| R²(readout, `KL_t`), primary model | 0.60 | 0.47 | 0.36 |
| Models where `C_t` > KL (probability readout) | 5 / 5 | 3 / 4 | 3 / 4 |
| Models where `C_t` > KL (logit readout) | 1 / 5 | 0 / 4 | 1 / 4 |

**Caveat.** The probe's probability saturates at many sites, and part of the "`C_t` is ahead of KL" result comes from that saturation. On the unsaturated logit scale, `C_t` and current KL explain the readout about equally well. Both views are reported.

**Temporal order matters.** A ridge regression from `h_t` predicts real-order `C_t` far better than `C_t` rebuilt from the same high-KL flags shuffled in time (current flag held fixed):

| | cartpole | reacher | pendulum |
|---|---|---|---|
| z (real vs. 20 scrambles) | +22.8 | +36.3 | +2.4 |

**Geometry.** The readout direction `v` lies almost entirely outside the high-variance subspace of `h_t`. On cartpole, 7.4% of `v` is in the top-50 principal components, against 19.6% for a random vector. Its top-10-PC share is below the random 5th percentile on 12 of 13 models.

### 4.2 External meaning: history predicts imagination error

**Correlation at every horizon.** `C_t` correlates positively with imagination error at K = 1, 5, 10 and 20 on every task ([fig03](outputs/camera_ready/figures_final/fig03_horizon.pdf)):

| r(`C_t`, `E_{t,K}`) | K=1 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| cartpole | +0.63 | +0.65 | +0.64 | +0.60 |
| reacher | +0.48 | +0.51 | +0.52 | +0.52 |
| pendulum | +0.28 | +0.25 | +0.19 | +0.11 |

**Beyond the baselines.** `C_t` adds variance in `E_{t,10}` beyond {KL, Rec, EMARec}, and beyond ensemble disagreement on cartpole. The out-of-sample version is fitted on 4/5 of the episodes and scored on the held-out 1/5.

| | cartpole | reacher | pendulum | walker-walk | cheetah-run |
|---|---|---|---|---|---|
| R² of baselines alone | 0.70 | 0.59 | 0.49 | 0.06 | 0.02 |
| ΔR² of `C_t`, in-sample | +0.0168 | +0.0053 | +0.0022 | +0.0021 | +0.0113 |
| **ΔR²_cv of `C_t`, out of sample** | **+0.0168** | **+0.0101** | **+0.0021** | **+0.0019** | **+0.0113** |
| out-of-sample CI excludes 0 | ✓ | ✓ | ✓ | ✓ | ✓ |

The increments are small but consistent. Out-of-sample R² can fall below in-sample R² and can be negative, so these are a fair test.

**History, not just smoothing.** Real-order `C_t` beats time-scrambled `C_t` at predicting imagination error on every task (z = +25.1 / +54.6 / +21.0; [fig05](outputs/camera_ready/figures_final/fig05_order_scramble.pdf)). In head-to-head regressions, `C_t` adds to EMARec on all three tasks.

### 4.3 Causal dissociation: the readout moves, prediction quality does not

At 500 sites per model, the hidden state is steered: `h_t' = h_t + λ·v` for λ ∈ {−2, −1, 0, +1, +2}·σ_v. The same steps are applied along 50 random directions, with all conditions sharing their random draws. Three outcomes are recorded:
- the readout immediately (k = 0);
- the readout after ten more real observation steps (k = 10);
- imagination error `E` from the steered state.

| | n models | readout z, k=0 | readout z, k=10 | imagination error z |
|---|---|---|---|---|
| cartpole | 5 | **+10.6 ± 1.5** | +6.2 ± 4.0 | −0.1 ± 0.4 |
| reacher | 4 | **+11.1 ± 0.8** | +1.2 ± 1.3 | −0.1 ± 0.2 |
| pendulum | 4 | **+11.4 ± 4.7** | −0.1 ± 0.1 | +0.2 ± 0.3 |
| walker-walk | 1 | +12.8 | +4.5 | **+2.30** |
| cheetah-run | 1 | +11.9 | +5.0 | +0.04 |

- **Dissociation:**
  - On all 13 models of the three original tasks, steering moves the readout far beyond random directions, but never moves imagination error (max |z| = 0.59). This also holds with observation noise (max |z| = 0.37) and when `v` is ablated instead of steered.
  - On walker-walk, imagination error does move (z = +2.30). The dissociation is therefore not universal.
- **Persistence:** the readout change survives ten recurrent steps (z > 2) on 4 of 5 cartpole models and on both new tasks. It does not survive on reacher or pendulum.

**Interpretation.** The direction that carries the model's report of its own difficulty is not the mechanism of its predictive quality. Had only the readout been measured, steering would have looked like strong causal evidence for a "difficulty mechanism". We recommend that causal claims about self-monitoring representations report both an internal and an external outcome.

### 4.4 Comparison with standard uncertainty signals

The six standard signals are KL, Rec, EMARec, EMA-KL, prior entropy and posterior entropy. Together they explain 0.76 / 0.65 / 0.61 of the variance in `E_{t,10}`. ([T2_comparison.md](outputs/camera_ready/T2_comparison.md), [CV_delta_r2.md](outputs/camera_ready/CV_delta_r2.md), [fig06](outputs/camera_ready/figures_final/fig06_comparison_forest.pdf).)

| Increment beyond all other signals | cartpole | reacher | pendulum |
|---|---|---|---|
| readout (logit), in-sample | +0.014 ✓ | +0.006 ✓ | +0.003 ✓ |
| readout (logit), out of sample | ✓ | ✓ | ✓ |
| `C_t`, out of sample | ✗ | ✗ | ✓ |

All increments are ≤ 0.015.

Against random directions, the readout's *standalone* association with imagination error clears the null on cartpole and reacher. Its *incremental* value beyond the baselines does not clear the null on any task ([T1_readout.md](outputs/camera_ready/T1_readout.md)).

### 4.5 Decision-making test (negative result)

Can the readout or `C_t^past` decide *which imagined rollouts to trust*? Sites are ranked by each signal available at decision time, and we measure the mean imagination error of the trusted fraction (risk–coverage curves, area under the curve AURC).

The pre-registered success rule required beating the best simple baseline (with a CI) and beating the random-direction null on at least 2 of 3 tasks.

**Result: the rule fails on all 3 tasks.** Lagged reconstruction error and its moving average rank sites better than either the readout or `C_t`. The readout does rank sites much better than random directions on cartpole and reacher. Details: [T3_decision.md](outputs/camera_ready/T3_decision.md), [fig07a–c](outputs/camera_ready/figures_final/).

### 4.6 New tasks: walker-walk and cheetah-run

Two extra world models were trained with the same recipe ([T4_generality.md](outputs/camera_ready/T4_generality.md)).

- **What carries over:**
  - The probe detects high-KL states (AUROC 0.76, 0.83).
  - `C_t` adds out-of-sample predictive value for imagination error on both tasks.
  - Steering moves the readout at k = 0 and k = 10.
- **What does not:**
  - `C_t` is not ahead of current KL in explaining the readout.
  - On walker, steering also moves imagination error.

### 4.7 Is it a single feature? Sparse autoencoders

Sparse autoencoders were trained on `h_t`, three per task. No small set of atoms captures `v`: the top-10 atoms carry ≤ 3% of it, and every selected atom has |cos(atom, v)| ≤ 0.08.

Ablating the atoms most correlated with imagination error changes imagination error in 8 of 9 runs, but does not consistently move the readout. The difficulty readout is a **distributed** direction, not a sparse feature ([fig09](outputs/camera_ready/figures_final/fig09_atoms.pdf)).

### 4.8 A filtering-theory account

For a correctly specified, stationary linear-Gaussian system, a Kalman filter's innovations are white, so a history of past surprise should add nothing. Under misspecification or non-stationarity, innovations become serially correlated and accumulated surprise becomes predictive.

A scalar Kalman-filter simulation (3 regimes × 10 seeds) reproduces this. A `C_t` analogue adds the following beyond the filter's own instantaneous signals:

| Regime | ΔR² | z |
|---|---|---|
| Correctly specified, stationary | +0.0004 ± 0.0003 | — |
| Misspecified | +0.0046 ± 0.0018 | +13.3 |
| Non-stationary | +0.083 ± 0.014 | +257.6 |

Script: `run_phase5_filtering_theory.py`.

---

## 5. What the results do not show

- **Not subjective uncertainty, and not a failure detector.** The representation is derived from the model's own surprise signal, so it reflects difficulty the model has already registered. It does not beat simple baselines for deciding when to trust imagination (§4.5).
- **Not the mechanism of predictive quality.** Its demonstrated causal role is limited to the model's internal report (§4.3).
- **Not scale- or modality-general.** The study uses one RSSM family at one small scale, with vector observations and data from a uniform-random policy. Pixels, larger models and learned-policy data are untested.
- **Probe limits.** The probe is linear and trained on a binary label. Other readouts could find more.
- **Some results are task-dependent:**
  - persistence of the steering effect (§4.3);
  - the walker exception to the dissociation (§4.3);
  - whether `C_t` adds value beyond *all* standard signals (§4.4);
  - a reconstruction-error-matched contrast whose direction follows the within-bin sign of r(Rec, `C_t`) on each task.

---

## 6. Experimental setup

### World model

Mini-DreamerV3, XS configuration:

| Component | Value |
|---|---|
| Deterministic state `h_t` | 256-d GRU |
| Stochastic latent `z_t` | 32 categoricals × 32 classes |
| Parameters | ≈ 12M |
| Observations | vector (proprioceptive) |
| Training | 100,000 environment steps, uniform-random policy |

The recurrent update is `h_t = GRU([z_{t−1}, a_{t−1}], h_{t−1})`. Every analysis logs states with exactly this pairing: observation `o_t` enters the posterior together with the action `a_{t−1}` that produced it, as in training. This is tested in `tests/test_camera_ready.py`.

### Tasks (dm-control)

| Task | Obs dim | Action dim | Models |
|---|---|---|---|
| cartpole-swingup | 5 | 1 | 5 (1 primary + 4 replicates) |
| reacher-easy | 6 | 2 | 4 |
| pendulum-swingup | 3 | 1 | 4 |
| walker-walk | 24 | 6 | 1 |
| cheetah-run | 17 | 6 | 1 |

Ten further cartpole models (5 full-observation / 5 partial-observation pairs) test partial observability.

### Data splits

All analyses use fresh episodes of 500 steps, split by episode with fixed seeds. The first 12 steps of each episode are burn-in and excluded.

| Split | Episode seeds | Used for |
|---|---|---|
| Probe-fit | 10000–10099 (100) | probe `v`, PCA, sparse autoencoders |
| Calibration | 20000–20049 (50) | KL threshold `τ` |
| Evaluation | 30000–30099 (100) | **every reported number** |

### Fixed constants

| Constant | cartpole | reacher | pendulum |
|---|---|---|---|
| Discount γ | 0.95 | 0.70 | 0.90 |
| EMARec α | 0.02 | 0.5 | 0.02 |

For walker-walk and cheetah-run, γ and α are chosen by the same selection rule on probe-fit episodes.

### Statistics

- **Bootstrap:** 1,000 resamples over episodes (seed 7).
- **Random-direction null:** 50 fixed directions shared by all analyses (`outputs/canonical_null/random_directions_50_dim256.npz`), with common random numbers across conditions.
- **Cross-validation:** grouped 5-fold by episode (seed 55).
- **Seeds:** every stochastic step is seeded.

---

## 7. Installation

```bash
git clone <this repository>
cd Signal-Check-Uncertainty
pip install -r requirements.txt     # torch, dm_control, scikit-learn, numpy, matplotlib, scipy
python -m pytest -q tests/test_camera_ready.py
```

Python 3.11. CPU is recommended: on Apple Silicon at this batch size, CPU is faster than MPS and more reproducible.

---

## 8. Reproducing every number and figure

Run from the repository root, in order. Outputs go to `outputs/camera_ready/` unless noted.

**Model checkpoints are not included in git** because of their size. Train them with the scripts in step 0, or place them under `outputs/checkpoints/`.

```bash
# 0. Train world models (several CPU hours each)
python run_experiment.py                  # cartpole primary + 3-member ensemble baseline
python run_second_env.py                  # reacher
python compare_environments.py            # pendulum
python run_multiseed.py                   # cartpole replicates
python run_multiseed_env.py               # reacher / pendulum replicates
python run_phase4_pomdp_training.py       # full / partial-observation cartpole pairs
python run_t4_train.py walker walk        # walker-walk
python run_t4_train.py cheetah run        # cheetah-run

# 1. Log states and run per-model analyses (emergence, geometry, external)
python run_cr_A_log.py
python run_cr_A_analysis.py
for m in steer noisy pomdp taskg; do python run_cr_A_causal.py --mode $m; done   # causal tests
python run_cr_A_report.py

# 2. Sparse autoencoders (3 per task)
CR_PROBE=frozen python run_cr_B_sae.py

# 3. Final numbers -> camera_ready_numbers.{json,md}
python run_cr_final_numbers.py
python run_cr_numbers_table.py

# 4. Readout analysis, comparison with standard signals, decision test
python run_t123.py                        # T1_readout, T2_comparison, T3_decision

# 5. New tasks
python run_t4_analyze.py                  # T4_generality

# 6. Out-of-sample ΔR²
python run_cv_delta_r2.py
python run_cv_delta_r2.py --t4
python run_cv_report.py                   # CV_delta_r2.md

# 7. Figures (every plotted value is checked against the result JSON)
python paper/figures/src/make_cr_figures_final.py

# Filtering-theory simulation (§4.8)
python run_phase5_filtering_theory.py
```

Optional environment variables:

- `CR_CONVENTION=buggy` logs with the wrong action pairing, as a control.
- `CR_STATE_DIR` and `CR_RAW_DIR` redirect outputs.

---

## 9. Repository map

```
src/
  model/              world model: encoder, RSSM (GRU + categorical latent), decoder
  training/           world-model training loop
  env/                dm-control wrappers
  probe/              probe training, C_t, intervention helpers
  rerun/              shared statistics: episode-bootstrap (GramSet), site masks, EMA
  camera_ready/
    core.py           state logging, episode splits, imagination error E_{t,K}, model registry
    causal.py         steering / ablation against the random-direction null

run_cr_A_*.py         logging and per-model analyses
run_cr_B_sae.py       sparse autoencoders
run_cr_final_numbers.py, run_cr_numbers_table.py    final numbers
run_t123.py           readout analysis, comparison, decision test
run_t4_train.py, run_t4_analyze.py                  walker / cheetah
run_cv_delta_r2.py, run_cv_report.py                out-of-sample ΔR²
run_phase5_filtering_theory.py                      Kalman-filter simulation

paper/figures/src/make_cr_figures_final.py          paper figures
tests/test_camera_ready.py                          action pairing, logging, determinism
outputs/camera_ready/                               results and figures (below)
```

---

## 10. Result files reference

All in `outputs/camera_ready/`.

| File | Contents |
|---|---|
| [camera_ready_numbers.md](outputs/camera_ready/camera_ready_numbers.md) / `.json` | Every number reported for emergence, external meaning, causal tests, geometry, partial observation and sparse autoencoders, with flags |
| [CV_delta_r2.md](outputs/camera_ready/CV_delta_r2.md) / `.json` | In-sample vs. out-of-sample ΔR² on all five tasks |
| [T1_readout.md](outputs/camera_ready/T1_readout.md) / `.json` | Direct predictive value of the readout, against the random-direction null |
| [T2_comparison.md](outputs/camera_ready/T2_comparison.md) / `.json` | Readout and `C_t` vs. six standard uncertainty signals |
| [T3_decision.md](outputs/camera_ready/T3_decision.md) / `.json` | Risk–coverage decision test (pre-registered, negative) |
| [T4_generality.md](outputs/camera_ready/T4_generality.md) / `.json` | walker-walk and cheetah-run |
| [A_consistency.md](outputs/camera_ready/A_consistency.md) / `.json` | Per-model results and an action-pairing control |
| `B_sae_corrected_refit.json` | Sparse-autoencoder results |
| [figures_final/](outputs/camera_ready/figures_final/) | Paper figures (PDF + PNG). [FIGURE_MANIFEST.md](outputs/camera_ready/figures_final/FIGURE_MANIFEST.md) lists each figure's data source and plotted values. |

---

## 11. Earlier experiments in this repository

The repository also contains the exploratory work that led to the paper:
- `run_task_*.py`, `run_phase*.py`, `run_sprint_*.py`;
- write-ups in [RESULTS_CONSOLIDATED.md](RESULTS_CONSOLIDATED.md) and [DEV_LOG.md](DEV_LOG.md).

Most of those runs logged states with a different action pairing (`o_t` with `a_t`) and used an earlier probe, so their numbers differ from the paper's. **Where they disagree, the results in `outputs/camera_ready/` are the ones the paper reports.** Files marked `SUPERSEDED` are kept only for the record.

---

*Author: Donald Aadithiyan, University of Moratuwa.*
