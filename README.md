# Endogenous Predictive-Difficulty Representations in Recurrent World Models

Code, results and figures for *Endogenous Predictive-Difficulty Representations in Recurrent World Models* (Donald Aadithiyan, ICAART 2027).

> **In one sentence.** The hidden state of a recurrent world model carries an informative report of its own recent predictive difficulty, but that report is not the mechanism that controls prediction quality.

Everything in this repository runs on a laptop CPU. No GPU is needed.

---

## Contents

1. [The question](#1-the-question)
2. [Key concepts](#2-key-concepts)
3. [The four findings at a glance](#3-the-four-findings-at-a-glance)
4. [Results in detail](#4-results-in-detail)
5. [Interpretation](#5-interpretation)
6. [Limitations](#6-limitations)
7. [Experimental setup](#7-experimental-setup)
8. [Installation](#8-installation)
9. [Reproducing every number and figure](#9-reproducing-every-number-and-figure)
10. [Repository map](#10-repository-map)
11. [Result files and figures](#11-result-files-and-figures)
12. [Earlier experiments in this repository](#12-earlier-experiments-in-this-repository)

---

## 1. The question

A world model lets an agent imagine the consequences of its actions without acting. Recurrent world models such as World Models, PlaNet and the Dreamer family keep a hidden state `h_t` that summarises the past for this purpose. No training objective asks that state to describe how hard its own recent predictions have been.

The obvious difficulty signals are *instantaneous*:
- how surprised the model is by the current observation (the KL divergence between its prior and posterior);
- how badly it reconstructs that observation.

A hidden state, however, is a running summary, so it could also carry a **history** of difficulty. If a model is systematically wrong in some regime, its surprise tends to arrive in streaks. Recent surprise then says more about the next few steps than a single step does.

**We ask: does the hidden state of a recurrent world model carry a history of its own predictive difficulty, is that signal useful, and is it the mechanism that determines prediction quality?**

Answering this takes more than fitting a probe:
- A readout that predicts current surprise is partly circular, because current surprise is what the probe is trained on.
- A readout that only tracks smoothed recent error adds nothing a running average could not.
- A readout that can be moved by intervention is not thereby shown to control the capability it appears to report.

So we find the signal, validate it against an external quantity defined without reference to the probe, test whether it can support a decision, and intervene on it while measuring **both** the readout and that external quantity.

---

## 2. Key concepts

| Term | Meaning |
|---|---|
| `h_t` | The world model's hidden state: 256 numbers, updated by a GRU as `h_t = GRU([z_{t−1}, a_{t−1}], h_{t−1})`. |
| `z_t` | A small random latent: 32 categorical variables with 32 classes each. |
| **Prior / posterior** | The model's guess about `z_t` before seeing observation `o_t` (prior, from `h_t` alone) and after seeing it (posterior). |
| `KL_t` | KL divergence between posterior and prior: how surprised the model was at step `t`. |
| `Rec_t` | Reconstruction error: how badly the model can redraw `o_t` from its state. |
| `EMARec_t` | Smoothed reconstruction error, an exponential moving average `α·Rec_t + (1−α)·EMARec_{t−1}`. |
| **Linear probe** | A logistic-regression classifier on `h_t` computing `σ(wᵀh_t + b)`. The unit vector `v = w/‖w‖` is the probe **direction**, `wᵀh_t + b` is the **logit**, and the output probability is the **readout**. |
| **`C_t`** | The **accumulated-difficulty score**: a discounted count of recent high-surprise steps, `C_t = Σ_{i=0..t} γ^i · 1[KL_{t−i} > τ]`. |
| **Imagination** | Rolling the model forward from step `t` using only the real actions and its own predictions, decoding each state back into an observation. |
| **`E_{t,K}`** | **Imagination error**: the mean distance between imagined and real observations over the next `K` steps, `(1/K) Σ_{k=1..K} ‖ô_{t+k} − o_{t+k}‖₂`. It is defined without reference to KL or to any probe. `K = 10` is the headline horizon. |
| **Site** | A starting step `t` at which `E` is measured. There are 46,700 per task. |
| **ΔR²_cv** | Out-of-sample incremental variance. Ordinary least squares is fit on 4 of 5 folds of evaluation trajectories and predicts the fifth. ΔR²_cv = R²_cv(baselines + signal) − R²_cv(baselines), computed on the pooled out-of-fold predictions. Unlike in-sample ΔR², it can be negative. |
| **`z` against the null** | A causal effect scored as `(effect − mean null effect) / sd of null effects`, where the null applies the same manipulation along 50 random directions of matched norm. |
| **Steering / ablation** | Steering adds a multiple of `v` to `h_t`. Ablation removes the component of `h_t` along a direction. |
| **Dissociation** | An intervention moves one quantity and not the other. |
| **Sparse autoencoder (SAE)** | Learns an overcomplete dictionary of directions in `h`-space (**atoms**), so that each hidden state is reconstructed from a few active atoms. |

---

## 3. The four findings at a glance

Task order throughout is cartpole / reacher / pendulum (the three main tasks), then walker / cheetah (two further tasks used to test generality).

| # | Finding | Key numbers | Figure | Result file |
|---|---|---|---|---|
| **1** | **A probe trained only to detect current surprise yields a direction whose readout tracks a discounted history of past surprise.** | R²(readout, `C_t`) = 0.84 / 0.50 / 0.89, against 0.60 / 0.47 / 0.36 for current KL | [fig02](outputs/camera_ready/figures_final/fig02_R2_Ct_vs_KL.pdf) | [camera_ready_numbers.md](outputs/camera_ready/camera_ready_numbers.md) |
| **2** | **On held-out trajectories, `C_t` predicts later imagination error beyond current surprise, reconstruction error and smoothed reconstruction error, on all five tasks.** | ΔR²_cv = +0.0168 / +0.0101 / +0.0021 / +0.0019 / +0.0113; every 95% interval excludes zero | [fig04](outputs/camera_ready/figures_final/fig04_cv_deltaR2_Ct.pdf) | [CV_delta_r2.md](outputs/camera_ready/CV_delta_r2.md) |
| **3** | **The readout adds predictive value beyond six standard uncertainty signals on two of the three main tasks, and supports a decision about when to trust imagination.** Trusting the half of sites it ranks safest lowers their imagination error by 24–64% relative to random trust. It does not beat the previous step's reconstruction error. | beyond all six: +0.0104 (cartpole), +0.0011 (pendulum), not reacher | [fig06](outputs/camera_ready/figures_final/fig06_comparison_forest.pdf), [fig07a–c](outputs/camera_ready/figures_final/) | [T2_comparison.md](outputs/camera_ready/T2_comparison.md), [T3_decision.md](outputs/camera_ready/T3_decision.md) |
| **4** | **Main contribution. Steering the probe direction moves the readout but not imagination error on 14 of 15 models. Ablating error-correlated SAE atoms, nearly orthogonal to that direction, raises imagination error in 8 of 9 dictionary runs.** | readout z = +10.6 to +12.8 per task; max \|z\| on E = 0.59 over the 13 main-task models; atom ablation z on E from +3.3 to +34.0 | [fig08](outputs/camera_ready/figures_final/fig08_steering.pdf), [fig09](outputs/camera_ready/figures_final/fig09_atoms.pdf) | [camera_ready_numbers.md](outputs/camera_ready/camera_ready_numbers.md) |

**How the findings fit together.** Findings 1–3 establish that the signal is present, informative and usable. Finding 4 shows that it is nevertheless not the lever on the model's predictions. Had only the readout been measured, the steering result would have read as strong evidence for a difficulty mechanism. Measuring an external outcome in the same intervention, against the same random-direction baseline, revealed the dissociation.

---

## 4. Results in detail

All quantities are measured on 100 held-out evaluation episodes per task (46,700 sites). Intervals are 95% bootstrap intervals with trajectories as the resampling unit (1,000 resamples). Multi-model results report mean ± standard deviation over independently trained models. Every number below is produced by a script into a JSON file in `outputs/camera_ready/`.

### 4.1 Finding 1: a probe direction that tracks accumulated difficulty

**The probe.** A logistic-regression probe is trained on the standardised hidden states of 100 probe-fit episodes. It predicts whether *current* surprise is high, `1[KL_t > τ]`. It is never shown `C_t` or any past surprise. On held-out evaluation episodes it separates high- from low-KL states:

| | cartpole | reacher | pendulum | walker | cheetah |
|---|---|---|---|---|---|
| Probe AUROC (primary model) | 0.98 (0.83 with σ=0.1 observation noise) | 0.88 | > 0.99 | 0.76 | 0.83 |

**Temporal accumulation.** On the three main tasks, the readout is explained better by `C_t` than by current KL:

| | cartpole | reacher | pendulum | walker | cheetah |
|---|---|---|---|---|---|
| R²(readout, `C_t`) | **0.84** | **0.50** | **0.89** | 0.27 | 0.26 |
| R²(readout, `KL_t`) | 0.60 | 0.47 | 0.36 | 0.29 | 0.44 |

- On the main tasks, `C_t` is ahead on 11 of 13 models (for the probe's output probability; see [Limitations](#6-limitations)).
- On walker and cheetah, the readout also tracks `C_t`, but current KL explains as much or more. The advantage of accumulated over current difficulty is specific to the main tasks.

**Order matters.** `C_t` is recomputed from the same high-KL flags permuted within each trajectory, which keeps their rate but destroys their order. A ridge regression on `h_t` then explains it much less well:

| | cartpole | reacher | pendulum |
|---|---|---|---|
| R², real vs. scrambled `C_t` | 0.82 vs 0.23 | 0.53 vs 0.02 | 0.94 vs 0.94 |
| z against 20 scrambles | +22.8 | +36.3 | +2.4 |

On pendulum the fit is carried mainly by the *rate* of high-KL steps within a trajectory.

As a sanity check, ablating the direction (`h_t − v vᵀh_t`) removes the readout far beyond random directions (z = −27.7 on the probe logit).

**Geometry.** Only a small share of `v` lies where the hidden state varies most. The share of ‖v‖² in the top ten principal components of `h_t` is 0.6% on cartpole and 1.3% on reacher, against 3.9% for a random direction. It is below the random 5th percentile on 12 of 13 models; the exception is the primary pendulum model (6.2%).

**Partial observation.** Five pairs of cartpole models are identical except that one model of each pair has velocity masked from its encoder input. The probe is near ceiling in both conditions (AUROC 0.97 vs 0.97). The readout fits `C_t` slightly better under partial observation on four of five pairs (R² = 0.82 vs 0.84).

### 4.2 Finding 2: `C_t` predicts later imagination error

**Horizon dependence.** `C_t` is positively correlated with `E_{t,K}` at every horizon on the three main tasks ([fig03](outputs/camera_ready/figures_final/fig03_horizon.pdf)):

| r(`C_t`, `E_{t,K}`) | K=1 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| cartpole | +0.63 | +0.65 | +0.64 | +0.60 |
| reacher | +0.48 | +0.51 | +0.52 | +0.52 |
| pendulum | +0.28 | +0.25 | +0.19 | +0.11 |

The association is strong and nearly horizon-independent on cartpole and reacher, and weaker and decaying on pendulum.

**Incremental information, out of sample.** Because `C_t` is built from KL, and KL may itself predict later error, a correlation is not sufficient. We fit `E_{t,10} ~ C_t + KL_t + Rec_t + EMARec_t` (plus ensemble disagreement on cartpole) and report the variance `C_t` explains beyond the baselines on held-out trajectories:

| | cartpole | reacher | pendulum | walker | cheetah |
|---|---|---|---|---|---|
| **ΔR²_cv of `C_t`** | **+0.0168** | **+0.0101** | **+0.0021** | **+0.0019** | **+0.0113** |
| 95% interval | [+0.0065, +0.0271] | [+0.0015, +0.0200] | [+0.0006, +0.0036] | [+0.0006, +0.0034] | [+0.0080, +0.0150] |
| ΔR², in-sample | +0.0168 | +0.0053 | +0.0022 | +0.0021 | +0.0113 |
| R²_cv of baselines alone | 0.70 | 0.53 | 0.49 | 0.06 | 0.02 |

- **`C_t` adds out-of-sample predictive value on all five tasks.** The in-sample values are of the same order, so the gains are not an artefact of fitting.
- The gains are small relative to baselines that already explain much of the variance on the main tasks.
- On cheetah, `C_t` is uncorrelated with `E` on its own (r = −0.01) and is informative only together with the baselines.

**Accumulated history and its order.**
- Adding `C_t` to a regression on `EMARec_t` alone raises the variance explained in `E_{t,10}` by +0.166 / +0.100 / +0.049 (in sample).
- Real-order `C_t` is more correlated with `E_{t,10}` than time-scrambled `C_t` on all three main tasks: r = 0.64 vs 0.12, 0.52 vs 0.00, 0.19 vs 0.01 (z = +25.1 / +54.6 / +21.0; [fig05](outputs/camera_ready/figures_final/fig05_order_scramble.pdf)).

### 4.3 Finding 3: comparison with standard signals, and a decision

**Predictive value of the readout itself.** The readout predicts later imagination error directly (r = 0.69 / 0.48 / 0.51). Its value beyond KL, Rec and EMARec ([CV_delta_r2.md](outputs/camera_ready/CV_delta_r2.md)):

| | cartpole | reacher | pendulum |
|---|---|---|---|
| ΔR²_cv of the readout | **+0.0429** [+0.0275, +0.0603] | +0.0021 [−0.0005, +0.0054] (not distinguishable from zero) | **+0.0255** [+0.0208, +0.0304] |

On cartpole and pendulum this is more than `C_t` adds.

Compared on the logit against random directions in `h_t` ([T1_readout.md](outputs/camera_ready/T1_readout.md)):
- the readout's standalone association exceeds theirs on cartpole and reacher (z = +5.5 and +32.3 for R²) but not on pendulum (z = +1.7);
- its incremental value does not exceed theirs on any task.

The logit is used here because a random direction passed through the probe's sigmoid saturates, which makes the probability uninformative for this comparison. Information about later imagination error is therefore spread across `h_t`. The probe direction is one interpretable way to read it out, not the only one.

**A decision: when to trust imagination** ([T3_decision.md](outputs/camera_ready/T3_decision.md), [fig07a–c](outputs/camera_ready/figures_final/)).

The setup is selective trust:
- At each site, an agent either trusts its ten-step imagined rollout or falls back to real observations.
- A signal ranks the sites, and at coverage `c` the agent trusts the fraction `c` with the lowest signal.
- Risk is the mean `E_{t,10}` over trusted sites, summarised by the area under the risk–coverage curve (AURC; lower is better).
- Every signal uses only information available before `o_t` is observed.

The success criterion was fixed before the analysis: lower AURC than the best baseline, with an interval excluding zero, and better than random directions in `h_t`, on at least two tasks.

| | cartpole | reacher | pendulum |
|---|---|---|---|
| Mean `E` of the safest half, readout | **0.141** | **0.163** | **0.153** |
| Mean `E` of the safest half, random trust | 0.386 | 0.215 | 0.235 |
| AURC, readout | 0.187 | 0.168 | 0.171 |
| AURC, best baseline (previous-step Rec or EMARec) | **0.182** | **0.159** | **0.166** |
| Readout vs random directions, z | +2.6 | +5.1 | +1.5 |

- The readout is a working basis for the decision: it cuts the error of trusted rollouts well below random trust.
- **It does not meet the pre-set criterion.** The previous step's reconstruction error or its moving average achieves a slightly lower AURC on every task (by about 3–6%, intervals excluding zero), and `C_t` does not do better.

**Comparison with six standard signals** ([T2_comparison.md](outputs/camera_ready/T2_comparison.md), [fig06](outputs/camera_ready/figures_final/fig06_comparison_forest.pdf)).

The six standard signals are KL, Rec, EMARec, EMA-KL, prior entropy and posterior entropy. Individually, the strongest are:
- posterior entropy and EMA-KL on cartpole (|r| = 0.80 each);
- EMA-KL on reacher (0.75);
- Rec on pendulum (0.63).

Together the six explain R²_cv = 0.75 / 0.63 / 0.61 of `E_{t,10}`.

| Beyond all six standard signals | cartpole | reacher | pendulum |
|---|---|---|---|
| readout, ΔR²_cv | **+0.0104** [+0.0041, +0.0183] | interval includes zero | **+0.0011** [+0.00004, +0.0020] |
| `C_t`, ΔR²_cv | interval includes zero | interval includes zero | **+0.0030** [+0.0013, +0.0046] |

The internal readout carries information about later imagination error that the six standard signals together miss on two of three tasks. The information in `C_t` is largely shared with EMA-KL and the entropies.

### 4.4 Finding 4: steering dissociates the readout from imagination error

**Question: does changing what the readout says about predictive difficulty change prediction quality?**

Every intervention is measured on two outcomes: the readout, and the imagination error `E_{t,10}` obtained by imagining open-loop from the intervened state.

- **Intervention:** `h_t' = h_t + λv` for λ ∈ {−2, −1, 0, +1, +2}·σ_v, where σ_v is the standard deviation of `vᵀh`.
- **Scoring:** the slope of each response against λ, relative to the 50-direction null, on the same sites and with the same random seeds.
- **Timing:** the readout is recorded immediately (k = 0) and after ten steps (k = 10), during which the model updates its state with the real observations.

| | models | readout z, k=0 | readout z, k=10 | imagination error z |
|---|---|---|---|---|
| cartpole | 5 | +10.6 ± 1.5 | +6.2 ± 4.0 (4 of 5 models > 2) | −0.1 ± 0.4 |
| reacher | 4 | +11.1 ± 0.8 | +1.2 ± 1.3 | −0.1 ± 0.2 |
| pendulum | 4 | +11.4 ± 4.7 | −0.1 ± 0.1 | +0.2 ± 0.3 |
| walker | 1 | +12.8 | +4.5 | **+2.30** |
| cheetah | 1 | +11.9 | +5.0 | +0.04 |

- **The readout changes on every model, but imagination error stays within the null on 14 of 15 models.** The effect on `E` never exceeds |z| = 0.59 on the 13 main-task models; walker is the exception.
- **The k = 0 effect is expected by construction,** since random directions barely project onto `v`. The informative outcomes are imagination error and the readout ten steps later.
- **Ten steps later,** the readout shift persists on cartpole, walker and cheetah. On reacher and pendulum the incoming observations restore it. Either way, imagination error does not follow the readout.

**Replication.**
- The null on `E` holds under observation noise (σ = 0.1; maximum |z| = 0.37 on the three primary models).
- It also holds in a separate protocol on the five full/partial-observation cartpole pairs, which are not among the 15 models. Ablating `v` gives z = −0.15 ± 0.51 on `E` with full observation and −0.38 ± 0.37 with partial observation.

**Caveats.**
- `v` is a low-variance direction. A null against matched-norm random directions is therefore also what we would expect if the decoder and prior were insensitive to perturbations off the region the state normally visits. The null was not restricted to other low-variance directions.
- The atom results below show that imagination error is *not* uniformly insensitive to edits of `h_t`.
- A null at the resolution of 50 random directions does not rule out small effects.

**Error-related atoms lie off the probe direction.** If steering `v` does not move imagination error, which directions of `h_t` do?

- **Dictionaries:** TopK sparse autoencoders (256 → 2048 atoms, 48 active per state, three independent runs), trained on pooled probe-fit hidden states of the primary models of the three main tasks.
- **`v` is diffuse in them:** the largest single atom carries 0.2–0.5% of its coefficient mass and the top ten at most 3%. A random vector is decomposed equally well, so this shows only that `v` is diffuse.
- **Atom selection:** for each task and run, take the atom whose activation correlates most with `E` among the 20 most variable atoms, using half of the evaluation episodes. Test it on the other half. This gives nine atoms; their identities differ across runs, as expected for non-canonical dictionaries.

| Ablating the selected atom | result |
|---|---|
| Effect on imagination error | raises it beyond the null in **8 of 9** runs (z from +3.3 to +34.0; the exception is one pendulum run, +0.95) |
| Alignment with `v` | nearly orthogonal: \|cos(atom, v)\| ≤ 0.08 |
| Effect on the readout (logit) | small or inconsistent: 5 of 9 within \|z\| < 2, the rest 2.3–9.5 with mixed signs |

The directions of `h_t` whose removal degrades imagination are largely separate from the direction that reports difficulty ([fig09](outputs/camera_ready/figures_final/fig09_atoms.pdf)).

---

## 5. Interpretation

**What is represented?** A probe direction whose readout tracks *accumulated predictive difficulty*: a discounted memory of recent high-surprise steps. Its score `C_t` predicts, out of sample, how far imagination will depart from reality on every task tested.

The strength of the representation and its external relevance do not coincide:
- pendulum has the strongest fit between readout and `C_t` but a small increment for imagination error;
- cheetah has a weak fit and a larger increment.

**Why could such a signal exist?** For a correctly specified, stationary linear-Gaussian system, a Kalman filter's innovations (its one-step prediction errors) are uncorrelated over time, so past innovations add nothing. Under misspecification or non-stationarity, innovations become serially correlated, and accumulated evidence of mismatch becomes predictive.

A scalar simulation (`run_phase5_filtering_theory.py`; 10 runs of 20,000 steps per regime) illustrates this:
- The truth is `x_{t+1} = a·x_t + w_t`, observed with noise.
- A `C_t` analogue counts exceedances of a fixed innovation threshold, with discounting.
- It is scored beyond the filter's instantaneous innovation and its reported posterior variance.

| Regime | Filter assumes | Truth | Incremental R² of the `C_t` analogue |
|---|---|---|---|
| Stationary | a = 0.85 | a = 0.85 | +0.0004 ± 0.0003 |
| Misspecified | a = 0.55 | a = 0.85 | +0.0046 ± 0.0018 |
| Non-stationary | a = 0.85 | a shifts to 0.97 midway | +0.0825 ± 0.0142 |

A world model trained on finite data is never a correctly specified filter. That is one candidate reason for a difficulty signal in its state.

**What is its causal scope?** The readout can be moved without changing imagination error, and imagination error can be changed through directions nearly orthogonal to the readout. The signal should be read as an **informative report**, not as subjective uncertainty, a universal failure detector, or a control signal.

It reflects difficulty the model has registered. At sites where the imagined reward is high while the real reward is low (789 of 46,700 on cartpole, 695 on reacher), `C_t` is lower than elsewhere (8.4 vs 9.0 on cartpole, 0.6 vs 1.6 on reacher).

---

## 6. Limitations

The study uses one RSSM family at one scale, with vector observations, on five tasks, with data from a uniform-random policy. Walker and cheetah have one model each, and their discount was selected at the lower edge of the search grid.

- **Saturation:** the readout is a probability that saturates on many sites. Its closer fit to `C_t` than to current KL holds for this output on the three main tasks, but not for the logit, nor on walker and cheetah.
- **Threshold:** the threshold of `C_t` in Finding 2 is the median of each evaluation trajectory, so it uses the whole trajectory. The decision analysis uses a calibration threshold and is not affected.
- **Missing baselines:** ground-truth state features (velocity, angle, action magnitude) were not used as baselines. The readout may partly track a dynamical regime that produces persistent surprise.
- **Small gains:** the out-of-sample gains are small (0.2–4% of variance), and random directions of `h_t` add comparable incremental value. The probe direction is not unique in carrying information about imagination error.
- **Decision test:** the readout does not outperform the previous step's reconstruction error.
- **Steering null:**
  - It is against random directions, not other low-variance directions.
  - Persistence of the readout shift at k = 10 is task-dependent.
- **Geometry:** the low-variance geometry describes a logistic-regression weight vector. Such weights can load on low-variance directions to cancel noise.
- **Atom ablations:** these are compared with random directions, not with other atoms, on one model per task.
- **Units:** `E` is an unnormalised norm over observation dimensions, so all analyses are within task.

Pixel observations, larger models, data from learned policies, and use of the signal for planning are outside the scope of this study.

---

## 7. Experimental setup

### World model

Mini-DreamerV3 (a reduced DreamerV3), XS configuration:

| Component | Value |
|---|---|
| Hidden state `h_t` | 256-d GRU |
| Latent `z_t` | 32 categorical variables × 32 classes |
| Parameters | ≈ 12M |
| Observations | vector (proprioceptive) |
| Training | 100,000 environment steps, uniform-random policy |

States are logged with observation `o_t` paired with the preceding action `a_{t−1}`, the convention used in training (tested in `tests/test_camera_ready.py`). All analyses query frozen models.

### Tasks (DeepMind Control Suite)

| Environment | Obs. | Act. | Models |
|---|---|---|---|
| cartpole-swingup | 5 | 1 | 5 |
| reacher-easy | 6 | 2 | 4 |
| pendulum-swingup | 3 | 1 | 4 |
| walker-walk | 24 | 6 | 1 |
| cheetah-run | 17 | 6 | 1 |

- On each main task, one model is the *primary* model, used wherever a single model is analysed.
- Five further pairs of cartpole models (full vs velocity-masked observation) serve as a partial-observation check.

### Data

For each task, three disjoint sets of episodes are collected with the frozen model, using fixed seeds:

| Set | Episode seeds | Used for |
|---|---|---|
| Probe-fit | 10000–10099 (100) | probe, principal components, sparse autoencoders |
| Calibration | 20000–20049 (50) | thresholds |
| Evaluation | 30000–30099 (100) | **every reported quantity** (46,700 sites per task) |

### Fixed constants

| | cartpole | reacher | pendulum | walker | cheetah |
|---|---|---|---|---|---|
| Discount γ of `C_t` | 0.95 | 0.70 | 0.90 | 0.70 | 0.70 |
| Effective memory 1/(1−γ) | 20 steps | 3.3 | 10 | 3.3 | 3.3 |
| EMARec α | 0.02 | 0.5 | 0.02 | 0.5 | 0.5 |

γ is chosen on data separate from the evaluation episodes, as the value whose `C_t` is best fit by the hidden state. α is chosen on separate data to maximise the correlation of EMARec with KL.

The threshold τ of `C_t` is:
- the calibration median of KL for Finding 1 and the decision analysis;
- the median of each evaluation trajectory for Finding 2.

### Statistics

- **Bootstrap:** 1,000 resamples over trajectories (seed 7).
- **Random-direction null:** 50 fixed directions (`outputs/canonical_null/random_directions_50_dim256.npz`), evaluated on the same sites and with the same random seeds as the probe direction.
- **Cross-validation:** grouped 5-fold by trajectory (seed 55).
- **Seeds:** every stochastic step is seeded.

---

## 8. Installation

```bash
git clone https://github.com/DonaldAadithiyan/Signal-Check-Uncertainty.git
cd Signal-Check-Uncertainty
pip install -r requirements.txt     # torch, dm_control, scikit-learn, numpy, matplotlib, scipy
python -m pytest -q tests/test_camera_ready.py
```

Python 3.11. CPU is recommended: at this batch size, CPU is faster than Apple-Silicon MPS and more reproducible.

---

## 9. Reproducing every number and figure

Run from the repository root, in order. Outputs go to `outputs/camera_ready/` unless noted.

**Model checkpoints are not included** because of their size. Step 0 trains them; they are written under `outputs/checkpoints/` and `outputs/t4_generality/`.

```bash
# 0. Train world models (several CPU hours each)
python run_experiment.py                  # cartpole primary + 3-model ensemble baseline
python run_second_env.py                  # reacher
python compare_environments.py            # pendulum
python run_multiseed.py                   # cartpole replicates
python run_multiseed_env.py               # reacher / pendulum replicates
python run_phase4_pomdp_training.py       # full / partial-observation cartpole pairs
python run_t4_train.py walker walk        # walker-walk
python run_t4_train.py cheetah run        # cheetah-run

# 1. Log states; probe, geometry and external analyses per model
python run_cr_A_log.py
python run_cr_A_analysis.py
for m in steer noisy pomdp taskg; do python run_cr_A_causal.py --mode $m; done   # interventions
python run_cr_A_report.py

# 2. Sparse autoencoders (3 runs)
CR_PROBE=frozen python run_cr_B_sae.py

# 3. Final numbers -> camera_ready_numbers.{json,md}
python run_cr_final_numbers.py
python run_cr_numbers_table.py

# 4. Readout analysis, comparison with six standard signals, decision test
python run_t123.py                        # T1_readout, T2_comparison, T3_decision

# 5. Walker and cheetah
python run_t4_analyze.py                  # T4_generality

# 6. Out-of-sample ΔR²
python run_cv_delta_r2.py
python run_cv_delta_r2.py --t4
python run_cv_report.py                   # CV_delta_r2.md

# 7. Figures (every plotted value is checked against the result JSON)
python figures/make_figures.py

# Filtering simulation (Section 5)
python run_phase5_filtering_theory.py
```

Optional environment variables:

- `CR_CONVENTION=buggy` logs with `o_t` paired with `a_t`, as a control.
- `CR_STATE_DIR` and `CR_RAW_DIR` redirect outputs.

---

## 10. Repository map

```
src/
  model/              world model: encoder, RSSM (GRU + categorical latent), decoder
  training/           world-model training loop
  env/                DeepMind Control wrappers
  probe/              probe training, C_t, intervention helpers
  rerun/              shared statistics: trajectory bootstrap (GramSet), site masks, EMA
  camera_ready/
    core.py           state logging, episode splits, imagination error E_{t,K}, model registry
    causal.py         steering / ablation against the random-direction null

run_cr_A_*.py         state logging and per-model analyses
run_cr_B_sae.py       sparse autoencoders
run_cr_final_numbers.py, run_cr_numbers_table.py    final numbers
run_t123.py           readout analysis, comparison with standard signals, decision test
run_t4_train.py, run_t4_analyze.py                  walker / cheetah
run_cv_delta_r2.py, run_cv_report.py                out-of-sample ΔR²
run_phase5_filtering_theory.py                      Kalman-filter simulation
figures/make_figures.py                             all figures

tests/test_camera_ready.py                          action pairing, logging, determinism
outputs/camera_ready/                               results and figures
outputs/canonical_null/                             the 50 random directions
```

---

## 11. Result files and figures

### Result files (`outputs/camera_ready/`)

| File | Contents |
|---|---|
| [camera_ready_numbers.md](outputs/camera_ready/camera_ready_numbers.md) / `.json` | Probe, accumulation, geometry, partial observation, external, steering and SAE numbers |
| [CV_delta_r2.md](outputs/camera_ready/CV_delta_r2.md) / `.json` | In-sample and out-of-sample ΔR² on all five tasks |
| [T1_readout.md](outputs/camera_ready/T1_readout.md) / `.json` | Predictive value of the readout, against the random-direction null |
| [T2_comparison.md](outputs/camera_ready/T2_comparison.md) / `.json` | Readout and `C_t` vs six standard uncertainty signals |
| [T3_decision.md](outputs/camera_ready/T3_decision.md) / `.json` | Selective-trust decision test (pre-registered criterion) |
| [T4_generality.md](outputs/camera_ready/T4_generality.md) / `.json` | Walker and cheetah |
| [A_consistency.md](outputs/camera_ready/A_consistency.md) / `.json` | Per-model results, with a control logged under the `o_t`/`a_t` pairing |
| `B_sae_corrected_refit.json` | Sparse-autoencoder results |

### Figures (`outputs/camera_ready/figures_final/`, PDF + PNG)

| File | Shows |
|---|---|
| `fig01_readout_vs_Ct` | Readout against `C_t` on cartpole |
| `fig02_R2_Ct_vs_KL` | Variance of the readout explained by `C_t` and by current KL |
| `fig03_horizon` | r(`C_t`, `E_{t,K}`) by horizon |
| `fig04_cv_deltaR2_Ct` | Out-of-sample ΔR² of `C_t` on all five tasks |
| `fig05_order_scramble` | Real-order vs time-scrambled `C_t` |
| `fig06_comparison_forest` | Readout and `C_t` beyond standard signals |
| `fig07a/b/c_risk_coverage_*` | Selective trust: risk–coverage curves per task |
| `fig08_steering` | Steering z for the readout and imagination error, 15 models |
| `fig09_atoms` | SAE atom ablations: z on imagination error vs on the readout |

---

## 12. Earlier experiments in this repository

The repository also contains the exploratory work that preceded the final analysis:
- `run_task_*.py`, `run_phase*.py`, `run_sprint_*.py`, `kaggle/`;
- notes in [RESULTS_CONSOLIDATED.md](RESULTS_CONSOLIDATED.md) and [DEV_LOG.md](DEV_LOG.md).

Most of those runs logged states with `o_t` paired with `a_t` rather than `a_{t−1}`, and used an earlier probe, so their numbers differ. **Where they disagree, the results in `outputs/camera_ready/` are authoritative.**

---

*Donald Aadithiyan, Department of Computer Science and Engineering, University of Moratuwa.*
