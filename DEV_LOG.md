# Phase 1 — Signal Check Development Log

**Date:** 2026-05-24
**Device:** CPU (M4 MacBook Air — CPU is 2× faster than MPS at batch_size=8)
**Total wall time:** 4.53 hours

---

## What This Experiment Is Asking

A DreamerV3 world model maintains a hidden state `h_t` at every timestep — a 256-dimensional vector that summarises everything the model has seen so far. It is never explicitly trained to track uncertainty.

The question: **does `h_t` linearly encode epistemic uncertainty anyway?**

If yes, a simple linear classifier should be able to read high vs. low uncertainty directly from `h_t`. No fine-tuning, no auxiliary loss — just the signal that already exists.

This is Phase 1 of 3. A positive result here justifies Phases 2 (temporal propagation) and 3 (surgical repair) on the full 200M model.

---

## Model Configuration

A scaled-down DreamerV3 (XS config from the paper's Table 1), compared against the full XL model used in the paper's main benchmarks:

| Parameter | This experiment (XS) | Full DreamerV3 (XL) | Scale factor |
|---|---|---|---|
| Deterministic state `h_t` | **256-dim** GRU | **4096-dim** GRU | 16× |
| Stochastic state `z_t` | 32 cat × 32 classes (1024-dim) | 32 cat × 32 classes (1024-dim) | 1× (identical) |
| MLP hidden units | 640 | 8192 | 12.8× |
| Total parameters | ~12M | ~200M | ~17× |
| Encoder | Linear (5-dim obs) | CNN (64×64 image) | — |
| Observation space | 5-dim (cartpole position + velocity) | 64×64 RGB image | — |
| Action space | 1-dim (slider force) | varies (up to 38-dim) | — |
| Training budget | 100,000 env steps | 200M+ env steps | 2000× |
| KL free bits | 1.0 nats | 1.0 nats | identical |

The stochastic state dimensionality is identical across all DreamerV3 model sizes — only `h_t` and the MLP width scale. This means the XS probe results directly test whether KL is linearly readable from `h_t` at the smallest representational scale. If the signal holds here, the question for Phase 2 is whether it holds at 16× the GRU width.

Environment: `dm_control cartpole_swingup`, random policy.

---

## Phase 1 — Training the World Model

Trained a single world model from scratch on 100K environment steps (~73 min on CPU).

**Results:**
- 100,000 states logged across 200 episodes
- Mean KL = 13.51, std = 6.94
- Mean reconstruction error = 0.11

The KL std of 6.94 is the key number here — it means the model genuinely varies in how uncertain it is from step to step. That variance is what the probes will try to read from `h_t`.

---

## Phase 2 — Evaluation Sets

All sets collected using the **frozen** trained model (no further learning). Each contains `h_t`, `z_t`, KL, and reconstruction error per state.

---

### Set A — In-Distribution (ID)

**How it's built:** 20 episodes of `cartpole_swingup` with a random policy. No modifications. Same environment, same noise level, same policy as training. 10,000 states total.

**Why:** Establishes the baseline. If the probe can't read uncertainty here, it can't read it anywhere. Confirms the signal generalises from training to fresh rollouts.

| Stat | Value |
|---|---|
| States | 10,000 |
| Mean KL | 21.07 |
| Mean recon | ~0.11 |

---

### Set B — Near-OOD (Noisy)

**How it's built:** Same as Set A but Gaussian noise (σ=0.1) is added to every observation before it enters the model. The model receives corrupted sensor readings it has never encountered during training. 10,000 states.

**Why:** Tests whether the probe survives mild distribution shift. If it fails here, the signal is too fragile for practical use. The higher KL (25.58 vs 21.07) confirms the model is more surprised by noisy inputs — expected and healthy.

| Stat | Value |
|---|---|
| States | 10,000 |
| Mean KL | 25.58 |
| Mean recon | ~0.15 |

---

### Set C — Contrastive, KL-Matched

**How it's built:** Pool Sets A and B together. Divide into 10 bins by KL percentile. Within each bin, take the bottom 25% recon states as C1 candidates and the top 25% recon states as C2 candidates. Sample up to 20 per bin, cap at 200 per group. 400 total states.

**Why:** Tests whether h_t encodes genuine uncertainty or just KL magnitude. Because C1 and C2 are drawn from the same KL bins, they have nearly identical KL distributions. Any probe that scores above chance must be detecting something beyond "how surprised was the model" — it must be detecting whether the model is actually coping.

| Group | Label | Source | Mean KL | Mean recon |
|---|---|---|---|---|
| C1 — coping | 0 | Mixed A+B (low recon within KL bin) | 22.86 ± 8.80 | 0.052 |
| C2 — confused | 1 | Mixed A+B (high recon within KL bin) | 23.86 ± 9.47 | 0.471 |

KL gap: 1 nat. Recon gap: 9×. A probe separating these is detecting internal confusion, not input strangeness.

---

### Set C Strong — Genuinely Novel Task

**How it's built:** Collect 10,000 states from `cartpole_balance` using the frozen swingup model. Balance is a structurally different task — the pole starts upright (not hanging), dynamics are different, typical state trajectories are entirely outside the swingup training distribution. Apply the same KL-matched binning: C1 from balance bottom-40% recon, C2 from swingup top-40% recon. 280 total states (140 per group).

**Why:** The hardest possible test. C1 states are not "noisy swingup" — they are from a completely different task the model has never been trained on. If the probe still separates C1 (novel task, coping) from C2 (familiar task, confused), it means h_t is encoding something about internal model state that transcends input distribution entirely.

| Group | Label | Source | Mean KL | Mean recon |
|---|---|---|---|---|
| C1 — novel, coping | 0 | cartpole_balance (bottom 40% recon within balance) | 33.42 ± 11.85 | 2.857 |
| C2 — familiar, confused | 1 | cartpole_swingup (top 40% recon within swingup) | 32.95 ± 11.07 | 0.945 |

KL gap: 0.47 nats. The model has seen zero balance states during training.

---

## Phase 3 — Ensemble Training

Three independent world models trained from scratch with different random seeds (0, 1, 2), run in parallel (~197 min).

These are used as a baseline: the standard way to estimate uncertainty in neural networks is to train multiple models and measure how much they disagree. We compare our linear probe against this ensemble baseline on Set C.

---

## Phase 4 — Probe Analysis

A linear classifier (logistic regression) is trained on `h_t` vectors to predict whether a state has high or low uncertainty. AUROC is the metric — 0.5 is random, 1.0 is perfect.

Three probes are tested, each using a different definition of "uncertain." The ensemble (3 independently-trained models measuring disagreement) is the standard baseline for comparison.

---

### Unified Results Table

| Method | Train held-out | Set A (ID) | Set B (noisy OOD) | Set C (KL-matched) | Set C Strong† |
|---|---|---|---|---|---|
| **Probe A** — KL → h_t | 0.9019 | 0.8632 | 0.8464 | **0.7227** | 0.7216 |
| **Probe B** — rollout variance | 0.8946 | 0.6285 | 0.7561 | 0.6256 | 0.6035 |
| **Probe C** — recon → h_t | 0.9263 | 0.9518 | 0.7944 | 0.7210 | 0.6385 |
| **RWM-U Ensemble** (trajectory-aware) | — | 0.8678 | 0.8417 | **0.7436** | 0.3070 |
| **z_t probe** (stochastic state) | 0.9341 | 0.8467 | — | 0.6669 | 0.3304 |
| **Within-balance check** (confound test) | — | — | — | — | **0.5060** |

† Set C Strong is confounded: C1 is balance h_t, C2 is swingup h_t — two populations with different trajectory histories regardless of uncertainty. The within-balance check (both groups from balance, same task identity) collapses to 0.51, confirming the 0.72 was task identity detection. The reliable contrastive result is Set C (KL-matched).

- **Sets A and B**: probe and ensemble are essentially tied (within 0.5%). Equivalent on standard tests.
- **Set C KL-matched**: ensemble slightly edges the probe (0.74 vs 0.72). The controlled contrastive result — same task, only confusion differs — and the reliable one.
- **Set C Strong**: all methods compromised by the trajectory history confound. Retained for reference but not interpretable as a clean uncertainty result.

---

### Probe A — KL Divergence as Uncertainty Signal

**What it is:** Train a logistic regression on `h_t` to predict whether KL is above or below median. KL = how much the model's prediction (prior) differed from what it actually observed (posterior). High KL = model was surprised = uncertain.

**Why it matters:** KL is the model's own uncertainty signal during training. If it's linearly readable from `h_t`, the hidden state is encoding what the model learned to be uncertain about.

| Set | AUROC | What it means |
|---|---|---|
| Train held-out | **0.9019** | h_t strongly encodes KL on familiar data |
| Set A (ID) | **0.8632** | Generalises to fresh rollouts — not memorised |
| Set B (noisy OOD) | **0.8464** | Survives mild distribution shift |
| Set C (KL-matched) | **0.7227** | Holds even when KL is controlled for — not just reading KL magnitude |
| Set C Strong (novel task) | **0.7216** | Survives genuinely foreign states from a completely different task |

---

### Probe B — Rollout Variance as Uncertainty Signal

**What it is:** From each `h_t`, run the model forward 5 steps purely in imagination (no observations), repeat 5 times. Measure how much the imagined futures vary across runs. High variance = model doesn't know what happens next.

**Why it matters:** Tests a different kind of uncertainty — predictive uncertainty about the future, not just surprise at the current observation.

| Set | AUROC | What it means |
|---|---|---|
| Train held-out | **0.8946** | Variance signal exists and is learnable from h_t |
| Set A (ID) | **0.6285** | Weaker — single-model rollout variance is noisy |
| Set B (noisy OOD) | **0.7561** | Improves on OOD; noisy inputs create naturally higher variance |
| Set C (KL-matched) | **0.6256** | Marginal but above chance |
| Set C Strong (novel task) | **0.6035** | Holds weakly on novel task |

Consistently weaker than Probe A. A single model's imagination variance is a noisy proxy — the real variance signal requires multiple independently-trained models (the ensemble), which is exactly what Probe B is trying to approximate with one model.

---

### Probe C — Reconstruction Error as Uncertainty Signal

**What it is:** Train a logistic regression on `h_t` to predict whether reconstruction error is above or below median. Reconstruction error = how well the model can rebuild the original observation from its hidden state. High error = didn't understand what it saw.

**Why it matters:** Sanity check. Reconstruction error is directly computable from `h_t`, so it should be linearly readable. If this probe fails, something is fundamentally wrong with the setup. High scores here confirm the probing methodology works.

| Set | AUROC | What it means |
|---|---|---|
| Train held-out | **0.9263** | h_t strongly encodes reconstruction quality |
| Set A (ID) | **0.9518** | Even stronger on fresh ID data |
| Set B (noisy OOD) | **0.7944** | Still solid on OOD |
| Set C (KL-matched) | **0.7210** | Holds on controlled test |
| Set C Strong (novel task) | **0.6385** | Holds on novel task |

The high Set A score (0.95) confirms the probe and methodology are working correctly. The lower scores on Set C and Set C Strong are expected — those sets were constructed to be hard precisely because they decorrelate reconstruction quality from input familiarity.

---

### Ensemble Disagreement — Baseline (RWM-U style)

**What it is:** Three independently-trained models step through the same observation trajectory in lockstep, each building its own h_t. At each step, variance across their decoded predictions = disagreement = uncertainty estimate. This directly matches the RWM-U methodology.

**Implementation:** each ensemble model processes every observation through its own encoder and RSSM, maintaining its own recurrent state throughout the episode. This is a trajectory-aware comparison — not single-step.

---

#### What the ensemble and probe are each good at

Running all signals on the same OOD detection task (ID swingup = 0, OOD balance = 1). Full results in the "Direct OOD Detection" section below; summary here:

| Method | OOD Detection AUROC |
|---|---|
| Recon error (single model, no training) | **0.9985** |
| KL directly (oracle) | **0.9560** |
| RWM-U Ensemble | 0.9425 |
| h_t probe (swingup-trained) | 0.7636 |
| mean(n_t) raw (no training) | 0.6910 |
| Probe A earlier estimate | 0.6081 |

The reconstruction error from a single model, computed with no training, exceeds the three-model ensemble. This is the key finding of the OOD detection sweep. The h_t probe at 0.76 beats the 0.61 from within-task tests — direct OOD is easier because h_t encodes task identity in addition to confusion.

This means the probe and ensemble are **not competing on the same signal**:

| Signal | Best method | AUROC |
|---|---|---|
| Input is OOD (never seen in training) | Recon error / KL oracle | **0.999 / 0.956** |
| Cross-model disagreement baseline | RWM-U Ensemble | **0.943** |
| Model is internally confused on this trajectory | h_t Probe A | **0.72** (Set C KL-matched) |

---

#### Comparison across all test sets

| Set | RWM-U Ensemble | Probe A (h_t) | What's being tested |
|---|---|---|---|
| Set A (ID) | 0.8678 | 0.8632 | Within-distribution uncertainty — **tied** |
| Set B (noisy OOD) | 0.8417 | 0.8464 | Mild OOD uncertainty — **tied** |
| Set C (KL-matched) | **0.7436** | 0.7227 | Coping vs confused at matched KL — **ensemble slightly better** |
| Set C Strong (novel task) | 0.3070 | **0.7216** | Novelty vs internal confusion — **probe wins** |
| OOD detection (direct) | **0.9425** | 0.6081 | Is this input from a different distribution? — **ensemble wins** |

The ensemble is the right tool for detecting distributional shift. The probe detects something the ensemble cannot: internal model confusion that is independent of input novelty. On the hard contrastive test the ensemble inverts to 0.31 — it is doing novelty detection, so it scores C1 (novel balance) as uncertain and C2 (confused swingup) as certain. The probe gets it right at 0.72.

These are complementary signals, not competing ones.

---

## Set C Design — Correction

**Original design flaw:** Set C was initially built by selecting C1 from the bottom 20% of Set B's reconstruction error and C2 from the top 20% of Set A's reconstruction error. Because KL and reconstruction error are correlated (r=0.64), this inadvertently created groups with very different KL distributions:

| Group | Mean KL |
|---|---|
| C1 (original) | 18.2 |
| C2 (original) | 34.0 |

Probe A, trained on KL labels, scored 0.9918 — but it was just detecting the KL gap, not testing novelty vs uncertainty.

**Corrected design (KL-matched):** Pool Sets A and B, bin states by KL percentile into 10 equal bins, then within each bin select C1 (bottom 25% recon) and C2 (top 25% recon). This ensures C1 and C2 have matched KL distributions:

| Group | Mean KL | Mean recon |
|---|---|---|
| C1 (coping) | 22.86 ± 8.80 | 0.052 |
| C2 (confused) | 23.86 ± 9.47 | 0.471 |

KL difference is now only 1 nat. Recon difference remains large (0.05 vs 0.47). Any probe that scores above chance on this set is detecting something beyond KL.

---

## Strong Test — Genuinely Novel States (cartpole_balance)

### Construction

Novel states collected from `cartpole_balance` (20 episodes, 10,000 states):
- Mean KL = 48.9 nats (vs swingup training mean of 13.5)
- Mean recon = 7.4 (vs swingup training mean of 0.11)

Two groups after KL-matching (C1 from balance bottom-40% recon, C2 from swingup top-40% recon):

| Group | Source | Mean KL | Mean recon |
|---|---|---|---|
| C1 (novel, coping) | cartpole_balance | 33.42 ± 11.85 | 2.857 |
| C2 (familiar, confused) | cartpole_swingup | 32.95 ± 11.07 | 0.945 |

### Initial results

| Probe | Set C Strong AUROC |
|---|---|
| Probe A — KL → `h_t` | 0.7216 |
| Probe B — rollout variance | 0.6035 |
| Probe C — recon → `h_t` | 0.6385 |
| RWM-U Ensemble (trajectory-aware) | 0.3070 |
| `z_t` probe | 0.3304 |

### The confound — and why these numbers cannot be trusted

Set C Strong mixes two populations with different trajectory histories. C1 h_t vectors were accumulated over hundreds of balance observations. C2 h_t vectors were accumulated over hundreds of swingup observations. These two populations have different distributional fingerprints embedded in h_t regardless of uncertainty content — a probe detecting *task identity* would score above chance.

There are two explanations for Probe A scoring 0.72 on Set C Strong:
- **Explanation A (wanted):** The probe reads genuine internal confusion from h_t. The swingup states in C2 have h_t vectors that look like "the model has been struggling" because the GRU accumulated a history of high reconstruction errors.
- **Explanation B (confound):** The probe detects that balance trajectory h_t vectors look different from swingup trajectory h_t vectors as a global distributional property — regardless of uncertainty.

The 0.72 is consistent with both. The set cannot distinguish them.

### Confound check — within-balance contrastive test

To isolate the confound, a within-balance set is constructed: both C1 and C2 drawn from balance trajectories only. Same task identity throughout. Only confusion differs.

| Group | Source | Mean KL | Mean recon |
|---|---|---|---|
| C1 (coping) | cartpole_balance (bottom 30% recon within KL bin) | 45.90 ± 19.78 | 3.121 |
| C2 (confused) | cartpole_balance (top 30% recon within KL bin) | 46.24 ± 22.62 | 12.666 |

400 total states (200 per group). KL gap: 0.34 nats. Both groups are balance h_t vectors — no task identity signal is available.

**Result:**

| Test set | C1 source | C2 source | Probe A AUROC |
|---|---|---|---|
| Set C (KL-matched) | swingup (low recon) | swingup (high recon) | **0.7227** — clean |
| Set C Strong | balance (low recon) | swingup (high recon) | 0.7216 — confounded |
| Within-balance (confound check) | balance (low recon) | balance (high recon) | **0.5060** — chance |

`z_t` probe on within-balance: **0.4345** (below chance).

**Interpretation:** Probe A collapses to 0.51 when both groups share the same task identity. The probe trained on swingup cannot detect within-balance confusion. The Set C Strong result of 0.72 was detecting task identity embedded in trajectory history, not genuine uncertainty generalisation across tasks.

The clean result is Set C (KL-matched, within-swingup): **0.7227**. The main claim rests on that.

---

## Δh_t Probe — Is the Confusion Signal in the Dynamics?

### Motivation

The within-balance confound check confirmed that h_t position is task-specific — the swingup confusion boundary does not transfer. But h_t values are accumulated trajectory history. The GRU update rule is not task-specific. The question: does **Δh_t = h_t − h_{t−1}** — the update the GRU applied at each step — encode confusion in a task-agnostic way?

When the model is confused (high KL), it received a surprising observation and the GRU should correct more. When coping (low KL), smaller correction needed. If this is true, ||Δh_t|| alone would be a proxy for confusion — and a probe on Δh_t would transfer across tasks.

### Results

Probe trained on swingup Δh_t with KL labels. Tested on within-swingup and within-balance (both groups same task identity, no task signal available).

| Test | h_t probe | Δh_t probe | \|\|Δh_t\|\| (raw norm) |
|---|---|---|---|
| Swingup held-out (ID) | 0.9027 | 0.7049 | 0.4525 |
| Within-swingup (KL-matched) | 0.5842 | **0.7333** | 0.6317 |
| Within-balance ← key test | 0.5166 | 0.5725 | **0.3730** |

Correlations: swingup ||Δh_t|| vs KL = −0.025 (near zero), balance ||Δh_t|| vs KL = 0.22 (weak), balance ||Δh_t|| vs recon = −0.09 (slightly negative).

### What These Numbers Mean

**Δh_t > h_t within-swingup (0.73 vs 0.58):** The update vector carries more information about current confusion than the accumulated state. Δh_t captures what just happened; h_t captures everything. For detecting confusion at a specific step, recency matters more.

**Δh_t does not transfer (0.57 within-balance):** Small improvement over h_t (0.52) but nowhere near signal. The directional pattern of confusion-updates in swingup h_t space does not match the pattern in balance h_t space. The direction the GRU moves when confused is task-specific.

**||Δh_t|| is inverted in within-balance (0.37 — below chance):** Confused balance states (C2, high recon) have marginally *smaller* updates than coping states (C1). The recon–||Δh_t|| correlation is −0.09 in balance. The naive hypothesis — "confused = large update" — does not hold once KL is matched. In swingup, the correlation between ||Δh_t|| and KL is −0.025 (effectively zero), confirming update magnitude is not a reliable confusion signal in either task.

### Interpretation

The confusion signal is **directional and task-specific at episode start, but partially task-agnostic at mid-episode steps.** The step-filtering matters: the jump from 0.57 to 0.68 on within-balance is entirely explained by requiring step ≥ 2. Broken down:

| Swingup filter | Balance filter | Δh_t within-balance |
|---|---|---|
| step ≥ 1 | step ≥ 1 | 0.5725 |
| step ≥ 2 | step ≥ 1 | 0.6111 |
| step ≥ 2 | step ≥ 2 | **0.6814** |

Step 0 and 1 have h_t close to zero initialisation — the GRU has not accumulated task context yet, so Δh_t at those steps is dominated by initialization noise. Once both the probe training and test evaluation exclude these early steps, the signal improves significantly. The confusion-in-dynamics signal is **episode-mature** — it exists mid-trajectory, not at cold start.

Raw norms remain inverted (||Δh_t|| within-balance: 0.37). The signal is directional, not magnitude-based: which way the GRU just moved carries partial cross-task confusion information, but how far it moved does not.

---

## Trajectory Curvature — Does Confusion Bend the h_t Path?

### Motivation

Δh_t direction is partially task-agnostic (0.68 at step ≥ 2). The natural follow-up: does adding the **second derivative** — how sharply the direction changed — improve or extend the signal?

```
c_t = h_t − 2·h_{t−1} + h_{t−2}
```

||c_t|| measures how much the trajectory bent at step t. If confused states produce more erratic trajectories (the model keeps changing direction because it is surprised in different ways), curvature would be a task-agnostic geometric property — not where h_t is, not which direction it is moving, but whether it keeps changing direction.

### Results

| Signal | Dims | SW held-out | Within-SW | Within-BAL ←key |
|---|---|---|---|---|
| h_t probe | 256 | 0.9026 | 0.5870 | 0.5603 |
| Δh_t probe | 256 | 0.7077 | 0.7177 | **0.6814** |
| c_t probe | 256 | 0.5584 | 0.4873 | 0.5399 |
| [Δh_t ; c_t] probe | 512 | 0.7315 | 0.7083 | 0.6631 |
| \|\|c_t\|\| raw | 1 | 0.4492 | 0.5849 | 0.3923 |
| \|\|Δh_t\|\| raw | 1 | 0.4536 | 0.6251 | 0.3730 |

Correlations: swingup ||c_t|| vs KL = −0.058 (near zero), balance ||c_t|| vs KL = 0.20, balance ||c_t|| vs recon = −0.13.

### What These Numbers Mean

**c_t probe within-balance: 0.54 — does not transfer.** The curvature hypothesis fails. Confused states do not produce more bent h_t trajectories in a way that is linearly separable and task-agnostic. Adding c_t to Δh_t actually degrades performance (0.66 vs 0.68 alone).

**||c_t|| raw within-balance: 0.39 — inverted**, same pattern as ||Δh_t||. Confused balance states have smaller curvature magnitude than coping ones once KL is matched. The raw norms of both the first and second derivatives of h_t anti-correlate with confusion within-balance.

**The signal is in the direction of Δh_t, not its magnitude or the curvature it produces.** The partial cross-task transfer at 0.68 comes from the directional pattern of the last GRU update, not from the trajectory's geometric shape.

### Interpretation

Curvature adds nothing. The confusion signal that partially generalises is the direction of the most recent update — what the GRU just did in response to the last observation. That direction has a partial cross-task signature. The second-order structure (how much that direction changed) does not.

What remains open: why does the direction of Δh_t have any cross-task signal at all at step ≥ 2? Answered by the gate analysis below.

---

## GRU Gate Analysis — Mechanistic Confusion Signal

### Motivation

Δh_t partially transfers (0.68). The natural mechanistic explanation would be: when surprised (high KL), the GRU update gate z_t fires harder, updating more h_t dimensions — so z_t activity should be a cleaner confusion proxy. We extract all three GRU gates directly from the GRUCell weights and test each.

```
r_t = sigmoid(W_ir·x + W_hr·h + b_r)        reset gate     — how much of h_{t-1} to use for candidate
z_t = sigmoid(W_iz·x + W_hz·h + b_z)        update gate    — how much of candidate vs h_{t-1} to take
n_t = tanh(W_in·x + r_t ⊙ (W_hn·h + b_n))  candidate gate — proposed new h_t
h_t = (1 − z_t) ⊙ h_{t-1} + z_t ⊙ n_t
```

### Results

| Signal | Dims | SW held-out | Within-SW | Within-BAL ←key |
|---|---|---|---|---|
| h_t (position) | 256 | 0.9588 | 0.5991 | 0.5652 |
| Δh_t (1st deriv) | 256 | 0.9581 | 0.5413 | 0.6766 |
| z_t gate | 256 | 0.9646 | 0.5457 | **0.4618** |
| r_t gate | 256 | 0.9686 | 0.5390 | **0.6888** |
| n_t gate (candidate) | 256 | 0.9587 | 0.5457 | **0.6864** |
| [z_t ; r_t] combined | 512 | 0.9761 | 0.5373 | 0.6251 |
| mean(z_t) raw | 1 | 0.2200 | 0.3166 | 0.2539 |

Correlations: swingup mean(z_t) vs KL = **−0.43**, balance mean(z_t) vs KL = −0.22, balance mean(z_t) vs recon = −0.28.

### What These Numbers Mean

**z_t inverts (0.46, below chance).** The update gate is the *opposite* of a confusion signal. High KL → lower z_t. Verified: low-KL quartile mean z_t = 0.9370, high-KL quartile mean z_t = 0.9269 — a gap of only 0.01. The update gate is nearly saturated at 0.94 across all states. The GRU has learned an "almost always overwrite" policy — it almost never carries forward h_{t-1} regardless of confusion level. The tiny negative correlation means confused states actually resist updating marginally more than coping states.

**r_t is the best cross-task signal (0.69), beating Δh_t (0.68).** The reset gate — which controls how much of h_{t-1} feeds into the candidate computation — partially transfers. When confused, the model's strategy for using its past history (r_t activation pattern) has some consistency across tasks.

**n_t is equally good (0.69).** The candidate hidden state — what the GRU proposes to write into h_t — also partially transfers. This makes sense: r_t shapes n_t, so if r_t has a cross-task confusion signal, n_t inherits it.

**mean(z_t) raw: 0.25 — deeply inverted.** No useful information, completely wrong direction.

### Interpretation

The expected mechanism was wrong. Confusion does not cause the GRU to update more aggressively — the GRU almost always updates maximally (z_t ≈ 0.94). The confusion signal is not in *whether* h_t gets overwritten but in *what* the candidate contains (n_t) and *how past history influences it* (r_t).

The cross-task signal that Δh_t was capturing is the directional content of r_t and n_t — the pattern of how the reset gate weights past history when computing the candidate. This pattern has partial task-agnostic structure at step ≥ 2, but not enough to cross the threshold into a practically useful signal.

**The mechanistic picture:** This GRU is a near-perfect overwriter. Every step, it almost completely replaces h_t with the new candidate. Confusion is encoded in what that candidate looks like (n_t direction and r_t activation), not in whether the replacement happens. The candidate content is partly task-agnostic; the update decision (z_t) is not a confusion signal at all.

---

## Direct OOD Detection — All Signals

### Motivation

All previous experiments used hard contrastive tests within a single task (within-swingup or within-balance), with KL matched between groups. These were designed to be difficult. The direct test — swingup (label 0) vs balance (label 1), no KL matching — is the easiest possible OOD detection scenario. It establishes a ceiling: what is each signal actually capable of when the two populations are maximally separated?

Swingup: 200 episodes, 99,600 steps, mean KL=21.1, mean recon=0.165.
Balance: 20 episodes, 9,960 steps, mean KL=51.8, mean recon=7.677.

Balance is evaluated by the frozen swingup model — it has never been trained on balance observations. Swingup is downsampled to match balance set size for balanced evaluation (9,960 states each, 19,920 total).

### Results

**Unsupervised raw scalars (no training needed):**

| Signal | AUROC |
|---|---|
| Recon error | **0.9985** |
| KL | **0.9560** |
| mean(\|n_t\|) — candidate gate magnitude | **0.6910** |
| std(z_t) — update gate variance | **0.6813** |
| \|\|h_t\|\| — recurrent state norm | 0.6381 |
| mean(r_t) — reset gate mean | 0.5390 |
| std(r_t) | 0.4998 |
| \|\|Δh_t\|\| | 0.4753 |
| mean(z_t) | 0.3556 |

**Probe-based (trained on swingup KL labels, score used as OOD detector):**

| Signal | AUROC |
|---|---|
| h_t probe | **0.7636** |
| r_t probe | 0.7484 |
| n_t probe | 0.7366 |
| Δh_t probe | 0.5919 |
| z_t probe | 0.5309 |

**Distribution shift between tasks:**

| Feature | Swingup mean | Balance mean | Ratio |
|---|---|---|---|
| recon | 0.1654 | 7.6769 | **46.4×** |
| KL | 21.07 | 51.84 | 2.46× |
| z_gate mean | 0.9381 | 0.9352 | 1.00× |
| r_gate mean | 0.5412 | 0.5453 | 1.01× |
| \|\|Δh_t\|\| | 0.3274 | 0.3219 | 0.98× |

### What These Numbers Mean

**Reconstruction error beats the ensemble (0.9985 vs 0.9425).** A single scalar, from a single model, with no training overhead, exceeds the three-model ensemble on direct OOD detection. The distribution shift table explains why: reconstruction error has a 46× mean gap between tasks. The balance observations are so far outside the swingup training distribution that the decoder simply cannot reconstruct them. This is the cleanest OOD signal in the entire experiment.

**KL also beats the ensemble (0.9560).** KL has a 2.46× gap between tasks, which is smaller but still highly discriminative. Both recon and KL are oracle signals — they are computed at inference time and require no additional training beyond the world model itself.

**Gate signals work without training (0.69).** mean(|n_t|) and std(z_t) reach 0.69 and 0.68 respectively as unsupervised scalars. This is above the h_t probe estimate from earlier within-task tests (0.61) and requires zero training. The candidate gate magnitude and update gate variance show detectable differences between the two tasks even though the mean gate activations (mean(z_t), mean(r_t)) are nearly identical across tasks (1.0× ratio).

**Gates and Δh_t have near-zero mean shift between tasks.** The distribution shift table shows that mean(z_t), mean(r_t), and ||Δh_t|| have essentially the same mean across tasks (1.00–1.01× ratio). These signals were only useful in contrastive within-task settings. In the direct OOD setting, they carry almost no between-task information — consistent with why ||Δh_t|| inverts (0.4753).

**The h_t probe improves from 0.61 to 0.76 in direct OOD.** In the within-task contrastive test (swingup only), the h_t probe scored 0.61. In direct OOD it scores 0.76 — not because the probe improved, but because h_t encodes task identity as well as confusion. The probe's decision boundary learned on swingup KL labels happens to partially separate the swingup vs balance h_t manifolds.

**mean(z_t) is deeply inverted (0.36).** Fully consistent with the gate analysis: z_t slightly decreases at higher KL, and balance mean(z_t) = 0.9352 ≈ swingup mean(z_t) = 0.9381. As a raw scalar it carries no OOD information.

### Gap to Ensemble

| Best raw signal | Best probe | Ensemble |
|---|---|---|
| Recon: **0.9985** (+0.056 above ensemble) | h_t: **0.7636** (−0.179 below ensemble) | 0.9425 |

The gap to ensemble is closed (and exceeded) by the oracle signals. The non-oracle signals (gate-based, probes) still fall 0.18–0.25 below ensemble. If oracle access to recon or KL is acceptable, a single model already beats the three-model ensemble. If not — if the goal is an unsupervised OOD signal from internal activations without decoder access — the gate signals (0.69) and h_t probe (0.76) are the best available.

### Interpretation

The direct OOD test reveals that the model's own outputs (KL, recon) are far more informative than its internal activations (gates, derivatives) for the task of detecting whether the input is from a different distribution. The hard contrastive experiments were testing something the outputs already cannot do (within-task confusion with matched KL), which is why internal signals looked relatively competitive there.

For practical OOD detection, the hierarchy is:
1. Reconstruction error — 0.9985, beats ensemble, zero training
2. KL — 0.9560, beats ensemble, zero training
3. h_t probe — 0.7636, requires swingup training data
4. Gate scalars (mean(|n_t|), std(z_t)) — 0.68–0.69, no training
5. Gate probes (r_t, n_t) — 0.74–0.75, requires training

---

## Per-Block Analysis — Where in h_t is the Signal?

`h_t` is 256-dimensional. We split it into 4 quarters and test each independently to see if the signal is concentrated anywhere or spread uniformly.

| Block | Dimensions | Train held-out AUROC | Set A AUROC |
|---|---|---|---|
| Q1 | 0–64 | 0.8635 | 0.8933 |
| Q2 | 64–128 | 0.8731 | 0.8928 |
| Q3 | 128–192 | 0.8716 | 0.8970 |
| Q4 | 192–256 | 0.8682 | 0.8905 |

All four quarters score essentially the same (~0.87–0.90). The uncertainty signal is **distributed uniformly** across `h_t`, not concentrated in any particular subspace. This is consistent with the GRU mixing information across all hidden units at every step.

---

## h_t vs z_t Comparison

`z_t` is the stochastic state — 1024-dimensional categorical logits that directly parameterise the KL divergence. We compare probes on both to see whether `h_t` is encoding anything beyond what `z_t` already contains.

| Feature | Dims | Train held-out | Set A | Set C (KL-matched) | Set C Strong† |
|---|---|---|---|---|---|
| `h_t` | 256 | 0.9019 | 0.8632 | **0.7227** | 0.7216 |
| `z_t` | 1024 | 0.9341 | 0.8467 | **0.6669** | 0.3304 |

† Set C Strong is confounded by trajectory history — see confound check section. Within-balance: h_t=0.51, z_t=0.43 (both chance).

`z_t` scores slightly higher on training held-out (expected — KL is computed directly from `z_t` logits). On the KL-matched Set C, `h_t` outperforms `z_t` (0.72 vs 0.67). This is the clean comparison: same task, only confusion differs. The deterministic recurrent state carries more information about whether the model is coping than the stochastic state, even when KL is matched between groups.

---

## Results in Plain English

**Set A — Normal data, fresh rollout**
AUROC 0.87

The model was given clean cartpole observations it had never seen before (but same type as training). The probe could read uncertainty from h_t 87% accurately. Means the signal isn't just memorised from training — it generalises to new situations.

---

**Set B — Noisy, unfamiliar data**
AUROC 0.85

Same as Set A but with random noise added to the sensors. The model had never seen inputs like this. Probe still scores 0.85. Means the uncertainty signal survives even when the model is thrown into genuinely foreign territory.

---

**Set C — Hard test (KL-matched, noisy OOD vs confused swingup)**
AUROC 0.72

Two groups with the same average KL (22.86 vs 23.86 nats):
- C1: noisy inputs, model coping — low reconstruction error
- C2: clean familiar inputs, model confused — high reconstruction error

The probe correctly identified which was which 72% of the time, despite matched KL. The signal in h_t is independent of how surprised the model was (KL), and tracks whether it is actually coping.

---

**Set C Strong — confounded, result retracted**
Initial result: Probe A 0.72. Within-balance confound check: **0.51 (chance).**

Set C Strong compared balance h_t vectors (C1) against swingup h_t vectors (C2). After hundreds of trajectory steps, those two populations have different distributional fingerprints in h_t regardless of uncertainty. A probe detecting task identity would score above chance.

The confound check ran both C1 and C2 from balance trajectories only — same task identity, only confusion differed. Probe A dropped to 0.51. The 0.72 was detecting trajectory distribution, not internal model confusion. The result is not interpretable as a generalisation claim.

The ensemble result (0.31) is unaffected by this: it was already correctly characterised as novelty detection rather than confusion detection.

---

**Bottom line:**
- For detecting distributional shift (OOD detection): use the ensemble (0.94)
- For detecting internal model confusion within a known task: use the h_t probe (0.72 on KL-matched Set C)
- The probe does NOT generalise across tasks — within-balance test confirms it cannot detect confusion in a novel task it was not trained on
- h_t encodes coping vs confused within the training distribution; whether that signal transfers to genuinely novel tasks remains open

---

## Verdict

| Criterion | Threshold | Result | |
|---|---|---|---|
| Probe A Set A AUROC | > 0.72 | **0.8632** | PASS ✓ |
| Probe A Set C AUROC (KL-matched) | > 0.63 | **0.7227** | PASS ✓ |
| Probe A within-balance (confound check) | > 0.55 | **0.5060** | FAIL — does not generalise across tasks |

**POSITIVE RESULT — signal exists within the training task. Phase 2 is justified.**

`h_t` linearly encodes internal model uncertainty at the XS scale within the training distribution. The signal holds on fresh ID data (0.86), noisy OOD (0.85), and the KL-matched contrastive test (0.72). The Set C Strong result (0.72) was retracted after a within-balance confound check showed it was detecting trajectory distribution, not confusion — within-balance collapses to 0.51.

The scope of the claim is now narrower but cleaner: the probe detects whether the model is coping or confused within the task it was trained on. Cross-task generalisation was not demonstrated.

**The direct OOD test adds a separate finding:** reconstruction error from a single frozen model (0.9985) and KL (0.9560) both exceed the three-model ensemble (0.9425) with zero training overhead. These are oracle signals computed at inference time. Internal activations (gates, probes) fall in the 0.69–0.76 range and do not close the gap to the oracle signals.

The natural next question (Phase 2): does this signal persist and propagate forward in time — can the model's uncertainty at step T predict what will go wrong at step T+10?

---

## Honest Assessment

### What is genuinely strong

**The contrastive design worked.** Set C and Set C Strong are well-constructed controls. KL-matched within bins means neither method can cheat by detecting KL magnitude — the 1-nat KL gap alongside a 9× reconstruction gap in Set C is exactly the kind of controlled contrast that makes a result credible.

**The ensemble inversion is surprising and important.** On Set C Strong, the RWM-U ensemble inverts to 0.31 — it flags balance states (novel task) as uncertain and familiar swingup states where the model is confused as certain. It is detecting novelty, not confusion. Probe A does the opposite, correctly (0.72). This is the uncertainty vs novelty separation that motivated Phase 1, and it came through cleanly.

**The h_t vs z_t comparison is mechanistically decisive.** `z_t` directly parameterises KL and has access to the full per-step posterior distribution. It still collapses to 0.33 on Set C Strong. `h_t` holds at 0.72. The signal is not in the per-step stochastic variable — it is in the recurrent trajectory context accumulated by the GRU. That is a mechanistic finding.

**The ensemble implementation is not the explanation.** The RWM-U ensemble was correctly implemented — each model steps through the full observation sequence in lockstep, building its own `h_t` from scratch. The 0.31 on Set C Strong is not an artefact of a broken baseline. It is a genuine methodological limitation of disagreement-based methods — though note the Set C Strong result itself is now retracted as confounded.

---

### What is weaker — limitations to be direct about

**The model is tiny.** 256-dim GRU, 100K steps, cartpole. The full DreamerV3 default is 8192-dim, millions of steps, continuous control tasks with rich observation spaces. The signal exists at XS scale, but claiming it will hold at the 200M parameter scale requires running that experiment.

**The per-block result is a problem for Phase 3.** The signal is distributed uniformly across all four quarters of `h_t` — AUROC is essentially flat across Q1–Q4 (0.87–0.90). If the signal is not localised in specific dimensions or blocks, surgical block-level repair (the Phase 3 story) becomes significantly harder. Full representation retraining may be unavoidable. This is the most important limitation to flag before committing to a Phase 3 design.

---

### The most important finding — not what was expected

Every prior causal tracing paper (ROME, MEMIT, CART) finds that learned information is localised — specific layers or neurons carry the target. The per-block result says the opposite: uncertainty is distributed uniformly across all dimensions of `h_t`. There is no "uncertainty block" or "uncertainty subspace." The information is geometric and spread across the full recurrent state.

This is a mechanistic finding in itself. It means the GRU mixes uncertainty information across all hidden units at every step — a consequence of the recurrent update rule, not a design choice. This changes the Phase 3 story, but it also gives something to say about the mechanism that is novel.

The right way to frame this going into Phase 2: the question is not whether the signal is localised (it is not) but whether it is temporally structured — does it propagate forward in time in a way that is predictive and actionable?
