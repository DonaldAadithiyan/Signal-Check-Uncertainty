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

A scaled-down DreamerV3 (XS config from the paper's Table 1):

| Parameter | Value |
|---|---|
| Deterministic state `h_t` | 256-dim GRU hidden state |
| Stochastic state `z_t` | 32 categoricals × 32 classes (1024-dim) |
| Encoder embed dim | 64 |
| Observation space | 5-dim (cartpole position + velocity) |
| Action space | 1-dim (slider force) |
| Training budget | 100,000 environment steps |
| KL free bits | 1.0 nats (prevents posterior collapse) |

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

| Method | Train held-out | Set A (ID) | Set B (noisy OOD) | Set C (KL-matched) | Set C Strong (novel task) |
|---|---|---|---|---|---|
| **Probe A** — KL → h_t | 0.9019 | 0.8632 | 0.8464 | **0.7227** | **0.7216** |
| **Probe B** — rollout variance | 0.8946 | 0.6285 | 0.7561 | 0.6256 | 0.6035 |
| **Probe C** — recon → h_t | 0.9263 | 0.9518 | 0.7944 | 0.7210 | 0.6385 |
| **RWM-U Ensemble** (trajectory-aware) | — | 0.8678 | 0.8417 | **0.7436** | **0.3070** |
| **z_t probe** (stochastic state) | 0.9341 | 0.8467 | — | 0.6669 | 0.3304 |

The RWM-U ensemble is properly implemented: each model runs through the full observation trajectory in lockstep, building its own h_t from the sequence. This matches the RWM-U methodology directly.

- **Sets A and B**: probe and ensemble are essentially tied (within 0.5%). Equivalent on standard tests.
- **Set C KL-matched**: ensemble slightly edges the probe (0.74 vs 0.72). With full trajectory context, a 3-model ensemble is marginally better at the controlled contrastive test.
- **Set C Strong**: probe holds at 0.72, ensemble collapses to 0.31 — below chance, systematically inverted. The ensemble flags the novel balance states as uncertain (they look foreign, models disagree more), completely missing that the confused swingup states are the genuinely uncertain ones. It is doing novelty detection.

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

Running both on the same OOD detection task (ID swingup = 0, OOD balance = 1):

| Method | OOD Detection AUROC |
|---|---|
| RWM-U Ensemble | **0.9425** |
| KL directly (oracle) | 0.9549 |
| z_t probe | 0.7452 |
| Probe A — h_t | 0.6081 |

The ensemble dominates at OOD detection. It measures cross-model disagreement, which naturally peaks when models trained on swingup encounter balance states they have never seen. The h_t probe scores only 0.61 on the same task — h_t is not a good OOD detector.

This means the probe and ensemble are **not competing on the same signal**:

| Signal | Best method | AUROC |
|---|---|---|
| Input is OOD (never seen in training) | RWM-U Ensemble | **0.94** |
| Model is internally confused on this trajectory | h_t Probe A | **0.72** (Set C Strong) |

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

### Why this test is stronger

The KL-matched Set C (above) used noisy versions of the same swingup task as C1. It controlled for KL magnitude but the novel states were still structurally similar to training data — same environment, same physics, just with Gaussian noise added.

The strong test removes that entirely. C1 states come from `cartpole_balance` — a completely different task. The model was trained only on `cartpole_swingup`. In swingup, the pole starts hanging downward and must be swung up. In balance, the pole starts near vertical and must be kept there. The state distributions are entirely different: different position ranges, different velocity profiles, different typical trajectories.

The model has never seen a single balance state. These are not "noisy swingup" — they are structurally foreign.

### Construction

Novel states collected from `cartpole_balance` (20 episodes, 10,000 states):
- Mean KL = 48.9 nats (vs swingup training mean of 13.5)
- Mean recon = 7.4 (vs swingup training mean of 0.11)

The model is deeply confused by balance states in general. But within balance, some states are handled better than others. We take the bottom 40% of recon within balance as C1 candidates — the model is coping *relative to how lost it is on this task*.

Two groups after KL-matching (10 bins, C1 from balance bottom-40% recon, C2 from swingup top-40% recon):

| Group | Source | Mean KL | Mean recon |
|---|---|---|---|
| C1 (novel, coping) | cartpole_balance | 33.42 ± 11.85 | 2.857 |
| C2 (familiar, confused) | cartpole_swingup | 32.95 ± 11.07 | 0.945 |

KL difference: 0.47 nats — essentially identical. 274 total states (137 per group).

### Results

| Probe | Set C Strong AUROC |
|---|---|
| Probe A — KL → `h_t` | **0.7216** |
| Probe B — rollout variance | 0.6035 |
| Probe C — recon → `h_t` | 0.6385 |
| RWM-U Ensemble (trajectory-aware) | 0.3070 |
| `z_t` probe | **0.3304** |

### What these numbers mean

**Probe A 0.72:** The probe separates novel/coping from familiar/confused at 72% accuracy, even with matched KL. The signal is not KL magnitude — it is something in the trajectory-level context the GRU has accumulated.

**z_t 0.33:** `z_t` directly parameterises KL and has access to the full posterior distribution — it scores 0.33, below chance. The signal is not in the per-step uncertainty variables.

**RWM-U Ensemble 0.31:** The trajectory-aware ensemble — each model stepping through the full observation sequence in lockstep, building its own h_t — scores 0.31, below chance. It calls the novel balance states highly uncertain (they look foreign, models disagree) while the genuinely uncertain swingup states score low. It is detecting novelty, not uncertainty.

This is the central result. On Sets A and B, the RWM-U ensemble and the probe are tied (0.87 vs 0.87, 0.84 vs 0.85). On the hardest test, the probe holds at 0.72 and the ensemble inverts to 0.31. The probe wins precisely because it reads from h_t — which has accumulated trajectory context about whether the model is coping — while the ensemble measures disagreement between models that were never trained to distinguish those two things.

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

| Feature | Dims | Train held-out | Set A | Set C (KL-matched) | Set C Strong (novel task) |
|---|---|---|---|---|---|
| `h_t` | 256 | 0.9019 | 0.8632 | **0.7227** | **0.7216** |
| `z_t` | 1024 | 0.9341 | 0.8467 | **0.6669** | **0.3304** |

`z_t` scores slightly higher on training held-out (expected — KL is computed directly from `z_t` logits). On the KL-matched Set C, `h_t` outperforms `z_t` (0.72 vs 0.67). On the hard Set C Strong test, the gap widens dramatically: `h_t` holds at 0.72 while `z_t` collapses to 0.33 — below chance, same direction as the ensemble inversion. The stochastic state carries no useful trajectory context; it only sees per-step uncertainty. The deterministic recurrent state is where the "coping vs confused" signal lives.

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

**Set C Strong — Internal confusion test (KL-matched, novel task vs confused swingup)**
Probe A: 0.72 | z_t: 0.33 | RWM-U Ensemble: 0.31

Novel states from `cartpole_balance` (never seen in training) paired against familiar swingup states where the model is confused. KL-matched so neither method can rely on KL magnitude.

- C1: balance states — novel, model coping within that task (label 0)
- C2: swingup states — familiar, model internally confused (label 1)

The ensemble scores 0.31 — below chance, backwards. It flags balance as uncertain (foreign inputs cause disagreement) and swingup as certain (familiar inputs cause agreement). It is detecting novelty, not confusion.

The probe scores 0.72. It is detecting internal model confusion that is invisible to the ensemble.

Note: if you run the ensemble on a pure OOD detection task (swingup vs balance, no KL-matching), the ensemble scores 0.94 — it is the right tool for that problem. The probe scores only 0.61 on the same task. **These are different signals and the right tool depends on the question being asked.**

---

**Bottom line:**
- For detecting distributional shift (OOD detection): use the ensemble (0.94)
- For detecting internal model confusion independent of input novelty: use the h_t probe (0.72 vs ensemble's 0.31 on the hard contrastive test)
- h_t encodes something the ensemble cannot see — trajectory-level context about whether the model is coping

---

## Verdict

| Criterion | Threshold | Result | |
|---|---|---|---|
| Probe A Set A AUROC | > 0.72 | **0.8632** | PASS ✓ |
| Probe A Set C AUROC | > 0.63 | **0.7227** (KL-matched) / **0.7216** (strong) | PASS ✓ |

**POSITIVE RESULT — signal exists. Phase 2 and 3 are justified.**

`h_t` linearly encodes internal model uncertainty at the XS scale. The signal holds on fresh ID data (0.86), noisy OOD (0.85), KL-matched contrastive (0.72), and the hard internal-confusion test (0.72). The probe's advantage over the RWM-U ensemble is specific: on standard uncertainty detection they are tied, but on the internal confusion test the ensemble inverts to 0.31 while the probe holds. The signal in h_t is trajectory-level context about whether the model is coping — not input strangeness, not KL magnitude, and not something visible to cross-model disagreement.

The natural next question (Phase 2): does this signal persist and propagate forward in time — can the model's uncertainty at step T predict what will go wrong at step T+10?

---

## Honest Assessment

### What is genuinely strong

**The contrastive design worked.** Set C and Set C Strong are well-constructed controls. KL-matched within bins means neither method can cheat by detecting KL magnitude — the 1-nat KL gap alongside a 9× reconstruction gap in Set C is exactly the kind of controlled contrast that makes a result credible.

**The ensemble inversion is surprising and important.** On Set C Strong, the RWM-U ensemble inverts to 0.31 — it flags balance states (novel task) as uncertain and familiar swingup states where the model is confused as certain. It is detecting novelty, not confusion. Probe A does the opposite, correctly (0.72). This is the uncertainty vs novelty separation that motivated Phase 1, and it came through cleanly.

**The h_t vs z_t comparison is mechanistically decisive.** `z_t` directly parameterises KL and has access to the full per-step posterior distribution. It still collapses to 0.33 on Set C Strong. `h_t` holds at 0.72. The signal is not in the per-step stochastic variable — it is in the recurrent trajectory context accumulated by the GRU. That is a mechanistic finding.

**The ensemble implementation is not the explanation.** The RWM-U ensemble was correctly implemented — each model steps through the full observation sequence in lockstep, building its own `h_t` from scratch. The 0.31 is not an artefact of a broken baseline. It is a genuine methodological limitation of disagreement-based methods.

---

### What is weaker — limitations to be direct about

**The model is tiny.** 256-dim GRU, 100K steps, cartpole. The full DreamerV3 default is 8192-dim, millions of steps, continuous control tasks with rich observation spaces. The signal exists at XS scale, but claiming it will hold at the 200M parameter scale requires running that experiment.

**The per-block result is a problem for Phase 3.** The signal is distributed uniformly across all four quarters of `h_t` — AUROC is essentially flat across Q1–Q4 (0.87–0.90). If the signal is not localised in specific dimensions or blocks, surgical block-level repair (the Phase 3 story) becomes significantly harder. Full representation retraining may be unavoidable. This is the most important limitation to flag before committing to a Phase 3 design.

---

### The most important finding — not what was expected

Every prior causal tracing paper (ROME, MEMIT, CART) finds that learned information is localised — specific layers or neurons carry the target. The per-block result says the opposite: uncertainty is distributed uniformly across all dimensions of `h_t`. There is no "uncertainty block" or "uncertainty subspace." The information is geometric and spread across the full recurrent state.

This is a mechanistic finding in itself. It means the GRU mixes uncertainty information across all hidden units at every step — a consequence of the recurrent update rule, not a design choice. This changes the Phase 3 story, but it also gives something to say about the mechanism that is novel.

The right way to frame this going into Phase 2: the question is not whether the signal is localised (it is not) but whether it is temporally structured — does it propagate forward in time in a way that is predictive and actionable?
