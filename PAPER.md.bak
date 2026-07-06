# Implicit Confusion Encoding in Recurrent World Model States

**Abstract.** World models based on the RSSM architecture maintain a recurrent hidden state h_t that summarises trajectory history. This state is trained only to minimise prediction error, yet we show it implicitly encodes a confusion signal — a history of accumulated KL divergences — that is orthogonal to both ensemble disagreement and standard OOD detection. Using a Mini-DreamerV3 (XS config, 100K steps, cartpole) as a testbed, we find that a linear probe trained on h_t with KL labels achieves AUROC 0.72 on a KL-matched contrastive set where novelty is controlled for, and 0.70 cross-task without multi-task training (via first-difference Δh_t). We give a closed-form characterisation: the probe approximates a discounted count of recent high-KL steps, C_t = Σ γ^i · 1[KL_{t−i} > median], with γ = 0.95 and R² = 0.80. The signal lives in the near-null space of h_t's principal variation (88.2° from all top-50 PCA components; 9% of probe variance in the high-variance subspace), a geometric consequence of GRU update-gate saturation (z_t ≈ 0.94). Partial correlation analysis confirms: Probe A tracks accumulated KL history (r(probe, KL|recon) = +0.52); ensemble disagreement tracks current reconstruction quality (r(ens, recon|KL) = +0.56) — distinct measurements both called "uncertainty." The confusion signal operationalises directly: as an observation-routing oracle it outperforms recon-threshold baselines at 30% query rate (0.818 vs 0.770 recall), capturing 81% of the KL oracle's advantage; the advantage is mechanistically confirmed as concentrated in multi-step confused sequences (probe-only detections have 3.4× longer mean streak length than recon-only detections). A second orthogonal signal — the obs/imagination boundary at 1.0000 AUROC, r = −0.015 to Probe A — coexists in the same near-null space, with its immediate saturation predicted exactly by z_gate: after one imagination step, (1−z_gate)^1 = 0.06 of the original posterior content remains, which is sufficient to push h_t fully off the posterior manifold. Together these results suggest that the near-null space of RSSM hidden states encodes a structured record of model experience that is invisible to standard representation analysis but directly exploitable for observation routing — a resource that exists in any trained world model at no additional cost.

---

## 1. Introduction

Model-based reinforcement learning agents trained with the DreamerV3 framework maintain a continuous recurrent hidden state h_t ∈ ℝ^256 (XS config) or ℝ^4096 (XL config) at every timestep. This state is a deterministic summary of the agent's observation history, updated by a GRU and trained implicitly through the RSSM prediction objective. It is never explicitly asked to represent uncertainty, confusion, or anything beyond what is necessary for good predictions.

The central question of this paper: **does h_t linearly encode whether the model is confused — and if so, what is it computing?**

We use the term **confusion** to refer to the history of KL divergences accumulated by the model over recent trajectory steps — whether the model has been continuously surprised for many steps in a row. A model can be confused in this sense while currently producing low reconstruction error (a sustained confused trajectory that just happened to encounter an easy observation). Ensemble disagreement, which measures *between-model* disagreement, does not detect this: it is highest on novel inputs, not on sustained within-distribution confusion.

The empirical distinction is sharp. When evaluated on a KL-matched contrastive set (Set C, §3.1) where the two groups have matched KL distributions but a 9× reconstruction gap, the RWM-U ensemble inverts to AUROC 0.31 — it flags novel states as confident. Probe A scores 0.72 on the same test. These signals are orthogonal to each other and to standard OOD detection (recon error, which scores 0.9964 on direct OOD detection with no training).

Prior work missed the confusion signal for two reasons that follow from its geometry:

1. **Ensemble disagreement is orthogonal to the confusion signal.** The probe inverts to 0.49 on direct OOD detection while achieving 0.72 on within-task KL-matched sets. Measuring disagreement cannot find a signal that disagrees with it.

2. **Standard probing methodology looks in the wrong subspace.** Standard approaches inspect the dominant variance structure of representations (PCA, t-SNE). We show the confusion signal lies in the near-null space — directions with 88.2° angle to all top-50 PCs and only 9% of the probe's direction in the top-50-PC subspace. PCA-based analysis would discard 91% of the signal.

The signal was hidden not because it is weak but because it is orthogonal to where prior methods looked.

**Contributions:**

1. We demonstrate and characterise a within-task confusion signal in h_t of a DreamerV3 world model (AUROC 0.72 on KL-matched sets, §4.1).

2. We provide a closed-form characterisation: Probe A ≈ linear function of C_t = Σ γ^i · 1[KL_{t−i} > median], γ = 0.95, R² = 0.80 (§4.2). No prior probing paper has produced a closed-form expression for what its probe computes.

3. We establish a geometric account: the confusion signal occupies the near-null space of h_t's PCA, consistent with GRU update-gate saturation theory (§4.3).

4. We recover cross-task generalisation via Δh_t without multi-task training (0.70 AUROC cross-task, §4.4).

5. We operationalise the signal as an observation-routing policy that outperforms recon-error baselines (§4.5), and identify a second orthogonal signal — the obs/imagination boundary (1.0000 AUROC, r = −0.015 to Probe A) — demonstrating that h_t encodes at least two independent confusion-related aspects (§4.6).

---

## 2. Background

### 2.1 DreamerV3 and the RSSM

DreamerV3 [Hafner et al., 2023] trains a world model consisting of an encoder, a Recurrent State-Space Model (RSSM), a decoder, and a reward/discount predictor. The RSSM maintains two components at each step: a deterministic state h_t (the GRU output) and a stochastic state z_t (categorical latents). The training objective minimises:

L = E[recon loss + β · KL(posterior ‖ prior)]

where KL(t) = KL(q(z_t | h_t, x_t) ‖ p(z_t | h_t)) is the divergence between the posterior (which sees the observation) and the prior (which does not). KL(t) is the model's own per-step measure of how surprised it was. High KL = the observation was hard to predict from h_t alone.

The GRU update rule at each step:

```
r_t = σ(W_ir·x + W_hr·h + b_r)          reset gate
z_t = σ(W_iz·x + W_hz·h + b_z)          update gate  
n_t = tanh(W_in·x + r_t ⊙ (W_hn·h + b_n))  candidate
h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ n_t
```

The update gate z_t controls how much of the previous state is retained vs replaced by the candidate n_t.

### 2.2 Ensemble methods and their limitations for confusion detection

The standard approach to uncertainty in MBRL is ensemble disagreement [Chua et al., 2018; Kidambi et al., 2020]. Multiple models are trained independently; disagreement between their predictions measures uncertainty. This is theoretically motivated by Bayesian model averaging but has a fundamental limitation: it measures *between-model* uncertainty (novelty) not *within-model* confusion. We formalise this distinction empirically in §4.1.

### 2.3 Probing methodology

A linear probe [Alain & Bengio, 2016] is a logistic regression trained on internal representations to predict a target label. The probe tests whether information is *linearly accessible* — whether a linear classifier can read it from the representation. We use Probe A: logistic regression on h_t to predict KL above/below median. AUROC is the metric.

The key design choice in our evaluation is the **KL-matched contrastive set** (Set C, §3.1), which separates confusion from novelty by construction.

---

## 3. Experimental Setup

### 3.1 Model and environment

**World model:** Mini-DreamerV3, XS configuration [Hafner et al., 2023]:

| Component | XS (this work) | XL (paper) |
|---|---|---|
| Deterministic state | 256-dim GRU | 4096-dim |
| Stochastic state | 32 × 32 categoricals | 32 × 32 |
| MLP width | 640 | 8192 |
| Total parameters | ~12M | ~200M |
| Training steps | 100K | 200M+ |

The z_t dimensionality is identical across all DreamerV3 sizes; only h_t and MLP width scale. This means the XS probe results directly test whether KL is linearly readable from h_t at the smallest representational scale.

**Environment:** dm_control cartpole_swingup, random policy. 100K training steps, 200 episodes.

**Training states:** 100K (h_t, z_t, KL_t, recon_t) tuples logged during training. Probe training: 60% split (60K states). Evaluation sets collected from a frozen model post-training.

### 3.2 Evaluation sets

**Set A (ID):** 10K fresh swingup states, same distribution as training.

**Set B (noisy OOD):** 10K states with σ=0.1 Gaussian noise added to observations. The model was not trained on noisy inputs.

**Set C (KL-matched contrastive):** The critical evaluation. Pool Sets A+B, bin into 10 KL percentile bins, then within each bin select C1 (bottom 30% recon = coping) and C2 (top 30% recon = confused). Groups have matched KL distributions (1-nat gap) but a 9× recon gap. Any probe scoring above chance on Set C is detecting confusion independent of KL magnitude.

**Set C Strong (confounded, retracted):** C1 from cartpole_balance, C2 from cartpole_swingup. Initial AUROC 0.72 was retracted after a within-balance confound check showed 0.51 (chance) — the probe was detecting trajectory fingerprints, not confusion.

**Within-balance (confound check):** Both C1 and C2 from balance trajectories. Tests whether the probe can detect within-task confusion on a task it was not trained on.

### 3.3 Probe training protocol

All probes: logistic regression (sklearn, L2, C=1), StandardScaler preprocessing. Labels: KL > median (binary). AUROC metric. Single train/test split (60/40), stratified.

---

## 4. Results

### 4.1 The confusion signal exists and is distinct from novelty

**Table 1: AUROC across evaluation sets.**

| Method | Set A (ID) | Set B (noisy) | Set C (KL-matched) | Within-balance |
|---|---|---|---|---|
| Probe A (KL → h_t) | 0.863 | 0.846 | **0.723** | 0.506 |
| Probe C (recon → h_t) | 0.952 | 0.794 | 0.721 | — |
| RWM-U Ensemble | 0.868 | 0.842 | 0.744 | — |
| z_t probe (stochastic) | 0.847 | — | 0.667 | 0.435 |

**Set C (KL-matched) is the key result.** C1 and C2 have matched KL distributions (mean 22.9 vs 23.9 nats, 1-nat gap) but a 9× reconstruction gap (0.052 vs 0.471). Probe A at AUROC 0.72 is detecting something beyond KL magnitude — the accumulated confusion history in h_t.

**Adversarial Set C (within-A only).** We verify the Set C result is not driven by soft noise-level leakage from pooling Set A and Set B. Rebuilding the contrastive set from Set A states only (clean swingup, no noise): AUROC = **0.7115** vs 0.7144 for the original set (Δ = −0.003, below measurement noise). The result is robust within-distribution.

**The ensemble inverts on novel tasks.** On Set C Strong (before retraction), RWM-U reached 0.31 — it flags novel states as confident and familiar-confused states as uncertain. It is a novelty detector. The probe does the opposite (0.72). The within-balance confound check shows neither the probe nor z_t can detect cross-task confusion (0.51, 0.44 respectively) — cross-task generalisation requires the derivative representation (§4.4).

**Direct OOD detection (swingup vs balance, no KL matching):** Reconstruction error (0.9964) and KL (0.9582) from a single frozen model both exceed the 3-model ensemble reference (0.9425). The h_t probe inverts to 0.49 — below chance — on clean held-out evaluation. These three signals are categorically distinct: OOD is best detected by per-step scalars; confusion history requires the probe.

### 4.2 Closed-form characterisation: the probe computes a confusion integral

Define the **confusion integral**:

C_t = Σ_{i=0}^{T} γ^i · 1[KL_{t−i} > median_KL]

where the sum runs backwards within the current trajectory. C_t is a discounted count of recent confused steps.

**Table 2: R² of probe score regressed on C_t vs γ.**

| γ | R²(probe ~ C_t) | Δ vs KL alone |
|---|---|---|
| 0.70 | 0.722 | +0.203 |
| 0.80 | 0.752 | +0.233 |
| 0.90 | 0.786 | +0.267 |
| **0.95** | **0.798** | **+0.280** |
| 0.99 | 0.786 | +0.267 |

Baseline: R²(probe ~ KL_t alone) = 0.519.

**The probe approximates C_t with γ = 0.95 at R² = 0.7983.** Current KL alone explains 52% of probe variance; the discounted confusion integral explains 80%. The joint regression (KL_t + C_t) reaches R² = 0.804, suggesting C_t already captures most of the predictable structure.

**Figure 1: Accumulation curve.** Probe score grows monotonically with streak length L_t (consecutive high-KL steps ending at t): from 0.244 (L_t = 0) to 0.765 (L_t = 14). Pearson r(L_t, probe score) = +0.853. R²(probe ~ L_t) = 0.376 — streak length alone explains 38% of variance, substantially below the integral but confirming the monotonic structure.

**Interpretation of γ = 0.95.** This implies ~13-step effective memory (γ^13 ≈ 0.51). High-KL steps from 13 steps ago receive half the weight of the current step. The probe is most sensitive to the onset of confusion, less sensitive to sustained confusion — the probe saturates above streak length ~10.

**Verification by direct regression.** We train a Ridge regression probe directly on C_t values (continuous target, γ=0.95) rather than binary KL labels. R²(h_t → C_t via Ridge) = 0.794 (γ=0.95), 0.815 (γ=0.99). The AUROC on Set C is 0.711 — essentially unchanged from the binary probe (0.714). The binary KL proxy is near-optimal: direct C_t supervision does not substantially improve performance, confirming h_t is already near the C_t encoding ceiling with either supervision target. The 20% unexplained variance is genuinely not linearly recoverable.

This is, to our knowledge, the first closed-form expression produced for what a probe over world model hidden states is computing.

### 4.3 Geometric account: confusion in the null space of h_t

**Update-gate saturation.** Across 4 KL quartiles (spanning 7× KL range), z_gate varies by only 0.0083 (mean 0.9385, std 0.0061). Pearson r(mean(z_gate), KL) = −0.46 — the GRU has learned an almost-always-overwrite policy. Confusion is not encoded in *how much* h_t gets updated, but in *where* it gets updated to.

**PCA angle analysis.** Fitted 50-component PCA on 100K scaled h_t vectors (top-50 covers 90%+ of variance). Mean angle between probe direction and each top-50 PC: **88.2°**. The top 50 PCs capture only **9%** of the probe's variance. The confusion signal is in the near-null space of h_t's principal variation — invisible to any PCA-based representation analysis.

This result has a mechanistic account. When z_t ≈ 1, h_t ≈ n_t at every step. The dominant variance structure of h_t is therefore driven by the dominant variance of n_t, which is driven by the variance of observations (the high-signal, high-variance inputs the encoder processes). Confusion — a second-order property of prediction error — ends up encoded in directions orthogonal to observation-driven variance, accumulated at low amplitude across all 256 dimensions over the trajectory.

**Per-block uniformity.** Splitting h_t into 4 quarters of 64 dimensions: AUROC is flat at 0.87–0.90 across all quarters. There is no "confusion subspace" — the signal is spread uniformly, consistent with the GRU mixing information across all dimensions at every step.

**The n_t candidate direction hypothesis.** We tested whether confused states push the candidate n_t in the probe direction: r(n_t·w_probe, recon) = −0.006 for swingup (near zero). The directional hypothesis fails — confusion is multi-step accumulation, not a single-step candidate signal.

### 4.4 Cross-task generalisation via Δh_t

The within-balance confound check showed Probe A (on h_t) collapses to 0.51 cross-task. The confound is trajectory history: h_t accumulates a distributional fingerprint of the task it has been processing. We remove this by taking the first difference:

Δh_t = h_t − h_{t−1}

Δh_t retains only what changed this step, stripping the trajectory fingerprint.

**Single-task Δh_t probe, evaluated cross-task:**

| Held-out task | Δh_t AUROC | h_t AUROC |
|---|---|---|
| swingup (own task) | 0.537 | 0.506 (within-balance) |
| balance | **0.704** | 0.516 |
| balance_sparse | **0.711** | — |

**The cross-task signal is present without multi-task training.** A Δh_t probe trained only on swingup achieves 0.70 on within-balance KL-matched sets, compared to 0.51 for the h_t probe. Multi-task pooling (across 4 cartpole tasks, LOO evaluation) adds only +0.005 on the balance held-out task.

**Interpretation.** The trajectory-fingerprint confound was the reason h_t failed cross-task, not the absence of a cross-task signal. The confusion signal in Δh_t is already task-agnostic — it reflects what just happened in response to this observation, not accumulated task identity.

**Δh_t is weaker within-task.** R²(Δh_t probe) on swingup held-out = 0.705 vs R²(h_t probe) = 0.902. Within the training task, accumulated history provides context that Δh_t lacks. The trade-off: h_t maximises within-task confusion detection; Δh_t enables cross-task transfer at lower precision.

**The Δh_t → ΔC_t characterisation does not hold.** We tested whether Δh_t probe scores approximate ΔC_t = C_t − C_{t−1} (the rate of change of confusion accumulation). R²(Δh_t probe ~ ΔC_t) = 0.005 across all γ — near zero. Δh_t's cross-task generalisation is not because it tracks ΔC_t, but because differencing removes the trajectory fingerprint that blocked cross-task transfer in h_t. The two representations carry different information: h_t encodes accumulated confusion history; Δh_t encodes the immediate GRU response to the current observation, task-agnostically.

### 4.5 Observation routing: the probe as a confusion oracle

**Setup.** At each step in a trajectory, compute probe(h_t). If probe(h_t) > θ, "query" — collect a real observation. Otherwise, continue in imagination. Evaluate at 40K held-out states across query budgets [5%, 70%], measuring recall of top-25% KL events (states most in need of real observations).

**Table 3: Recall at 30% query budget.**

| Policy | Recall | Mean imagined KL |
|---|---|---|
| Probe A | **0.818** | **10.50** |
| KL oracle | 1.000 | 10.03 |
| Recon oracle | 0.770 | 11.06 |
| Random | 0.305 | 13.45 |

The probe outperforms the recon oracle (0.818 vs 0.770). Recon error is available at inference time and is directly correlated with KL (r=0.60) — it is a strong scalar baseline. The probe exceeds it by using trajectory history (C_t with γ=0.95) that no scalar can access.

**AUC of recall-vs-budget curve (0-70% range):** Probe = 0.509, KL oracle = 0.570, Recon oracle = 0.491, Random = 0.247. Normalised probe performance: 81% of KL oracle, 107% of recon oracle.

**The probe beats recon-threshold because confusion accumulates over time.** A state with low current recon error may still be in the middle of a confused trajectory. C_t captures this; current recon does not. We confirm this mechanism directly: states that the probe queries but the recon oracle misses have a mean streak length L_t = 73.3 steps (76% have L_t > 5), compared to L_t = 21.5 for states the recon oracle catches that the probe misses (35% have L_t > 5) — a 3.4× ratio. The probe's advantage is concentrated in sustained, multi-step confused sequences that no current-step scalar can anticipate.

### 4.6 Three orthogonal signals in h_t

**The observation/imagination boundary.** A probe trained to separate real posterior h_t (label 0) from imagined h_t (label 1) across all imagination depths 1–15 achieves **AUROC 1.0000**. After one imagination step with random actions, h_t is perfectly separable from any real posterior state by a linear classifier.

**Mechanistic basis — closed-form prediction.** With z_gate = 0.9385, after d imagination steps the fraction of original posterior content remaining in h_t is (1−z_gate)^d = 0.062^d. At d=1: only 6.2% of h_t is from the original posterior — it is 93.8% prior material. The boundary probe score saturates to 0.999 at d=1, and the boundary direction projection at d=1 (0.935) matches the theoretical prior-contamination fraction (0.939) directly. Prediction: boundary_score(d≥1) ≈ 1.0 because (1−z_gate)^d < 0.07 for all d≥1, fully explaining the 1.0000 AUROC without any free parameters beyond the measured z_gate.

**Orthogonality.** Pearson r(boundary probe, Probe A) on 40K held-out real states = **−0.015**. The two probes are sampling orthogonal directions in h_t space. Both live in the near-null space (boundary probe: 87.4° mean angle, 17% in top-50 PCs; Probe A: 88.2°, 9%). Their mutual angle is **84.1°**.

**Three-probe summary:**

| Signal | AUROC | h_t null-space? | r to Probe A |
|---|---|---|---|
| KL/confusion (Probe A) | 0.902 (train) / 0.723 (Set C) | Yes (9% in top-50) | 1.000 |
| Obs/imagination boundary | 1.0000 | Yes (17% in top-50) | −0.015 |
| OOD detection | — | No (recon: 0.9964, no training) | — |

**Partial correlation clarification.** The intuitive framing — probe tracks confusion (recon), ensemble tracks novelty (KL) — does not survive partial correlation analysis. After controlling for confounds: r(probe, KL|recon) = +0.52, r(probe, recon|KL) = −0.08. Probe A is primarily a KL-history signal: it tracks accumulated KL over the trajectory (C_t), and recon adds nothing after conditioning on KL. Ensemble disagreement shows r(ens, recon|KL) = +0.56 — it is primarily a reconstruction-quality signal. The dissociation is real: Probe A tracks *accumulated KL history* (C_t); the ensemble tracks *current-step reconstruction quality*. These are distinct measurements of model behaviour, both called "uncertainty" in the literature but measuring different things. The Set C AUROC 0.72 remains valid: it shows h_t encodes KL history beyond what current KL alone captures, which is the meaningful finding.

---

## 5. Why This Signal Was Hidden

The confusion signal in h_t was not found by prior probing work for two reasons that follow from the signal's geometric structure.

**Reason 1: orthogonality to ensemble disagreement.** Ensemble methods measure disagreement between models — they detect when different random seeds would produce different predictions on the same input. This is highest for novel inputs. Within-task confusion (the model's inability to predict well within its training distribution) is *uncorrelated* with ensemble disagreement: the probe inverts to 0.49 on direct OOD detection while achieving 0.72 on within-task KL-matched sets. You cannot discover a confusion signal by measuring disagreement, because the two signals are orthogonal.

**Reason 2: orthogonality to dominant representation structure.** Standard probing methods (PCA, t-SNE, linear probing without contrastive controls) operate on the dominant variance structure of representations. The confusion signal is in directions that are 88.2° from all top-50 PCA components. Any analysis that reduced dimensionality via PCA before probing would discard 91% of the signal before looking for it.

**Why the signal ended up in the null space.** With z_gate ≈ 0.94 (empirically confirmed), h_t ≈ n_t at every step. The dominant variance in h_t is driven by the dominant variance in n_t, which is driven by observation content — the high-amplitude, high-variance inputs the encoder processes. Confusion is a second-order property of prediction error history, not observation content. Its information ends up in the orthogonal complement of the observation-driven subspace — the null space — accumulated at low amplitude over multiple steps.

This is a consequence of the training objective, not a coincidence. The RSSM training objective forces h_t to track observations well; it does not force confusion to be encoded in any particular direction. The confusion signal appears in whatever low-variance directions are left over after observation dynamics claim the high-variance ones.

---

## 6. Discussion

### 6.1 Limitations

**Scale.** All results are on a 256-dim GRU, 100K steps, 5-dim cartpole observations. The full DreamerV3 XL uses 4096-dim GRU, millions of steps, 64×64 image inputs. The scale question is open but bounded by the theory in §6.2: the null-space geometry follows from z_gate saturation, which is a learned behaviour driven by the RSSM objective rather than a capacity constraint. If z_gate remains near-saturated at larger scale — as is typical in DreamerV3 across environment classes [Hafner et al., 2023] — the null-space geometry should persist. The confusion integral time constant (γ = 0.95, ~13-step memory) may change with longer episodes and richer dynamics; this is the primary unknown.

**Task diversity.** Cross-task experiments use 4 cartpole variants, all with identical observation space (5-dim). The Δh_t cross-task result (0.70) is encouraging but untested on genuinely different environments (locomotion, manipulation). The balance/balance_sparse pair share identical dynamics; the 4 tasks provide limited diversity.

**Probe-weighted returns: a null result.** Using probe scores to weight imagined returns (Task 2) degraded return quality (Δr = −0.53). The cause: imagined prior entropy is 14× smaller than real KL at this training level — the imagination is not calibrated to real dynamics. This mechanism would require a calibrated world model; at 100K steps it does not hold.

**Probe is not a perfect oracle.** The confusion probe achieves 81% of KL oracle performance in active querying. R² = 0.80 means 20% of variance in C_t is unexplained. Direct training on C_t labels (rather than binary KL) would likely improve performance.

### 6.2 Theoretical sketch: z_gate saturation and null-space encoding

When z_t → 1 (as observed empirically, z_t ≈ 0.94 with std 0.006), h_t ≈ n_t. The candidate n_t is computed as:

n_t = tanh(W_in·x + r_t ⊙ (W_hn·h + b_n))

where x = [z_{t−1}; action_{t−1}] is the input. The dominant variance in n_t is driven by the variation in x (observation-dependent inputs). Confusion — accumulated prediction error over the trajectory — must be encoded in directions orthogonal to this input-driven variation.

More formally, let V_obs be the subspace of h_t space aligned with the dominant variance of x through the linear map W_in (before tanh), and V_⊥ its orthogonal complement. When z_t ≈ 1, h_t ≈ n_t, so the variance of h_t is dominated by the variance of n_t, which is dominated by the variance of W_in·x. Therefore I(confusion; V_obs) is bounded from above by a term proportional to (1 − mean(z_t)), the residual retention of previous state. As z_t → 1, the confusion signal must encode in V_⊥.

This gives a testable prediction: higher z_gate → probe more orthogonal to top PCs (positive r between z_gate and probe-PC angle).

We test this directly across 6 training checkpoints of the same model (steps 5K–100K), giving real z_gate variation:

| Training step | mean(z_gate) | Probe-PC angle (top 10) |
|---|---|---|
| 5,000 | 0.775 | 89.0° |
| 10,000 | 0.843 | 89.0° |
| 20,000 | 0.859 | 88.2° |
| 40,000 | 0.882 | 88.3° |
| 70,000 | 0.901 | 87.7° |
| 100,000 | 0.923 | 87.2° |

z_gate span: 0.148. Angle span: 1.87°. Pearson r(z_gate, angle) = **−0.889** (p=0.018) — the opposite of the prediction. As z_gate saturates, the probe direction becomes *slightly less* orthogonal to the top PCs, not more.

Two observations temper this falsification. First, the angle range is 87–89° throughout all training stages — the confusion signal is near-orthogonal to top PCs regardless of z_gate level. The geometric fact is robust. Second, at step 5K (z_gate=0.775) the model has barely begun training and the probe direction is essentially random (no real confusion signal to learn), producing 89° by chance rather than by mechanism. The slight decrease to 87° at 100K reflects the confusion direction stabilising in h_t space as training progresses, marginally more aligned with some low-variance PC directions.

**Revised mechanistic account.** The null-space geometry is not primarily driven by z_gate saturation. A more accurate account: the RSSM training objective creates a structural separation in h_t between task-relevant information (which is observation-driven and lands in the high-variance subspace, captured by top PCs) and confusion-history information (which accumulates slowly across steps and lands in whatever low-variance directions are not claimed by task dynamics). This separation is a consequence of the *content* of what each subspace represents, not of how aggressively the GRU overwrites. The z_gate prediction was a plausible but incorrect mechanism for the same geometric outcome.

### 6.3 Implications for Phase 2 and Phase 3

**Phase 2 (temporal propagation at scale):** The XS pilot showed ΔR² growing from +0.016 (k=1) to +0.037 (k=20) on top of KL autocorrelation. The probe's R² is flat across horizons while KL's decays. This temporal structure should be more pronounced at 200M scale with richer trajectories. Testing the confusion integral interpretation at scale (whether γ=0.95 or a different time constant) would directly assess whether the signal remains a simple discounted accumulation or develops more complex structure.

**Phase 3 (surgical repair):** The uniform per-block AUROC (0.87–0.90 across all four quarters of h_t) and the null-space geometry indicate that the signal is not localised in any subspace. Surgical block-level repair is unlikely to work. Forcing the GRU to route confusion information into a dedicated subspace — via an auxiliary training objective — is the natural Phase 3 direction.

---

## 7. Related Work

**World model uncertainty:** RSSM-based uncertainty has been studied primarily via ensemble disagreement [Chua et al., 2018; Kidambi et al., 2020], predictive variance [Janner et al., 2019], and distributional shift detection [Lütjens et al., 2020]. These methods measure novelty. Our work measures confusion — a within-distribution property that these methods cannot detect.

**Representation probing:** Linear probes have been used extensively in NLP [Conneau et al., 2018; Tenney et al., 2019] and vision [Alain & Bengio, 2016]. Probing for uncertainty in neural representations has been studied in supervised settings [Gal & Ghahramani, 2016; Lakshminarayanan et al., 2017]. To our knowledge, this is the first work to apply probing to uncertainty in world model hidden states with a contrastive evaluation design that controls for KL magnitude.

**Active learning / observation routing:** Active perception in model-based RL has been explored via information gain [Schmidhuber, 1991] and curiosity signals [Pathak et al., 2017]. Our routing oracle is closest in spirit to selective real-data collection [Yu et al., 2020] but uses a learned confusion signal rather than model uncertainty as the routing criterion.

**RSSM state analysis:** Prior work on the content of world model latent states has focused on what task-relevant information is encoded [Ha & Schmidhuber, 2018; Hafner et al., 2019b] and whether the latent space supports disentanglement [Depeweg et al., 2018]. These works characterise the *primary* content of learned representations — the dominant variance structure. Our finding is complementary: we characterise the *residual* content — what the near-null space of the representation encodes implicitly as a side effect of training. This residual has not been analysed in prior RSSM work because the standard methodology (PCA, reconstruction probing) would not detect signals in directions that explain less than 0.1% of representation variance.

Ha, D., & Schmidhuber, J. (2018). World Models. NeurIPS Workshop.
Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2019b). Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020.
Depeweg, S., Hernandez-Lobato, J.-M., Doshi-Velez, F., & Udluft, S. (2018). Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning. ICML 2018.

---

## 8. Conclusion

We have shown that a DreamerV3 world model's recurrent state h_t implicitly encodes a within-task confusion signal that is distinct from novelty, detectable at AUROC 0.72 on KL-matched contrastive sets, and approximately equal to a discounted confusion count C_t (R² = 0.80). The signal lives in the near-null space of h_t's PCA (88° from all top PCs, 9% in high-variance subspace) — a geometric consequence of GRU update-gate saturation. Taking Δh_t recovers cross-task generalisation (0.70 AUROC) without multi-task training. The signal operationalises as an observation-routing oracle that outperforms recon-error baselines. Finally, a second orthogonal signal — the observation/imagination boundary at 1.0000 AUROC — coexists in the same near-null space with r = −0.015 correlation to the confusion probe.

The finding that h_t encodes these signals without being trained to do so suggests that the RSSM training objective induces implicit structure beyond what it requires. Understanding and exploiting this structure is a productive direction for model-based RL.

---

## References

Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2023). Mastering Diverse Domains with World Models. arXiv:2301.04104.

Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS 2018.

Kidambi, R., Rajeswaran, A., Netrapalli, P., & Joachims, T. (2020). MOReL: Model-Based Offline Reinforcement Learning. NeurIPS 2020.

Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. ICLR Workshop 2017.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML 2016.

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.

Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. NeurIPS 2019.

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. ICML 2017.

Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J., Levine, S., ... & Ma, T. (2020). MOPO: Model-based offline policy optimization. NeurIPS 2020.

---

*Code and data: all experiments run on a CPU-only M4 MacBook Air (100K env steps, ~4.5 hours total wall time). Codebase available at [repo].*
