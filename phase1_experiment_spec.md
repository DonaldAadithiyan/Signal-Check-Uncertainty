# Phase 1 Signal Check — Mini-DreamerV3 Uncertainty Probe
## Pilot Experiment Specification

**Goal:** Determine whether a linearly readable epistemic uncertainty signal exists inside the recurrent hidden state `h_t` of a scaled-down DreamerV3 world model — using only signals the model already produces internally, with no external labels.

**Scope:** This is a go/no-go experiment for the full research programme. It covers Phase 1 only. If no signal is found at this scale, the programme pivots to the auxiliary training objective angle. If a signal is found, Phases 2 and 3 are justified.

**Hardware target:** MacBook Air (CPU only). Estimated wall-clock time: 3–4 hours.

---

## 1. Repository and Model Configuration

Clone the PyTorch reimplementation of DreamerV3:

```
https://github.com/NM512/dreamer-pytorch
```

This is a faithful reimplementation that preserves the RSSM architecture and the KL posterior/prior training objective exactly. The supervisor can verify it against the original JAX codebase — the architecture is identical. The reason for using this over the official repo is that the official codebase requires JAX with GPU support; this runs cleanly on CPU.

### Model size: XS configuration

Set the following hyperparameters in the config to match DreamerV3's own XS ablation setting:

| Parameter | Value |
|---|---|
| `rssm.hidden` | 256 |
| `rssm.deter` | 256 |
| `rssm.stoch` | 32 |
| `rssm.classes` | 32 |
| `encoder.mlp_layers` | 1 |
| `decoder.mlp_layers` | 1 |
| `actor.layers` | 1 |
| `critic.layers` | 1 |

This is the smallest configuration the DreamerV3 paper itself defines. The KL training objective — which is the source of the probe signal — is fully intact at this size.

---

## 2. Environment

**Primary:** `dm_control` — `cartpole_swingup`

This is the standard benchmark used in the DreamerV3 paper. It is a continuous-control environment with non-trivial dynamics, which means the KL gap has meaningful variance across states. It runs on CPU without a display via the `dm_control` headless backend.

Install:
```bash
pip install dm_control gymnasium numpy torch scikit-learn matplotlib
```

**Why not MiniGrid or Atari:** MiniGrid dynamics are nearly deterministic — the KL gap barely varies, making the probe signal trivially easy or trivially absent. Atari requires an ALE ROM and image observations that slow down CPU training significantly. DMControl Cartpole is the right balance of speed and credibility.

---

## 3. Training the World Model

Train the world model only — the policy quality is irrelevant to this experiment. You need a world model that has learned partial dynamics, not a well-performing agent.

**Training budget:** 100,000 environment steps.

This is approximately 2% of a full DreamerV3 training run (5M steps). It is enough for the world model to fit the common states and begin to show variance in its KL gap across the state space. It will not produce a competent policy. That is fine — you are probing the world model internals, not evaluating the policy.

**What to log during training:** At every environment step, record and save to disk:

- `h_t` — the full GRU hidden state vector (256-dimensional)
- `kl_t` — the scalar KL divergence between posterior `q(z_t | h_t, x_t)` and prior `p(z_t | h_t)` computed at that step
- `recon_error_t` — the scalar L2 reconstruction error between the decoded observation and the actual observation
- `step_index` — the global environment step number
- `trajectory_id` — which episode this step belongs to

Save these as a single `.npz` file at the end of training. This is your probe dataset.

**Estimated training time on MacBook Air CPU:** 45–90 minutes at 100K steps.

---

## 4. Constructing the Three State Sets

After training, collect three additional trajectory sets by running the **frozen world model** (no weight updates) on new rollouts. The policy used for collection does not matter — a random policy is fine.

### Set A — In-Distribution (ID) States

Run 20 episodes using the trained policy (or random policy) in the **same** `cartpole_swingup` environment the model was trained on. Collect `(h_t, kl_t, recon_error_t)` tuples. These are the states the model knows well — expected to have low KL gap on average.

Target size: ~2,000 state vectors.

### Set B — Near-OOD States

Run 20 episodes with a **modified** version of the environment. Apply one of the following interventions — choose whichever is simplest to implement in `dm_control`:

- Change the pole length by 50% (standard perturbation in MBRL literature)
- Add Gaussian noise with std=0.1 to the observation vector at every step
- Set the initial state to extreme angles the training policy never visited (initialise with `physics.set_state`)

The model's weights are unchanged. The environment dynamics are slightly different. Collect the same tuples. These are the near-OOD states — expected to have higher KL gap than ID states on average.

Target size: ~2,000 state vectors.

### Set C — Contrastive States (Critical)

This set directly tests whether the probe captures genuine epistemic uncertainty or merely novelty. It requires manual construction of two sub-groups:

**C1 — OOD-but-accurate:** States where the observation looks unusual (high L2 distance from the training distribution mean in observation space) but where `recon_error_t` is low — the model still predicts well. Find these by scanning Set B for timesteps where recon error is in the bottom 20th percentile despite being OOD. Label these as **low uncertainty** ground truth.

**C2 — ID-but-failing:** States where the observation looks familiar (low L2 distance from training distribution mean) but where `recon_error_t` is high — the model is unreliable despite the familiar appearance. Find these by scanning Set A for timesteps where recon error is in the top 20th percentile. Label these as **high uncertainty** ground truth.

Target size: ~200 states per sub-group (400 total). The contrastive set is small by design — it is testing a specific qualitative property, not statistical power.

A probe that passes the main test but fails on Set C is a **novelty detector**, not an uncertainty detector. The distinction matters for the proposal's core claim.

---

## 5. Probe Training

Train all probes using **logistic regression** from `scikit-learn`. No neural probes — linear separability is the claim. A nonlinear probe passing would not confirm linear structure in `h_t`.

### Label construction

Binarise `kl_t` values using the **median** of the training set as the threshold:
- `kl_t > median` → label 1 (high uncertainty)
- `kl_t ≤ median` → label 0 (low uncertainty)

This avoids any arbitrary threshold choice and is fully unsupervised — the label comes from the model's own training signal.

### Probe A — KL gap signal

- Features: `h_t` vectors (256-dim) from the training data
- Labels: binarised `kl_t`
- Train/val/test split: 60/20/20, stratified
- Regularisation: L2 with C=1.0 (default)

### Probe B — Rollout variance signal

For each state in the dataset, run 5 imagination rollouts of length 5 from that `h_t` using the frozen model's dynamics predictor `p(ẑ_t | h_t)`. Compute the variance of the predicted next observations across the 5 rollouts. Binarise by median. Train the same logistic regression on `h_t` to predict high vs. low rollout variance.

This is the second candidate signal from the proposal. It does not require access to the actual next observation — it is fully internal to the model's imagination.

### Probe C — Recon error signal (sanity check)

Same as Probe A but using binarised `recon_error_t` as the label instead of `kl_t`. This checks whether prediction error is readable from `h_t` at all. It is an easier target than KL gap. If this fails, everything else will fail too — it is a sanity check, not a scientific result.

---

## 6. Ensemble Baseline

Train **3 independent copies** of the same XS world model on the same data, with different random seeds. At this model size and training budget, each copy takes ~45 minutes — run them sequentially, or start them before the probe analysis step.

For each state in Set C, compute **ensemble disagreement** as the variance of predicted next observations across the 3 models. Binarise by median. This gives you an ensemble-based uncertainty score for Set C.

The comparison that matters most for the supervisor: **does the single-model probe (Probe A or B) achieve higher AUROC on Set C than the ensemble disagreement baseline?** If yes, this is the central claim of the paper — that internal structure in a single model contains information about its own uncertainty that an ensemble of that same model cannot recover.

---

## 7. Evaluation Metrics

Compute the following for each probe and for the ensemble baseline:

### Primary metric: AUROC

Use `sklearn.metrics.roc_auc_score`. AUROC of 0.5 = chance, 1.0 = perfect. No threshold tuning needed.

Report AUROC separately on:
- Held-out ID states (Set A test split)
- Set B OOD states
- Set C contrastive states (the critical evaluation)

### Secondary metric: Per-block contribution

The GRU hidden state is 256-dimensional. Split it into 4 quarters of 64 dimensions each (a lightweight proxy for the 8-block structure in the full 200M model). Train separate logistic regressions on each quarter. Report which quarter has the highest AUROC. This gives the supervisor a preview of the causal tracing question in Phase 3.

### Tertiary: `h_t` vs `z_t`

The proposal predicts that `h_t` carries more uncertainty signal than `z_t` (the stochastic latent). Train the same Probe A logistic regression on `z_t` vectors instead of `h_t` vectors. Compare AUROCs. The expected result is `h_t` AUROC > `z_t` AUROC, which would be consistent with the architectural argument in the proposal (KL training forces uncertainty-awareness into `h_t`, not `z_t`).

---

## 8. Success and Failure Criteria

### Signal exists (positive result)

All three of the following hold:

| Criterion | Threshold | Interpretation |
|---|---|---|
| Probe A AUROC on Set A held-out | > 0.72 | KL gap is linearly readable from `h_t` |
| Probe A or B AUROC on Set C | > 0.63 | Signal captures genuine uncertainty, not just novelty |
| `h_t` AUROC > `z_t` AUROC | p < 0.10 (bootstrap) | Uncertainty lives in the recurrent state, not the latent |

If these hold, the programme proceeds. Phase 2 (temporal propagation) and Phase 3 (surgical repair) are worth pursuing with the full 200M model.

### Signal absent (negative result)

All probes hover near AUROC 0.50–0.55 regardless of signal type, layer partition, or state set. This is a clean negative result with the following interpretation: standard DreamerV3 training does not spontaneously encode uncertainty in `h_t` at XS scale and 100K step budget. This motivates the auxiliary training objective variant of the proposal — force a subspace of `h_t` to predict its own future reconstruction error as an explicit training target. That is still a publishable contribution (mechanistic evidence of structural overconfidence), and the negative result from this experiment is the motivation.

---

## 9. What to Present to Your Supervisor

Prepare one page with the following:

**The claim being tested:** Does a linear probe trained on `h_t` — using only the model's own KL gap as a label — achieve above-chance AUROC on held-out states, including states where novelty and uncertainty are deliberately decorrelated?

**The model:** XS DreamerV3 (PyTorch reimplementation), 256-unit GRU, trained for 100K steps on DMControl Cartpole-Swingup. This is the DreamerV3 paper's own XS ablation configuration.

**The result:** One AUROC table (3 rows: Set A held-out, Set B OOD, Set C contrastive) × (4 columns: Probe A KL, Probe B variance, Probe C recon sanity, Ensemble baseline).

**The interpretation:** If the probe beats the ensemble on Set C, the central claim of Phase 2 is plausible at scale — a single model's internal state encodes something about its own uncertainty that independent retraining cannot recover. If it does not, the programme still has a well-motivated negative result and a clear pivot.

**The framing sentence for the meeting:** *"This is a 100K step, XS-scale go/no-go experiment. It cannot prove the signal exists at full scale, but it can cheaply falsify the hypothesis before we invest months in it."*

---

## 10. File Checklist

By end of experiment, you should have produced:

- `training_states.npz` — `(h_t, z_t, kl_t, recon_error_t)` tuples from training
- `set_a_id.npz` — in-distribution evaluation states
- `set_b_ood.npz` — near-OOD evaluation states
- `set_c_contrastive.npz` — contrastive states with C1/C2 labels
- `probe_results.csv` — AUROC values for all probes and all state sets
- `block_auroc.csv` — per-quarter AUROC for the block contribution analysis
- `ht_vs_zt.csv` — AUROC comparison between `h_t` and `z_t` probes
- `figures/roc_curves.png` — ROC curves for the main probes on all three sets
- `figures/block_heatmap.png` — bar chart of per-block AUROC

---

*Research Programme: Epistemic Self-Awareness of World Models · Phase 1 Pilot · NeurIPS / ICLR Target*
