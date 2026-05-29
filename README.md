# Signal Check — Epistemic Uncertainty in World Model Hidden States

Does a DreamerV3 world model's recurrent state `h_t` linearly encode whether it is confused — without any explicit uncertainty training?

Phase 1 of a 3-phase research programme. Go/no-go pilot on a scaled-down model before committing to the full 200M-parameter experiment.

---

## Research Question

A DreamerV3 RSSM maintains a deterministic hidden state `h_t` at every timestep. It is trained to minimise reconstruction error and KL divergence — never to track its own uncertainty. The question: does a linear classifier trained on `h_t` predict high vs. low KL, even when KL magnitude is held constant between groups?

If yes, the GRU accumulates trajectory-level confusion context as a side effect of RSSM training. That enables a cheap within-task uncertainty signal: no ensemble, no fine-tuning, no auxiliary head.

---

## Model

Mini-DreamerV3 (XS config), trained from scratch on `dm_control cartpole_swingup`:

| Component | This experiment | Full DreamerV3 (XL) |
|---|---|---|
| Deterministic state `h_t` | 256-dim GRU | 4096-dim GRU |
| Stochastic state `z_t` | 32 cat × 32 classes (1024-dim) | 32 cat × 32 classes |
| Total parameters | ~12M | ~200M |
| Training steps | 100K | 200M+ |
| Observation space | 5-dim (cartpole state) | 64×64 RGB |

The stochastic state dimensionality is identical across all DreamerV3 sizes. Only `h_t` and MLP width scale. This is the smallest configuration the DreamerV3 paper defines.

---

## Key Results

### Phase 1 — Does the signal exist?

Linear probes (logistic regression) on `h_t`, evaluated on three sets:

| Method | Set A (ID) | Set B (noisy OOD) | Set C (KL-matched) |
|---|---|---|---|
| Probe A — KL → h_t | 0.8632 | 0.8464 | **0.7227** |
| Probe B — rollout variance | 0.6285 | 0.7561 | 0.6256 |
| Probe C — recon → h_t | 0.9518 | 0.7944 | 0.7210 |
| RWM-U Ensemble (3 models) | 0.8678 | 0.8417 | **0.7436** |

Set C is the decisive test: C1 and C2 are drawn from the same KL bins (1-nat gap), so any probe that scores above chance must detect something beyond KL magnitude. AUROC 0.72 on Set C is the main claim.

**Cross-task generalisation: not demonstrated.** A within-balance confound check (both groups from `cartpole_balance`, same task identity) collapses to 0.51. The probe detects within-task confusion; it does not transfer across tasks.

### Direct OOD detection

| Signal | AUROC | Notes |
|---|---|---|
| Recon error (no training) | **0.9964** | Beats 3-model ensemble |
| KL (no training) | **0.9582** | Beats ensemble |
| z_t probe (stochastic state, 1024-dim) | 0.8988 | |
| h_t probe | **0.4903 — inverted** | Below chance on clean eval |

The h_t probe inverts on direct OOD detection: it has no information about task identity. This sharpens the claim — within-task confusion and distributional shift are orthogonal signals in `h_t`.

### Phase 2 Pilot — Temporal structure

Does `probe(h_t)` predict confusion at step t+k beyond KL autocorrelation?

| k | r(probe, KL_{t+k}) | R²(KL_t) | R²(+probe) | ΔR² |
|---|---|---|---|---|
| 1  | +0.719 | 0.768 | 0.784 | +0.016 |
| 5  | +0.721 | 0.750 | 0.769 | +0.018 |
| 10 | +0.716 | 0.708 | 0.731 | +0.023 |
| 20 | +0.700 | 0.608 | 0.645 | **+0.037** |

The probe's R² is flat (0.49–0.52) while KL's decays (0.77→0.61). ΔR² grows with horizon — the probe carries trajectory-level context that outlasts scalar KL autocorrelation.

**Verdict:** Positive. Phase 2 at 200M scale is justified.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11. CPU recommended on Apple Silicon (MPS is slower for small-batch inference on M-series).

---

## Reproducing Experiments

All scripts read config from `src/config.py` (XS_CONFIG). Set `checkpoint_path` and `training_data_path` there before running.

**Train world model and collect training states:**
```bash
python run_experiment.py
```

**Rerun probes on evaluation sets (Sets A, B, C):**
```bash
python rerun_probes.py
python rerun_probes_strong.py   # Set C Strong + within-balance confound check
```

**Ensemble baseline (RWM-U):**
```bash
python rerun_rwmu_baseline.py
```

**Δh_t and curvature probes:**
```bash
python run_delta_ht_probe.py
python run_curvature_probe.py
```

**GRU gate analysis:**
```bash
python run_gate_analysis.py
```

**Within-balance confound check:**
```bash
python run_within_balance_probe.py
```

**Direct OOD detection (all signals):**
```bash
python run_ood_detection.py
```

**Imagination depth experiments (null result):**
```bash
python run_imagination_depth.py
```

**Phase 2 pilot — temporal prediction:**
```bash
python run_phase2_pilot.py
```

---

## Project Structure

```
src/
  config.py              XS_CONFIG — model hyperparameters and paths
  model/
    world_model.py       WorldModel: encoder, RSSM, decoder
    rssm.py              RSSM: observe_step, imagine_step, prior_net, GRU
  env/
    wrapper.py           CartpoleEnv (swingup / balance, optional noise)
  probe/
    linear_probe.py      train_probe, auroc, binarise_by_median
  training/
    trainer.py           world model training loop
  data/                  data collection utilities

run_experiment.py        full training + state collection pipeline
run_phase2_pilot.py      temporal prediction (Experiments 1, 2, 3)
run_ood_detection.py     direct OOD detection — all signals
run_imagination_depth.py imagination depth experiments (null result)
run_gate_analysis.py     GRU gate analysis (r_t, z_t, n_t)
run_delta_ht_probe.py    Δh_t probe
run_curvature_probe.py   trajectory curvature probe
rerun_probes.py          re-evaluate probes on Sets A/B/C
rerun_probes_strong.py   re-evaluate on Set C Strong + confound check
rerun_rwmu_baseline.py   trajectory-aware ensemble baseline

DEV_LOG.md               detailed experimental record with all results
phase1_experiment_spec.md original experiment specification
```

---

## Findings Summary

1. **Signal exists within-task (AUROC 0.72, Set C KL-matched).** `h_t` linearly encodes whether the model is coping or confused, even when KL magnitude is matched between groups.

2. **Signal does not transfer across tasks.** Within-balance confound check: 0.51 (chance). The probe is task-specific.

3. **h_t and OOD are orthogonal signals.** The h_t probe inverts to 0.49 on direct OOD detection. Reconstruction error (0.9964) and KL (0.9582) from a single model both exceed the 3-model ensemble for OOD — no training needed.

4. **Signal is distributed, not localised.** All four h_t quarters score 0.87–0.90 (flat). No "uncertainty subspace." Phase 3 surgical repair cannot target a specific block.

5. **Imagination depth hypothesis is wrong.** Uncertainty does not compound monotonically with rollout depth. One imagination step washes out the low-KL signal; the model reaches a characteristic confusion ceiling at depth 1 and plateaus. Mechanistic cause: GRU update gate z_t ≈ 0.94 — near-complete overwrite every step.

6. **Temporal structure confirmed at XS scale.** ΔR² grows from +0.016 (k=1) to +0.037 (k=20). The probe's predictive contribution increases relative to KL autocorrelation as the horizon extends.

---

*Research programme: Epistemic Self-Awareness of World Models · Phase 1 Pilot*
