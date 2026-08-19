# Reading the Room: Implicit Confusion Encoding in Recurrent World Model States

A DreamerV3 world model's recurrent hidden state `h_t` is trained only to reduce prediction error — yet it also **tracks its own confusion**: a history of being wrong for several steps in a row, distinct from novelty, hiding in the near-null space of `h_t` where variance-based analysis never looks. This repository contains the full experimental codebase, results, and write-ups behind that finding.

**Status:** paper complete and accepted at IJCAI-ECAI 2026 GlobalSouthAI (reviews 5/6/7); currently hardened and expanded for resubmission to a stronger venue. All core experiments (Tasks A–T) are done; a width-scaling study is in progress (see [Scaling work](#scaling-work-in-progress)).

> **Every result in this project was produced on a single laptop CPU (Apple M4).** No GPU, no cluster. A deliberate demonstration that causally-validated interpretability does not require frontier-scale compute.

---

## The finding in one paragraph

We use a **Mini-DreamerV3 (XS: 256-dim GRU, 32×32 categorical latents, ~12M params)** trained from scratch on three dm-control tasks. A linear probe on `h_t` reads a **confusion signal** — a discounted count of recent high-KL (high-surprise) steps — that is (1) **distinct from novelty and ensemble disagreement** (they point in *opposite* directions on a surprise-matched test), (2) **has a closed-form** (C_t, γ=0.95, R²=0.80), (3) **is causally load-bearing** (editing it out of `h_t` changes behaviour, verified against a 50-direction empirical null and by a real-value-substitution method), (4) **hides in the near-null space** (88° from the top-50 PCs, 9% of its own variance there), and (5) **generalises across three tasks**, with a practical use — deciding when to check reality instead of trusting imagination — that holds on two of the three.

---

## Model

| Component | This work (XS) | Full DreamerV3 (XL) |
|---|---|---|
| Deterministic state `h_t` | 256-dim GRU | 4096-dim GRU |
| Stochastic state `z_t` | 32 cat × 32 classes (1024-dim) | 32 cat × 32 classes (identical) |
| Total parameters | ~12M | ~200M |
| Training steps | 100K | 200M+ |
| Observation | vector (3–6 dim) | 64×64 RGB |

Only GRU/MLP width and the observation encoder differ between XS and XL; the stochastic-state size is identical, which is why the geometry finding is worth a scale check rather than assumed size-specific.

**Tasks:** dm-control `cartpole-swingup` (5-dim obs / 1-dim act), `reacher-easy` (6/2), `pendulum-swingup` (3/1).

---

## Headline results

### The signal exists and is not novelty (Set C)

| Method | Set A (ID) | Set C (KL-matched) | Direct novelty |
|---|---|---|---|
| Probe on `h_t` | 0.863 | **0.723** | 0.49 |
| Ensemble disagreement | 0.868 | 0.31† | 0.943 |
| Reconstruction error | — | — | 0.996 |

†cross-task construction; within-task ensemble on Set C is 0.744. **Set C** holds KL (surprise) fixed while confusion varies — so a probe scoring above chance there reads something beyond surprise magnitude. 5-seed mean **0.715 ± 0.074**, 95% CI [0.666, 0.763]. The probe and ensemble point in *opposite directions* on the same test — the cleanest proof they are different signals.

### Closed-form account

The probe ≈ a discounted count of recent high-KL steps, C_t = Σ γⁱ·1[KL_{t−i} > median], at **γ = 0.95, R² = 0.80** (γ identical across all 5 seeds; ~13-step memory). Current KL alone explains only R² = 0.52 — the gap is the history.

### Causally load-bearing (against a 50-direction null)

| Edit | Δ probe score | Null percentile |
|---|---|---|
| Ablation (subtract the direction) | −0.586 | 100th (z ≈ −22) |
| Real-value substitution (no synthetic edit) | −0.761 | 100th |

Effect decays across look-ahead at the same γ=0.95 the probe reads; replicates on all 5 seeds. Two forward-dynamics measures show the direction *reads* unreliable states without *being* the cause of the unreliability.

### Geometry, and it is not a gate artefact

Confusion direction sits **88.2°** from all top-50 PCs, 9% of its variance there. Ruled out (by direct test) as a gate-saturation artefact (forcing the gate 0.5–0.99 moves the angle 1.29°) and as a single-block artefact (all 4 quarters read it at 0.87–0.90).

### Generalises across three tasks (with one traceable exception)

| | Cartpole | Reacher | Pendulum |
|---|---|---|---|
| Geometry angle | 88.0° | 89.4° | 88.1° |
| C_t R² | 0.80 | 0.26 | 0.86 |
| Set C AUROC | 0.72 | 0.62–0.67 | **0.32–0.40** |
| Within-bin recon↔C_t corr | +0.39 | −0.09 | −0.12 |

Geometry and closed-form carry over; the exact memory length and Set C do not. Pendulum's Set C **inverts** despite the *strongest* C_t fit — traced exactly to the sign of the within-bin recon↔C_t correlation (a checkable diagnostic for any reconstruction-based contrastive test).

### Operational use: routing

Using the confusion score to decide when to query a real observation beats a reconstruction-error baseline on cartpole (+0.04) and pendulum (+0.30), and loses on reacher (−0.08) where reconstruction error is already strong. Real but task-dependent.

---

## Reviewer-driven additions (post-IJCAI)

Two fixes built in response to reviewer feedback, both reported as-is:

- **KL-only routing baseline** (Reviewer TAQ8) — "what does the probe add over thresholding KL?" **Mixed/unfavourable:** the probe beats a causally-fair KL threshold on reacher (+0.27, 3/3 seeds) but not on cartpole (no reliable difference) or pendulum (probe reliably worse). Narrows the routing claim. → `run_task_r_kl_routing.py`, [deliverable](outputs/deliverables/task_R_kl_routing_baseline.md).
- **5-model ensemble** (Reviewer R3TF) — a stronger ensemble might narrow the Set C gap. **Favourable:** the gap holds (ensemble Set C ~0.55 at n=2→5 vs probe 0.71; gap +0.16 unchanged), even as the ensemble genuinely improves at novelty. → `run_task_t_ensemble_size.py`, [deliverable](outputs/deliverables/task_T_ensemble_size.md).

---

## Scaling work (in progress)

The paper's most-repeated caveat is scale: all main results are on a 256-dim GRU, three orders of magnitude below full DreamerV3. Current effort to strengthen this for resubmission:

- **Width-scaling trend (Tier 1, on this codebase, CPU):** `run_width_sweep.py` extends the single deter=512 scale point (Task S) into a 256→512→1024→2048 trend across all three tasks, measuring the two load-bearing findings (null-space geometry, causal ablation) plus C_t R² and Set C at each width. Each (task, width) cell is independently checkpointed and resumable; `--summarize` collates finished cells into a trend table.
  ```bash
  python run_width_sweep.py --task cartpole --width 256 512 1024 2048
  python run_width_sweep.py --summarize
  ```
- **Image-scale existence proof (Tier 2, planned):** port the `h_t`-logging + geometry/ablation measurements to an image-based DreamerV3 (`dreamerv3-torch`) and test whether the geometry and causal effect survive a **convolutional** encoder on one DMC-vision task. Off-the-shelf DMC-vision checkpoints are not readily downloadable (checked: `dreamerv3-torch`, official `danijar/dreamerv3`, Hugging Face all publish code/scores, not loadable vision weights), so this requires a self-trained model.

---

## Setup

```bash
pip install -r requirements.txt   # torch, dm_control, scikit-learn, numpy, matplotlib, scipy
```

Python 3.11. **CPU recommended on Apple Silicon** — at this batch size kernel-launch overhead makes CPU faster than MPS, and the RSSM rollout more reproducible.

---

## Reproducing the results

Each task is a self-contained script writing JSON/CSV under `outputs/`. Full one-shot sequence and every hyperparameter are in [outputs/deliverables/APPENDIX_REPRODUCIBILITY.md](outputs/deliverables/APPENDIX_REPRODUCIBILITY.md). Highlights:

```bash
# train the three world models
python run_experiment.py            # cartpole → outputs/checkpoints/world_model.pt
python run_second_env.py            # reacher
python compare_environments.py     # pendulum

# core probe + closed-form
python run_experiment.py           # Sets A/B/C, Probe A, ROC/AUROC
python run_confusion_integral.py   # γ sweep, C_t R²

# causal validation
python run_causal_intervention.py  # Task A: ablation / amplification / route-flip
python run_task_g_null.py          # Task G: 50-direction empirical null
python run_task_l_swap.py          # Task L: real-value substitution
python run_task_i_multiseed_causal.py  # Task I: 5-seed causal
python run_task_h_attractor.py     # Task H: Berger attractor cross-check

# replication, mechanism, applications, scale
python run_multiseed.py            # 5 cartpole seeds
python run_multiseed_env.py        # reacher/pendulum × 4 seeds
python run_task_n_mechanism.py     # Set C inversion mechanism
python run_task_q_routing.py       # cross-env routing
python run_task_r_kl_routing.py    # KL-only routing baseline (reviewer fix)
python run_task_t_ensemble_size.py # 5-model ensemble (reviewer fix)
python run_task_s_scale.py         # deter=512 scale point
```

---

## Documentation

| Document | What it is |
|---|---|
| [PAPER.md](PAPER.md) | Full paper draft (extended version) |
| [outputs/deliverables/PAPER_WALKTHROUGH.md](outputs/deliverables/PAPER_WALKTHROUGH.md) | Section-by-section walkthrough of the camera-ready paper, from scratch |
| [outputs/deliverables/FULL_PROJECT_EXPLAINER.md](outputs/deliverables/FULL_PROJECT_EXPLAINER.md) | Complete zero-background explainer of the whole project |
| [outputs/deliverables/APPENDIX_REPRODUCIBILITY.md](outputs/deliverables/APPENDIX_REPRODUCIBILITY.md) | Full setup, hyperparameters, seeds, and raw-result tables |
| [RESULTS_CONSOLIDATED.md](RESULTS_CONSOLIDATED.md) | Single source of truth: every headline number with provenance |
| [DEV_LOG.md](DEV_LOG.md) | Detailed running experimental record |
| `outputs/deliverables/task_*.md` | One context-rich write-up per task (A–T): hypothesis, build, numbers, caveats |

---

## Project structure

```
src/
  config.py              XS_CONFIG — model hyperparameters and paths
  model/
    world_model.py       WorldModel: encoder, RSSM, decoder
    rssm.py              RSSM: observe_step, imagine_step, KL, GRU
  env/
    wrapper.py           CartpoleEnv (swingup / balance, optional noise)
    dmc_wrapper.py       DMCEnv (reacher, pendulum)
  probe/
    linear_probe.py      train_probe, auroc, ensemble_disagreement
    intervention.py      compute_ct, probe_direction, ablation/swap helpers
  training/trainer.py    world model training loop

run_experiment.py        full training + state collection (cartpole)
run_second_env.py        reacher training pipeline
compare_environments.py  pendulum training + three-env comparison
run_task_[A-T]*.py       one script per experiment (see Reproducing above)
run_width_sweep.py       Tier-1 width-scaling study (256→2048)
make_paper_figures.py    paper figures
outputs/                 checkpoints, results (JSON/CSV), figures, deliverables
```

---

## Findings summary

1. **A within-task confusion signal exists in `h_t`** (Set C AUROC 0.72, 5-seed) — distinct from novelty and ensemble disagreement, which it *opposes* on the surprise-matched test.
2. **It has a closed form** — a discounted count of recent high-KL steps (γ=0.95, R²=0.80), stable across seeds.
3. **It is causally load-bearing** — editing it out of `h_t` changes behaviour at the 100th percentile of a 50-direction null, confirmed by an independent real-value-substitution method and across 5 seeds. It *reads* unreliable states without *being* the cause of the unreliability.
4. **It hides in the near-null space** (88° from top PCs, 9% variance there) — invisible to variance-based analysis; shown *not* to be a gate-saturation or single-block artefact.
5. **It generalises across three tasks** in direction, geometry, and closed form; the exact time constant and the Set C metric are task-specific, with pendulum's inversion traced to a checkable within-bin correlation sign.
6. **It is operationally useful** for observation routing on two of three tasks — a real but task-dependent benefit.

Negative/qualifying results are kept visible throughout: gate-saturation hypothesis (falsified), probe-weighted returns (fails), imagination stopping rule (null), KL-only routing (probe's advantage is task-dependent).

---

*Research programme: epistemic uncertainty in world models. Author: Donald Aadithiyan, University of Moratuwa.*
