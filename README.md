# Reading the Room: Implicit Confusion Encoding in Recurrent World Model States

Code, results and figures for **ICAART 2027 Paper #388** (accepted, full paper).

A small DreamerV3-style world model is trained only to predict its observations. We ask whether its recurrent state `h_t` also carries a readable record of *recent* surprise: a discounted count of high-KL steps, `C_t = Σ_i γ^i · 1[KL_{t−i} > threshold]`. A linear probe on `h_t` reads such a signal. This repo tests what that signal is, what it predicts, and whether editing it changes anything downstream.

> Everything here runs on a laptop CPU (Apple M4). No GPU is needed.

---

## Status

| Item | State |
|---|---|
| Paper | Accepted at ICAART 2027 (#388) |
| Camera-ready deadline | **27 Nov 2026 (AOE)**, 12 pages (+4 paid) |
| Final numbers | Done: [outputs/camera_ready/camera_ready_numbers.md](outputs/camera_ready/camera_ready_numbers.md) |
| Reviewer analyses (Tasks 1–4 + cross-validated ΔR²) | Done: see [Camera-ready results](#camera-ready-results) |
| Final figures | Done: [outputs/camera_ready/figures_final/](outputs/camera_ready/figures_final/) (11 vector PDFs + PNG previews + [FIGURE_MANIFEST.md](outputs/camera_ready/figures_final/FIGURE_MANIFEST.md)) |
| Paper text (`paper/icaart_main.tex`) | **Not yet updated** to the final numbers |

Progress log: [outputs/camera_ready/STATUS.md](outputs/camera_ready/STATUS.md).

### Why the numbers changed after acceptance

The original loggers paired observation `o_t` with action `a_t`. The correct pairing is `a_{t−1}`. All camera-ready numbers use **corrected logging** and one fixed rule, applied before results were seen:

- **Refit on corrected probe-fit episodes, using the paper's recipe:** the probe direction `v`, the PCA, the SAEs, and the KL threshold. The threshold is now the calibration-episode median.
- **Kept from the paper:** γ = 0.95 / 0.70 / 0.90 (cartpole / reacher / pendulum), EMARec α = 0.02 / 0.5 / 0.02, each section's `C_t` definition, and the canonical 50-direction random null.
- **Readout:** the probe probability `σ(vᵀh + b)`. The logit is stored alongside, because the probability saturates and some nulls become degenerate. Those cases are flagged.

An old-pairing control run on identical code separates logging effects from refit effects. See [A_consistency.md](outputs/camera_ready/A_consistency.md).

---

## Camera-ready results

These are summaries. Every number comes from a JSON file written by a script; none is typed by hand.

### What holds

- **C_t adds predictive value for imagination error beyond {KL, Rec, EMARec}.** This is Table 4 ΔR² for `E_{t,10}`. It holds in-sample and out of sample (grouped 5-fold CV), on all five tasks:

  | | cartpole | reacher | pendulum | walker* | cheetah* |
  |---|---|---|---|---|---|
  | ΔR², in-sample | +0.0168 | +0.0053 | +0.0022 | +0.0021 | +0.0113 |
  | ΔR²_cv, out of sample | +0.0168 | +0.0101 | +0.0021 | +0.0019 | +0.0113 |

  \*New tasks, added for the camera-ready (Task 4).

- **Steering `h_t` along `v` moves the readout at k=0 but not imagination error.** Readout z is +10.6 / +11.1 / +11.4 against the 50-direction null; max |z| on E is 0.59 over 13 models.
- **v sits outside the high-variance subspace** on 12 of 13 models.

### What changed or failed (reported as is)

- **Steering persistence at k=10:** the readout effect is below z = 2 on 8 of 13 models. It holds on cartpole (4 of 5 models) only.
- **Walker:** steering also moves imagination error (E z = +2.30). The readout/E dissociation is therefore task-dependent.
- **"C_t ahead of current KL"** depends on the readout definition:

  | Readout | cartpole | reacher | pendulum | walker | cheetah |
  |---|---|---|---|---|---|
  | Probability | 5/5 | 3/4 | 3/4 | no | no |
  | Logit | 1/5 | 0/4 | 1/4 | — | — |

  Cells show the number of models where C_t is ahead.
- **SAE atoms:** ablating selected atoms changes imagination error in 8 of 9 runs. The atom that moved the readout, and the KL-independent atom, do not reappear.
- **Partial observation:** the signal does not strengthen on every pair. Probe AUROC is higher under partial observation on 2 of 5 pairs.
- **Sign flips:** reacher within-bin r(Rec, C_t) goes from −0.09 to +0.29, and pendulum's external scramble z goes from −11.7 to +21.0.

### Reviewer tasks

| Task | Question | Result | Write-up |
|---|---|---|---|
| 1 | Does the readout predict E directly? | Yes beyond the controls, but its ΔR² does **not** beat random directions in `h_t` on any task (logit null) | [T1_readout.md](outputs/camera_ready/T1_readout.md) |
| 2 | Readout and C_t vs six standard uncertainty signals | Readout adds value beyond all others on 3/3 in-sample. C_t is clearly above zero only on pendulum. All increments ≤ 0.015 | [T2_comparison.md](outputs/camera_ready/T2_comparison.md) |
| 3 | Is it useful for a decision (risk–coverage)? | **Pre-registered rule fails on 3/3.** Lagged Rec/EMARec rank sites better than the readout or C_t | [T3_decision.md](outputs/camera_ready/T3_decision.md) |
| 4 | Does it generalise to new tasks? | Partly: walker-walk and cheetah-run (see above) | [T4_generality.md](outputs/camera_ready/T4_generality.md) |
| CV | Out-of-sample ΔR² for rows of Tables 4 / T1 / T2 | Table 4 row holds on 5/5 tasks. Other rows are mixed | [CV_delta_r2.md](outputs/camera_ready/CV_delta_r2.md) |

---

## Setup

```bash
pip install -r requirements.txt     # torch, dm_control, scikit-learn, numpy, matplotlib, scipy
```

Use Python 3.11 on CPU. On Apple Silicon, CPU is faster than MPS at this batch size.

**Model:** Mini-DreamerV3 XS. It has a 256-d GRU state `h_t`, a 32×32 categorical `z_t`, about 12M parameters, and is trained for 100k steps on vector observations.

**Tasks:** dm-control cartpole-swingup, reacher-easy and pendulum-swingup, plus walker-walk and cheetah-run for Task 4.

**Episode splits:** fixed seeds for every task.

| Split | Episode seeds |
|---|---|
| Probe-fit | 10000–10099 |
| Calibration | 20000–20049 |
| Evaluation | 30000–30099 |

---

## Reproducing the camera-ready pipeline

Run these from the repo root, in order. Every script is seeded. Outputs go to `outputs/camera_ready/` unless noted. The canonical null (`outputs/canonical_null/random_directions_50_dim256.npz`) is reused, never regenerated.

```bash
# 0. tests: action pairing, logging matches the old pipeline, determinism
python -m pytest -q tests/test_camera_ready.py

# 1. corrected logging + per-model analyses (13 models + 10 partial-observation models)
python run_cr_A_log.py
python run_cr_A_analysis.py
for m in steer noisy pomdp taskg; do python run_cr_A_causal.py --mode $m; done   # steering, noisy steering, POMDP ablation, Task G null
python run_cr_A_report.py          # A_consistency.md

# 2. SAEs retrained on corrected states (3 runs per task)
CR_PROBE=frozen python run_cr_B_sae.py

# 3. final numbers under the refit rule
python run_cr_final_numbers.py     # camera_ready_numbers.json, fig_data.json
python run_cr_numbers_table.py     # camera_ready_numbers.md

# 4. reviewer Tasks 1–3
python run_t123.py                 # T1_readout, T2_comparison, T3_decision

# 5. Task 4: train and analyse two new tasks (several CPU hours)
python run_t4_train.py walker walk     # outputs/t4_generality/walker_walk/
python run_t4_train.py cheetah run    # outputs/t4_generality/cheetah_run/
python run_t4_analyze.py           # T4_generality.json

# 6. out-of-sample ΔR²
python run_cv_delta_r2.py
python run_cv_delta_r2.py --t4
python run_cv_report.py            # CV_delta_r2.md

# 7. figures (plotting only; every plotted value is checked against the JSON)
python paper/figures/src/make_cr_figures_final.py   # outputs/camera_ready/figures_final/
```

Relevant environment variables:

- `CR_CONVENTION=buggy` reruns logging with the old pairing, as a control.
- `CR_PROBE` selects `paper` or `frozen` probe handling.
- `CR_STATE_DIR` and `CR_RAW_DIR` redirect outputs.

---

## Repository map

### Camera-ready (current)

| Path | Contents |
|---|---|
| `src/camera_ready/core.py` | Corrected logging, episode splits, imagination error `E_{t,K}`, model registry, probe refit |
| `src/camera_ready/causal.py` | Steering / ablation runner, canonical null, z-scores |
| `run_cr_*.py`, `run_t123.py`, `run_t4_*.py`, `run_cv_*.py` | Camera-ready pipeline (above) |
| `paper/figures/src/make_cr_figures_final.py` | Final figures |
| `tests/test_camera_ready.py` | Pairing, logging and determinism tests |
| `outputs/camera_ready/` | Final numbers, task write-ups, figures |

### Paper

| Path | Contents |
|---|---|
| `paper/icaart_main.tex` | Paper source (pre-camera-ready numbers) |
| `paper/fill_paper.py` | Fills `\res{…}` macros from result JSON |
| `paper/PAPER_AGENT_HANDOFF.md` | Notes for writing the paper text |

### Shared library

| Path | Contents |
|---|---|
| `src/model/`, `src/training/`, `src/env/` | World model (encoder, RSSM, decoder), trainer, dm-control wrappers |
| `src/probe/` | Probe training, `C_t`, intervention helpers |
| `src/rerun/` | Pre-acceptance "sprint" pipeline: statistics (`GramSet` bootstrap), site masks, EMA |

### Legacy

Pre-acceptance experiments (Tasks A–T, Phases 1–6, the width sweep) are in `run_task_*.py`, `run_phase*.py` and `run_sprint_*.py`, with write-ups in `outputs/deliverables/`, [RESULTS_CONSOLIDATED.md](RESULTS_CONSOLIDATED.md) and [DEV_LOG.md](DEV_LOG.md). These results use the old action pairing and the original probe. Where they disagree with `outputs/camera_ready/`, the camera-ready numbers take precedence.

**Superseded files, kept for the record:**

- `outputs/camera_ready/figures_SUPERSEDED/`
- `outputs/camera_ready/*_oldpipeline_SUPERSEDED.*`
- `A_raw_superseded_probonly/`

---

## Notes on `outputs/`

`outputs/` is git-ignored because it holds checkpoints, logged states and raw arrays. The small camera-ready deliverables are force-added so the links above resolve:

- the `.md` and `.json` files in `outputs/camera_ready/`
- `outputs/camera_ready/figures_final/`

Checkpoints and logged states are not in git.

---

*Author: Donald Aadithiyan, University of Moratuwa.*
