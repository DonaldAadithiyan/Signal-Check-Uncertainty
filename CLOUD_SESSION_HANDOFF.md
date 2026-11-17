# Cloud Session Handoff — Signal-Check-Uncertainty

**Written:** 2026-09-27, from the local Claude Code session on the user's M4 MacBook Air.
**For:** the cloud Claude Code session that continues this work. You start with no memory of what happened before. This file is that memory. Read all of §0 and §1 before touching anything.

---

## 0. Read this first

### 0.1 What the cloud clone contains, and what it doesn't

The repo is `github.com/DonaldAadithiyan/Signal-Check-Uncertainty`, branch `main`. At handoff, HEAD was `b6d604d` ("Add scripts to generate figures for the paper"). The working tree was clean and pushed.

**`outputs/` is gitignored and is 13 GB locally.** A fresh clone therefore contains:

- all code (`run_*.py`, `src/`, `kaggle/`, `paper/figures/src/`)
- the finished figure PDFs in `paper/figures/`
- the Kaggle packages (`kaggle/taskB/`, `kaggle/taskB_kaggle_inputs.zip`, `kaggle/task12/`)
- `DEV_LOG.md`, `RESULTS_CONSOLIDATED.md`, `PAPER.md`, `README.md`

The clone does **not** contain:

- any results JSON (`outputs/phase*/`, `outputs/causal/`, `outputs/multiseed*/`, `outputs/taskA_regime_detection/`, ...)
- the deliverable write-ups in `outputs/deliverables/*.md` (about 45 files; they are the project's main written record)
- the trained checkpoints (`outputs/checkpoints/`, `outputs/second_env/`, `outputs/third_env/`, `outputs/multiseed*/`)
- `outputs/canonical_null/random_directions_50_dim256.npz`. A copy is tracked at `kaggle/task12/canonical_null/random_directions_50_dim256.npz`, and it is byte-identical to the one used everywhere.
- the saved training activations (`outputs/data/training_states.npz` and the per-env `*_training_states.npz`)

**Consequences:**

- The figure scripts in `paper/figures/src/` will **fail** in the cloud unless the results JSON is present. They read from `outputs/`.
- `run_taskB_reemergence_analysis.py` needs the Task B cell directories, which will come back from Kaggle.
- Anything that loads a frozen world model needs the checkpoints.

**If the user has not already done so**, ask them to run this locally before relying on the cloud clone. It pushes about 6.5 MB of text results and deliverables, and no checkpoints:

```bash
cd "~/Desktop/Work/Personal Learn Dev/Signal-Check-Uncertainty"
find outputs \( -name "*.json" -o -name "*.md" -o -name "*.csv" \) \
     -not -path "outputs/taskB_training_necessity/*" -print0 | xargs -0 git add -f
git add -f outputs/canonical_null/random_directions_50_dim256.npz
git commit -m "Track small results JSON, deliverables and canonical null for cloud sessions"
git push
```

Do not commit or push on the user's behalf unless they ask. Before anything is published, check whether the repo is public.

### 0.2 How this user works

- They are terse: "keep going", "whats happening". They prefer action to explanation, and short, direct status reports.
- "Keep going / don't wait" means run the whole sequence without stopping at check-ins.
- "Don't run anything" means **stop immediately and kill anything you started.** This happened once in the previous session. The user wanted an assessment of a Kaggle log, not new compute, and I had launched a local reacher run that had to be killed.
- They run the long jobs themselves on **Kaggle**. You cannot run Kaggle notebooks. Hand off packaged notebooks and zips instead.
- Commits: only when asked. The user commits and pushes themselves.
- Reporting standard (non-negotiable, applied across the whole project):
  - every number ships with N and a CI, sd, or null percentile
  - negative results are reported as plainly as positive ones
  - existing results are never softened or altered
  - ship rules are fixed before results are seen
  - when a "must match" value doesn't match, stop and report it; don't adjust the number or the plot

### 0.3 Technical ground rules

- Python is **`python3.11`** (pyenv 3.11.6). Locally: torch 2.12.0, numpy 2.4.2, sklearn 1.8.0, dm_control present. In the cloud, `pip install -r requirements.txt` (torch, dm_control, scikit-learn, numpy, matplotlib, scipy). dm_control needs MuJoCo, which comes via the `mujoco` pip wheel.
- **CPU, not GPU/MPS/TPU**, for this model. Batch 8 × seq 16, a small GRU, and one env step per gradient step: accelerators lose to kernel-launch and transfer overhead. Measured locally: CPU about 28 steps/s vs MPS about 14.
- Seeding convention (the "RNG-bug audit"):
  - Every stochastic path is seeded: `torch.manual_seed`, `np.random.seed`, env seeds, and **`random.seed`** (see §3.4).
  - RSSM latent sampling draws from torch's global RNG, so seed once at the top of each pipeline, in a fixed call order.
- Precondition checks are **hard stops** (raise, don't log and continue). This is the `assert_balance_ok()` lesson from Task 11.
- Local shell was zsh, where `echo ===` fails ("== not found"). Use `echo "---"`.

---

## 1. The research programme in one page

**Paper:** "Endogenous Predictive-Difficulty Representations in Recurrent World Models", earlier titled "Reading the Room". An earlier version was accepted at the IJCAI-ECAI 2026 GlobalSouthAI workshop (reviews 5/6/7).

**Current target:** **ICAART 2027** (CORE B, two-column SCITEPRESS, abstract ≤200 words).

- The user's latest revised `icaart_main.tex` compiles to 8 pages, with a 184-word abstract. It is **not in the repo and not found on disk** at handoff. The newest file in `~/Downloads/icaart_main.tex` is the older 6-figure version.
- An earlier planning doc named ICML 2027 (abstract Jan 16, paper Jan 22, 2027) with a **hard scope freeze on Nov 30, 2026**. The freeze still stands as the date after which there is no new experimental work.

**Model:** Mini-DreamerV3 XS RSSM.

- 256-d GRU deterministic state `h_t`, 32×32 categorical `z_t`, about 12M params.
- Trained 100k env steps with a uniform-random policy on dm_control **cartpole-swingup, reacher-easy, pendulum-swingup**.
- There is no policy or critic anywhere in the pipeline.

**Key objects:**

- **`v`:** a logistic probe on `h_t` predicting KL_t > median, unit-normalized in raw h space (`src/probe/intervention.py: probe_direction`).
  - `v` is **never saved to disk**. By convention every script refits it from that model's own `training_states.npz`: 60/40 stratified split, `random_state=0`, `train_probe` with C=1, lbfgs, standardized.
- **`C_t = Σ γ^i 1[KL_{t−i} > median]`**, with γ = 0.95 (cartpole), 0.70 (reacher), 0.90 (pendulum).
- **`E^state_{t,K}`:** mean L2 distance between decoded imagined and real observations over K steps.
  - Imagination replays the real action sequence.
  - Measured on 100 held-out trajectories, 6000 sites per task.

**Four closed pillars. Nothing may change these:**

| Pillar | Result |
|---|---|
| Emergence | Readout ≈ C_t: R² 0.80 cartpole vs 0.52 for current KL alone. `v` sits ~88° from the top-50 PCs. Temporal-scramble z ≥ +18.7 on all 3 tasks. |
| External validity | Incremental ΔR² of C_t over KL/Recon/EMARecon (+ensemble disagreement on cartpole only), K=10: cartpole +0.0280 [+0.0154,+0.0433]; reacher +0.0182 [+0.0083,+0.0309]; pendulum +0.0019 [+0.0001,+0.0051]. Trajectory-stratified bootstrap, 1000 resamples. |
| Mechanism | `v` is diffuse under a TopK SAE (256→2048, k=48, 3 SAE seeds, unanimous). Sparse atoms reacher #612 and cartpole #156 have distinct causal roles. |
| Causal dissociation | Steering along dense `v` moves the readout on every seed (z +5.2..+19.2) but not E^state (max z 1.48). #612 moves E^state only (z +2.67). Ablating #156 moves both (E^state +10.3, readout −5.0). |

**The pendulum pattern:** pendulum repeatedly behaves as the exception. It has the weakest ΔR², Set C inverts, and the scrambled C_t predicts E^state better than real order (z −11.7). The revised paper frames this as task dependence, not failure.

**Project history.** Details live in `DEV_LOG.md` (about 1900 lines), `RESULTS_CONSOLIDATED.md`, and `outputs/deliverables/`.

- **Phase 1b–1f (Tasks A–T):** causal hardening, multiseed, third environment, SAE, routing, and more. All complete.
- **"Reframed" Phases 0–6:** asset verification, external validity (Gate 1 passed 3/3), rigor controls, SAE, POMDP, filtering theory, causal steering. Also a hardening addendum, which:
  - retracted the Phase 6 "pendulum inverts" causal claim (it was an artifact of rank-correlation p-values)
  - restricted the KL-non-redundant atom claim to reacher only
- **Usefulness search:** Tasks 8–12 and the Task 11 extension. Every attempt to use the signal operationally failed against cheap baselines. The lesson: failure-predictive information is *distributed* across `h_t`, not concentrated in `v`.

---

## 2. The Final Bounded Usefulness Search (handoff of 2026-09-24)

The user supplied a handoff with three tasks: close Task 12, run Task A, run Task B.

- **Hard stops:**
  - no new detector after Task A
  - no new constraint design after Task B
  - no new tasks or environments added to rescue a weak result
  - no new downstream applications (routing, acquisition, replay, steering, red-teaming, repair, adaptive horizon)
- A failed ship rule is filed as a boundary condition in the paper and triggers no follow-up.
- The revised paper has since **removed all practical-use attempts**, so Task A and Task B outcomes matter only as limitations or extensions.

### 2.1 Task 12: CLOSED, documentation only

`outputs/deliverables/task_12_active_acquisition.md` records the run as "STOPPED EARLY — NO-SHIP on completed task; not a full run."

| Task | Status | Result |
|---|---|---|
| Cartpole | complete | `ct` 0.4681 was worse than `random` 0.4515, `kl` 0.4491 and `recon` 0.4476. Null z = +0.15. NO-SHIP. |
| Reacher | partial | Main conditions done; null sweep reached 12/50 before stopping. `ct` was worse than baseline and random. No verdict computed. |
| Pendulum | not run | — |

The Sekar reacher value (0.6921, about 2× everything else) is recorded as an uninvestigated anomaly. Do not cite it. The code lives in `kaggle/task12/` and is unmodified.

### 2.2 Task A: regime-change detection. CLOSED, NO-SHIP

- **Question:** does `s_t = v^T h_t` detect a dynamics shift earlier than cheap sequential detectors at the same false-alarm rate (FAR)?
- **Script:** `run_taskA_regime_detection.py`, inference only, about 7 minutes on CPU.
- **Outputs:** `outputs/taskA_regime_detection/{task}_results.json` and `outputs/figures/taskA_delay_vs_far.png`.
- **Deliverables:** `outputs/deliverables/task_A_regime_detection_preflight.md` and `task_A_regime_detection.md`.

**Preflight.** Task 10's logs were unusable: it ran open-loop imagination after the shift and saved only aggregate scalars. So fresh `observe_step`-every-step rollouts were run.

**Protocol.**

- Shifts are Task 10's: gravity ×2 for cartpole and pendulum, joint damping ×8 for reacher.
- Episodes are 250 steps with the shift at t*=150. Burn-in is 20 steps, H=20, and undetected delays are censored at 100.
- Seeds: calibration 20000–20099 (no shift). Evaluation 30000–30099 (shift) and 31000–31099 (no shift).
- Models: 5 cartpole (`outputs/multiseed/seed_{0..4}`), plus 4 each for reacher and pendulum. Those 4 are the original model in `second_env`/`third_env` plus `multiseed_env/{task}_seed{1,2,3}`.
- Detectors:
  - primary: `s_t`
  - secondary: closed-form C_t
  - baselines: KL, Recon, EMARecon, CUSUM(KL), Page-Hinkley(KL)
  - null: 50 canonical directions, sign-oriented on calibration data

**Results.**

- **Cartpole: UNINFORMATIVE on all 5 seeds.** No detector beat chance; detection probability was about 0 everywhere. It is excluded from the ship decision, as the protocol requires.
- **Reacher:** EMARecon detects in about 7–9 steps (dp 0.88–0.93). `s_t` essentially never detects (delay about 97). Delay difference `s_t` − EMARecon is +88 to +91 steps, CI excluding zero, on every seed.
- **Pendulum:** `s_t` is worse than the best of Recon/KL/CUSUM/PH on all 4 seeds (+8 to +39 steps, CIs exclude zero). `s_t` does beat the random-direction null on seeds 2 and 3 (z +4.2, +3.9).
- Realized FAR at N=100 calibration episodes was noisy for some detectors.
- **Verdict: 0/2 informative tasks pass.** Filed as a boundary condition: the temporal memory explains, but it is not a better change detector.
- **Known small quirk:** the preflight flag `kl_rise_clear` compares the post-shift mean to the pre-shift mean + 1 SD. That is too strict for cartpole. The decisive check, "any detector beats chance", is the right one, and it is also false for cartpole.

---

## 3. Task B: training-time necessity. IN PROGRESS on Kaggle. This is the live item.

### 3.1 Question and design (fixed before any result)

- **Question:** is the predictive-difficulty representation necessary for learning, or a by-product?
- **Conditions,** each trained fresh through an identical code path:

| Condition | What it does |
|---|---|
| **A_unconstrained** | Projection hook disabled. |
| **B_vconstrained** | `h̃ = h − v_t(v_tᵀh)` applied immediately after the GRU output, so every consumer sees `h̃`: the next GRU step, the prior head, the posterior head, and the decoder. There is no reward head. `v_t` is stop-gradient. |
| **C_randomdir** | The same projection against a fresh random unit vector (RNG seed = training seed + 777), on the same schedule. |

- **Schedule:** no constraint for the first 10,000 env steps. After that, refit every 5,000 env steps on the most recent 20,000 logged `h_t`, with labels `KL > median` of that window and the standard probe settings.
  - The logged `h_t` is the training loop's own online-inference log, i.e. the same thing as `training_states.npz`.
  - Every `v_t` and the cosine between consecutive refits are logged to `v_history.json`.
- **Grid:** 3 tasks × 3 conditions × seeds **3101, 3102, 3103** = 27 runs × 100k env steps.
- **Declared fallback:** cartpole only (9 runs). It must be decided *before* the full run and not switched mid-run. So far the full 27 are being attempted, which is the recorded decision.
- **Primary outcome:** E^state at K=10 on held-out evaluation seeds 40000–40029, never used for `v_t`.
  - Also measured: K ∈ {1,5,10,20} and next-step reconstruction.
  - "Worse" means the same sign on all 3 seeds **and** a pooled bootstrap CI (1000 draws, seeded) that excludes zero.
- **Re-emergence.** Run on each final B model; **all four** criteria must hold, measured against condition A:
  1. new probe AUROC ≥ A mean − 0.03
  2. R²(h̃ → C_t) ≥ A mean − 0.05
  3. the new probe clears the canonical 50-direction null on the same model at z > 2
  4. the Phase 1 incremental R² of the new probe's C_t over KL/Recon/EMA has a CI excluding zero
- **Classification per task:**

| Outcome | Criteria |
|---|---|
| Necessity | B worse than A, B worse than C, and re-emergence fails |
| Substitution | B not worse than A, and re-emergence passes |
| Nonessential | B not worse than A, and re-emergence fails |
| Nonspecific damage | B and C both worse than A, and B not worse than C |
| Mixed | results differ by task; never pooled |

- **Ship rule:** Necessity or Substitution on ≥2/3 tasks (or on cartpole under the fallback). Otherwise NO-SHIP, which becomes a limitations paragraph on "load-bearing" language. If NO-SHIP, there is no redesign of geometry, projection point, refit schedule, or seeds.

### 3.2 Files

| File | Purpose |
|---|---|
| `src/model/constrained_rssm_hook.py` | `ProjectionHookedGRU` wraps `rssm.gru` in place; the base RSSM is unmodified. `install_projection_hook(rssm)` returns the hook, and `hook.set_direction(v)` updates it (None means passthrough). |
| `run_taskB_training_necessity.py` | Local runner. Flags: `--pilot` (cartpole seed 3101, all conditions), `--tasks`, `--seeds`, `--conditions`. Skips cells whose `model.pt` exists. Writes `model.pt`, `states.npz`, `loss_history.json`, `v_history.json` per cell, plus `manifest.json`. |
| `run_taskB_repro_check.py` | Trains the first 15k steps of condition B twice and checks byte-identical h/kl/recon. |
| `run_taskB_reemergence_analysis.py` | Outcome, re-emergence and classification analysis. **Not integration-tested; has known gaps (§3.6).** |
| `kaggle/taskB/taskB_program.py` | Kaggle port of `train_condition`. **A duplicate of the local training loop: keep the two in sync.** Adds `cell_result.json` per cell (the resumability key), `set_scale()`, and `run_all(tasks, conditions, seeds, total_env_steps)`. Output dir comes from env var `TASKB_OUT`, default `/kaggle/working/taskB_results`. |
| `kaggle/taskB/taskB_kaggle_notebook.ipynb` | Cells: setup (`INPUT_ROOT = /kaggle/input/taskb-signal-check-inputs`; edit to match the dataset slug), smoke test, pilot, repro check, full run (`FULL_TASKS`), manifest. |
| `kaggle/taskB/README.md` | Upload and run steps, the timing, the RNG bug note. |
| `kaggle/taskB_kaggle_inputs.zip` | Built from `kaggle/taskB/` (src + program + README), excluding the notebook. Rebuild it after any code change. |
| `outputs/deliverables/task_B_training_necessity_pilot_status.md` | Status write-up (local only unless force-added). |

The cell directory name is `{task}_{condition}_{seed}`, for example `cartpole_B_vconstrained_3101`.

### 3.3 Pilot: PASSED on Kaggle (all 4 checks)

Source: the Kaggle log the user pasted on 2026-09-25.

| Check | Result |
|---|---|
| Constraint effectiveness | `|vᵀh̃|` = 2.75e-8 (B) and 4.48e-8 (C), measured on steps after the last refit that governs any logged step. Machine zero. |
| Stability | Final losses 32.0012 (A), 32.0012 (B), 32.0017 (C). 200 episodes, 99,001 gradient steps each. |
| Reproducibility | 15k-step rerun of B was byte-identical for h, kl and recon (resid 7.34e-8 both times). |
| Refits | 19 per constrained condition, matching the expected (100k−10k)/5k + 1. The cosine series is in `v_history.json`, not yet summarized. |

Wall time per 100k-step cell on Kaggle CPU was **172–186 min**. Locally it was about 135 min per cell, uncontended.

### 3.4 The bug found and fixed during the pilot (important)

The first local repro check **failed**. h, kl and recon diverged at exactly step 1000, which is `cfg['warmup_steps']`, the first gradient step.

- **Cause:** `EpisodeReplayBuffer.sample()` (`src/training/replay_buffer.py`) uses Python's global **`random`** module (`random.choice`, `random.randint`). Neither `torch.manual_seed` nor `np.random.seed` seeds it.
- **Fix:** `random.seed(seed)` at the top of `train_condition`, in both the local and the Kaggle program.
- **After the fix:** the check passes, locally and on Kaggle.
- **The same gap still exists, unfixed, in `src/training/trainer.py` and `run_multiseed_env.py`.** Those scripts never claimed training-loop byte-reproducibility; the RNG audit covered only inference on frozen models. So this is a newly surfaced requirement, not a regression.
- If you ever retrain elsewhere and need reproducibility, add `random.seed`.

### 3.5 Full-run status and compute

- **Last observed (2026-09-25 log):**
  - cartpole A/B/C for seed 3101 are done (the pilot cells, reused by the resumable run)
  - `cartpole_A_unconstrained_3102` was at 60k/100k, about 11.8 h into the Kaggle session
- The current state is unknown. Ask the user for the latest log, or for the downloaded `taskB_results/`.
- Kaggle sessions cap at about 12 h. **A cell interrupted mid-run is lost and restarts from zero** (cells are atomic). Completed cells are skipped on restart via `cell_result.json`. Expect about 3–4 cells per session.
- About 24 cells remain × about 180 min ≈ **72 h serial.**
- **Switching Kaggle accounts restarts from scratch** unless the finished `taskB_results/` is uploaded as a second input dataset and copied back into place first:

  ```python
  shutil.copytree('/kaggle/input/<results-slug>/taskB_results', '/kaggle/working/taskB_results', dirs_exist_ok=True)
  ```

  Simpler: split the grid across accounts, e.g. `FULL_TASKS=['reacher']` on one and `['pendulum']` on another.
- **TPU: advised against.** The user has 20 TPU-hours per week. The code is PyTorch without `torch_xla`, and the workload is batch 8, a tiny GRU, and one env step per gradient step, so a TPU would be idle or slower.
- **Recommended speed-up: parallel cells.** Each run uses about 1.2 cores and Kaggle CPU sessions have 4 vCPU, so running about 3 cells concurrently gives about 3×.
  - I offered to add a parallel mode (N worker processes over the cell list, each still writing its own `cell_result.json`). **It is not implemented yet.**
  - If the user asks for it: implement it in `kaggle/taskB/taskB_program.py` with `multiprocessing` (spawn context; each worker imports torch and sets `torch.set_num_threads(1)`), then rebuild the zip.
- The local disk has one extra finished cell, `outputs/taskB_training_necessity/cartpole_A_unconstrained_3101/` (final loss 32.0014, which matches Kaggle's 32.0012). It has no `cell_result.json` because the local runner doesn't write one. It is a cross-check only; prefer the Kaggle cells.

### 3.6 Known gaps in `run_taskB_reemergence_analysis.py`. Fix before trusting its output.

It imports cleanly but has never run on real results.

1. **Criterion 4 is not implemented.** `reemergence_overall` uses only criteria 1–3. This must be implemented (the Phase 1 incremental R², with a bootstrap CI, of the new probe's C_t against E^state(K=10) on the evaluation trajectories) before any classification is reported.
2. `ema_alpha` is hardcoded to 0.1. Use `fit_ema_alpha` from `run_phase1_external_validation.py` on the calibration/training states, which is the project convention.
3. **The new probe is fit on all logged states, including the first 10,000 unconstrained steps.** Restrict it to post-warmup (constrained) states, or re-emergence will be contaminated by unprojected `h`. The same applies to criterion 3's null and to the angle-to-running-`v_t` check.
4. `bootstrap_diff_ci` resamples the two conditions independently over pooled evaluation sites. That is acceptable as specified ("pooled bootstrap over evaluation sites"), but state it in the deliverable.
5. `R²(h̃→C_t)` is computed as corr(probe score, C_t)². That matches the "readout vs C_t" convention. The spec says `R²(h̃_t → C_t)`, which Phase 2 computes as a ridge regression from full `h`. Pick one, apply it identically to A and B, and state which.
6. Condition-A reference AUROC and R² are computed per seed from the A cells. Fine.

Then write the final `outputs/deliverables/task_B_training_necessity.md` with:

- pilot results
- the fallback decision
- per-seed tables
- the re-emergence table
- classification per task
- ship verdict
- caveats, including the RNG bug and the §3.6 fixes

---

## 4. Paper figures (done; `paper/figures/`)

The user asked for the figures only; no LaTeX edits. The spec: single-column 7.50 cm wide, fixed heights, vector PDF, Times, Okabe–Ito colours, markers for greyscale, no titles, and must-match values asserted programmatically.

**Current filenames, after the user's renumbering to match the revised tex outline:**

| File | Size (cm) | Script |
|---|---|---|
| `fig1_emergence.pdf` | 7.5 × 6.4 | `src/make_fig2.py` |
| `fig2_external.pdf` | 7.5 × 7.6 | `src/make_fig3.py` |
| `fig3_mechanism.pdf` | 7.5 × 5.6 | `src/make_fig4.py` |
| `fig4_dissociation.pdf` (hero) | 7.5 × 7.2 | `src/make_fig5.py` |
| `fig1_overview.pdf` | 7.5 × 6.8 | `src/make_fig1.py`. **Possibly unused now**; the user was asked whether to delete it. |
| `fig6_filtering.pdf` | 7.5 × 4.6 | `src/make_fig6.py`. **Possibly unused now.** |

- Script numbers don't match the new figure numbers; the docstrings say "Fig N (was Fig M)". Heights were kept from the old spec because the revised tex wasn't available. **Confirm the heights against the revised `\figslot` lines.**
- `paper/figures/src/style.py` holds the shared style, `SIZES`, `check()` (raises on a must-match failure) and `save()`. Setting `FIG_PREVIEW_DIR` also writes PNG previews.
- `make_all.sh` rebuilds all six.
- `_cache_fig2a.npz` is tracked; it lets fig1_emergence(a) rebuild without the 1 GB `training_states.npz`.
- `__pycache__` got committed by accident and can be removed.
- Verified: sizes exact, all fonts embedded (TimesNewRoman, STIXGeneral, DejaVuSans for ✓/✗).
- The paper was **not compiled**: no LaTeX locally. SCITEPRESS template files are at `~/Downloads/SCITEPRESS_Conference_Latex/`.

**Must-match failures (plotted as the data says; the author decides).** In fig3_mechanism, the paper says the top-1 share is 0.3–0.5% and the top-4 share 0.9–1.2% "all tasks/seeds". Three runs fall outside:

- cartpole SAE seed 2: top-1 is 0.24%
- reacher SAE seed 1: top-4 is 1.31%
- reacher SAE seed 2: top-4 is 1.40%

**Data that was never saved, so panels are reduced:**

- mechanism (a): top-10 atom shares only, and no random-vector control
- mechanism (b): only the atom selected per SAE seed on the held-out half, not the full candidate pool
- filtering: mean ± sd only, no per-seed points
- emergence (a): points regenerated deterministically with `run_confusion_integral.py`'s own functions on the saved training states. This reproduces R² 0.7983 and KL 0.5187 exactly.

**Paper-text discrepancies found:**

- **Table 2 mixes model sets.** Cartpole 0.80 and all three angles are the original model; the 4-seed mean angles would be 89.31 and 88.46. Reacher 0.26 and pendulum 0.86 are 4-seed means; the single-model values are 0.216 and 0.886. Reacher and pendulum KL-only R² were saved for seed 0 only.
- **Fig emergence (d):** the statistic is ridge R²(h_t → C_t), not "variance of the readout".
- **"+5.2..+19.2"** pools k=0 and k=10: the minimum is a k=10 value (reacher seed 1) and the maximum a k=0 value (pendulum seed 2).
- **#156 was ablated,** not amplified as the text says.
- **Bootstrap** is by trajectory; the text says by site.

**VERIFY-tag answers:**

| Tag | Answer |
|---|---|
| E^state norm and actions | L2 in decoded observation space; real actions replayed |
| Headline K | 10 |
| Ensemble disagreement baseline | Cartpole only |
| EMA comparison target (+0.177/+0.049/+0.013) | E^state, not the readout |
| "Removal" | `h' = h − (h·d)d` along the atom's unit decoder column, re-encoded through the SAE: #612 0.0054→1.5e-6, #156 0.0038→~0 |
| k=10 steps after steering | Posterior updates (`observe_step` with real obs); E^state uses `imagine_step` |
| Sparse-atom null | 50 random unit directions (rng 3033), not random atoms |
| Bootstrap | 1000 resamples, trajectory-stratified |
| Kalman baselines | \|innovation\| and filter variance P |

**Data sources used by the figure scripts** (all under `outputs/`):

- `phase1_external_validation/phase1_results.json` and `addendum_1_1_1_2_results.json`
- `phase2_rigor_controls/phase2_results.json` (scrambling, EMA head-to-head)
- `third_env/three_env_comparison.json` and `multiseed_env/aggregate.json`
- `phase3_sae/phase3_results.json`, `addendum_2_1_2_2_2_3_results.json`, `phase3b_causal_results.json`, `phase3b_causal_results_atom156.json`
- `phase6_causal_steering/seed_extension/phase6_seed_extension_aggregate.json`
- `phase5_filtering_theory/phase5_results.json`

---

## 5. Repo map (the parts you'll touch)

- `src/model/rssm.py`
  - `observe_step` and `imagine_step` compute `h_next = self.gru(inp, h)` identically; only `z` differs (posterior vs prior). Repair can therefore only fix `z` (Task 10).
  - Also has `kl_divergence(post, prior, free_bits)` and `_straight_through_sample`, which uses the global torch RNG.
- `src/model/world_model.py`: encoder, RSSM and decoder; `compute_loss(obs, actions, kl_free, kl_scale)`. No reward head.
- `src/training/replay_buffer.py`: `EpisodeReplayBuffer`. **Uses the Python `random` module.**
- `src/probe/linear_probe.py`: `train_probe` (StandardScaler + LogisticRegression C=1 lbfgs `random_state=0`), `binarise_by_median`, `auroc`.
- `src/probe/intervention.py`: `probe_direction`, `compute_ct`, `random_matched_direction`, bootstrap helpers, and a GRU gate override.
- `src/env/dmc_wrapper.py` has `DMCEnv(domain, task, noisy, seed)`; `src/env/wrapper.py` has `CartpoleEnv`.
- `src/config.py`: `XS_CONFIG` (deter 256, stoch 32×32, seq_len 16, batch 8, warmup 1000, 100k steps, lr 3e-4, `episode_max_steps` 500, device cpu).
- `run_phase1_external_validation.py`: `load_model`, `collect_trajectories`, `imagined_vs_real_obs`, `e_state`, `fit_ema_alpha`, `ema_series`, the `TASKS` specs (checkpoint and `training_states` paths, `gamma_ct`).
- Checkpoints:
  - cartpole: `outputs/checkpoints/world_model.pt`
  - reacher: `outputs/second_env/reacher_easy_world_model.pt`
  - pendulum: `outputs/third_env/pendulum_swingup_world_model.pt`
  - multiseed: `outputs/multiseed/seed_{0..4}/world_model.pt` (+ `training_states.npz`) and `outputs/multiseed_env/{reacher,pendulum}_seed{1,2,3}/model.pt` (+ `states.npz`)
- Canonical null: `outputs/canonical_null/random_directions_50_dim256.npz` (key `directions`, shape (50,256), seed 14024). Always reuse it; never regenerate it.

---

## 6. What to do next, in priority order

1. **Get the Task B state from the user**: the latest Kaggle log, or the downloaded `taskB_results/`. Don't guess at pending results.
2. If the user wants faster Kaggle throughput: implement the parallel-cell mode (§3.5), rebuild `kaggle/taskB_kaggle_inputs.zip`, and hand it back. **Don't run Kaggle yourself.**
3. **Fix the gaps in `run_taskB_reemergence_analysis.py` (§3.6) now,** while waiting. Test it end to end on the one finished local A cell plus a small synthetic B cell. Use a tiny `train_condition` run with `set_scale`-style overrides if needed, and don't leave long runs going.
4. Once all 27 cells (or the declared 9) are downloaded into `outputs/taskB_training_necessity/`, run the analysis and write `task_B_training_necessity.md`. Apply the ship rule exactly as fixed; do not add seeds or metrics.
5. **Paper:** when the user shares the revised `icaart_main.tex`, confirm figure heights and filenames. Then flag the text discrepancies in §4 for the author. Don't edit the tex unless asked.
6. **Scope freeze: 2026-11-30.** After that, only reproducibility fixes.

---

## 7. Things not to do

- Don't redesign the Task B constraint, projection point, refit schedule, seeds or metrics after seeing results.
- Don't add new detectors, applications, tasks or environments to rescue a result.
- Don't regenerate the canonical null or refit `v` with different settings.
- Don't use a GPU, TPU or MPS for this model.
- Don't commit or push without being asked.
- Don't cite the Sekar reacher 0.6921 value.
- Don't average results away across tasks or seeds; report per task and per seed.
