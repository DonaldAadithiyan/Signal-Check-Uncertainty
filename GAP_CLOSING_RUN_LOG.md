# Gap-Closing Spec — Run Log

Live log of every task from `Task_Spec_Closing_Gap_To_Target_Claim.md` and its
addendum (`Pillar 4 Decision: Path A Committed, Path B Gated`), covering what
was run, every result/metric produced, and every finding — updated as work
lands. This is a working log, not a polished deliverable; the per-task
deliverables in `outputs/deliverables/` are the citable write-ups.

**Session start:** 2026-09-16 · **Latest update:** 2026-09-18, Task 8 added +
synthesis precision fix + Task 6/Path B paused mid-evaluation for a real
actor-collapse finding (investigation not yet concluded) · **Status:** Tasks
1, 2, 3, 4, 4a, 5, 8 all complete, plus the standalone Phase-3-sign diagnostic
and the full RNG-bug audit (Phases 2, 3, 6 + addenda fixed and re-verified)
and both carried-forward cleanup items (cartpole atom #156 re-run,
difficulty-matched FULL-vs-PARTIAL comparison). Task 6 (Path B) has 13
actor-critic policies trained but paused for an actor-collapse finding — see
its own section below, not resolved. Task 7 (learned correction, revised
design) remains NOT STARTED — both correctly gated (Task 6 on Tasks
1–5 closing with time/budget remaining **and the venue/timeline question**,
which is explicitly flagged as a decision for the user, not the coding agent).
See `GAP_CLOSING_FINAL_SUMMARY.md` for the consolidated final deliverable.

---

## RNG-Bug Scope Audit and Carried-Forward Cleanup

**Status:** Complete. Full write-up: `outputs/deliverables/rng_bug_audit.md`.

**Audit finding:** the only RNG-reproducibility issue anywhere in the
codebase is RSSM stochastic-latent sampling (`Categorical.sample()`), the
same bug class Task 1 fixed for Phase 1. No dropout exists in the model. One
`torch.randn` call (SAE decoder init) is training-time-only and moot for
already-trained, checkpoint-loaded SAEs. Confirmed present and unfixed in:
`run_phase2_rigor_controls.py`, `run_phase3_causal_features.py`,
`run_phase6_causal_steering.py`, `run_addendum_phase6_hardening.py`.
Confirmed **already correctly seeded** (no fix needed): `run_phase4_seed_
sweep_analysis.py`, `run_phase4_pomdp_analysis.py` (both from the prior
session's Task 1-pattern fixes). `run_addendum_phase3_hardening.py` confirmed
unaffected (no RSSM sampling calls at all).

**Fixed and re-verified (byte-identical across independent reruns), in
priority order:**
1. `run_phase4_seed_sweep_analysis.py` — already correct, verified, no
   change to Task 3's MIXED (3/6) verdict.
2. `run_phase3_causal_features.py` — bug confirmed (two pre-fix reruns
   differed), fixed, verified. Small magnitude corrections to §9.5–9.7's
   numbers, no verdict changes on reacher/pendulum.
3. `run_phase6_causal_steering.py` + `run_addendum_phase6_hardening.py` —
   bug confirmed, fixed, verified. Small magnitude corrections, no change to
   the "probe decisive / E^state null" verdict on any task.
4. `run_phase2_rigor_controls.py` — bug confirmed, fixed, verified. Moderate
   magnitude corrections (largest: cartpole's external z, 26.9→19.6), no sign
   or verdict changes on any of the three tasks.

**Carried-forward cleanup #1 — cartpole's retracted atom, resolved.** Re-ran
the causal ablation protocol on cartpole's split-sample-confirmed atom (#156,
replacing the retracted #139). **Result: #156 is the strongest causal result
in the entire project** — z=+10.3 on `E^state` and z=−5.0 on the probe
readout, the only atom to meet §9.6's original "simultaneous effect" bar.
Verified byte-reproducible. Propagated to `phase3_sae_decomposition.md` and
`task_5_causal_mechanism_synthesis.md`'s Claim 2.

**Carried-forward cleanup #2 — Task 3's difficulty-matched comparison,
resolved.** Built `run_phase4_difficulty_matched_comparison.py`: pools
FULL+PARTIAL sites, bins by KL quantile across the pooled distribution,
compares causal z-scores within matched bins across all 5 seeds. **Result:
PARTIAL is stronger in 2/3 matched bins (lowest and highest KL) but FULL is
stronger in the middle bin** — a KL-dependent, non-monotonic relationship,
not a clean confirmation or refutation of the raw aggregate gap. The
lowest-KL bin shows a striking sign flip (PARTIAL positive, FULL negative)
that is flagged as needing a closer look rather than built upon as-is (noisy,
smallest/most-variable sample of the three bins). Does not overturn the
MIXED (3/6) verdict.

**What did not change:** no previously-reported qualitative verdict in
Phases 1, 2, 3, or 6 flipped sign or significance as a result of this audit
— all corrections are magnitude-level (within expected stochastic-sampling
noise), except cartpole's atom-156 re-run, which is a genuine new, stronger
finding rather than a rounding correction.

---

## Task 1 — Reconcile pendulum's incremental-R² point estimate

**Status:** Complete. **Scope grew beyond the original ask** — this wasn't a
reporting/aggregation inconsistency, it was a genuine unseeded-RNG bug.

**What was run:**
- Diagnostic: compared `run_phase1_external_validation.py`'s saved JSON
  output against a fresh rerun; found the two disagreed (pendulum incremental
  R²: 0.001683 vs 0.001958) despite the pipeline appearing fully seeded
  (`random_state=0` on the probe split, `SEED=2024` for env/site RNG).
- Root-cause isolation: `collect_trajectories()` A/B tested with fixed inputs
  — same env seed, same model weights — produced different `h_t` arrays
  across two calls (`np.allclose` → False). Traced to
  `RSSM._straight_through_sample()` → `torch.distributions.Categorical.sample()`,
  which draws from PyTorch's **global, unseeded RNG**. `torch.manual_seed()`
  was never called anywhere in `run_phase1_external_validation.py`.
- First fix attempt (seed inside `collect_trajectories` only) was
  **insufficient** — `imagined_vs_real_obs()` (called ~6000×/task, after
  `collect_trajectories` returns) also samples via `imagine_step`, consuming
  more of the same unseeded global stream. Residual drift persisted at a
  smaller scale (pendulum 0.001958 vs 0.001924 across two "fixed" reruns).
- Final fix: `torch.manual_seed(SEED)` once at the top of `run_task()`,
  covering trajectory collection and every downstream imagination-rollout
  sampling call in one fixed sequential order. Same fix mirrored in the
  addendum script's `rebuild_site_dataset()` (needed its own `torch` import,
  which wasn't present).
- Verification: two independent reruns of `run_task('pendulum', ...)`
  produced **byte-identical** `incremental_r2_ct` (0.001945358002854669 both
  times). Full 3-task pipeline rerun, then addendum bootstrap rerun on top —
  addendum's independent reconstruction now matches the canonical run exactly
  on all 3 tasks.

**Results (canonical, reproducible, post-fix):**

| Task | r(C_t, E^state) @ K=10 | incremental R² | 95% bootstrap CI | Old value |
|---|---:|---:|---:|---:|
| cartpole | +0.608 (p=0) | **+0.0280** | [+0.0154, +0.0433] | was +0.0277 |
| reacher | +0.256 (p=1.5e-90) | **+0.0182** | [+0.0083, +0.0309] | was +0.0186 |
| pendulum | +0.174 (p=4.5e-42) | **+0.0019** | [+0.0001, +0.0051] | was +0.0006 / +0.0021 (inconsistent) |

Gate 1 verdict unchanged: PASS on all 3 tasks, CI excludes zero on all 3.

**Files updated:** `run_phase1_external_validation.py` (fix + docstrings),
`run_addendum_phase1_hardening.py` (fix + `import torch`),
`outputs/deliverables/phase1_external_validation.md` (all 3 internal tables +
new reproducibility note), `outputs/deliverables/phase2_rigor_controls.md`,
`outputs/deliverables/phase6_causal_steering.md`,
`outputs/deliverables/pendulum_outlier_synthesis.md` (cross-references to the
old number), `Reframed_AAMAS2027_Project.md` (§6.1 table + heterogeneity
block, both copies in Downloads).

**Finding flagged, not yet acted on:** the identical unseeded-RNG bug is
present in Phases 2, 3, 6, and their addenda (`run_phase2_rigor_controls.py`,
`run_phase3_sae_decomposition.py`, `run_phase3_causal_features.py`,
`run_addendum_phase3_hardening.py`, `run_phase6_causal_steering.py`,
`run_addendum_phase6_hardening.py` — none call `torch.manual_seed`). Not
re-run here (out of Task 1's literal scope, real compute cost) — **awaiting a
decision on whether to re-verify those phases too.**

---

## Task 2 — Confirm atom identity consistency between selection and causal test

**Status:** Complete.

**What was checked:** whether reacher's SAE atom used in the §9.5–9.7 causal
ablation test (pre-split-sample-correction selection) is the same feature as
the atom §9.3.1's split-sample procedure independently confirmed on held-out
data.

**Finding:** **Confirmed identical — atom #612 for reacher, both procedures.**
§9.3.1's held-out split-sample selection independently re-derives #612 as
reacher's top externally-predictive candidate; this is the same index used in
the causal ablation test. No re-run needed. Reacher's full evidentiary chain
is now closed without a gap: #612 is simultaneously (a) held-out
KL-non-redundant and externally predictive, (b) causally connected to
`E^state` beyond the 50-direction null (z=+2.6, 98th pct), and (c) removes its
own incremental-R² advantage on ablation (+0.0038→+0.0004).

**Explicitly out of scope, not silently ignored:** cartpole (#139) and
pendulum (#310) were causally tested on their now-retracted, pre-correction
atom selections, not their split-sample-confirmed replacements (cartpole's
replacement is #156). Re-running the causal test on #156 was not done —
flagged as a real open item for cartpole specifically, but outside what Task 2
asked to check (which was reacher-scoped by its own framing: "reacher was
already the strongest case").

**Files updated:** `outputs/deliverables/phase3_sae_decomposition.md` (new
explicit "Atom-identity check" subsection after §9.7).

---

## Task 3 — Phase 4 seed sweep and multi-step causal-protocol upgrade

**Status:** IN PROGRESS — training running in background, long pole of the
whole spec. Analysis script written and ready, not yet run.

**What was built:**
- `run_phase4_seed_sweep_training.py` + `run_phase4_seed_sweep_one.py` — reuse
  the original `train_condition()` unmodified (same architecture, optimizer,
  100K-step budget, velocity-masking scheme), redirected to per-seed output
  dirs so they never collide with the original seed=4242 pair. New seeds:
  1717, 2929, 5151, 8383 (fixed before any run, no cherry-picking).
- `run_phase4_seed_sweep_analysis.py` — full aggregation pipeline, with the
  two upgrades the spec requires:
  1. **Multi-step causal protocol**, replacing the original's single-step
     static probe-decay proxy: ablates `h_t`, then continues the trajectory
     via `imagine_step` for k∈{0,1,5,10} steps, re-measuring both the probe
     readout and the `E^state` metric from the actual continuation rather
     than an instantaneous re-score.
  2. **Difficulty base-rate / matched-bin check**: bins causal-test sites into
     KL terciles, repeats the ablation z-score within each KL-matched bin, to
     rule out "PARTIAL's larger effect is just because its sites are harder"
     as a confound.
- All 4 seed pairs' training launched concurrently as separate background
  processes (`OMP_NUM_THREADS=2` each, to fit 10 cores without oversubscribing
  alongside Tasks 1/4's concurrent CPU use at the time).

**Status:** COMPLETE. All 4 seed pairs finished training (full: ~230-235 min
each; partial: ~251-257 min each, slower than the original ~87min benchmark
due to concurrent compute load from other tasks running at the same time —
loss/KL curves verified consistent with the original benchmark before use).
Analysis run across all 5 seed pairs (original 4242 + 4 new).

**Results — aggregate across 5 seed pairs:**

| Metric | FULL mean±std | PARTIAL mean±std | Partial stronger? |
|---|---:|---:|:---:|
| Confusion AUROC | 0.871±0.028 | **0.951±0.013** | ✅ |
| R²(h_t, C_t) | 0.761±0.035 | **0.824±0.013** | ✅ |
| r(C_t, E^state) | **0.568±0.060** | 0.522±0.028 | ❌ |
| Incremental R² (C_t over KL+Recon) | **0.062±0.027** | 0.009±0.003 | ❌ |
| Causal z (probe, multi-step) | −6.93±2.27 | **−9.92±1.85** | ✅ |
| Causal z (E^state, multi-step) | −0.37±0.36 | 0.13±0.22 | ❌ |

**VERDICT: MIXED (3/6), revised down from the original single-pair
"SUPPORTS H6" (4/6).**

**The most important correction:** the multi-step-corrected causal effect on
genuine external imagination quality (`E^state`) is small and statistically
indistinguishable from zero **in both conditions** — the original single-pair
report's dramatic causal z-score doubling (−4.69→−10.58) used a simplified
single-step probe-decay proxy and does not survive the multi-step-continuation
upgrade. This retracts a headline claim, in the same spirit as Phase 6's own
addendum-driven retraction of its external-causal-effect claim.

**What survives robustly, multi-seed-confirmed:** confusion AUROC and
`R²(h_t,C_t)` — both stronger under PARTIAL on every seed, tight cross-seed
spread.

**New finding from the difficulty base-rate / matched-bin check:** PARTIAL's
evaluation sites are systematically higher-KL than FULL's at every matched
tercile, on every seed (e.g. seed 4242's high-KL bin: FULL mean KL 19.5 vs
PARTIAL's 43.5) — and the causal z-score scales strongly with local KL
*within* both conditions (often an order of magnitude between low- and
high-KL bins). This means part of the raw FULL-vs-PARTIAL causal-magnitude
gap is attributable to a KL/difficulty confound, not cleanly isolated to the
observability manipulation — a real qualification the original single-pair
report could not have surfaced.

**Files:** `run_phase4_seed_sweep_training.py`, `run_phase4_seed_sweep_one.py`,
`run_phase4_seed_sweep_analysis.py`,
`outputs/phase4_pomdp/seed_sweep/phase4_seed_sweep_results.json`,
`outputs/deliverables/task_3_phase4_seed_sweep.md`. Cross-references updated
in `outputs/deliverables/phase4_pomdp_stress_test.md` (superseded-status
banner), `outputs/deliverables/task_5_causal_mechanism_synthesis.md`
(cross-phase update section), and `Reframed_AAMAS2027_Project.md` (header +
progress note, both copies in Downloads).

---

## Task 4 — Distribution-shift test

**Status:** Complete.

**What was built:** `run_task4_distribution_shift.py` — reuses Phase 1's
`imagined_vs_real_obs`/incremental-R² pipeline unmodified, adds a
`noisy=True` env-construction path (`src/env/dmc_wrapper.DMCEnv` and
`CartpoleEnv` already support `noise_std=0.1` Gaussian observation noise,
matching the original cartpole-only Set A/B convention, generalized here to
all 3 tasks). Probe/scaler/EMA-α fit once on original clean data, never
refit per condition. Also reruns Phase 6's causal-steering dose-response
(λ∈{−2,−1,0,+1,+2}) on both conditions, with a reduced null (20 dirs/150
sites vs Phase 6's 50/600 — stated explicitly as a replication-check
cost/precision tradeoff, not a full-power rerun).

**Note:** this script was launched once, then killed and relaunched after
Task 1's seeding-bug fix was discovered — its first run's results were
pre-fix and discarded rather than reported.

**Results — incremental R² over KL/Recon/EMA, Set A (clean) vs Set B (σ=0.1 noise):**

| Task | R² (Set A) | R² (Set B) | Verdict |
|---|---:|---:|:---:|
| cartpole | +0.0279 | +0.0684 | **STRENGTHENS** |
| reacher | +0.0184 | +0.0087 | **DEGRADES** |
| pendulum | +0.0019 | +0.0259 | **STRENGTHENS** |

(Pendulum's Set A value, +0.0019, matches Task 1's independently-reconciled
canonical number exactly — a useful cross-check that Task 1's fix is solid.)

**Results — causal steering under shift, z-score at λ=+2 vs 20-direction null:**

| Task | z(probe), Set A | z(probe), Set B | z(E^state), Set A | z(E^state), Set B |
|---|---:|---:|---:|---:|
| cartpole | +3.47 | +3.22 | −0.25 | +0.27 |
| reacher | +10.44 | +9.52 | +0.81 | +0.03 |
| pendulum | +2.84 | +3.25 | +1.22 | −1.43 |

**Findings:**
- Probe-readout causal effect **replicates cleanly under shift on all 3
  tasks** (comparable magnitude both conditions).
- External `E^state` causal effect **remains null on all 3 tasks, both
  conditions** — no z-score exceeds |1.43|. Distribution shift neither
  resurrects nor introduces a causal effect on genuine imagination quality;
  Phase 6's null result replicates cleanly under shift.
- Correlational advantage is genuinely mixed: 2/3 tasks strengthen under
  noise (opposite of the naive "noise just adds noise everywhere" prior),
  reacher degrades — plausible mechanism offered (reacher's `r(C_t,KL)` is
  already highest at baseline; noise likely pushes more of the shared
  variance into KL directly, same ceiling-effect pattern Phase 4 found for
  partial observability).

**Verdict on the "...and distribution shift" claim:** earned, with an honest
task-dependent account rather than a uniform "yes" — the causal dissociation
(Claim 1 from Task 5's synthesis) is stable under shift; the correlational
advantage is not uniform across tasks.

**Files:** `run_task4_distribution_shift.py`,
`outputs/task4_distribution_shift/task4_distribution_shift.json`,
`outputs/deliverables/task_4_distribution_shift.md`.

---

## Task 4a — Pillar 4 write-up (Path A, committed)

**Status:** Complete.

**What was done:** written up the existing Task R (KL-only routing baseline,
Reviewer TAQ8 response) result as the committed Pillar 4, at the same
evidentiary standard as every other pillar. No new computation — Task R's
existing analysis already had the right rigor (task-dependent, multi-seed,
not averaged, honest about the unfavorable pendulum result).

**Results reproduced from Task R (unchanged, just reframed):**

| Task | seeds | Probe-A recall@30% | KL-only (prior-step) | Δ recall | probe wins? |
|---|---:|---:|---:|---:|:---:|
| cartpole | 5 | 0.626 ± 0.047 | 0.632 ± 0.068 | −0.006 ± 0.025 | 3/5, sign-unstable |
| reacher | 3 | 0.539 ± 0.013 | 0.267 ± 0.007 | **+0.273 ± 0.020** | 3/3 |
| pendulum | 3 | 0.729 ± 0.017 | 0.760 ± 0.023 | **−0.030 ± 0.006** | 0/3 |

**Framing applied:** explicitly scoped as a perception-triggering signal
evaluated by recall at fixed query budget, not a validated active-perception
method (no policy/critic exists in this pipeline to test task return).
Renamed away from any branded name implying a working method — used
"confusion-gated observation querying," descriptive per the spec's own
suggestion (could not locate the original "AAP-WM" name anywhere in this repo
or the AAMAS doc to know exactly what it referred to).

**Files:**
`outputs/deliverables/pillar4_confusion_gated_observation_querying.md`.

---

## Task 5 — Synthesis: the causal mechanism section

**Status:** Complete. Pure writing/reconciliation, no new computation.

**What was done:** consolidated Phase 3 and Phase 6's separately-reported
(and partially self-retracted) causal findings into one internally-consistent
account, incorporating Task 2's atom-identity confirmation.

**The consolidated finding:** the representation's causal role **bifurcates**
into two independent objects with opposite footprints:
1. **Dense direction `v`**: decisively, bidirectionally causal for the
   model's own confusion **readout** (z=+7 to +16 vs. 50-direction null, all
   3 tasks, both k=0 and k=10 lookahead) — but **no detectable causal effect
   on genuine external imagination quality** on any task (|z|<1.1 everywhere
   post-correction). Governs self-report, not demonstrated to govern
   behavior.
2. **Sparse SAE atom #612 (reacher only)**: the mirror image — ablating it
   **does** move genuine `E^state` beyond the null (z=+2.6, 98th pct) while
   leaving the probe readout untouched (|z|<1), and removes its own
   incremental-R² advantage (+0.0038→+0.0004) on ablation. Confirmed (Task 2)
   to be the same atom across correlational and causal testing.

These are not conflicting results about one object — they are two different,
independently-verified claims about two different, causally-independent
directions in `h_t`-space (the atom carries negligible weight in `v`'s own
reconstruction, on every task).

**Files:**
`outputs/deliverables/task_5_causal_mechanism_synthesis.md`.

---

## Task 6 — Path B (real actor-critic gated perception)

**Status:** STARTED (venue question resolved: "build it properly," full
rigor, all 3 tasks/full seed counts), IN PROGRESS, PAUSED on a real finding
before the gated-perception evaluation could be trusted.

**What was built:** `src/model/actor_critic.py` (Actor/Critic networks,
lambda-return, imagine_rollout, a differentiable torch port of Phase 1's
`reward_from_decoded_obs`), `run_task6_actor_critic_training.py` (DreamerV3-
style alternating loop: collect real episodes under the current actor →
fine-tune the world model on the growing replay buffer → train actor/critic
via imagination through the updated model — a real scope expansion from
"reuse the frozen world model as-is," made necessary after pure-imagination
training against the truly-frozen model showed zero learning: cartpole's
frozen model was trained on uniform-random data that empirically never gets
anywhere near the swingup goal, confirmed directly, max cos_pole ≈ −0.72 over
500 random real steps vs. +1.0 = upright).

**Pendulum needed reward shaping.** Its exact reward (orientation cos >
0.9903) was found in 0/500 random real steps and 0/2000 in a larger check —
essentially zero gradient signal for pure-imagination training. Added a
dense training-only proxy reward (`(orientation_zz+1)/2`, same rescaling
style as cartpole's existing proxy) for pendulum's actor-critic training
only; real reward stays unshaped for all evaluation. Confirmed via real-env
eval: the shaped-trained actor reaches orientation_zz up to 0.99999 (genuine
swing to vertical) but doesn't yet stabilize there long enough to trigger the
tight exact-reward window — a specific, real, reportable finding (learns to
swing up, not yet to balance), not a training failure.

**All 13 policies trained** (cartpole ×5, reacher ×4, pendulum ×4), each via
12 alternating loops, real reward tracked throughout. Final real rewards:
cartpole 0.14–0.20, reacher 0.08–0.24 (genuine multi-fold improvement over
random baselines), pendulum's exact reward stayed near-zero (0.0001–0.0038)
consistent with the swing-up-not-balance finding above.

**Actor collapse found, not yet resolved.** Building the gated-perception
evaluation harness surfaced a serious concern: real-env action traces show
**10 of 13 policies (all cartpole, all pendulum, 1/4 reacher) collapsed to a
near-constant, single-direction saturated action** (std ≈0.05, pinned near
±1.0 on one action dimension) — confirmed across full trajectories, not just
at initialization. Reward still improved during training because a
near-constant hard push is a real, partially effective swing-up strategy for
these underactuated tasks — **this may not be a pathology**: cartpole/
pendulum swingup are exactly the class of problem where time-optimal control
theory (Pontryagin's maximum principle) predicts bang-bang-like solutions are
genuinely close to optimal, not just a degenerate local minimum. Reacher, by
contrast, is target-dependent and multi-dimensional — no fixed action solves
it regardless of target location — and 3/4 reacher seeds retained genuine
closed-loop action diversity (std≈0.998, using both action dimensions).

**Two competing explanations, not yet distinguished:** (1) genuine entropy-
collapse pathology leaving real reward on the table, fixable with stronger/
adaptive entropy regularization; (2) the collapsed policies are already
close to these tasks' actual near-bang-bang optimum, and more entropy would
only add noise, not capability. **The recommended cheap diagnostic — compare
achieved reward against known DreamerV3/comparable benchmark performance on
cartpole-swingup and pendulum-swingup, no new training needed — was proposed
but not yet run** when work paused here to address a scope/sequencing
question and produce Task 8 instead. Whichever explanation holds, reacher's
3 non-collapsed seeds remain valid for gated-perception evaluation
regardless, and were not blocked by this finding — evaluating them was not
resumed in this pass either, pending the explicit decision to prioritize
Task 8 first (see below).

**Files:** `src/model/actor_critic.py`, `run_task6_actor_critic_training.py`,
`run_task6_one.py`, `run_task6_gated_perception_eval.py` (harness built,
found the collapse before producing a trustworthy full-batch result — one
real bug already fixed in it: `make_env` was called with an invalid keyword
argument, caught before any reported numbers depended on it),
`outputs/task6_actor_critic/{cartpole,reacher,pendulum}/seed*/actor_critic.pt`
(13 trained policies, all retained — none discarded).

**Decision, per explicit instruction:** the diagnostic and any retrain
decision are deferred until after the diagnostic actually runs — not decided
in this pass either way. Task 8 (below) was prioritized instead, as agreed,
since it's cheap, independent, and provides a fallback utility result
regardless of how Path B resolves.

---

## Task 8 — Error-Reduction Adaptive Reliance (distinct from Task 6)

**Status:** Complete. Explicitly a separate task from Task 6/Path B, not a
redefinition of it — Task 6's sunk work (13 trained policies, the collapse
finding) is retained and untouched by this task.

**The question:** at a fixed real-observation query budget, does selecting
states using `C_t` reduce actual accumulated future imagination error
(`E^state`, Phase 1's own metric) more than KL/Recon/EMARecon/ensemble
disagreement? Distinct from Task 4a's routing result, which scored recall
against a KL-*derived* label — this scores genuine error reduction against
real ground truth.

**Built:** `run_task8_error_reduction.py`, reusing Phase 1's trajectory
collection and `imagined_vs_real_obs` entirely — no new training. 6,000
sites/task, 50/50 calibration/evaluation split (calibration-only threshold
selection, no evaluation-split leakage), 5 query budgets (5–50%).

**A causal-availability bug caught by the user, before this result was
trusted — reverses the entire finding.** The first version computed
KL/Recon/EMARecon (and even `C_t`'s own lag-0 term) from the **same-step**
observation, exactly the tautology Task 4a's own write-up warns against
("using same-step KL_t would be tautological, since it is the exact
variable that defines the label"). This gave the baselines unrestricted
access to information `C_t` never had in the same way — not a fair,
matched-information comparison. The symptom that gave it away: the buggy
run found KL beats `C_t` on reacher, the *opposite* of Task 4a's own
correctly-lagged finding on that identical task. The user flagged this
discrepancy specifically and asked whether the signals had been lagged —
they hadn't. Fixed by lagging every signal to `t-1`
(`prior_step_value`, matching `run_task_r_kl_routing.py`'s established
`prior_step_kl` pattern exactly); `C_t` recomputed as `C_{t-1}` from a
prior-step-only KL series, not a further lag of the already-computed
same-step `C_t` array (which would double-lag it).

**A second, independent bug found while re-verifying the fix:** `C_t`
(now `C_{t-1}`) has a large exact plateau at its maximum value — verified
**52% of cartpole sites and 85% of pendulum sites** tied at the single max
(reacher: 0.08%, negligible). A strict `>` threshold comparison excluded
every tied site when the calibration cutoff landed on that plateau. Fixed
by changing to `>=`, applied identically to every signal (confirmed this
cannot introduce a new asymmetry — no signal-specific logic, purely whether
an exact tie at the boundary is included).

**Corrected result — `C_t` now WINS on all three tasks, but by two
different mechanisms, not a uniform result — and on further check, the
cartpole/pendulum "win" needed a real correction, not just a caveat.**
Reacher (no plateau): clean, fine-grained, budget-scaling win at every
budget 5–50%, now *agreeing with* Task 4a's reacher result rather than
contradicting it.

**Follow-on check (user-requested) found the cartpole/pendulum "win" is not
what it first looked like.** Two things were checked: (1) how ties within
`C_t`'s plateau are actually broken, and (2) whether the apparent pendulum
win is reconcilable with Task 4a's pendulum result. **Tie-breaking:** the
selection is a single boolean mask (`signal_eval >= thresh`) with no sort or
secondary key — every tied site is included unconditionally. Verified
directly that on cartpole, the calibration split's 50th/70th/80th/90th/95th
percentiles of `C_{t-1}` are ALL exactly equal to its maximum (the 52%
plateau swallows every one of these cutoffs); pendulum's 85% plateau does
the same. **This means `C_t`'s "5% budget" and "50% budget" conditions are
not different queries on these two tasks — the threshold is bit-for-bit
identical at every nominal budget from 5–50%, so `C_t` silently spends its
real ~52%/~85% budget regardless of what was nominally requested.** Only
cartpole's 50% row is a genuinely budget-matched comparison (where `C_t`
does win, narrowly: +0.3351 vs Recon's +0.3296); every other cartpole row
and **every pendulum row** compares `C_t`'s inflated real spend against
baselines' correctly-matched smaller spend — not a fair comparison, and not
evidence `C_t` "wins" at those nominal budgets.

**Task 4a/Task 8 pendulum reconciliation: checked, and it turns out the two
results were never comparable in the first place** — not the informative
recall-vs-error-reduction dissociation the reconciliation task anticipated.
Two independent problems, both confirmed by reading the actual code/data:
(1) Task 4a's −0.030 pendulum figure (3/3 seeds) is for **Probe-A**, not
`C_t` — `run_task_r_kl_routing.py` has a separate `ct_direct` router
(ridge-regressed from `h_t`) but it was never included in the multiseed
comparison that produced −0.030; the single-split `ct_direct` number that
does exist (0.744 vs. `kl_prior`'s 0.808) is directionally consistent but
was never seed-replicated. (2) Task 4a and Task 8 use entirely different
site pools for pendulum — Task 4a uses the original 100K-step
`training_states.npz` split 60/40 by timestep; Task 8 uses freshly-collected
held-out evaluation episodes via `collect_trajectories`. Given both (1) and
(2), and pendulum's own budget-matching failure above, **no comparison
between the two tasks' pendulum results is possible without a fresh,
matched re-run** — not attempted here, flagged as the concrete next step if
this specific comparison is ever needed. Pendulum should not be cited as a
`C_t`-wins finding in Task 8 regardless of the Task 4a question, independent
of this cross-task issue.

**This does not contradict Gate 1** (C_t's incremental regression
information is unaffected either way). The corrected, precise verdict:
`C_t` genuinely wins on reacher (matched budget, fine-grained); wins on
cartpole only at the one matched budget (50%); has no supported win claim on
pendulum at any tested budget.

**Files:** `run_task8_error_reduction.py`,
`outputs/task8_error_reduction/task8_results.json`,
`outputs/deliverables/task_8_error_reduction.md`.

---

## Synthesis precision fix — Task 5's causal-mechanism document

**Status:** Complete. Documentation-only, no new compute.

**What changed:** the bottom-line "the representation's causal role
bifurcates, independently confirmed twice" framing treated both legs of the
finding at the same confidence level, which is inaccurate. Split into:
**Leg A (dense-direction dissociation)** — kept at full strength, no hedge —
rests on Phase 6's direct steering and Task 3's POMDP seed sweep, neither of
which depends on the SAE pipeline or any SAE-derived number. **Leg B
(sparse-atom mirror image)** — reacher's atom #612 stays fully confirmed;
cartpole's atom #156 (currently the strongest single causal result in the
project) is now explicitly flagged **provisional**, pending independent
verification of its incremental-R² figure (suspected copy/reuse issue) and
the SAE pipeline's own reproducibility — neither of which this session has
independently verified or found; the hedge was applied on instruction, not
after confirming the underlying bug myself.

**Files updated:** `outputs/deliverables/task_5_causal_mechanism_synthesis.md`
(bottom-line statement, the "both halves" sentence, the "What remains open"
item), `outputs/deliverables/phase3_sae_decomposition.md` (header verdict,
the atom-156 subsection's "What this changes" note),
`GAP_CLOSING_FINAL_SUMMARY.md` (Section 5).

**Open item this creates:** §11.1 (the suspected #156 incremental-R² copy/
reuse bug) and §11.2 (SAE pipeline reproducibility) have not actually been
investigated yet — the hedge is applied precautionarily, per instruction,
not because the bug was found and confirmed. Investigating and resolving
§11.1/§11.2 themselves remains open.

---

## Diagnostic — Sign of the Phase 3 causal ablation effect on E^state

**Status:** Complete. Near-zero cost — pure re-read of an existing output
file (`outputs/phase3_sae/phase3b_causal_results.json`), no new computation.

**What was done:** the file already stored `ablation_effect.d_e_state` (the
signed mean Δ`E^state`, ablated − original) at the time of the original Phase
3 run, but the deliverable's prose only ever reported the z-score magnitude,
never the sign. Extracted and surfaced it.

**Results:**

| Task | Atom | Signed mean Δ`E^state` | z (vs null) | Percentile |
|---|---:|---:|---:|---:|
| cartpole | #139 | **+0.0449** | +3.76 | 98% |
| reacher | #612 | **+0.0304** | +2.62 | 98% |
| pendulum | #310 | +0.0055 | −0.17 | 50% |

**Finding:** the sign is positive and unambiguous on both tasks where the
effect clears the null — **ablating the atom makes `E^state` larger, i.e.
imagination gets *worse*, not better**, on cartpole and reacher. Pendulum
shows no reliable effect either way.

**Why this matters, stated for the record:** this closes off "suppress this
atom as a correction mechanism" as a dead end, not an open question — the
atom is load-bearing information the model needs, consistent with (not
contradicting) the workshop paper's own Appendix E finding that two earlier
direct-correction attempts (down-weighting, early stopping) both failed. A
future correction mechanism would need a different design premise. Per the
task's explicit scope boundary, no correction mechanism was designed and no
decision about pursuing one was made here — this task's only job was to
surface the sign, which is now done.

**Files updated:** `outputs/deliverables/phase3_sae_decomposition.md` (new
"Diagnostic addendum: the sign of the d(E^state) causal effect" subsection
after §9.5–9.6).

---

## Task 7 — Confusion-gated residual correction of the transition output

**Status:** SPECCED (revised once), NOT STARTED — correctly gated on Task 6,
which has not itself started. No design or implementation work has begun;
this entry exists only to record the current spec for continuity.

**Revision history:** originally specced as correcting `h_t` itself before
rollout (`h_t' = h_t + f_θ(a_i, h_t)`, evaluated against an unconstrained
mean-regression baseline). **Superseded** by a revised design that instead
corrects the transition function's *output* `h_{t+1}`, gated on the confusion
atom read from `h_t`, and adds an explicit relative-norm magnitude budget
(`‖g_θ(...)‖ ≤ ε·‖h_{t+1}_raw‖`) on the correction itself. The budget is a
real methodological improvement, not cosmetic: it directly narrows the room
for the mean-regression-sandbagging failure mode, and forces the baseline
comparison to be magnitude-matched (baseline 3 is now a population-mean shift
capped at the *same* ε as the learned correction, not an unconstrained shift)
— without that matching, an unbudgeted baseline could look artificially
weak or artificially strong relative to the learned correction for reasons
having nothing to do with whether the correction is doing real work.

**What it is now (for continuity):** given the diagnostic finding above
(ablating the atom makes `E^state` worse on cartpole/reacher, so the atom is
load-bearing, not a removable flaw), Task 7 asks whether a small,
norm-budgeted, confusion-gated correction added to the transition's output
can recover real predictive information beyond what a magnitude-matched
trivial shift achieves. Requires: calibration/evaluation split discipline,
ε swept across 2-3 values (selected on calibration data) to show the finding
isn't an artifact of one budget choice, three mandatory magnitude-matched
conditions (no-correction / learned `g_θ` / magnitude-matched mean-regression
baseline), multi-seed where existing replication allows, and per-task (not
averaged) honest reporting of one of three explicit outcomes (beats the
matched baseline / matches it / underperforms no-correction).

**Sequencing, as specified:** do not start until Task 6 (Path B) is complete.
Same priority tier as other gated stretch items, not urgent.

---

## Open items carried forward (not part of this spec's acceptance criteria, surfaced along the way)

1. **Unseeded-RNG bug in Phases 2/3/6 and their addenda** (see Task 1) —
   awaiting a decision on whether to re-verify.
2. **Cartpole's SAE causal test uses a retracted atom (#139, not the
   split-sample-confirmed #156)** (see Task 2) — flagged, not fixed, since it
   was outside Task 2's reacher-scoped check.
3. **`Reframed_AAMAS2027_Project.md` lives only in `~/Downloads`, not tracked
   in this git repo** (surfaced during an earlier reconciliation pass) — still
   true, still worth considering moving into the repo if it's meant to be
   canonical.
